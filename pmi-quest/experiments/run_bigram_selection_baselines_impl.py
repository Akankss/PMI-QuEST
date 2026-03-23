"""
run_bigram_selection_baselines.py
==================================
Compares three bigram selection criteria on LibriSpeech word mode:

  1. PMI selection     — keep bigrams with PMI(a,b) > tau  (proposed)
  2. Frequency selection — keep the top-N most frequent bigrams (BPE-style)
  3. Random selection  — keep N randomly chosen bigrams (control)

All three use the same HNSW + SW pipeline as PMI-QuEST.
N is set to match PMI at tau=0.5 (~1270) and tau=1.5 (~700) so feature
counts are directly comparable.

Usage
-----
python run_bigram_selection_baselines.py \\
    --corpus_csv   tokenised/wav2vec2-base_l0_k100/corpus.csv \\
    --query_csv    tokenised/wav2vec2-base_l0_k100/queries.csv \\
    --relevance    qbe_librispeech/metadata/relevance.json \\
    --n_bigrams    700 1270 \\
    --random_seeds 42 123 456
"""

import argparse, csv, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    seqs = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            fname = row.get('Filename') or ''
            data  = row.get('Data')     or ''
            stem  = Path(fname).stem
            toks  = [int(x) for x in data.strip().split(',') if x.strip()]
            if stem and toks:
                seqs[stem] = toks
    return seqs


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Bigram statistics from corpus
# ─────────────────────────────────────────────────────────────────────────────

def compute_bigram_stats(corpus_seqs, k=100):
    """Return unigram counts, bigram counts, and PMI for all (a,b) pairs."""
    uni_count = np.zeros(k,      dtype=np.float64)
    big_count = np.zeros((k, k), dtype=np.float64)
    total_uni = 0
    total_big = 0

    for seq in corpus_seqs:
        for t in seq:
            uni_count[t] += 1
        total_uni += len(seq)
        for a, b in zip(seq, seq[1:]):
            big_count[a, b] += 1
        total_big += max(len(seq) - 1, 0)

    p_uni = uni_count / max(total_uni, 1)
    p_big = big_count / max(total_big, 1)

    # PMI(a,b) = log p(a,b) / (p(a)*p(b))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.outer(p_uni, p_uni)
        pmi   = np.where(denom > 0, np.log(np.where(denom > 0, p_big / denom, 1)), 0.0)

    return uni_count, big_count, p_uni, p_big, pmi


def select_pmi(pmi, tau):
    """Return set of (a,b) pairs with PMI > tau."""
    k = pmi.shape[0]
    return set(zip(*np.where(pmi > tau)))


def select_frequency(big_count, n):
    """Return top-N bigrams by raw corpus frequency."""
    k = big_count.shape[0]
    pairs = [(big_count[a, b], a, b) for a in range(k) for b in range(k)
             if big_count[a, b] > 0]
    pairs.sort(reverse=True)
    return {(a, b) for _, a, b in pairs[:n]}


def select_random(big_count, n, seed):
    """Return N randomly chosen bigrams that appear at least once."""
    rng = np.random.default_rng(seed)
    k   = big_count.shape[0]
    candidates = [(a, b) for a in range(k) for b in range(k)
                  if big_count[a, b] > 0]
    if len(candidates) <= n:
        return set(candidates)
    chosen = rng.choice(len(candidates), size=n, replace=False)
    return {candidates[i] for i in chosen}


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_matrix(corpus_seqs, bigram_set, uni_idf, big_idf,
                       k=100, alpha=0.5):
    """
    Build (N, k+|B|) TF-IDF matrix for a given bigram set.
    bigram_set: set of (a,b) tuples to include as features.
    Returns: csr_matrix (L2-normalised), bigram_list (ordered index).
    """
    bigram_list = sorted(bigram_set)
    big_to_idx  = {b: i for i, b in enumerate(bigram_list)}
    N = len(corpus_seqs)
    F = k + len(bigram_list)

    mat = lil_matrix((N, F), dtype=np.float32)

    for doc_i, seq in enumerate(corpus_seqs):
        L = len(seq)
        if L == 0:
            continue
        # unigrams
        uc = defaultdict(int)
        for t in seq:
            uc[t] += 1
        for t, c in uc.items():
            mat[doc_i, t] = (c / L) * uni_idf[t]
        # bigrams
        if L > 1 and bigram_list:
            bc = defaultdict(int)
            for a, b in zip(seq, seq[1:]):
                if (a, b) in big_to_idx:
                    bc[(a, b)] += 1
            for (a, b), c in bc.items():
                mat[doc_i, k + big_to_idx[(a, b)]] = (
                    alpha * (c / (L - 1)) * big_idf[a, b])

    mat = csr_matrix(mat)
    norms = np.sqrt(np.array(mat.power(2).sum(axis=1))).ravel()
    norms[norms == 0] = 1.0
    mat = csr_matrix(mat.multiply(1.0 / norms[:, None]))
    return mat, bigram_list, big_to_idx


def query_vec(q_seq, bigram_set, uni_idf, big_idf,
              k=100, alpha=0.5, big_to_idx=None):
    """Build query TF-IDF vector matching the corpus matrix layout."""
    L   = len(q_seq)
    F   = k + len(bigram_set)
    vec = np.zeros(F, dtype=np.float32)
    # unigrams
    uc = defaultdict(int)
    for t in q_seq:
        uc[t] += 1
    for t, c in uc.items():
        vec[t] = (c / L) * uni_idf[t]
    # bigrams
    if L > 1 and bigram_set and big_to_idx:
        bc = defaultdict(int)
        for a, b in zip(q_seq, q_seq[1:]):
            if (a, b) in big_to_idx:
                bc[(a, b)] += 1
        for (a, b), c in bc.items():
            vec[k + big_to_idx[(a, b)]] = (
                alpha * (c / (L - 1)) * big_idf[a, b])
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# SW reranking
# ─────────────────────────────────────────────────────────────────────────────

def sw_score(q, d, match=2.0, mismatch=-1.0, gap=-2.0):
    m, n = len(q), len(d)
    if m == 0 or n == 0:
        return 0.0
    H = np.zeros((m + 1, n + 1), dtype=np.float32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = match if q[i-1] == d[j-1] else mismatch
            H[i, j] = max(0, H[i-1,j-1]+s, H[i-1,j]+gap, H[i,j-1]+gap)
    return float(H.max()) / m


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one bigram selection
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_selection(label, bigram_set, corpus_seqs, corpus_ids,
                       queries, relevance, uni_idf, big_idf,
                       k=100, alpha=0.5, hnsw_k=50):
    print(f"\n  [{label}]  |B|={len(bigram_set)}  F={k+len(bigram_set)}")
    t0 = time.time()

    mat, bigram_list, big_to_idx = build_tfidf_matrix(
        corpus_seqs, bigram_set, uni_idf, big_idf, k, alpha)
    dense = mat.toarray()

    aps, p1s, p5s, p10s = [], [], [], []

    for qid, q_seq in queries.items():
        if qid not in relevance or not relevance[qid]:
            continue
        rel_set = set(relevance[qid])

        q_vec = query_vec(q_seq, bigram_set, uni_idf, big_idf,
                          k, alpha, big_to_idx)
        sims  = dense @ q_vec
        top_k = np.argsort(-sims)[:hnsw_k].tolist()

        # SW rerank
        sw = [(sw_score(q_seq, corpus_seqs[i]), i) for i in top_k]
        sw.sort(key=lambda x: -x[0])
        top_k_set = set(top_k)

        remaining = [(float(sims[i]), i)
                     for i in np.argsort(-sims) if i not in top_k_set]
        ranked_ids = [corpus_ids[i] for _, i in sw] + \
                     [corpus_ids[i] for _, i in remaining]

        # AP
        hits = ap = 0.0
        for rank, did in enumerate(ranked_ids, 1):
            if did in rel_set:
                hits += 1; ap += hits / rank
        aps.append(ap / len(rel_set))
        p1s.append(1.0 if ranked_ids and ranked_ids[0] in rel_set else 0.0)
        p5s.append(sum(1 for d in ranked_ids[:5]  if d in rel_set) / 5)
        p10s.append(sum(1 for d in ranked_ids[:10] if d in rel_set) / 10)

    result = {
        'label':  label,
        'n_bigrams': len(bigram_set),
        'MAP':  round(float(np.mean(aps)),  4),
        'P@1':  round(float(np.mean(p1s)),  4),
        'P@5':  round(float(np.mean(p5s)),  4),
        'P@10': round(float(np.mean(p10s)), 4),
        'N':    len(aps),
        'time': round(time.time() - t0, 1),
    }
    print(f"    MAP={result['MAP']:.4f}  P@1={result['P@1']:.4f}  "
          f"P@5={result['P@5']:.4f}  P@10={result['P@10']:.4f}  "
          f"[{result['time']}s]")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus_csv',   required=True)
    ap.add_argument('--query_csv',    required=True)
    ap.add_argument('--relevance',    required=True)
    ap.add_argument('--k',            type=int,   default=100)
    ap.add_argument('--alpha',        type=float, default=0.5)
    ap.add_argument('--hnsw_k',       type=int,   default=50)
    ap.add_argument('--pmi_taus',     type=float, nargs='+', default=[0.5, 1.5])
    ap.add_argument('--n_bigrams',    type=int,   nargs='+', default=[700, 1270])
    ap.add_argument('--random_seeds', type=int,   nargs='+', default=[42, 123, 456])
    args = ap.parse_args()

    corpus   = load_csv(args.corpus_csv)
    queries  = load_csv(args.query_csv)
    relevance = load_json(args.relevance)

    corpus_list = list(corpus.values())
    corpus_ids  = list(corpus.keys())
    k = args.k

    print("=" * 65)
    print("Bigram Selection Criterion Comparison")
    print(f"  N={len(corpus_list)}  Q={len(queries)}  k={k}  alpha={args.alpha}")
    print("=" * 65)

    # Compute corpus statistics
    print("\nComputing bigram statistics...")
    _, big_count, p_uni, p_big, pmi = compute_bigram_stats(corpus_list, k)

    # IDF values
    N = len(corpus_list)
    uni_df = np.zeros(k, dtype=np.float32)
    big_df = np.zeros((k, k), dtype=np.float32)
    for seq in corpus_list:
        for t in set(seq):
            uni_df[t] += 1
        for a, b in set(zip(seq, seq[1:])):
            big_df[a, b] += 1
    uni_idf = np.log(N / (1.0 + uni_df)).astype(np.float32)
    big_idf = np.log(N / (1.0 + big_df)).astype(np.float32)

    n_observed = int((big_count > 0).sum())
    print(f"  Observed bigrams: {n_observed:,} / {k*k:,} possible")

    results = []

    # ── H-QuEST reference (unigrams only) ────────────────────────────────────
    results.append(evaluate_selection(
        "H-QuEST (unigrams, no bigrams)", set(),
        corpus_list, corpus_ids, queries, relevance,
        uni_idf, big_idf, k, args.alpha, args.hnsw_k))

    # ── PMI selection ─────────────────────────────────────────────────────────
    for tau in args.pmi_taus:
        bset = select_pmi(pmi, tau)
        results.append(evaluate_selection(
            f"PMI τ={tau} ({len(bset)} bigrams)",
            bset, corpus_list, corpus_ids, queries, relevance,
            uni_idf, big_idf, k, args.alpha, args.hnsw_k))

    # ── Frequency (BPE-style) selection ──────────────────────────────────────
    for n in args.n_bigrams:
        bset = select_frequency(big_count, n)
        results.append(evaluate_selection(
            f"Frequency top-{n} (BPE-style)",
            bset, corpus_list, corpus_ids, queries, relevance,
            uni_idf, big_idf, k, args.alpha, args.hnsw_k))

    # ── Random selection (multiple seeds → mean ± std) ────────────────────────
    for n in args.n_bigrams:
        seed_results = []
        for seed in args.random_seeds:
            bset = select_random(big_count, n, seed)
            r = evaluate_selection(
                f"Random {n} bigrams (seed={seed})",
                bset, corpus_list, corpus_ids, queries, relevance,
                uni_idf, big_idf, k, args.alpha, args.hnsw_k)
            seed_results.append(r)
        # Aggregate
        maps  = [r['MAP']  for r in seed_results]
        p1s   = [r['P@1']  for r in seed_results]
        print(f"\n  Random {n} bigrams — mean over {len(args.random_seeds)} seeds:")
        print(f"    MAP={np.mean(maps):.4f}±{np.std(maps):.4f}  "
              f"P@1={np.mean(p1s):.4f}±{np.std(p1s):.4f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  {'System':<40} {'|B|':>5} {'MAP':>7} {'P@1':>7} {'P@5':>7} {'P@10':>7}")
    print("  " + "-" * 63)
    for r in results:
        print(f"  {r['label']:<40} {r['n_bigrams']:>5} "
              f"{r['MAP']:>7.4f} {r['P@1']:>7.4f} "
              f"{r['P@5']:>7.4f} {r['P@10']:>7.4f}")

    print()
    print("LaTeX rows (for ablation table):")
    for r in results:
        print(f"  {r['label']:<45} & {r['MAP']:.4f} & {r['P@1']:.4f} "
              f"& {r['P@5']:.4f} & {r['P@10']:.4f} \\\\")


if __name__ == '__main__':
    main()
