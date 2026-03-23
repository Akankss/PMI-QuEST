"""
experiments/run_bestsd_comparison.py
==========================
Head-to-head comparison on BEST-STD tokens:

  BEST-STD tokens + BEST-STD retrieval (inverted index)
  BEST-STD tokens + H-QuEST            (TF-IDF + HNSW + SW)
  BEST-STD tokens + PMI-QuEST          (PMI-TF-IDF + HNSW + SW)

All three systems use IDENTICAL input tokens so the only variable
is the retrieval mechanism. This directly answers:
"Is PMI-QuEST a better retrieval method than BEST-STD's own system
 when given the same tokens?"

Usage
-----
python run_bestsd_vs_pmiquest.py \\
    --corpus_csv  tokenised/best_std/corpus.csv \\
    --query_csv   tokenised/best_std/queries.csv \\
    --relevance   qbe_librispeech/metadata/relevance.json \\
    --k 512
"""

import argparse, csv, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    seqs = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            stem = Path(row['Filename']).stem
            toks = [int(x) for x in row['Data'].split(',') if x.strip()]
            if stem and toks:
                seqs[stem] = toks
    return seqs


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def average_precision(ranked, rel_set):
    if not rel_set:
        return 0.0
    hits = ap = 0.0
    for rank, did in enumerate(ranked, 1):
        if did in rel_set:
            hits += 1
            ap += hits / rank
    return ap / len(rel_set)


def precision_at_k(ranked, rel_set, k):
    return sum(1 for d in ranked[:k] if d in rel_set) / k


def mrr(ranked, rel_set):
    for rank, did in enumerate(ranked, 1):
        if did in rel_set:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SW reranking (shared by all methods)
# ─────────────────────────────────────────────────────────────────────────────

def sw_score(q, d, match=2.0, mismatch=-1.0, gap=-2.0):
    m, n = len(q), len(d)
    if m == 0 or n == 0:
        return 0.0
    H = np.zeros((m + 1, n + 1), dtype=np.float32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = match if q[i-1] == d[j-1] else mismatch
            H[i, j] = max(0,
                          H[i-1, j-1] + s,
                          H[i-1, j] + gap,
                          H[i, j-1] + gap)
    return float(H.max()) / m


def sw_rerank(q_seq, candidate_ids, corpus, top_k=50):
    top = candidate_ids[:top_k]
    rest = candidate_ids[top_k:]
    scored = sorted(
        [(sw_score(q_seq, corpus[did]), did) for did in top],
        key=lambda x: -x[0]
    )
    return [did for _, did in scored] + rest


# ─────────────────────────────────────────────────────────────────────────────
# System 1: BEST-STD retrieval (inverted index)
# ─────────────────────────────────────────────────────────────────────────────

def build_inverted_index(corpus):
    """
    Builds two indexes:
      unigram: token → {doc_id: count}
      bigram:  (a,b) → {doc_id: count}
    """
    uni = defaultdict(lambda: defaultdict(int))
    bi  = defaultdict(lambda: defaultdict(int))
    for doc_id, seq in corpus.items():
        for t in seq:
            uni[t][doc_id] += 1
        for a, b in zip(seq, seq[1:]):
            bi[(a, b)][doc_id] += 1
    return uni, bi


def best_std_retrieve(q_seq, uni_idx, bi_idx, corpus, top_k=50):
    """
    BEST-STD-style retrieval:
      1. Score by bigram match count (ordered token pairs)
      2. Fall back to unigram if no bigram matches
      3. SW rerank top-K
    This captures the ordered sequence matching that is BEST-STD's
    core contribution over plain TF-IDF.
    """
    scores = defaultdict(float)
    q_bgs = list(zip(q_seq, q_seq[1:]))

    if q_bgs:
        for bg in q_bgs:
            if bg in bi_idx:
                for doc_id, cnt in bi_idx[bg].items():
                    scores[doc_id] += cnt / len(q_bgs)

    # If no bigram matches, fall back to unigram
    if not scores:
        for t in q_seq:
            if t in uni_idx:
                for doc_id, cnt in uni_idx[t].items():
                    scores[doc_id] += cnt / len(q_seq)

    ranked = [did for did, _ in
              sorted(scores.items(), key=lambda x: -x[1])]

    # Add unmatched docs at the end
    matched = set(ranked)
    unmatched = [did for did in corpus if did not in matched]
    ranked += unmatched

    # SW rerank
    return sw_rerank(q_seq, ranked, corpus, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# System 2: H-QuEST (TF-IDF + cosine + SW)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf(corpus_list, k):
    N = len(corpus_list)
    df = np.zeros(k, dtype=np.float32)
    for seq in corpus_list:
        for t in set(seq):
            df[t] += 1
    idf = np.log(N / (1.0 + df)).astype(np.float32)

    mat = lil_matrix((N, k), dtype=np.float32)
    for i, seq in enumerate(corpus_list):
        L = len(seq)
        if L == 0:
            continue
        counts = defaultdict(int)
        for t in seq:
            counts[t] += 1
        for t, c in counts.items():
            mat[i, t] = (c / L) * idf[t]

    mat = csr_matrix(mat)
    norms = np.sqrt(np.array(mat.power(2).sum(axis=1))).ravel()
    norms[norms == 0] = 1.0
    mat = csr_matrix(mat.multiply(1.0 / norms[:, None]))
    return mat, idf


def tfidf_vec(seq, idf, k):
    L = len(seq)
    if L == 0:
        return np.zeros(k, dtype=np.float32)
    v = np.zeros(k, dtype=np.float32)
    counts = defaultdict(int)
    for t in seq:
        counts[t] += 1
    for t, c in counts.items():
        v[t] = (c / L) * idf[t]
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v


def hquest_retrieve(q_seq, corpus_mat, idf, k, corpus_ids, corpus, top_k=50):
    q_vec = tfidf_vec(q_seq, idf, k)
    sims  = corpus_mat.toarray() @ q_vec
    ranked = [corpus_ids[i] for i in np.argsort(-sims)]
    return sw_rerank(q_seq, ranked, corpus, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# System 3: PMI-QuEST (PMI-TF-IDF + cosine + SW)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pmi(corpus_list, k):
    uni_count = np.zeros(k, dtype=np.float64)
    bi_count  = np.zeros((k, k), dtype=np.float64)
    total_uni = total_bi = 0
    for seq in corpus_list:
        for t in seq:
            uni_count[t] += 1
        total_uni += len(seq)
        for a, b in zip(seq, seq[1:]):
            bi_count[a, b] += 1
        total_bi += max(len(seq) - 1, 0)
    p_uni = uni_count / max(total_uni, 1)
    p_bi  = bi_count  / max(total_bi,  1)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.outer(p_uni, p_uni)
        pmi   = np.where(denom > 0,
                         np.log(np.where(denom > 0, p_bi / denom, 1.0)),
                         0.0)
    return pmi


def build_pmi_tfidf(corpus_list, corpus_ids, k, tau=0.5, alpha=0.5):
    pmi = compute_pmi(corpus_list, k)
    bigram_set = set(zip(*np.where(pmi > tau)))
    bigram_list = sorted(bigram_set)
    big_to_idx  = {b: i for i, b in enumerate(bigram_list)}
    F = k + len(bigram_list)
    print(f"  PMI bigrams at τ={tau}: {len(bigram_list)}")

    N = len(corpus_list)
    df_uni = np.zeros(k, dtype=np.float32)
    df_big = np.zeros((k, k), dtype=np.float32)
    for seq in corpus_list:
        for t in set(seq):
            df_uni[t] += 1
        for a, b in set(zip(seq, seq[1:])):
            if (a, b) in big_to_idx:
                df_big[a, b] += 1
    idf_uni = np.log(N / (1.0 + df_uni)).astype(np.float32)
    idf_big = np.log(N / (1.0 + df_big)).astype(np.float32)

    mat = lil_matrix((N, F), dtype=np.float32)
    for i, seq in enumerate(corpus_list):
        L = len(seq)
        if L == 0:
            continue
        uc = defaultdict(int)
        for t in seq:
            uc[t] += 1
        for t, c in uc.items():
            mat[i, t] = (c / L) * idf_uni[t]
        if L > 1:
            bc = defaultdict(int)
            for a, b in zip(seq, seq[1:]):
                if (a, b) in big_to_idx:
                    bc[(a, b)] += 1
            for (a, b), c in bc.items():
                mat[i, k + big_to_idx[(a, b)]] = (
                    alpha * (c / (L - 1)) * idf_big[a, b])

    mat = csr_matrix(mat)
    norms = np.sqrt(np.array(mat.power(2).sum(axis=1))).ravel()
    norms[norms == 0] = 1.0
    mat = csr_matrix(mat.multiply(1.0 / norms[:, None]))
    return mat, idf_uni, idf_big, bigram_list, big_to_idx, F


def pmi_vec(seq, idf_uni, idf_big, k, alpha, big_to_idx, F):
    v = np.zeros(F, dtype=np.float32)
    L = len(seq)
    if L == 0:
        return v
    uc = defaultdict(int)
    for t in seq:
        uc[t] += 1
    for t, c in uc.items():
        v[t] = (c / L) * idf_uni[t]
    if L > 1:
        bc = defaultdict(int)
        for a, b in zip(seq, seq[1:]):
            if (a, b) in big_to_idx:
                bc[(a, b)] += 1
        for (a, b), c in bc.items():
            v[k + big_to_idx[(a, b)]] = (
                alpha * (c / (L - 1)) * idf_big[a, b])
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v


def pmi_retrieve(q_seq, mat, idf_uni, idf_big, k, alpha,
                 big_to_idx, F, corpus_ids, corpus, top_k=50):
    q_vec = pmi_vec(q_seq, idf_uni, idf_big, k, alpha, big_to_idx, F)
    sims  = mat.toarray() @ q_vec
    ranked = [corpus_ids[i] for i in np.argsort(-sims)]
    return sw_rerank(q_seq, ranked, corpus, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# Run one system
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(label, ranked_fn, queries, relevance):
    aps, p1s, p5s, p10s, mrrs = [], [], [], [], []
    t0 = time.time()
    for qid, q_seq in queries.items():
        if qid not in relevance or not relevance[qid]:
            continue
        rel = set(relevance[qid])
        ranked = ranked_fn(q_seq)
        aps.append(average_precision(ranked, rel))
        p1s.append(precision_at_k(ranked, rel, 1))
        p5s.append(precision_at_k(ranked, rel, 5))
        p10s.append(precision_at_k(ranked, rel, 10))
        mrrs.append(mrr(ranked, rel))

    r = dict(label=label,
             MAP=round(float(np.mean(aps)),  4),
             MRR=round(float(np.mean(mrrs)), 4),
             P1=round(float(np.mean(p1s)),   4),
             P5=round(float(np.mean(p5s)),   4),
             P10=round(float(np.mean(p10s)), 4),
             N=len(aps), time=round(time.time()-t0, 1))
    print(f"\n  [{label}]")
    print(f"    MAP={r['MAP']:.4f}  MRR={r['MRR']:.4f}  "
          f"P@1={r['P1']:.4f}  P@5={r['P5']:.4f}  [{r['time']}s]")
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus_csv', required=True)
    p.add_argument('--query_csv',  required=True)
    p.add_argument('--relevance',  required=True)
    p.add_argument('--k',    type=int,   default=512)
    p.add_argument('--tau',  type=float, default=0.5)
    p.add_argument('--alpha',type=float, default=0.5)
    p.add_argument('--top_k',type=int,   default=50)
    args = p.parse_args()

    corpus    = load_csv(args.corpus_csv)
    queries   = load_csv(args.query_csv)
    relevance = load_json(args.relevance)

    corpus_list = list(corpus.values())
    corpus_ids  = list(corpus.keys())
    k = args.k

    print("=" * 65)
    print("BEST-STD tokens: their retrieval vs PMI-QuEST retrieval")
    print(f"  N={len(corpus_list)}  Q={len(queries)}  k={k}")
    print("=" * 65)

    # ── Build all indexes ────────────────────────────────────────────────
    print("\nBuilding indexes...")
    uni_idx, bi_idx = build_inverted_index(corpus)
    tfidf_mat, idf  = build_tfidf(corpus_list, k)
    pmi_mat, idf_u, idf_b, bg_list, bg_idx, F = build_pmi_tfidf(
        corpus_list, corpus_ids, k, args.tau, args.alpha)

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\nRunning systems...")

    results = []

    results.append(evaluate(
        f"BEST-STD retrieval (bigram index + SW)",
        lambda q: best_std_retrieve(q, uni_idx, bi_idx, corpus, args.top_k),
        queries, relevance))

    results.append(evaluate(
        f"H-QuEST (TF-IDF + SW)",
        lambda q: hquest_retrieve(q, tfidf_mat, idf, k,
                                   corpus_ids, corpus, args.top_k),
        queries, relevance))

    results.append(evaluate(
        f"PMI-QuEST (τ={args.tau}, PMI-TF-IDF + SW)",
        lambda q: pmi_retrieve(q, pmi_mat, idf_u, idf_b, k, args.alpha,
                                bg_idx, F, corpus_ids, corpus, args.top_k),
        queries, relevance))

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY — BEST-STD tokens, three retrieval methods")
    print("=" * 65)
    print(f"  {'System':<42} {'MAP':>7} {'MRR':>7} {'P@1':>7}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['label']:<42} {r['MAP']:>7.4f} "
              f"{r['MRR']:>7.4f} {r['P1']:>7.4f}")

    print()
    print("LaTeX rows:")
    for r in results:
        print(f"  {r['label']:<45} & {r['MAP']:.4f} & {r['MRR']:.4f} "
              f"& {r['P1']:.4f} & {r['P5']:.4f} & {r['P10']:.4f} \\\\")


if __name__ == '__main__':
    main()
