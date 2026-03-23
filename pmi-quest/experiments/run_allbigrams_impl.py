"""
run_allbigrams_eval.py
======================
Evaluates the all-bigrams TF-IDF system on LibriSpeech word mode and
reports MAP, P@1, P@5, P@10 to fill in the ablation table.

All-bigrams TF-IDF: same as PMI-QuEST but retains ALL k^2 = 10,000
bigrams (no PMI threshold filtering). Uses the same HNSW + SW pipeline
as H-QuEST so the comparison is fair.

Usage
-----
python run_allbigrams_eval.py \\
    --corpus_csv  tokenised/wav2vec2-base_l0_k100/corpus.csv \\
    --query_csv   tokenised/wav2vec2-base_l0_k100/queries.csv \\
    --relevance   qbe_librispeech/metadata/relevance.json
"""

import argparse, csv, json, sys, time
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
            fname = (row.get('Filename') or row.get('filename') or '')
            data  = (row.get('Data')     or row.get('data')     or '')
            stem  = Path(fname).stem
            tokens = [int(x) for x in data.strip().split(',') if x.strip()]
            if stem and tokens:
                seqs[stem] = tokens
    return seqs


def load_relevance(path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# All-bigrams TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

class AllBigramsTFIDF:
    """
    TF-IDF with ALL k^2 ordered bigrams as features (no PMI filtering).
    Feature vector: [unigram weights (k) || bigram weights (k^2)]
    = k + k^2 = 100 + 10,000 = 10,100 dimensions.
    HNSW + SW reranking identical to H-QuEST/PMI-QuEST.
    alpha=0.5 (same as PMI-QuEST default).
    """

    def __init__(self, k=100, hnsw_k=50, alpha=0.5,
                 sw_match=2.0, sw_mismatch=-1.0, sw_gap=-2.0):
        self.k           = k
        self.hnsw_k      = hnsw_k
        self.alpha       = alpha
        self.sw_match    = sw_match
        self.sw_mismatch = sw_mismatch
        self.sw_gap      = sw_gap

    def _unigram_idx(self, t):
        return t

    def _bigram_idx(self, a, b):
        return self.k + a * self.k + b   # offset by k unigram dims

    @property
    def n_features(self):
        return self.k + self.k * self.k   # 10,100

    def fit(self, corpus_seqs):
        self._corpus_seqs = corpus_seqs
        N = len(corpus_seqs)
        F = self.n_features

        print(f"  Building all-bigrams TF-IDF: N={N}, F={F:,}")
        t0 = time.time()

        # ── document frequency ──────────────────────────────────────────────
        unigram_df = np.zeros(self.k, dtype=np.float32)
        bigram_df  = np.zeros((self.k, self.k), dtype=np.float32)

        for seq in corpus_seqs:
            present_uni = set(seq)
            for t in present_uni:
                unigram_df[t] += 1
            present_big = set()
            for i in range(len(seq) - 1):
                present_big.add((seq[i], seq[i+1]))
            for (a, b) in present_big:
                bigram_df[a, b] += 1

        self._uni_idf = np.log(N / (1.0 + unigram_df)).astype(np.float32)
        self._big_idf = np.log(N / (1.0 + bigram_df)).astype(np.float32)

        # ── corpus matrix ───────────────────────────────────────────────────
        # Build sparse (N × F) matrix
        mat = lil_matrix((N, F), dtype=np.float32)

        for doc_idx, seq in enumerate(corpus_seqs):
            L = len(seq)
            if L == 0:
                continue
            # unigrams
            uni_counts = defaultdict(int)
            for t in seq:
                uni_counts[t] += 1
            for t, cnt in uni_counts.items():
                tf = cnt / L
                mat[doc_idx, self._unigram_idx(t)] = tf * self._uni_idf[t]
            # bigrams
            big_counts = defaultdict(int)
            for i in range(L - 1):
                big_counts[(seq[i], seq[i+1])] += 1
            for (a, b), cnt in big_counts.items():
                tf = cnt / (L - 1)
                val = self.alpha * tf * self._big_idf[a, b]
                mat[doc_idx, self._bigram_idx(a, b)] = val

        self._matrix = csr_matrix(mat)  # (N, F)

        # ── L2-normalise rows ────────────────────────────────────────────────
        norms = np.sqrt(np.array(self._matrix.power(2).sum(axis=1))).ravel()
        norms[norms == 0] = 1.0
        self._matrix = self._matrix.multiply(1.0 / norms[:, None])
        self._matrix = csr_matrix(self._matrix)

        print(f"  Matrix built in {time.time()-t0:.1f}s "
              f"({self._matrix.nnz:,} non-zeros)")

        # ── HNSW index ───────────────────────────────────────────────────────
        # Use brute-force cosine for top-K (10,100 dims is too large for
        # the lightweight HNSW we have; use dense retrieval on CPU)
        # For N=2620 this is fast enough (~0.5s/query).
        self._dense = self._matrix.toarray()   # (N, F) float32
        print(f"  Dense matrix shape: {self._dense.shape}")
        return self

    def _query_vec(self, query_seq):
        L = len(query_seq)
        vec = np.zeros(self.n_features, dtype=np.float32)
        uni_counts = defaultdict(int)
        for t in query_seq:
            uni_counts[t] += 1
        for t, cnt in uni_counts.items():
            tf = cnt / L
            vec[self._unigram_idx(t)] = tf * self._uni_idf[t]
        if L > 1:
            big_counts = defaultdict(int)
            for i in range(L - 1):
                big_counts[(query_seq[i], query_seq[i+1])] += 1
            for (a, b), cnt in big_counts.items():
                tf = cnt / (L - 1)
                vec[self._bigram_idx(a, b)] = self.alpha * tf * self._big_idf[a, b]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _sw_score(self, q, d):
        m, n = len(q), len(d)
        if m == 0 or n == 0:
            return 0.0
        H = np.zeros((m + 1, n + 1), dtype=np.float32)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = self.sw_match if q[i-1] == d[j-1] else self.sw_mismatch
                H[i, j] = max(0,
                              H[i-1, j-1] + match,
                              H[i-1, j]   + self.sw_gap,
                              H[i, j-1]   + self.sw_gap)
        return float(H.max()) / m

    def rank(self, query_seq, corpus_seqs=None):
        """Return corpus indices sorted by score (SW on top-K, cosine for rest)."""
        q_vec = self._query_vec(query_seq)
        sims  = self._dense @ q_vec            # (N,)
        top_k = np.argsort(-sims)[:self.hnsw_k].tolist()

        # SW rerank top-K
        sw_scores = [
            (self._sw_score(query_seq, self._corpus_seqs[i]), i)
            for i in top_k
        ]
        sw_scores.sort(key=lambda x: -x[0])
        top_k_set = set(top_k)

        # Remaining by cosine
        remaining = [(float(sims[i]), i)
                     for i in np.argsort(-sims) if i not in top_k_set]

        return [i for _, i in sw_scores] + [i for _, i in remaining]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def ap(ranked, rel_set):
    if not rel_set:
        return 0.0
    hits = score = 0.0
    for rank, idx in enumerate(ranked, 1):
        if idx in rel_set:
            hits += 1
            score += hits / rank
    return score / len(rel_set)

def precision_at_k(ranked, rel_set, k):
    return sum(1 for d in ranked[:k] if d in rel_set) / k

def evaluate(system, corpus_list, corpus_ids, queries, relevance):
    aps, p1s, p5s, p10s = [], [], [], []
    n = len(queries)
    for qi, (qid, q_seq) in enumerate(queries.items()):
        if qid not in relevance or not relevance[qid]:
            continue
        rel_set = set(relevance[qid])
        ranked_ids = [corpus_ids[i] for i in
                      system.rank(q_seq) if i < len(corpus_ids)]
        aps.append(ap(ranked_ids, rel_set))
        p1s.append(precision_at_k(ranked_ids, rel_set, 1))
        p5s.append(precision_at_k(ranked_ids, rel_set, 5))
        p10s.append(precision_at_k(ranked_ids, rel_set, 10))
        if (qi + 1) % 20 == 0:
            print(f"    {qi+1}/{n} queries done  "
                  f"(running MAP={np.mean(aps):.4f})")

    return {
        'MAP':  round(float(np.mean(aps)),  4),
        'P@1':  round(float(np.mean(p1s)),  4),
        'P@5':  round(float(np.mean(p5s)),  4),
        'P@10': round(float(np.mean(p10s)), 4),
        'N':    len(aps),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus_csv',  required=True,
                   help='Tokenised corpus CSV (Filename,Data)')
    p.add_argument('--query_csv',   required=True,
                   help='Tokenised query CSV (Filename,Data)')
    p.add_argument('--relevance',   required=True,
                   help='relevance.json: {query_id -> [corpus_ids]}')
    p.add_argument('--k',           type=int, default=100,
                   help='k-means vocabulary size (default 100)')
    p.add_argument('--hnsw_k',      type=int, default=50,
                   help='Top-K candidates before SW (default 50)')
    p.add_argument('--alpha',       type=float, default=0.5,
                   help='Bigram weight alpha (default 0.5)')
    args = p.parse_args()

    print("=" * 60)
    print("All-Bigrams TF-IDF Evaluation")
    print(f"  corpus:    {args.corpus_csv}")
    print(f"  queries:   {args.query_csv}")
    print(f"  k={args.k}  hnsw_k={args.hnsw_k}  alpha={args.alpha}")
    print("=" * 60)

    corpus    = load_csv(args.corpus_csv)
    queries   = load_csv(args.query_csv)
    relevance = load_relevance(args.relevance)

    corpus_list = list(corpus.values())
    corpus_ids  = list(corpus.keys())

    print(f"\nCorpus: {len(corpus_list)} docs  Queries: {len(queries)}")
    q_with_rel = sum(1 for qid in queries
                     if qid in relevance and relevance[qid])
    print(f"Queries with relevance: {q_with_rel}")

    # Fit
    system = AllBigramsTFIDF(
        k=args.k, hnsw_k=args.hnsw_k, alpha=args.alpha)
    system.fit(corpus_list)

    # Evaluate
    print("\nEvaluating all-bigrams TF-IDF ...")
    t0 = time.time()
    results = evaluate(system, corpus_list, corpus_ids, queries, relevance)
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("RESULTS — All-Bigrams TF-IDF + HNSW + SW")
    print("=" * 60)
    print(f"  MAP   = {results['MAP']:.4f}")
    print(f"  P@1   = {results['P@1']:.4f}")
    print(f"  P@5   = {results['P@5']:.4f}")
    print(f"  P@10  = {results['P@10']:.4f}")
    print(f"  N queries evaluated = {results['N']}")
    print(f"  Time  = {elapsed:.1f}s")
    print()

    # LaTeX row
    hq_map = 0.6330
    pct = (results['MAP'] - hq_map) / hq_map * 100
    print("LaTeX ablation table row:")
    print(f"All bigrams ($V^2{{=}}10{{,}}000$) & "
          f"{results['MAP']:.4f} & "
          f"{results['P@1']:.4f} & "
          f"{results['P@5']:.4f} & "
          f"{results['P@10']:.4f} \\\\")
    print(f"  (H-QuEST MAP = {hq_map}, "
          f"all-bigrams change = {pct:+.1f}%)")


if __name__ == '__main__':
    main()
