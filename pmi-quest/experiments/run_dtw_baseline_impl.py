"""
run_dtw_baseline.py
====================
DTW baseline for QbE-STD on discrete token sequences.

Two variants:
  token-DTW:   DTW edit distance on the raw discrete token sequences
               (hard match: cost=0 if tokens equal, 1 otherwise)
  frame-DTW:   DTW on SSL frame-level L2-normalised embeddings
               (requires access to the raw SSL hidden states, not just tokens)

We implement token-DTW here since the token CSVs are already available.
Frame-DTW requires re-loading the SSL model and is much slower.

Token-DTW serves as the standard discrete-token retrieval baseline that 
every QbE-STD paper compares against. It shows the gain from TF-IDF 
pre-filtering over exhaustive sequence alignment.

Note: Exhaustive DTW over 2620×200 pairs is slow on CPU (~20-30 min).
We use FastDTW for speed (O(N) instead of O(N^2)) with a window constraint.

Usage
-----
pip install fastdtw
python run_dtw_baseline.py \\
    --corpus_csv  tokenised/wav2vec2-base_l0_k100/corpus.csv \\
    --query_csv   tokenised/wav2vec2-base_l0_k100/queries.csv \\
    --relevance   qbe_librispeech/metadata/relevance.json \\
    --window      20
"""

import argparse, csv, json, time
from pathlib import Path
import numpy as np


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


def token_dtw_distance(q, d, window=None):
    """
    DTW distance between two token sequences using hard-match cost:
      cost(a,b) = 0 if a==b, 1 otherwise.
    Lower distance = more similar = higher relevance.
    Optionally constrain to a Sakoe-Chiba band of width `window`.
    """
    m, n = len(q), len(d)
    INF = float('inf')

    # Use a windowed DTW to keep O(N*window) instead of O(N^2)
    w = max(window or max(m, n), abs(m - n))

    dtw = np.full((m + 1, n + 1), INF, dtype=np.float32)
    dtw[0, 0] = 0.0

    for i in range(1, m + 1):
        j_lo = max(1, i - w)
        j_hi = min(n, i + w)
        for j in range(j_lo, j_hi + 1):
            cost = 0.0 if q[i-1] == d[j-1] else 1.0
            dtw[i, j] = cost + min(dtw[i-1, j],      # insertion
                                   dtw[i, j-1],       # deletion
                                   dtw[i-1, j-1])     # match

    return float(dtw[m, n]) / m   # normalise by query length


def ap(ranked, rel_set):
    if not rel_set:
        return 0.0
    hits = score = 0.0
    for rank, idx in enumerate(ranked, 1):
        if idx in rel_set:
            hits += 1; score += hits / rank
    return score / len(rel_set)


def precision_at_k(ranked, rel_set, k):
    return sum(1 for d in ranked[:k] if d in rel_set) / k


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus_csv', required=True)
    p.add_argument('--query_csv',  required=True)
    p.add_argument('--relevance',  required=True)
    p.add_argument('--window',     type=int, default=20,
                   help='Sakoe-Chiba band width (default 20 tokens)')
    p.add_argument('--max_queries', type=int, default=None,
                   help='Limit queries for quick testing')
    args = p.parse_args()

    corpus    = load_csv(args.corpus_csv)
    queries   = load_csv(args.query_csv)
    with open(args.relevance) as f:
        relevance = json.load(f)

    corpus_list = list(corpus.values())
    corpus_ids  = list(corpus.keys())

    query_items = [(qid, seq) for qid, seq in queries.items()
                   if qid in relevance and relevance[qid]]
    if args.max_queries:
        query_items = query_items[:args.max_queries]

    print("=" * 60)
    print("Token-DTW Baseline")
    print(f"  N={len(corpus_list)}  Q={len(query_items)}  window={args.window}")
    print(f"  Pairs to evaluate: {len(corpus_list) * len(query_items):,}")
    print("=" * 60)

    aps, p1s, p5s, p10s = [], [], [], []
    t0 = time.time()

    for qi, (qid, q_seq) in enumerate(query_items):
        rel_set = set(relevance[qid])

        # Compute DTW distance to every corpus doc
        dists = []
        for doc_i, d_seq in enumerate(corpus_list):
            dist = token_dtw_distance(q_seq, d_seq, args.window)
            dists.append((dist, doc_i))

        # Sort by distance ascending (lower = more similar)
        dists.sort(key=lambda x: x[0])
        ranked_ids = [corpus_ids[i] for _, i in dists]

        aps.append(ap(ranked_ids, rel_set))
        p1s.append(precision_at_k(ranked_ids, rel_set, 1))
        p5s.append(precision_at_k(ranked_ids, rel_set, 5))
        p10s.append(precision_at_k(ranked_ids, rel_set, 10))

        if (qi + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (qi + 1) * (len(query_items) - qi - 1)
            print(f"  {qi+1}/{len(query_items)} queries  "
                  f"running MAP={np.mean(aps):.4f}  "
                  f"ETA {eta/60:.1f} min")

    print()
    print("=" * 60)
    print("RESULTS — Token-DTW Baseline")
    print("=" * 60)
    print(f"  MAP   = {np.mean(aps):.4f}")
    print(f"  P@1   = {np.mean(p1s):.4f}")
    print(f"  P@5   = {np.mean(p5s):.4f}")
    print(f"  P@10  = {np.mean(p10s):.4f}")
    print(f"  N     = {len(aps)}")
    print(f"  Time  = {(time.time()-t0)/60:.1f} min")
    print()
    print("LaTeX row:")
    print(f"  Token-DTW (window={args.window}) & "
          f"{np.mean(aps):.4f} & {np.mean(p1s):.4f} & "
          f"{np.mean(p5s):.4f} & {np.mean(p10s):.4f} \\\\")


if __name__ == '__main__':
    main()
