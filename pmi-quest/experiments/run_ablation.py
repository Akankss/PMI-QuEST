"""
run_ablation.py
===============
Runs the full component ablation (Table II of the paper):

  Block 1: Cosine-only (no HNSW, no SW)
    - TF-IDF unigrams cosine
    - All-bigrams cosine

  Block 2: Full pipeline (HNSW + SW)
    - H-QuEST (unigrams)
    - All-bigrams + HNSW + SW

  Block 3: Bigram selection criterion
    - Random N=700 (mean±std, 3 seeds)
    - Random N=1270 (mean±std, 3 seeds)
    - Frequency top-700 (BPE-style)
    - Frequency top-1270 (BPE-style)
    - PMI tau=1.5
    - PMI tau=0.5  ← proposed

  Block 4: Bigram weight alpha
    - alpha=0.25, 0.50, 1.00

Usage
-----
python experiments/run_ablation.py \
    --corpus_csv tokenised/wavlm_l7_k100/corpus.csv \
    --query_csv  tokenised/wavlm_l7_k100/queries.csv \
    --relevance  data/qbe_librispeech/metadata/relevance.json
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import subprocess
from pathlib import Path

def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus_csv', required=True)
    p.add_argument('--query_csv',  required=True)
    p.add_argument('--relevance',  required=True)
    p.add_argument('--k',   type=int,   default=100)
    args = p.parse_args()

    base = Path(__file__).parent

    print("Running bigram selection ablation...")
    subprocess.run([
        sys.executable,
        str(base / '../experiments/run_bigram_selection_baselines_impl.py'),
        '--corpus_csv', args.corpus_csv,
        '--query_csv',  args.query_csv,
        '--relevance',  args.relevance,
        '--k',          str(args.k),
    ], check=True)

    print("\nRunning all-bigrams variants...")
    subprocess.run([
        sys.executable,
        str(base / '../experiments/run_allbigrams_impl.py'),
        '--corpus_csv', args.corpus_csv,
        '--query_csv',  args.query_csv,
        '--relevance',  args.relevance,
        '--k',          str(args.k),
    ], check=True)

if __name__ == '__main__':
    main()
