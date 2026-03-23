"""
run_baselines.py
================
Runs all non-PMI-QuEST baselines on an existing token CSV:
  1. Token-DTW  — hard-match DTW with Sakoe-Chiba band
  2. SSL mean-pool cosine — continuous frame vectors, no discretisation

These correspond to the baseline rows in Table I of the paper.
For TF-IDF and H-QuEST baselines use run_main_comparison.py.

Usage
-----
python experiments/run_baselines.py \
    --corpus_csv  tokenised/wavlm_l7_k100/corpus.csv \
    --query_csv   tokenised/wavlm_l7_k100/queries.csv \
    --relevance   data/qbe_librispeech/metadata/relevance.json \
    --dtw_window  20

For SSL mean-pool baseline also supply:
    --corpus_audio_dir data/qbe_librispeech/corpus/ \
    --query_audio_dir  data/qbe_librispeech/queries/ \
    --ssl_model        wavlm-base \
    --ssl_layer        6
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Re-use individual baseline scripts
from pathlib import Path
import argparse
import subprocess

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus_csv',       required=True)
    p.add_argument('--query_csv',        required=True)
    p.add_argument('--relevance',        required=True)
    p.add_argument('--dtw_window',       type=int, default=20)
    p.add_argument('--corpus_audio_dir', default=None)
    p.add_argument('--query_audio_dir',  default=None)
    p.add_argument('--ssl_model',        default='wavlm-base')
    p.add_argument('--ssl_layer',        type=int, default=6)
    args = p.parse_args()

    base = Path(__file__).parent

    print("=" * 60)
    print("Baseline 1: Token-DTW")
    print("=" * 60)
    subprocess.run([
        sys.executable, str(base / '../experiments/run_dtw_baseline_impl.py'),
        '--corpus_csv', args.corpus_csv,
        '--query_csv',  args.query_csv,
        '--relevance',  args.relevance,
        '--window',     str(args.dtw_window),
    ], check=True)

    if args.corpus_audio_dir:
        print()
        print("=" * 60)
        print("Baseline 2: SSL mean-pool cosine")
        print("=" * 60)
        subprocess.run([
            sys.executable, str(base / '../experiments/run_ssl_cosine_impl.py'),
            '--corpus_dir', args.corpus_audio_dir,
            '--query_dir',  args.query_audio_dir,
            '--relevance',  args.relevance,
            '--model',      args.ssl_model,
            '--layer',      str(args.ssl_layer),
        ], check=True)

if __name__ == '__main__':
    main()
