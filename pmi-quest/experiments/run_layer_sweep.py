"""
Layer Sweep — PMI-QuEST across transformer layers
===================================================
Tokenises corpus + queries for each (model, layer) combination and runs
PMI-QuEST evaluation, producing a single CSV and printed table suitable
for Figure X / Table Y in the paper.

Default sweep:
  wav2vec2-base  layers: 0 (CNN), 3, 6, 7, 9, 12
  hubert-base    layers: 3, 6, 7, 9, 12
  wavlm-base     layers: 3, 6, 7, 9, 12

Each config produces:
  out_dir/<model>_l<layer>_k100/corpus.csv
  out_dir/<model>_l<layer>_k100/queries.csv
  out_dir/<model>_l<layer>_k100/kmeans_centroids.npy   (cached)

Already-tokenised configs are loaded from cache and skipped.

Usage
-----
python run_layer_sweep.py \\
    --corpus_dir  qbe_librispeech/corpus_audio \\
    --query_dir   qbe_librispeech/query_audio \\
    --relevance   qbe_librispeech/metadata/relevance.json \\
    --out_dir     tokenised/ \\
    --out_csv     results/layer_sweep.csv

# Run only specific models
python run_layer_sweep.py \\
    --models hubert-base wavlm-base \\
    --layers 3 6 7 9 12 \\
    --corpus_dir  qbe_librispeech/corpus_audio \\
    --query_dir   qbe_librispeech/query_audio \\
    --relevance   qbe_librispeech/metadata/relevance.json \\
    --out_dir     tokenised/ \\
    --out_csv     results/layer_sweep.csv

# Skip tokenisation if CSVs already exist (retrieval-only re-run)
python run_layer_sweep.py --skip_tokenisation ...
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── import tokeniser ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from audio_tokenizer_v2 import KMeansTokenizer, load_audio, _write_csv
except ImportError:
    raise ImportError(
        "audio_tokenizer_v2.py not found. "
        "Place it in the same directory as this script."
    )

# ── import retrieval systems ──────────────────────────────────────────────────
try:
    from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest
except ImportError:
    raise ImportError(
        "pmiquest_system.py not found. "
        "Place it in the same directory as this script."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Default sweep configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIGS = [
    # (model_name,    layers_to_sweep)
    ("wav2vec2-base", [0, 3, 6, 7, 9, 12]),   # layer 0 = CNN (context-free)
    ("hubert-base",   [3, 6, 7, 9, 12]),
    ("wavlm-base",    [3, 6, 7, 9, 12]),
]

PMI_TAU     = 0.5    # best MAP config from ablation
BIGRAM_W    = 0.5    # alpha
N_CLUSTERS  = 100
K_CANDS     = 50


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict:
    """Load corpus/query CSV → {stem: [int, ...]}."""
    seqs = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fname  = row.get("Filename") or row.get("filename") or ""
            data   = row.get("Data")     or row.get("data")     or ""
            stem   = Path(fname).stem
            data   = data.strip()
            if data.startswith("["):
                tokens = json.loads(data)
            else:
                tokens = [int(x) for x in data.split(",") if x.strip()]
            seqs[stem] = tokens
    return seqs


def load_relevance(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation (with caching)
# ─────────────────────────────────────────────────────────────────────────────

def tokenise_config(
    model_name:   str,
    layer:        int,
    corpus_paths: list,
    query_paths:  list,
    out_dir:      Path,
    max_frames:   int = 500_000,
    force:        bool = False,
) -> tuple:
    """
    Tokenise corpus + queries for one (model, layer) config.
    Returns (corpus_csv_path, query_csv_path).
    Uses cached CSVs if they exist and force=False.
    """
    tok      = KMeansTokenizer(model_name, layer, N_CLUSTERS)
    tag      = tok.tag()
    config_dir = out_dir / tag
    config_dir.mkdir(parents=True, exist_ok=True)

    corpus_csv   = config_dir / "corpus.csv"
    query_csv    = config_dir / "queries.csv"
    centroid_npy = config_dir / "kmeans_centroids.npy"

    all_exist = corpus_csv.exists() and query_csv.exists()

    if all_exist and not force:
        print(f"  [cache] {tag} — CSVs already exist, skipping tokenisation.")
        return str(corpus_csv), str(query_csv)

    print(f"\n{'─'*60}")
    print(f"Tokenising: {tag}")
    print(f"{'─'*60}")

    # Load or fit k-means
    if centroid_npy.exists() and not force:
        print(f"  Loading cached centroids from {centroid_npy}")
        tok.load_centroids(str(centroid_npy))
    else:
        tok.fit([str(p) for p in corpus_paths], max_frames=max_frames)
        tok.save_centroids(str(centroid_npy))

    # Tokenise corpus
    print(f"  Tokenising {len(corpus_paths)} corpus files …")
    corpus_rows = _tokenize_files(corpus_paths, tok, "corpus")
    _write_csv(corpus_rows, str(corpus_csv))

    # Tokenise queries
    print(f"  Tokenising {len(query_paths)} query files …")
    query_rows = _tokenize_files(query_paths, tok, "queries")
    _write_csv(query_rows, str(query_csv))

    return str(corpus_csv), str(query_csv)


def _tokenize_files(paths: list, tok: KMeansTokenizer, desc: str) -> list:
    from tqdm import tqdm
    rows = []
    for path in tqdm(paths, desc=desc):
        try:
            tokens = tok.tokenize_file(str(path))
            rows.append((Path(path).name, tokens))
        except Exception as e:
            print(f"  WARNING: {Path(path).name}: {e}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval evaluation for one config
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_config(
    model_name:  str,
    layer:       int,
    corpus_csv:  str,
    query_csv:   str,
    relevance:   dict,
    verbose:     bool = True,
) -> dict:
    """Run TF-IDF, H-QuEST, PMI-QuEST on one tokenised config."""
    corpus  = load_csv(corpus_csv)
    queries = load_csv(query_csv)

    corpus_list = list(corpus.values())
    corpus_ids  = list(corpus.keys())
    query_ids   = list(queries.keys())

    vocab_size  = len(set(t for s in corpus_list for t in s))
    mean_q      = np.mean([len(s) for s in queries.values()])
    mean_c      = np.mean([len(s) for s in corpus_list])
    rho         = mean_q / mean_c
    tag         = f"{model_name}_l{layer}_k{N_CLUSTERS}"

    if verbose:
        print(f"\n  Evaluating {tag}  "
              f"V={vocab_size}  mean_q={mean_q:.1f}  rho={rho:.3f}")

    row = {
        "model": model_name, "layer": layer, "tag": tag,
        "vocab_size": vocab_size, "mean_q_len": mean_q,
        "mean_c_len": mean_c, "rho": rho,
    }

    def rank_and_translate(system):
        return {
            qid: [corpus_ids[i] for i in system.rank(queries[qid])
                  if i < len(corpus_ids)]
            for qid in query_ids if qid in relevance
        }

    def _eval(ranked_lists):
        aps, p1s, p5s, p10s = [], [], [], []
        for qid, ranked in ranked_lists.items():
            if qid not in relevance:
                continue
            rel = set(relevance[qid])
            n_rel, ap = 0, 0.0
            for rank, did in enumerate(ranked, 1):
                if did in rel:
                    n_rel += 1
                    ap += n_rel / rank
            aps.append(ap / max(len(rel), 1))
            p1s.append(1.0 if ranked and ranked[0] in rel else 0.0)
            p5s.append(sum(1 for d in ranked[:5]  if d in rel) / 5)
            p10s.append(sum(1 for d in ranked[:10] if d in rel) / 10)
        return {
            "map": float(np.mean(aps)), "p1": float(np.mean(p1s)),
            "p5": float(np.mean(p5s)),  "p10": float(np.mean(p10s)),
        }

    # TF-IDF
    t0 = time.time()
    bl = TFIDFBaseline()
    bl.fit(corpus_list)
    m = _eval(rank_and_translate(bl))
    row.update({f"tfidf_{k}": v for k, v in m.items()})
    if verbose:
        print(f"    TF-IDF:    MAP={m['map']:.4f}  P@1={m['p1']:.4f}  [{time.time()-t0:.1f}s]")

    # H-QuEST
    t0 = time.time()
    hq = HQuEST(hnsw_k=K_CANDS)
    hq.fit(corpus_list)
    m = _eval(rank_and_translate(hq))
    row.update({f"hquest_{k}": v for k, v in m.items()})
    if verbose:
        print(f"    H-QuEST:   MAP={m['map']:.4f}  P@1={m['p1']:.4f}  [{time.time()-t0:.1f}s]")

    # PMI-QuEST
    t0 = time.time()
    pq = PMIQuest(pmi_tau=PMI_TAU, bigram_weight=BIGRAM_W,
                  use_pmitd=False, hnsw_k=K_CANDS)
    pq.fit(corpus_list)
    m = _eval(rank_and_translate(pq))
    row.update({f"pmi_{k}": v for k, v in m.items()})
    if verbose:
        print(f"    PMI-QuEST: MAP={m['map']:.4f}  P@1={m['p1']:.4f}  [{time.time()-t0:.1f}s]")

    # Gains
    hq_map = row["hquest_map"]
    hq_p1  = row["hquest_p1"]
    row["pmi_map_gain"] = (row["pmi_map"] - hq_map) / max(hq_map, 1e-9)
    row["pmi_p1_gain"]  = (row["pmi_p1"]  - hq_p1)  / max(hq_p1,  1e-9)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────

def print_sweep_table(all_rows: list):
    print(f"\n{'='*80}")
    print("LAYER SWEEP — PMI-QuEST MAP across tokeniser layers")
    print(f"{'='*80}")

    # Group by model
    from itertools import groupby
    all_rows_sorted = sorted(all_rows, key=lambda r: (r["model"], r["layer"]))

    for model, group in groupby(all_rows_sorted, key=lambda r: r["model"]):
        rows = list(group)
        print(f"\n── {model} ──────────────────────────────────────────────────────────────")
        print(f"  {'Layer':<6}  {'TF-IDF':>7}  {'H-QuEST':>7}  {'PMI-QuEST':>9}  "
              f"{'vs HQ MAP':>9}  {'vs HQ P@1':>9}  {'#PMI-bigrams':>12}")
        print(f"  {'─'*72}")
        best_map = max(r["pmi_map"] for r in rows)
        for r in rows:
            layer_label = f"L{r['layer']}" if r['layer'] > 0 else "CNN"
            marker = " ←" if r["pmi_map"] == best_map else ""
            print(f"  {layer_label:<6}  "
                  f"{r['tfidf_map']:>7.4f}  "
                  f"{r['hquest_map']:>7.4f}  "
                  f"{r['pmi_map']:>9.4f}  "
                  f"{r['pmi_map_gain']*100:>+8.1f}%  "
                  f"{r['pmi_p1_gain']*100:>+8.1f}%"
                  f"{marker}")

    print(f"\n{'─'*80}")
    best = max(all_rows, key=lambda r: r["pmi_map"])
    print(f"Overall best: {best['tag']}  MAP={best['pmi_map']:.4f}  P@1={best['pmi_p1']:.4f}")
    print(f"{'─'*80}")


def save_csv(all_rows: list, path: str):
    fields = [
        "tag", "model", "layer", "vocab_size", "rho",
        "tfidf_map", "tfidf_p1", "tfidf_p5", "tfidf_p10",
        "hquest_map", "hquest_p1", "hquest_p5", "hquest_p10",
        "pmi_map", "pmi_p1", "pmi_p5", "pmi_p10",
        "pmi_map_gain", "pmi_p1_gain",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layer sweep: tokenise + evaluate PMI-QuEST across layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--corpus_dir",  required=True)
    parser.add_argument("--query_dir",   required=True)
    parser.add_argument("--relevance",   required=True)
    parser.add_argument("--out_dir",     default="tokenised",
                        help="Directory for tokenised CSVs (default: tokenised/).")
    parser.add_argument("--out_csv",     default="results/layer_sweep.csv")
    parser.add_argument("--models",      nargs="+",
                        default=["wav2vec2-base", "hubert-base", "wavlm-base"],
                        help="Models to sweep.")
    parser.add_argument("--layers",      nargs="+", type=int,
                        default=None,
                        help="Layers to sweep. Default: 0,3,6,7,9,12 for wav2vec2; "
                             "3,6,7,9,12 for hubert/wavlm.")
    parser.add_argument("--n_clusters",  type=int, default=100)
    parser.add_argument("--max_frames",  type=int, default=500_000)
    parser.add_argument("--skip_tokenisation", action="store_true",
                        help="Skip tokenisation step; use existing CSVs only.")
    parser.add_argument("--force",       action="store_true",
                        help="Re-tokenise even if CSVs already exist.")
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    relevance   = load_relevance(args.relevance)
    out_dir     = Path(args.out_dir)

    corpus_paths = sorted(
        list(Path(args.corpus_dir).glob("*.wav")) +
        list(Path(args.corpus_dir).glob("*.flac"))
    )
    query_paths  = sorted(
        list(Path(args.query_dir).glob("*.wav")) +
        list(Path(args.query_dir).glob("*.flac"))
    )
    print(f"Corpus: {len(corpus_paths)} files | Queries: {len(query_paths)} files")

    # Build config list
    configs = []
    model_default_layers = {
        "wav2vec2-base": [0, 3, 6, 7, 9, 12],
        "hubert-base":   [3, 6, 7, 9, 12],
        "wavlm-base":    [3, 6, 7, 9, 12],
    }
    for model in args.models:
        layers = args.layers if args.layers else model_default_layers.get(model, [6])
        for layer in layers:
            configs.append((model, layer))

    print(f"\nSweep: {len(configs)} configs")
    for m, l in configs:
        print(f"  {m}_l{l}_k{args.n_clusters}")

    # ── Phase 1: Tokenisation ─────────────────────────────────────────────────
    csv_paths = {}   # (model, layer) → (corpus_csv, query_csv)

    if args.skip_tokenisation:
        print("\n[--skip_tokenisation] Loading existing CSVs …")
        for model, layer in configs:
            tag = f"{model}_l{layer}_k{args.n_clusters}"
            corpus_csv = out_dir / tag / "corpus.csv"
            query_csv  = out_dir / tag / "queries.csv"
            if corpus_csv.exists() and query_csv.exists():
                csv_paths[(model, layer)] = (str(corpus_csv), str(query_csv))
            else:
                print(f"  WARNING: {tag} CSVs not found, skipping.")
    else:
        print("\n── Phase 1: Tokenisation ────────────────────────────────────────────")
        for model, layer in configs:
            c_csv, q_csv = tokenise_config(
                model, layer, corpus_paths, query_paths,
                out_dir, args.max_frames, args.force,
            )
            csv_paths[(model, layer)] = (c_csv, q_csv)

    # ── Phase 2: Retrieval evaluation ─────────────────────────────────────────
    print("\n── Phase 2: Retrieval Evaluation ────────────────────────────────────────")
    all_rows = []
    for model, layer in configs:
        if (model, layer) not in csv_paths:
            continue
        c_csv, q_csv = csv_paths[(model, layer)]
        row = evaluate_config(model, layer, c_csv, q_csv, relevance, verbose=True)
        all_rows.append(row)

    # ── Results ───────────────────────────────────────────────────────────────
    print_sweep_table(all_rows)
    save_csv(all_rows, args.out_csv)
