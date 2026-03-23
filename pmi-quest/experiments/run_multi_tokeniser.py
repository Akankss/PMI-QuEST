"""
Multi-Tokeniser PMI-QuEST Comparison
======================================
Runs PMI-QuEST (best config: no PMI-TD, tau=0.5, alpha=0.5) and H-QuEST
across all tokeniser CSVs in a directory, producing a single comparison table.

This generates Table IV in the paper:
  "PMI-QuEST across tokenisers: wav2vec2-base / HuBERT-base / WavLM-base"

Usage
-----
# After running audio_tokenizer_v2.py --all_models, you will have:
#   tokenised/wav2vec2-base_l6_k100/corpus.csv + queries.csv
#   tokenised/hubert-base_l6_k100/corpus.csv   + queries.csv
#   tokenised/wavlm-base_l6_k100/corpus.csv    + queries.csv
#
# Then run:
python run_multi_tokeniser.py \\
    --tokeniser_dir  tokenised/ \\
    --relevance      relevance.json \\
    --out            results/multi_tokeniser.csv

# Or specify individual CSV sets:
python run_multi_tokeniser.py \\
    --configs \\
        "wav2vec2-base l0 k100:qbe_librispeech/corpus_merged.csv:qbe_librispeech/queries_merged.csv" \\
        "hubert-base l6 k100:tokenised/hubert-base_l6_k100/corpus.csv:tokenised/hubert-base_l6_k100/queries.csv" \\
        "wavlm-base l6 k100:tokenised/wavlm-base_l6_k100/corpus.csv:tokenised/wavlm-base_l6_k100/queries.csv" \\
    --relevance relevance.json \\
    --out results/multi_tokeniser.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

# Import the retrieval systems from pmiquest_system.py
# (assumes pmiquest_system.py is in the same directory or on PYTHONPATH)
try:
    from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest, evaluate
except ImportError:
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent))
    from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest, evaluate


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading  (handles both comma-separated and JSON list formats)
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict:
    """Load corpus or queries CSV → {stem: [int, ...]}."""
    seqs = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Support both "Filename" and "filename" headers
            fname = row.get("Filename") or row.get("filename") or ""
            data  = row.get("Data") or row.get("data") or ""
            stem  = Path(fname).stem
            # Parse: JSON list "[1,2,3]" or plain "1,2,3"
            data = data.strip()
            if data.startswith("["):
                tokens = json.loads(data)
            else:
                tokens = [int(x) for x in data.split(",") if x.strip()]
            seqs[stem] = tokens
    return seqs


def load_relevance(path: str) -> dict:
    """Load relevance JSON → {query_id: [relevant_doc_ids]}."""
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Single-config runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_config(
    label:     str,
    corpus:    dict,
    queries:   dict,
    relevance: dict,
    verbose:   bool = True,
) -> dict:
    """
    Run TF-IDF Baseline, H-QuEST, and PMI-QuEST on one tokeniser config.

    Returns dict with keys:
      label, vocab_size, mean_query_len, mean_corpus_len, rho,
      tfidf_map, tfidf_p1, tfidf_p5, tfidf_p10,
      hquest_map, hquest_p1, hquest_p5, hquest_p10,
      pmi_map, pmi_p1, pmi_p5, pmi_p10,
      pmi_map_gain_vs_hquest, pmi_p1_gain_vs_hquest
    """
    corpus_list  = list(corpus.values())
    query_list   = list(queries.values())
    query_ids    = list(queries.keys())

    vocab_size      = len(set(t for s in corpus_list for t in s))
    mean_q_len      = np.mean([len(s) for s in query_list])
    mean_c_len      = np.mean([len(s) for s in corpus_list])
    rho             = mean_q_len / mean_c_len

    # Build (query_tokens, relevant_set) pairs
    eval_pairs = []
    for qid in query_ids:
        if qid not in relevance:
            continue
        rel_set = set(relevance[qid])
        eval_pairs.append((queries[qid], rel_set))

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Config: {label}")
        print(f"  Corpus: {len(corpus_list)}  Queries: {len(eval_pairs)}")
        print(f"  V={vocab_size}  mean_q={mean_q_len:.1f}  "
              f"mean_c={mean_c_len:.1f}  rho={rho:.3f}")

    # Index-to-doc-ID lookup (rank() returns int indices, relevance uses string IDs)
    corpus_ids = list(corpus.keys())   # position i → doc_id

    results = {"label": label, "vocab_size": vocab_size,
               "mean_query_len": mean_q_len, "mean_corpus_len": mean_c_len,
               "rho": rho}

    # ── TF-IDF Baseline ───────────────────────────────────────────────────────
    t0 = time.time()
    baseline = TFIDFBaseline()
    baseline.fit(corpus_list)
    scores_baseline = {
        qid: [corpus_ids[i] for i in baseline.rank(queries[qid]) if i < len(corpus_ids)]
        for qid in query_ids if qid in relevance
    }
    m = _eval(scores_baseline, relevance)
    results.update({
        "tfidf_map": m["map"], "tfidf_p1": m["p1"],
        "tfidf_p5": m["p5"],  "tfidf_p10": m["p10"],
    })
    if verbose:
        print(f"  TF-IDF:    MAP={m['map']:.4f}  P@1={m['p1']:.4f}  [{time.time()-t0:.1f}s]")

    # ── H-QuEST ───────────────────────────────────────────────────────────────
    t0 = time.time()
    hquest = HQuEST(hnsw_k=50)
    hquest.fit(corpus_list)
    scores_hquest = {
        qid: [corpus_ids[i] for i in hquest.rank(queries[qid]) if i < len(corpus_ids)]
        for qid in query_ids if qid in relevance
    }
    m = _eval(scores_hquest, relevance)
    results.update({
        "hquest_map": m["map"], "hquest_p1": m["p1"],
        "hquest_p5": m["p5"],  "hquest_p10": m["p10"],
    })
    if verbose:
        print(f"  H-QuEST:   MAP={m['map']:.4f}  P@1={m['p1']:.4f}  [{time.time()-t0:.1f}s]")

    # ── PMI-QuEST (best config: no PMI-TD, tau=0.5, alpha=0.5) ───────────────
    t0 = time.time()
    pmiquest = PMIQuest(
        pmi_tau       = 0.5,    # best MAP threshold from ablation
        bigram_weight = 0.5,    # alpha in the paper
        use_pmitd     = False,  # no PMI-TD: best MAP config
        hnsw_k        = 50,
    )
    pmiquest.fit(corpus_list)
    scores_pmi = {
        qid: [corpus_ids[i] for i in pmiquest.rank(queries[qid]) if i < len(corpus_ids)]
        for qid in query_ids if qid in relevance
    }
    m = _eval(scores_pmi, relevance)
    results.update({
        "pmi_map": m["map"], "pmi_p1": m["p1"],
        "pmi_p5": m["p5"],  "pmi_p10": m["p10"],
    })
    if verbose:
        print(f"  PMI-QuEST: MAP={m['map']:.4f}  P@1={m['p1']:.4f}  [{time.time()-t0:.1f}s]")

    # Gains over H-QuEST
    hq_map = results["hquest_map"]
    hq_p1  = results["hquest_p1"]
    results["pmi_map_gain_vs_hquest"] = (results["pmi_map"] - hq_map) / max(hq_map, 1e-9)
    results["pmi_p1_gain_vs_hquest"]  = (results["pmi_p1"]  - hq_p1)  / max(hq_p1,  1e-9)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper (MAP, P@k)
# ─────────────────────────────────────────────────────────────────────────────

def _eval(ranked_lists: dict, relevance: dict) -> dict:
    """
    ranked_lists: {query_id: [doc_id_str, ...]}  (full ranked list of string IDs)
    relevance:    {query_id: [relevant_doc_id_strs]}
    Returns MAP, P@1, P@5, P@10.
    """
    aps, p1s, p5s, p10s = [], [], [], []
    for qid, ranked in ranked_lists.items():
        if qid not in relevance:
            continue
        rel = set(relevance[qid])
        # AP
        n_rel, ap = 0, 0.0
        for rank, did in enumerate(ranked, 1):
            if did in rel:
                n_rel += 1
                ap += n_rel / rank
        aps.append(ap / max(len(rel), 1))
        # P@k
        p1s.append(1.0 if len(ranked) >= 1 and ranked[0] in rel else 0.0)
        p5s.append(sum(1 for d in ranked[:5] if d in rel) / 5)
        p10s.append(sum(1 for d in ranked[:10] if d in rel) / 10)
    return {
        "map": np.mean(aps),
        "p1":  np.mean(p1s),
        "p5":  np.mean(p5s),
        "p10": np.mean(p10s),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print and save results table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(all_results: list):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("MULTI-TOKENISER COMPARISON — PMI-QuEST")
    print(f"{'='*80}")
    hdr = f"{'Tokeniser':<28} {'V':>4}  {'rho':>5}  {'MAP':>6}  {'P@1':>6}  {'P@5':>6}  {'P@10':>6}"
    print(f"\n── TF-IDF Baseline ──────────────────────────────────────────────────────")
    print(hdr)
    print("─"*80)
    for r in all_results:
        print(f"  {r['label']:<26} {r['vocab_size']:>4}  {r['rho']:>5.3f}"
              f"  {r['tfidf_map']:>6.4f}  {r['tfidf_p1']:>6.4f}"
              f"  {r['tfidf_p5']:>6.4f}  {r['tfidf_p10']:>6.4f}")

    print(f"\n── H-QuEST ──────────────────────────────────────────────────────────────")
    print(hdr)
    print("─"*80)
    for r in all_results:
        print(f"  {r['label']:<26} {r['vocab_size']:>4}  {r['rho']:>5.3f}"
              f"  {r['hquest_map']:>6.4f}  {r['hquest_p1']:>6.4f}"
              f"  {r['hquest_p5']:>6.4f}  {r['hquest_p10']:>6.4f}")

    print(f"\n── PMI-QuEST (τ=0.5, no PMI-TD) ────────────────────────────────────────")
    print(hdr + f"  {'vs H-QuEST MAP':>14}  {'vs H-QuEST P@1':>14}")
    print("─"*80)
    for r in all_results:
        gmap = r['pmi_map_gain_vs_hquest'] * 100
        gp1  = r['pmi_p1_gain_vs_hquest']  * 100
        print(f"  {r['label']:<26} {r['vocab_size']:>4}  {r['rho']:>5.3f}"
              f"  {r['pmi_map']:>6.4f}  {r['pmi_p1']:>6.4f}"
              f"  {r['pmi_p5']:>6.4f}  {r['pmi_p10']:>6.4f}"
              f"  {gmap:>+13.1f}%  {gp1:>+13.1f}%")
    print("─"*80)


def save_csv(all_results: list, path: str):
    """Save results to CSV."""
    fields = [
        "label", "vocab_size", "rho",
        "tfidf_map", "tfidf_p1", "tfidf_p5", "tfidf_p10",
        "hquest_map", "hquest_p1", "hquest_p5", "hquest_p10",
        "pmi_map", "pmi_p1", "pmi_p5", "pmi_p10",
        "pmi_map_gain_vs_hquest", "pmi_p1_gain_vs_hquest",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\n  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PMI-QuEST across multiple tokeniser configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tokeniser_dir", default=None,
        help="Directory produced by audio_tokenizer_v2.py --all_models. "
             "Scans for subdirectories containing corpus.csv + queries.csv.",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Explicit configs in format 'label:corpus.csv:queries.csv'. "
             "Used instead of --tokeniser_dir.",
    )
    parser.add_argument(
        "--relevance", required=True,
        help="Path to relevance.json.",
    )
    parser.add_argument(
        "--out", default="results/multi_tokeniser.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    relevance = load_relevance(args.relevance)

    # Collect configs
    run_configs = []

    if args.tokeniser_dir:
        base = Path(args.tokeniser_dir)
        for subdir in sorted(base.iterdir()):
            corpus_csv = subdir / "corpus.csv"
            query_csv  = subdir / "queries.csv"
            if corpus_csv.exists() and query_csv.exists():
                run_configs.append((subdir.name, str(corpus_csv), str(query_csv)))
        if not run_configs:
            raise FileNotFoundError(
                f"No subdirectories with corpus.csv + queries.csv in {args.tokeniser_dir}"
            )

    elif args.configs:
        for spec in args.configs:
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError(f"Expected 'label:corpus_csv:query_csv', got '{spec}'")
            run_configs.append(tuple(parts))

    else:
        parser.error("Provide either --tokeniser_dir or --configs.")

    print(f"\nRunning {len(run_configs)} tokeniser configs …")

    all_results = []
    for label, corpus_csv, query_csv in run_configs:
        corpus  = load_csv(corpus_csv)
        queries = load_csv(query_csv)
        result  = run_one_config(label, corpus, queries, relevance, verbose=True)
        all_results.append(result)

    print_table(all_results)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_csv(all_results, args.out)
