"""
experiments/run_main_comparison.py
==========================
Loads LibriSpeech token CSV files and runs the three-way comparison:
    TF-IDF Baseline  vs  H-QuEST  vs  PMI-QuEST (proposed)

CSV format (both corpus and query files):
    Filename,Data
    1089-134686-0000.wav,"10,0,8,57,6,52,15,57,..."

Relevance format (JSON):
    {
      "q_THINGS_5639-40744-0014": ["6930-75918-0016", "1320-122617-0006", ...],
      ...
    }

Usage
-----
    python run_pmiquest_comparison.py \
        --corpus    corpus_tokens.csv \
        --queries   query_tokens.csv  \
        --relevance relevance.json    \
        --out       results/word_mode.csv \
        --ablation
"""

from __future__ import annotations
import argparse, csv as _csv, json, os, sys, time
from pathlib import Path
from typing import List, Dict, Set
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmiquest_system import (
    TFIDFBaseline, HQuEST, PMIQuest,
    run_comparison, run_pmiquest_ablation, evaluate,
)


# =============================================================================
# Loaders
# =============================================================================

def load_token_csv(csv_path: str) -> Dict[str, List[int]]:
    """
    Load:  Filename,Data
           1089-134686-0000.wav,"10,0,8,57,..."
    Returns {utt_id: [tok, tok, ...]}  utt_id = stem without extension.
    """
    tokens: Dict[str, List[int]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        fname_col = next((c for c in cols if c.strip().lower() in
                          ("filename","file","name","id")), cols[0])
        data_col  = next((c for c in cols if c.strip().lower() in
                          ("data","tokens","token","sequence")),
                         cols[1] if len(cols) > 1 else cols[0])
        for row in reader:
            fname  = row[fname_col].strip()
            utt_id = Path(fname).stem
            raw    = row[data_col].strip()
            toks   = [int(t.strip()) for t in raw.split(",")
                      if t.strip().lstrip("-").isdigit()]
            if toks:
                tokens[utt_id] = toks

    all_lens = [len(v) for v in tokens.values()]
    print(f"  {len(tokens):,} seqs  len={min(all_lens)}–{max(all_lens)}"
          f"  mean={np.mean(all_lens):.1f}   [{csv_path}]")
    return tokens


def load_relevance(path: str) -> Dict[str, List[str]]:
    with open(path) as f:
        return json.load(f)


def build_groundtruth(
    query_ids:   List[str],
    corpus_ids:  List[str],
    relevance:   Dict[str, List[str]],
) -> List[Set[int]]:
    """
    Map relevance lists → corpus index sets.
    Handles keys like "q_THINGS_5639-40744-0014"  or  "5639-40744-0014".
    """
    corpus_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    # Build reverse: last-segment of key → full key
    utt_to_key: Dict[str, str] = {}
    for k in relevance:
        parts = k.split("_")
        utt_to_key[parts[-1]] = k   # "5639-40744-0014" → "q_THINGS_5639-..."

    gt = []
    missing = 0
    for qid in query_ids:
        rel_ids = relevance.get(qid) or relevance.get(utt_to_key.get(qid, ""), [])
        rel_idxs = {corpus_idx[cid] for cid in rel_ids if cid in corpus_idx}
        if not rel_idxs:
            missing += 1
        gt.append(rel_idxs)

    if missing:
        print(f"  ⚠  {missing}/{len(query_ids)} queries matched no relevant docs")
    return gt


# =============================================================================
# Output
# =============================================================================

def save_csv(rows: List[Dict], outpath: str) -> None:
    if not rows: return
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    with open(outpath, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  ✓  Saved → {outpath}")


def print_stats(corpus_seqs, query_seqs, ground_truth):
    q = [len(s) for s in query_seqs]
    c = [len(s) for s in corpus_seqs]
    r = [len(g) for g in ground_truth]
    rho = np.mean(q) / np.mean(c)
    print(f"\n  Corpus  : {len(corpus_seqs):,}  |  Queries : {len(query_seqs):,}")
    print(f"  Q len   : mean={np.mean(q):.1f}  range={min(q)}–{max(q)}")
    print(f"  C len   : mean={np.mean(c):.1f}  range={min(c)}–{max(c)}")
    print(f"  ρ = {rho:.3f}  → {'SW regime' if rho < 0.15 else 'TF-IDF regime'}")
    print(f"  Rel/qry : mean={np.mean(r):.1f}  range={min(r)}–{max(r)}")
    V = len(set(t for s in corpus_seqs for t in s))
    print(f"  Vocab   : {V} unique tokens in corpus")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus",    required=True,  help="Corpus tokens CSV")
    p.add_argument("--queries",   required=True,  help="Query tokens CSV")
    p.add_argument("--relevance", required=True,  help="relevance.json")
    p.add_argument("--out",       default="results/comparison.csv")
    p.add_argument("--ablation",  action="store_true",
                   help="Also run component ablation sweep")
    p.add_argument("--rho_star",  type=float, default=0.15)
    args = p.parse_args()

    print("=" * 65)
    print("PMI-QuEST  vs  H-QuEST  vs  TF-IDF Baseline  [TASLP]")
    print("=" * 65)

    # ── load ──────────────────────────────────────────────────────
    print("\nLoading data…")
    corpus_tok = load_token_csv(args.corpus)
    query_tok  = load_token_csv(args.queries)
    relevance  = load_relevance(args.relevance)
    print(f"  {len(relevance):,} entries in relevance file")

    corpus_ids = sorted(corpus_tok.keys())
    query_ids  = sorted(query_tok.keys())

    corpus_seqs  = [corpus_tok[cid] for cid in corpus_ids]
    query_seqs   = [query_tok[qid]  for qid in query_ids]
    ground_truth = build_groundtruth(query_ids, corpus_ids, relevance)

    print_stats(corpus_seqs, query_seqs, ground_truth)

    # ── three-way comparison ──────────────────────────────────────
    print()
    results = run_comparison(corpus_seqs, query_seqs, ground_truth)

    save_csv([{"system": k, **v} for k, v in results.items()], args.out)

    # ── optional ablation ─────────────────────────────────────────
    if args.ablation:
        print("\nRunning component ablation…")
        ablation_rows = run_pmiquest_ablation(corpus_seqs, query_seqs, ground_truth)
        save_csv(ablation_rows, args.out.replace(".csv", "_ablation.csv"))

    print("\nDone.")


if __name__ == "__main__":
    main()
