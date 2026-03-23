"""
data/build_kathbath.py
========================
Converts the qbe_indicsuperb dataset into per-language
corpus_files.csv / query_files.csv / relevance.json / all_pairs.json
ready for run_kathbath_pmiquest.py.

ACTUAL dataset structure (confirmed from inspection):
  qbe_indicsuperb/
    <lang>/
      Audio/           ← corpus utterances (.wav, already 16 kHz)
      dev_queries/     ← 10 spoken queries (.wav)
      eval_queries/    ← 40 spoken queries (.wav)
      scoring/
        dev/
          <lang>.ecf.xml    ← which Audio/ files are in scope
          <lang>.tlist.xml  ← query term IDs (termid = query filename stem)
        eval/
          <lang>.ecf.xml
          <lang>.tlist.xml
          <lang>.rttm       ← ground truth (spoken term occurrences)

Usage
-----
# Use eval split — it has 40 queries + ground truth rttm
python prepare_kathbath_qbe.py \\
    --data_dir qbe_data/qbe_indicsuperb \\
    --out_dir  kathbath_ready/ \\
    --split    eval

# Dev split (10 queries, no rttm — MTWV cannot be computed without gt)
python prepare_kathbath_qbe.py \\
    --data_dir qbe_data/qbe_indicsuperb \\
    --out_dir  kathbath_ready/ \\
    --split    dev
"""

import argparse, csv, json, sys
import xml.etree.ElementTree as ET
from pathlib import Path

ALL_LANGS = [
    "bengali","gujarati","hindi","kannada","malayalam",
    "marathi","odia","punjabi","sanskrit","tamil","telugu","urdu"
]

def parse_ecf(ecf_path):
    """ecf.xml → list of corpus document stems in scope."""
    root = ET.parse(ecf_path).getroot()
    return [Path(ex.get("audio_filename","")).stem
            for ex in root.findall("excerpt")
            if ex.get("audio_filename","")]

def parse_tlist(tlist_path):
    """tlist.xml → list of query termids (= query wav filename stems)."""
    root = ET.parse(tlist_path).getroot()
    return [t.get("termid","") for t in root.findall("term") if t.get("termid","")]

def parse_rttm(rttm_path):
    """
    Parse IndicSUPERB rttm (transcript format, not NIST STD term-detection).

    Format observed:
      SPEAKER  <doc_stem>  1  <tbeg>  <dur>  <NA>  <NA>  SELF  <NA>
      LEXEME   <doc_stem>  1  <tbeg>  <dur>  <termid>  lex  SELF  <NA>

    Every LEXEME line is a positive occurrence:
      col[1] = corpus document stem
      col[5] = termid (matches tlist termid = query wav filename stem)

    Returns: {termid: [doc_stem, ...]}
    """
    rel = {}
    with open(rttm_path) as f:
        raw_lines = f.readlines()

    lexeme_lines = [l.strip() for l in raw_lines if l.strip().startswith("LEXEME")]
    print(f"    rttm: {len(raw_lines)} lines, {len(lexeme_lines)} LEXEME entries")
    for l in lexeme_lines[:3]:
        print(f"      {l}")

    for line in raw_lines:
        p = line.strip().split()
        if len(p) < 6 or p[0] != "LEXEME":
            continue
        doc_stem = Path(p[1]).stem   # corpus document
        termid   = p[5]              # query term ID
        rel.setdefault(termid, []).append(doc_stem)

    return rel

def process_language(lang, lang_dir, split, out_dir):
    audio_dir   = lang_dir / "Audio"
    query_dir   = lang_dir / ("dev_queries" if split == "dev" else "eval_queries")
    scoring_dir = lang_dir / "scoring" / split
    ecf_path    = scoring_dir / f"{lang}.ecf.xml"
    tlist_path  = scoring_dir / f"{lang}.tlist.xml"
    rttm_path   = scoring_dir / f"{lang}.rttm"

    for p, label in [(audio_dir,"Audio/"), (query_dir,f"{split}_queries/"),
                     (ecf_path,"ecf.xml"), (tlist_path,"tlist.xml")]:
        if not Path(p).exists():
            print(f"  [{lang}] Missing {label} — skipping"); return None

    doc_stems   = parse_ecf(ecf_path)
    query_stems = parse_tlist(tlist_path)
    print(f"\n  [{lang.upper()}]  {len(query_stems)} queries  {len(doc_stems)} docs")

    # Verify files exist
    corpus_wavs = [(s, str(audio_dir / f"{s}.wav"))
                   for s in doc_stems if (audio_dir / f"{s}.wav").exists()]
    query_wavs  = [(s, str(query_dir / f"{s}.wav"))
                   for s in query_stems if (query_dir / f"{s}.wav").exists()]

    missing = len(doc_stems) - len(corpus_wavs)
    if missing: print(f"    WARNING: {missing} corpus wavs not on disk")

    # Ground truth
    if rttm_path.exists():
        relevance = parse_rttm(rttm_path)
        n_pos = sum(len(v) for v in relevance.values())
        print(f"    Relevance (rttm): {len(relevance)} queries / {n_pos} positive pairs")
    else:
        relevance = {}
        print(f"    WARNING: no rttm — relevance empty. "
              f"Run with --split eval for proper MTWV.")

    all_pairs = [[q, d] for q,_ in query_wavs for d,_ in corpus_wavs]
    print(f"    Pairs: {len(all_pairs):,}")

    lang_out = Path(out_dir) / lang / split
    lang_out.mkdir(parents=True, exist_ok=True)

    with open(lang_out / "corpus_files.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem","path"])
        for s,p in corpus_wavs: w.writerow([s,p])

    with open(lang_out / "query_files.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stem","path"])
        for s,p in query_wavs: w.writerow([s,p])

    with open(lang_out / "relevance.json", "w") as f:
        json.dump(relevance, f, indent=2)

    with open(lang_out / "all_pairs.json", "w") as f:
        json.dump(all_pairs, f)

    return {
        "lang": lang, "split": split,
        "n_queries": len(query_wavs), "n_docs": len(corpus_wavs),
        "n_pairs": len(all_pairs), "n_positives": sum(len(v) for v in relevance.values()),
        "has_ground_truth": len(relevance) > 0,
        "corpus_csv":     str(lang_out / "corpus_files.csv"),
        "query_csv":      str(lang_out / "query_files.csv"),
        "relevance_json": str(lang_out / "relevance.json"),
        "pairs_json":     str(lang_out / "all_pairs.json"),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--data_dir", required=True,
                        help="Path to qbe_indicsuperb/ (contains bengali/, hindi/, etc.)")
    parser.add_argument("--out_dir",  default="kathbath_ready")
    parser.add_argument("--split",    default="eval", choices=["dev","eval"],
                        help="eval (recommended): 40 queries + rttm ground truth. "
                             "dev: 10 queries, no rttm. Default: eval")
    parser.add_argument("--langs",    nargs="+", default=ALL_LANGS)
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.exists():
        print(f"ERROR: {root} does not exist"); sys.exit(1)

    available = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    print(f"Found dirs: {sorted(available.keys())}")
    print(f"Split: {args.split}  |  Languages: {args.langs}\n")

    manifest = []
    for lang in args.langs:
        lang_dir = available.get(lang.lower())
        if lang_dir is None:
            print(f"WARNING: no directory for '{lang}'"); continue
        info = process_language(lang, lang_dir, args.split, args.out_dir)
        if info: manifest.append(info)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_{args.split}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*55}")
    print(f"Done. {len(manifest)}/{len(args.langs)} languages.")
    print(f"With ground truth: {sum(1 for m in manifest if m['has_ground_truth'])}/{len(manifest)}")
    print(f"Manifest → {manifest_path}")
    print(f"\nNext:")
    print(f"  python run_kathbath_pmiquest.py \\")
    print(f"      --manifest {manifest_path} \\")
    print(f"      --out_dir  kathbath_results/")
