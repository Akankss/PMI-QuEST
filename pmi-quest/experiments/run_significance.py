"""
experiments/run_significance.py
==========================
Runs paired Wilcoxon signed-rank tests comparing PMI-QuEST vs H-QuEST
on per-query AP scores for:

  1. LibriSpeech word mode — primary config (wav2vec2-base CNN)
  2. LibriSpeech word mode — best config   (WavLM-base L7)
  3. Kathbath cross-lingual — all 12 languages combined (XLS-R-300M L7)

Outputs:
  - Console table of W-statistic, p-value, effect size (Cohen's d)
  - LaTeX snippet ready to paste into Table I and Table V footnotes
  - CSV of per-query AP scores for all systems

Usage
-----
python run_significance_tests.py \\
    --libri_corpus_w2v   tokenised/wav2vec2-base_l0_k100/corpus.csv \\
    --libri_query_w2v    tokenised/wav2vec2-base_l0_k100/queries.csv \\
    --libri_corpus_wlm   tokenised/wavlm-base_l7_k100/corpus.csv \\
    --libri_query_wlm    tokenised/wavlm-base_l7_k100/queries.csv \\
    --relevance          qbe_librispeech/metadata/relevance.json \\
    --kathbath_manifest  kathbath_ready/manifest_eval.json \\
    --kathbath_results   kathbath_results/ \\
    --out_dir            results/significance/
"""

import argparse, csv, json, sys
from pathlib import Path

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    """Load corpus/query CSV → {stem: [int, ...]}.
    Handles both column-name formats:
      DictReader style: Filename,Data  (layer sweep output)
      Positional style: stem,tokens    (pmiquest_system output)
    Strips .wav extension from stems so keys match relevance.json.
    """
    seqs = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try both known column names
            fname = (row.get("Filename") or row.get("filename")
                     or row.get("stem")   or "")
            data  = (row.get("Data")     or row.get("data")
                     or row.get("tokens") or "")
            if not fname or not data:
                # Fallback: first two columns positionally
                vals = list(row.values())
                fname, data = vals[0], vals[1] if len(vals) > 1 else ("", "")
            stem = Path(fname).stem          # strips .wav extension
            data = data.strip()
            if data.startswith("["):
                import json as _json
                tokens = _json.loads(data)
            else:
                tokens = [int(x) for x in data.split(",") if x.strip()]
            if stem:
                seqs[stem] = tokens
    return seqs


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Per-query AP computation
# ─────────────────────────────────────────────────────────────────────────────

def per_query_ap(ranked_lists, relevance):
    """
    Returns dict {qid: AP} for every query that has at least one relevant doc.
    ranked_lists: {qid: [doc_id, ...]}
    relevance:    {qid: [doc_id, ...]}
    """
    aps = {}
    for qid, ranked in ranked_lists.items():
        rel = set(relevance.get(qid, []))
        if not rel:
            continue
        n_rel, ap = 0, 0.0
        for rank, did in enumerate(ranked, 1):
            if did in rel:
                n_rel += 1
                ap += n_rel / rank
        aps[qid] = ap / len(rel)
    return aps


def rank_system(system, queries, corpus_ids, relevance):
    """Run system.rank() for all queries; return {qid: [doc_ids]}."""
    ranked = {}
    for qid, q_seq in queries.items():
        if qid not in relevance:
            continue
        indices = system.rank(q_seq)
        ranked[qid] = [corpus_ids[i] for i in indices if i < len(corpus_ids)]
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon test + effect size
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_test(ap_a, ap_b, name_a="PMI-QuEST", name_b="H-QuEST"):
    """
    Paired Wilcoxon signed-rank test on matched per-query AP scores.
    Returns dict with W, p, effect size (Cohen's d on differences), direction.
    Only uses queries present in both dicts.
    """
    common = sorted(set(ap_a) & set(ap_b))
    if len(common) < 10:
        return {"error": f"Only {len(common)} common queries — too few"}

    a = np.array([ap_a[q] for q in common])
    b = np.array([ap_b[q] for q in common])
    diff = a - b

    # Wilcoxon signed-rank (two-sided)
    W, p = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')

    # One-sided p (PMI > H-QuEST)
    _, p_greater = stats.wilcoxon(a, b, alternative='greater', zero_method='wilcox')

    # Effect size: Cohen's d on the differences
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)

    # r = Z / sqrt(N)  (matched-pairs effect size)
    N = len(common)
    Z = stats.norm.ppf(1 - p / 2) * np.sign(np.mean(diff))
    r = Z / np.sqrt(N)

    return {
        "n_queries":  N,
        "mean_a":     float(np.mean(a)),
        "mean_b":     float(np.mean(b)),
        "mean_diff":  float(np.mean(diff)),
        "W":          float(W),
        "p_twosided": float(p),
        "p_greater":  float(p_greater),
        "cohens_d":   float(d),
        "r":          float(r),
        "sig_01":     p < 0.01,
        "sig_05":     p < 0.05,
        "name_a":     name_a,
        "name_b":     name_b,
    }


def sig_marker(result):
    """Return significance marker string."""
    if result.get("error"):
        return "n/a"
    if result["p_twosided"] < 0.001:
        return "$p{<}0.001$"
    elif result["p_twosided"] < 0.01:
        return "$p=" + f"{result['p_twosided']:.3f}$"
    elif result["p_twosided"] < 0.05:
        return "$p=" + f"{result['p_twosided']:.3f}$"
    else:
        return "$p=" + f"{result['p_twosided']:.3f}$ (n.s.)"


def print_result(label, result):
    if result.get("error"):
        print(f"  {label}: {result['error']}")
        return
    p = result["p_twosided"]
    pg = result["p_greater"]
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    print(f"  {label}")
    print(f"    N={result['n_queries']}  "
          f"MAP_PMI={result['mean_a']:.4f}  MAP_HQ={result['mean_b']:.4f}  "
          f"Δ={result['mean_diff']:+.4f}")
    print(f"    W={result['W']:.1f}  p(2-sided)={p:.4f}{sig}  "
          f"p(PMI>HQ)={pg:.4f}  d={result['cohens_d']:.3f}  r={result['r']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# LibriSpeech evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_librispeech(corpus_csv, query_csv, relevance, label):
    try:
        from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest

    corpus     = load_csv(corpus_csv)
    queries    = load_csv(query_csv)
    corpus_list = list(corpus.values())
    corpus_ids  = list(corpus.keys())

    print(f"\n  [{label}] N={len(corpus)}  Q={len(queries)}")

    # Fit systems
    bl = TFIDFBaseline(); bl.fit(corpus_list)
    hq = HQuEST(hnsw_k=50); hq.fit(corpus_list)
    pq = PMIQuest(pmi_tau=1.5, bigram_weight=0.5,
                  use_pmitd=False, hnsw_k=50); pq.fit(corpus_list)

    # Rank
    rl_bl = rank_system(bl, queries, corpus_ids, relevance)
    rl_hq = rank_system(hq, queries, corpus_ids, relevance)
    rl_pq = rank_system(pq, queries, corpus_ids, relevance)

    # Per-query AP
    ap_bl = per_query_ap(rl_bl, relevance)
    ap_hq = per_query_ap(rl_hq, relevance)
    ap_pq = per_query_ap(rl_pq, relevance)

    print(f"    TF-IDF   MAP={np.mean(list(ap_bl.values())):.4f}")
    print(f"    H-QuEST  MAP={np.mean(list(ap_hq.values())):.4f}")
    print(f"    PMI-QuEST MAP={np.mean(list(ap_pq.values())):.4f}")

    # Wilcoxon: PMI vs H-QuEST
    res_pmi_hq = wilcoxon_test(ap_pq, ap_hq, "PMI-QuEST", "H-QuEST")
    # Wilcoxon: PMI vs TF-IDF
    res_pmi_bl = wilcoxon_test(ap_pq, ap_bl, "PMI-QuEST", "TF-IDF")
    # Wilcoxon: H-QuEST vs TF-IDF
    res_hq_bl  = wilcoxon_test(ap_hq, ap_bl, "H-QuEST",   "TF-IDF")

    return {
        "ap_bl": ap_bl, "ap_hq": ap_hq, "ap_pq": ap_pq,
        "pmi_vs_hq": res_pmi_hq,
        "pmi_vs_bl": res_pmi_bl,
        "hq_vs_bl":  res_hq_bl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Kathbath evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_kathbath(manifest_path, results_dir):
    """
    Load per-query AP scores from cached Kathbath token CSVs.
    Runs PMI-QuEST and H-QuEST fresh (using cached token files).
    Pools all 12 × 40 = 480 query AP scores for the combined test.
    Also runs per-language tests.
    """
    try:
        from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest

    manifest = load_json(manifest_path)
    results_dir = Path(results_dir)

    # Pool across all languages
    all_ap_hq  = {}
    all_ap_pq  = {}
    per_lang   = {}

    for entry in manifest:
        lang = entry["lang"]
        # Find cached token CSVs
        # Standard path: kathbath_results/<lang>/xlsr-300m_l7_k100/corpus.csv
        tok_dir = results_dir / lang / "xlsr-300m_l7_k100"
        corpus_csv = tok_dir / "corpus.csv"
        query_csv  = tok_dir / "queries.csv"

        if not corpus_csv.exists() or not query_csv.exists():
            print(f"  [{lang.upper()}] token CSVs not found — skipping")
            continue

        corpus  = load_csv(str(corpus_csv))
        queries = load_csv(str(query_csv))
        relevance = load_json(entry["relevance_json"])

        corpus_list = list(corpus.values())
        corpus_ids  = list(corpus.keys())
        query_ids   = [q for q in queries if q in relevance and relevance[q]]

        if not query_ids:
            print(f"  [{lang.upper()}] no queries with relevance — skipping")
            continue

        queries_filtered = {q: queries[q] for q in query_ids}

        hq = HQuEST(hnsw_k=50); hq.fit(corpus_list)
        pq = PMIQuest(pmi_tau=0.5, bigram_weight=0.5,
                      use_pmitd=False, hnsw_k=50); pq.fit(corpus_list)

        rl_hq = rank_system(hq, queries_filtered, corpus_ids, relevance)
        rl_pq = rank_system(pq, queries_filtered, corpus_ids, relevance)

        ap_hq = per_query_ap(rl_hq, relevance)
        ap_pq = per_query_ap(rl_pq, relevance)

        # Namespace query IDs by language to avoid clashes
        for qid, val in ap_hq.items():
            all_ap_hq[f"{lang}::{qid}"] = val
        for qid, val in ap_pq.items():
            all_ap_pq[f"{lang}::{qid}"] = val

        # Per-language test
        per_lang[lang] = wilcoxon_test(ap_pq, ap_hq)
        hq_map = np.mean(list(ap_hq.values()))
        pq_map = np.mean(list(ap_pq.values()))
        p = per_lang[lang]["p_twosided"]
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "n.s."))
        print(f"  [{lang.upper():10s}]  HQ={hq_map:.4f}  PMI={pq_map:.4f}  "
              f"p={p:.4f}{sig}")

    # Combined test across all languages
    combined = wilcoxon_test(all_ap_pq, all_ap_hq, "PMI-QuEST", "H-QuEST")

    return {"per_lang": per_lang, "combined": combined,
            "all_ap_hq": all_ap_hq, "all_ap_pq": all_ap_pq}


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX output
# ─────────────────────────────────────────────────────────────────────────────

def make_latex_footnote(res_w2v, res_wlm, res_kath):
    """Generate LaTeX footnote text for significance markers."""

    def fmt(res):
        if res.get("error"):
            return "n/a"
        p = res["p_twosided"]
        d = res["cohens_d"]
        if p < 0.001:
            return f"$p{{<}}0.001$, $d={d:.2f}$"
        else:
            return f"$p={p:.3f}$, $d={d:.2f}$"

    lines = [
        r"% ── Significance test footnotes ─────────────────────────────",
        r"% Add to Table I caption:",
        r"% $^\dagger$ Paired Wilcoxon signed-rank test (PMI-QuEST vs.\ H-QuEST).",
        f"% wav2vec2-base CNN ({res_w2v.get('n_queries','?')} queries): {fmt(res_w2v)}",
        f"% WavLM-base L7    ({res_wlm.get('n_queries','?')} queries): {fmt(res_wlm)}",
        r"",
        r"% Add to Table V caption:",
        r"% $^\ddagger$ Combined Wilcoxon test across all 12 languages.",
        f"% Kathbath combined ({res_kath.get('n_queries','?')} queries): {fmt(res_kath)}",
    ]
    return "\n".join(lines)


def make_latex_table_markers(results_dict):
    """
    Returns a string showing significance markers (*, **, ***)
    to add as a row to Table I and Table V.
    """
    out = []
    out.append("\n% ── Significance markers for Table I ───────────────────")
    out.append("% Add below PMI-QuEST row:")
    for label, res in results_dict.items():
        if res.get("error"):
            continue
        p = res["p_twosided"]
        marker = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "n.s."))
        out.append(f"%   {label}: {marker} (p={p:.4f}, d={res['cohens_d']:.3f})")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Save per-query AP CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_ap_csv(path, ap_dict_by_system):
    """Save per-query APs for all systems to CSV."""
    all_qids = sorted(set().union(*[set(d.keys()) for d in ap_dict_by_system.values()]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        systems = list(ap_dict_by_system.keys())
        w.writerow(['query_id'] + systems)
        for qid in all_qids:
            row = [qid] + [f"{ap_dict_by_system[s].get(qid, ''):.6f}"
                           if qid in ap_dict_by_system[s] else ''
                           for s in systems]
            w.writerow(row)
    print(f"  Per-query AP saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--libri_corpus_w2v',  required=True)
    p.add_argument('--libri_query_w2v',   required=True)
    p.add_argument('--libri_corpus_wlm',  required=True)
    p.add_argument('--libri_query_wlm',   required=True)
    p.add_argument('--relevance',         required=True)
    p.add_argument('--kathbath_manifest', required=True)
    p.add_argument('--kathbath_results',  required=True)
    p.add_argument('--out_dir',           default='results/significance')
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    relevance = load_json(args.relevance)

    print("=" * 62)
    print("Paired Wilcoxon Signed-Rank Tests")
    print("PMI-QuEST vs H-QuEST  (per-query AP)")
    print("=" * 62)

    # ── LibriSpeech wav2vec2-base CNN ────────────────────────────────────────
    print("\n── LibriSpeech wav2vec2-base CNN ──")
    r_w2v = eval_librispeech(
        args.libri_corpus_w2v, args.libri_query_w2v, relevance,
        "wav2vec2-base CNN")
    print_result("PMI-QuEST vs H-QuEST",  r_w2v["pmi_vs_hq"])
    print_result("PMI-QuEST vs TF-IDF",   r_w2v["pmi_vs_bl"])
    print_result("H-QuEST   vs TF-IDF",   r_w2v["hq_vs_bl"])
    save_ap_csv(out / "libri_w2v_ap.csv",
                {"TF-IDF":     r_w2v["ap_bl"],
                 "H-QuEST":    r_w2v["ap_hq"],
                 "PMI-QuEST":  r_w2v["ap_pq"]})

    # ── LibriSpeech WavLM-base L7 ────────────────────────────────────────────
    print("\n── LibriSpeech WavLM-base L7 ──")
    r_wlm = eval_librispeech(
        args.libri_corpus_wlm, args.libri_query_wlm, relevance,
        "WavLM-base L7")
    print_result("PMI-QuEST vs H-QuEST",  r_wlm["pmi_vs_hq"])
    print_result("PMI-QuEST vs TF-IDF",   r_wlm["pmi_vs_bl"])
    save_ap_csv(out / "libri_wlm_ap.csv",
                {"TF-IDF":     r_wlm["ap_bl"],
                 "H-QuEST":    r_wlm["ap_hq"],
                 "PMI-QuEST":  r_wlm["ap_pq"]})

    # ── Kathbath cross-lingual ───────────────────────────────────────────────
    print("\n── Kathbath (12 languages) ──")
    r_kath = eval_kathbath(args.kathbath_manifest, args.kathbath_results)
    print()
    print_result("PMI-QuEST vs H-QuEST (combined, 480 queries)",
                 r_kath["combined"])
    save_ap_csv(out / "kathbath_ap.csv",
                {"H-QuEST":   r_kath["all_ap_hq"],
                 "PMI-QuEST": r_kath["all_ap_pq"]})

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("Summary — PMI-QuEST vs H-QuEST")
    print("=" * 62)
    print(f"  {'Evaluation':<35} {'N':>5} {'ΔMAP':>7} {'p':>8} {'d':>7} {'sig':>5}")
    print("  " + "─" * 60)
    rows = [
        ("LibriSpeech w2v-base CNN (Table I)",  r_w2v["pmi_vs_hq"]),
        ("LibriSpeech WavLM-base L7 (best)",    r_wlm["pmi_vs_hq"]),
        ("Kathbath combined 12 lang (Table V)", r_kath["combined"]),
    ]
    for label, res in rows:
        if res.get("error"):
            print(f"  {label:<35}  ERROR: {res['error']}")
            continue
        p   = res["p_twosided"]
        sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "n.s."))
        print(f"  {label:<35} {res['n_queries']:>5} "
              f"{res['mean_diff']:>+7.4f} {p:>8.4f} {res['cohens_d']:>7.3f} {sig:>5}")

    # ── LaTeX snippets ───────────────────────────────────────────────────────
    latex_path = out / "significance_latex.tex"
    with open(latex_path, 'w') as f:
        f.write("% Significance test results for Paper A\n")
        f.write("% Generated by run_significance_tests.py\n\n")

        # Per-language Kathbath table with p-values
        f.write("% ── Per-language Kathbath p-values (add to Table V) ──\n")
        f.write("% Language      HQ MAP   PMI MAP   p-value   sig\n")
        langs_order = ["sanskrit","kannada","gujarati","odia","punjabi",
                       "tamil","telugu","marathi","urdu","bengali",
                       "hindi","malayalam"]
        for lang in langs_order:
            res = r_kath["per_lang"].get(lang)
            if res and not res.get("error"):
                p = res["p_twosided"]
                sig = "***" if p<0.001 else("**" if p<0.01 else("*" if p<0.05 else "n.s."))
                hq  = res["mean_b"]
                pmi = res["mean_a"]
                f.write(f"% {lang:<12}  {hq:.4f}   {pmi:.4f}   {p:.4f}   {sig}\n")

        f.write("\n")
        f.write(make_latex_footnote(
            r_w2v["pmi_vs_hq"], r_wlm["pmi_vs_hq"], r_kath["combined"]))
        f.write("\n\n")

        # Table I caption addition
        f.write("% ── Updated Table I caption with significance ──\n")
        w2v_res = r_w2v["pmi_vs_hq"]
        if not w2v_res.get("error"):
            p = w2v_res["p_twosided"]
            d = w2v_res["cohens_d"]
            p_str = "$p{<}0.001$" if p < 0.001 else "$p=" + f"{p:.3f}$"
            f.write(f"% PMI-QuEST improvement over H-QuEST is statistically\n")
            f.write(f"% significant ({p_str}, Cohen's $d={d:.2f}$, paired\n")
            f.write(f"% Wilcoxon signed-rank test, $N={w2v_res['n_queries']}$ queries).\n")

    print(f"\n  LaTeX snippets → {latex_path}")
    print(f"  Per-query AP CSVs → {out}/")


if __name__ == '__main__':
    main()
