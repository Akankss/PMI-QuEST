"""
experiments/run_cross_lingual.py — Step 2
===================================
Runs PMI-QuEST on the prepared Kathbath QbE data (output of
prepare_kathbath_qbe.py) using XLS-R-300M tokens, evaluating
with MTWV (Maximum Term Weighted Value) across all 12 languages.

Key design decisions:
  1. Multilingual tokeniser: facebook/wav2vec2-xls-r-300m
     — Pretrained on 128 languages including all 12 Indic ones
     — Layer 7 (our sweep found this to be the peak; transferable
       to XLS-R since it has the same transformer architecture)
     — k=100 clusters (same as LibriSpeech experiments)

  2. MTWV scoring, not MAP
     — IndicSUPERB QbE is a pairwise detection task (match/no-match)
     — Our PMI-TF-IDF cosine score serves directly as the detection
       score; MTWV is then maximised over a threshold sweep

  3. Caching
     — k-means centroids and token CSVs are cached per language
     — Re-runs use cached files unless --force

  4. All three systems reported: TF-IDF, H-QuEST, PMI-QuEST
     — Exactly matching our LibriSpeech comparison table

Usage
-----
python run_kathbath_pmiquest.py \\
    --manifest  kathbath_ready/manifest_dev.json \\
    --out_dir   kathbath_results/ \\
    --model     xlsr-300m \\
    --layer     7

# Quick test on one language
python run_kathbath_pmiquest.py \\
    --manifest  kathbath_ready/manifest_dev.json \\
    --langs     hindi \\
    --out_dir   kathbath_results/

# Already tokenised — skip to evaluation
python run_kathbath_pmiquest.py \\
    --manifest  kathbath_ready/manifest_dev.json \\
    --out_dir   kathbath_results/ \\
    --skip_tokenisation
"""

import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "tokeniser"))
sys.path.insert(0, str(_ROOT))

MULTILINGUAL_MODELS = {
    "xlsr-300m":     "facebook/wav2vec2-xls-r-300m",
    "xlsr-1b":       "facebook/wav2vec2-xls-r-1b",
    "indicwav2vec":  "ai4bharat/indicwav2vec",
    "mms-300m":      "facebook/mms-300m",
    # English baselines (suboptimal, useful as ablation)
    "hubert-base":   "facebook/hubert-base-ls960",
    "wavlm-base":    "microsoft/wavlm-base",
}

DEFAULT_LAYER = {
    "xlsr-300m":    7,
    "xlsr-1b":      9,
    "indicwav2vec": 6,
    "mms-300m":     7,
    "hubert-base":  7,
    "wavlm-base":   7,
}


# ─────────────────────────────────────────────────────────────────────────────
# MTWV
# ─────────────────────────────────────────────────────────────────────────────

def compute_mtwv(scores_and_labels: list, n_thresholds: int = 300) -> dict:
    """
    MTWV = max_θ { 1 − [P_miss(θ) + β·P_fa(θ)] }
    β = N_false / N_true  (standard SUPERB formulation)
    """
    arr     = np.array(scores_and_labels, dtype=float)   # (N,2): score, label
    labels  = arr[:, 1].astype(int)
    scores  = arr[:, 0]
    n_true  = labels.sum()
    n_false = len(labels) - n_true

    if n_true == 0:
        return {"mtwv": 0.0, "p_miss": 1.0, "p_fa": 0.0,
                "n_true": 0, "n_false": int(n_false)}

    beta    = n_false / n_true
    thetas  = np.linspace(scores.min(), scores.max(), n_thresholds)
    best    = -np.inf
    best_pm = 1.0
    best_pf = 0.0

    for theta in thetas:
        predicted = scores >= theta
        pm = ((labels == 1) & ~predicted).sum() / n_true
        pf = ((labels == 0) &  predicted).sum() / max(n_false, 1)
        mtwv = 1.0 - (pm + beta * pf)
        if mtwv > best:
            best, best_pm, best_pf = mtwv, float(pm), float(pf)

    return {"mtwv": float(best), "p_miss": best_pm, "p_fa": best_pf,
            "n_true": int(n_true), "n_false": int(n_false), "beta": float(beta)}


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_16k(path: str) -> np.ndarray:
    """Load audio and resample to 16 kHz."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


class XLSRTokenizer:
    """
    Thin wrapper around facebook/wav2vec2-xls-r-300m (or any wav2vec2-family
    model) that extracts a specific transformer layer and applies k-means.
    """
    def __init__(self, hf_id: str, layer: int, n_clusters: int = 100):
        self.hf_id      = hf_id
        self.layer      = layer
        self.n_clusters = n_clusters
        self._model     = None
        self._proc      = None
        self._device    = None
        self.kmeans     = None

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Loading {self.hf_id} → layer {self.layer} "
              f"(device={self._device}) …")
        self._proc  = Wav2Vec2FeatureExtractor.from_pretrained(self.hf_id)
        self._model = Wav2Vec2Model.from_pretrained(
            self.hf_id, output_hidden_states=True
        ).to(self._device).eval()
        n = sum(p.numel() for p in self._model.parameters())
        print(f"    Loaded. Params: {n/1e6:.0f}M")

    def extract(self, audio: np.ndarray) -> np.ndarray:
        import torch
        self._load_model()
        inputs = self._proc(audio, sampling_rate=16000,
                            return_tensors="pt", padding=True)
        iv = inputs.input_values.to(self._device)
        with torch.no_grad():
            if self.layer == 0:
                feats = self._model.feature_extractor(iv)
                feats = feats.squeeze(0).T
            else:
                out   = self._model(iv, output_hidden_states=True)
                feats = out.hidden_states[self.layer].squeeze(0)
        return feats.cpu().numpy().astype(np.float32)

    def fit(self, paths: list, max_frames: int = 500_000):
        from sklearn.cluster import MiniBatchKMeans
        from tqdm import tqdm
        self._load_model()
        all_feats, total = [], 0
        for p in tqdm(paths, desc="  k-means fit"):
            try:
                feats = self.extract(load_audio_16k(str(p)))
                all_feats.append(feats)
                total += len(feats)
                if total >= max_frames:
                    break
            except Exception as e:
                print(f"    WARN {Path(p).name}: {e}")
        X = np.vstack(all_feats)
        if len(X) > max_frames:
            idx = np.random.default_rng(42).choice(len(X), max_frames, replace=False)
            X   = X[idx]
        print(f"    Fitting k={self.n_clusters} on {len(X):,} frames …")
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters,
                                      batch_size=10_000, n_init=10,
                                      random_state=42, verbose=0)
        self.kmeans.fit(X)
        print(f"    Inertia: {self.kmeans.inertia_:.2f}")

    def tokenize(self, path: str) -> list:
        feats = self.extract(load_audio_16k(path))
        return self.kmeans.predict(feats).tolist()

    def save(self, path: str):
        np.save(path, self.kmeans.cluster_centers_)

    def load_centroids(self, path: str):
        from sklearn.cluster import MiniBatchKMeans
        centers = np.load(path)
        self.kmeans = MiniBatchKMeans(n_clusters=len(centers))
        self.kmeans.cluster_centers_ = centers
        self._load_model()


def tokenise_language(lang_info: dict, tok_dir: Path,
                      tok: XLSRTokenizer, force: bool = False) -> tuple:
    corpus_csv = tok_dir / "corpus.csv"
    query_csv  = tok_dir / "queries.csv"
    centroid   = tok_dir / "kmeans_centroids.npy"

    tok_dir.mkdir(parents=True, exist_ok=True)

    if corpus_csv.exists() and query_csv.exists() and not force:
        print(f"  Cached token CSVs found — skipping tokenisation.")
        return str(corpus_csv), str(query_csv)

    # Load file lists
    corpus_rows = []
    with open(lang_info["corpus_csv"]) as f:
        corpus_rows = list(csv.DictReader(f))
    query_rows = []
    with open(lang_info["query_csv"]) as f:
        query_rows = list(csv.DictReader(f))

    corpus_paths = [r["path"] for r in corpus_rows]

    # Fit or load k-means
    if centroid.exists() and not force:
        print(f"  Loading cached centroids …")
        tok.load_centroids(str(centroid))
    else:
        tok.fit(corpus_paths)
        tok.save(str(centroid))

    def _write_csv(rows, out_path, desc):
        from tqdm import tqdm
        results = []
        for r in tqdm(rows, desc=f"  tokenise {desc}"):
            try:
                tokens = tok.tokenize(r["path"])
                results.append((r["stem"], tokens))
            except Exception as e:
                print(f"    WARN {r['stem']}: {e}")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Filename", "Data"])
            for stem, tokens in results:
                w.writerow([stem + ".wav", ",".join(map(str, tokens))])
        print(f"  → {out_path} ({len(results)} rows)")

    _write_csv(corpus_rows, str(corpus_csv), "corpus")
    _write_csv(query_rows,  str(query_csv),  "queries")
    return str(corpus_csv), str(query_csv)


# ─────────────────────────────────────────────────────────────────────────────
# Load token CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_token_csv(path: str) -> dict:
    seqs = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stem   = Path(row.get("Filename","")).stem
            data   = row.get("Data","").strip()
            tokens = json.loads(data) if data.startswith("[") else \
                     [int(x) for x in data.split(",") if x.strip()]
            seqs[stem] = tokens
    return seqs


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — pairwise scoring + MTWV
# ─────────────────────────────────────────────────────────────────────────────



def _eval(ranked_lists: dict, relevance: dict) -> dict:
    """MAP, P@1, P@5, P@10 — identical to LibriSpeech evaluation."""
    aps, p1s, p5s, p10s = [], [], [], []
    for qid, ranked in ranked_lists.items():
        if qid not in relevance:
            continue
        rel = set(relevance[qid])
        if not rel:
            continue
        n_rel, ap = 0, 0.0
        for rank, did in enumerate(ranked, 1):
            if did in rel:
                n_rel += 1
                ap += n_rel / rank
        aps.append(ap / len(rel))
        p1s.append(1.0  if ranked and ranked[0] in rel else 0.0)
        p5s.append(sum(1 for d in ranked[:5]  if d in rel) / 5)
        p10s.append(sum(1 for d in ranked[:10] if d in rel) / 10)
    return {
        "map":  float(np.mean(aps))  if aps  else 0.0,
        "p1":   float(np.mean(p1s))  if p1s  else 0.0,
        "p5":   float(np.mean(p5s))  if p5s  else 0.0,
        "p10":  float(np.mean(p10s)) if p10s else 0.0,
    }


def evaluate_language(lang_info: dict, corpus_tok: str,
                      query_tok: str, verbose: bool = True) -> dict:
    """
    Run TF-IDF, H-QuEST, PMI-QuEST on one Kathbath language.
    Reports MAP, P@1, P@5, P@10 — identical to LibriSpeech evaluation.
    """
    try:
        from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest

    lang    = lang_info["lang"]
    corpus  = load_token_csv(corpus_tok)
    queries = load_token_csv(query_tok)
    with open(lang_info["relevance_json"]) as f:
        relevance = json.load(f)

    corpus_list = list(corpus.values())
    corpus_ids  = list(corpus.keys())
    query_ids   = [qid for qid in queries if qid in relevance and relevance[qid]]

    vocab  = len(set(t for s in corpus_list for t in s))
    mean_q = float(np.mean([len(queries[q]) for q in query_ids])) if query_ids else 0.0
    rho    = mean_q / (float(np.mean([len(s) for s in corpus_list])) or 1.0)

    if verbose:
        print(f"\n  [{lang.upper()}]  V={vocab}  mean_q={mean_q:.1f}  rho={rho:.3f}  "
              f"queries_with_relevance={len(query_ids)}/{len(queries)}")

    row = {"lang": lang, "vocab": vocab, "rho": float(rho),
           "n_queries": len(query_ids), "n_docs": len(corpus_ids)}

    def rank_and_translate(system):
        """Run rank() for each query → {qid: [corpus_doc_ids]}"""
        ranked = {}
        for qid in query_ids:
            indices = system.rank(queries[qid])   # list of int indices into corpus_list
            ranked[qid] = [corpus_ids[i] for i in indices if i < len(corpus_ids)]
        return ranked

    # ── TF-IDF Baseline ───────────────────────────────────────────────────────
    t0 = time.time()
    bl = TFIDFBaseline()
    bl.fit(corpus_list)
    m = _eval(rank_and_translate(bl), relevance)
    row.update({f"tfidf_{k}": v for k, v in m.items()})
    if verbose:
        print(f"    TF-IDF:    MAP={m['map']:.4f}  P@1={m['p1']:.4f}  "
              f"P@5={m['p5']:.4f}  P@10={m['p10']:.4f}  [{time.time()-t0:.0f}s]")

    # ── H-QuEST ───────────────────────────────────────────────────────────────
    t0 = time.time()
    hq = HQuEST(hnsw_k=50)
    hq.fit(corpus_list)
    m = _eval(rank_and_translate(hq), relevance)
    row.update({f"hquest_{k}": v for k, v in m.items()})
    if verbose:
        print(f"    H-QuEST:   MAP={m['map']:.4f}  P@1={m['p1']:.4f}  "
              f"P@5={m['p5']:.4f}  P@10={m['p10']:.4f}  [{time.time()-t0:.0f}s]")

    # ── PMI-QuEST (proposed) ──────────────────────────────────────────────────
    t0 = time.time()
    pq = PMIQuest(pmi_tau=0.5, bigram_weight=0.5, use_pmitd=False, hnsw_k=50)
    pq.fit(corpus_list)
    m = _eval(rank_and_translate(pq), relevance)
    row.update({f"pmi_{k}": v for k, v in m.items()})
    if verbose:
        print(f"    PMI-QuEST: MAP={m['map']:.4f}  P@1={m['p1']:.4f}  "
              f"P@5={m['p5']:.4f}  P@10={m['p10']:.4f}  [{time.time()-t0:.0f}s]")

    hq_map = row["hquest_map"]
    row["pmi_map_gain"] = (row["pmi_map"] - hq_map) / max(hq_map, 1e-9)
    row["pmi_p1_gain"]  = (row["pmi_p1"]  - row["hquest_p1"]) / max(row["hquest_p1"], 1e-9)
    return row


def print_results(rows: list):
    print(f"\n{'='*80}")
    print("PMI-QuEST on KATHBATH — MAP / P@1 by language  (XLS-R-300M L7 k=100)")
    print(f"{'='*80}")
    print(f"  {'Language':<14}  {'TF-IDF MAP':>10}  {'HQ MAP':>8}  "
          f"{'PMI MAP':>8}  {'vs HQ':>7}  {'PMI P@1':>8}  {'HQ P@1':>7}")
    print("─" * 80)
    for r in sorted(rows, key=lambda x: x["pmi_map"], reverse=True):
        gain = r["pmi_map_gain"] * 100
        print(f"  {r['lang']:<14}  {r['tfidf_map']:>10.4f}  {r['hquest_map']:>8.4f}  "
              f"{r['pmi_map']:>8.4f}  {gain:>+6.1f}%  {r['pmi_p1']:>8.4f}  {r['hquest_p1']:>7.4f}")
    print("─" * 80)
    if rows:
        for key, label in [("tfidf_map","TF-IDF"), ("hquest_map","H-QuEST"),
                           ("pmi_map","PMI-QuEST")]:
            avg = np.mean([r[key] for r in rows])
            print(f"  AVG {label:<12}  MAP={avg:.4f}  "
                  f"P@1={np.mean([r[key.replace('map','p1')] for r in rows]):.4f}  "
                  f"P@5={np.mean([r[key.replace('map','p5')] for r in rows]):.4f}  "
                  f"P@10={np.mean([r[key.replace('map','p10')] for r in rows]):.4f}")
    print(f"{'='*80}")


def save_results(rows: list, path: str):
    fields = ["lang","n_queries","n_docs","vocab","rho",
              "tfidf_map","tfidf_p1","tfidf_p5","tfidf_p10",
              "hquest_map","hquest_p1","hquest_p5","hquest_p10",
              "pmi_map","pmi_p1","pmi_p5","pmi_p10",
              "pmi_map_gain","pmi_p1_gain"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n  Results saved → {path}")



# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--manifest",  required=True)
    parser.add_argument("--out_dir",   default="kathbath_results")
    parser.add_argument("--out_csv",   default=None)
    parser.add_argument("--model",     default="xlsr-300m",
                        choices=list(MULTILINGUAL_MODELS.keys()))
    parser.add_argument("--layer",     type=int, default=None)
    parser.add_argument("--n_clusters",type=int, default=100)
    parser.add_argument("--max_frames",type=int, default=500_000)
    parser.add_argument("--langs",     nargs="+", default=None)
    parser.add_argument("--skip_tokenisation", action="store_true")
    parser.add_argument("--force",     action="store_true")
    args = parser.parse_args()

    layer   = args.layer if args.layer is not None else DEFAULT_LAYER[args.model]
    hf_id   = MULTILINGUAL_MODELS[args.model]
    out_dir = Path(args.out_dir)
    out_csv = args.out_csv or str(out_dir / "kathbath_results.csv")

    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.langs:
        manifest = [m for m in manifest if m["lang"] in args.langs]

    print(f"\nKathbath QbE  —  {len(manifest)} languages")
    print(f"Model: {args.model}  HF: {hf_id}  Layer: {layer}  k={args.n_clusters}")

    # Build shared tokeniser (model loaded once, reused across languages)
    tok = XLSRTokenizer(hf_id, layer, args.n_clusters)

    all_rows = []

    for lang_info in manifest:
        lang    = lang_info["lang"]
        tok_dir = out_dir / lang / f"{args.model}_l{layer}_k{args.n_clusters}"

        print(f"\n{'='*60}")
        print(f"Language: {lang.upper()}")
        print(f"{'='*60}")

        # Tokenise
        if args.skip_tokenisation:
            corpus_tok = str(tok_dir / "corpus.csv")
            query_tok  = str(tok_dir / "queries.csv")
            if not Path(corpus_tok).exists():
                print(f"  Token CSVs not found — skipping."); continue
        else:
            try:
                corpus_tok, query_tok = tokenise_language(
                    lang_info, tok_dir, tok, args.force)
            except Exception as e:
                print(f"  Tokenisation failed: {e}"); continue

        # Evaluate
        try:
            row = evaluate_language(lang_info, corpus_tok, query_tok)
            all_rows.append(row)
        except Exception as e:
            import traceback
            print(f"  Evaluation failed: {e}")
            traceback.print_exc()

    print_results(all_rows)
    save_results(all_rows, out_csv)
