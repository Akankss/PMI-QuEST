"""
Audio → Discrete Token Sequences
=================================
Supports wav2vec 2.0, HuBERT, and WavLM, all producing integer token
sequences compatible with pmiquest_system.py.

All three models share the same 20ms frame rate (CNN stride), so token
counts are directly comparable across models for a given audio file.

Recommended layer per model
---------------------------
  wav2vec2-base / large   : CNN output (layer 0) — current default
                            OR transformer layer 6 (base) / 9 (large)
  hubert-base             : transformer layer 6  ← standard for QbE-STD
  hubert-large            : transformer layer 9
  wavlm-base              : transformer layer 6
  wavlm-base-plus         : transformer layer 6
  wavlm-large             : transformer layer 6 or 9

Layer 6 (base) and layer 9 (large) are the standard choices in the HuBERT
and WavLM papers for extracting discrete speech units.  The CNN-only
extraction used for wav2vec2 in our QbE-STD work is also valid but gives
context-free features; transformer layers give contextual features that
typically yield cleaner k-means clusters for phoneme-like units.

Usage examples
--------------
# Tokenise with HuBERT-base layer 6, k=100
python pmiquest/tokeniser.py \\
    --corpus_dir  qbe_librispeech/corpus_audio \\
    --query_dir   qbe_librispeech/query_audio \\
    --model       hubert-base \\
    --layer       6 \\
    --n_clusters  100 \\
    --out_dir     qbe_librispeech/hubert_base_l6_k100

# Tokenise with WavLM-base layer 6, k=200
python audio_tokenizer_v2.py \\
    --corpus_dir  qbe_librispeech/corpus_audio \\
    --query_dir   qbe_librispeech/query_audio \\
    --model       wavlm-base \\
    --layer       6 \\
    --n_clusters  200 \\
    --out_dir     qbe_librispeech/wavlm_base_l6_k200

# Load existing centroids (skip k-means fitting)
python audio_tokenizer_v2.py \\
    --corpus_dir  qbe_librispeech/corpus_audio \\
    --model       hubert-base \\
    --layer       6 \\
    --load_kmeans kmeans_hubert_base_l6_k100.npy \\
    --out_dir     qbe_librispeech/hubert_base_l6_k100

# Then run PMI-QuEST on the new tokens
python run_pmiquest_comparison.py \\
    --corpus    qbe_librispeech/hubert_base_l6_k100/corpus.csv \\
    --queries   qbe_librispeech/hubert_base_l6_k100/queries.csv \\
    --relevance relevance.json \\
    --out       results/hubert_base_l6_k100.csv \\
    --ablation
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

# Maps CLI name → HuggingFace model ID
MODEL_IDS = {
    # wav2vec 2.0
    "wav2vec2-base":         "facebook/wav2vec2-base",
    "wav2vec2-large":        "facebook/wav2vec2-large",
    "wav2vec2-large-robust": "facebook/wav2vec2-large-robust",

    # HuBERT
    "hubert-base":           "facebook/hubert-base-ls960",
    "hubert-large":          "facebook/hubert-large-ll60k",
    "hubert-xlarge":         "facebook/hubert-xlarge-ll60k",

    # WavLM
    "wavlm-base":            "microsoft/wavlm-base",
    "wavlm-base-plus":       "microsoft/wavlm-base-plus",
    "wavlm-large":           "microsoft/wavlm-large",

    # Multilingual / Indic — for Kathbath evaluation
    "mms-300m":              "facebook/mms-300m",
    "xlsr-53":               "facebook/wav2vec2-large-xlsr-53",
    "indicwav2vec":          "ai4bharat/indicwav2vec_v2_all",
}

# Recommended transformer layer per model (for phoneme-quality tokens)
RECOMMENDED_LAYER = {
    "wav2vec2-base":         6,    # CNN (layer=0) or transformer layer 6
    "wav2vec2-large":        9,
    "wav2vec2-large-robust": 9,
    "hubert-base":           6,    # from HuBERT paper
    "hubert-large":          9,    # from HuBERT paper
    "hubert-xlarge":         12,
    "wavlm-base":            6,
    "wavlm-base-plus":       6,
    "wavlm-large":           6,    # WavLM paper recommends layer 6 for tokens
    "mms-300m":              6,
    "xlsr-53":               9,
    "indicwav2vec":          6,
}

# Feature dimension per model
FEATURE_DIM = {
    "wav2vec2-base":         768,   # transformer hidden; CNN is 512
    "wav2vec2-large":        1024,
    "wav2vec2-large-robust": 1024,
    "hubert-base":           768,
    "hubert-large":          1024,
    "hubert-xlarge":         1280,
    "wavlm-base":            768,
    "wavlm-base-plus":       768,
    "wavlm-large":           1024,
    "mms-300m":              768,
    "xlsr-53":               1024,
    "indicwav2vec":          768,
}


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file → float32 mono numpy array at 16kHz."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        wav_t = torch.tensor(audio).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        audio = wav_t.squeeze().numpy()
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor — unified for all three model families
# ─────────────────────────────────────────────────────────────────────────────

class SpeechFeatureExtractor:
    """
    Unified feature extractor for wav2vec2, HuBERT, and WavLM.

    Parameters
    ----------
    model_name : str
        Short name from MODEL_IDS (e.g. 'hubert-base', 'wavlm-large').
    layer : int
        Transformer layer index (0 = CNN output, 1..N = transformer layers).
        layer=0 gives context-free CNN features (fast, wav2vec2-style).
        layer=6 or 9 gives contextual transformer features (HuBERT-style).
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model_name: str = "hubert-base",
        layer: int = 6,
        device: Optional[str] = None,
    ):
        if model_name not in MODEL_IDS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(MODEL_IDS.keys())}"
            )
        self.model_name  = model_name
        self.model_id    = MODEL_IDS[model_name]
        self.layer       = layer
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model      = None
        self._processor  = None
        self._family     = self._get_family(model_name)

        rec = RECOMMENDED_LAYER.get(model_name, 6)
        if layer > 0 and layer != rec:
            print(f"  [INFO] Recommended layer for {model_name} is {rec}; "
                  f"you chose {layer}.")

    @staticmethod
    def _get_family(name: str) -> str:
        if name.startswith("wav2vec2") or name in ("mms-300m", "xlsr-53"):
            return "wav2vec2"
        elif name.startswith("hubert") or name == "indicwav2vec":
            return "hubert"
        elif name.startswith("wavlm"):
            return "wavlm"
        raise ValueError(f"Cannot determine family for '{name}'")

    def _load_model(self):
        if self._model is not None:
            return

        from transformers import (
            Wav2Vec2Model, Wav2Vec2Processor,
            HubertModel,   Wav2Vec2FeatureExtractor,
            WavLMModel,
        )

        print(f"Loading {self.model_id}  (layer={self.layer}, "
              f"device={self.device}) …")

        if self._family == "wav2vec2":
            self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            self._model     = Wav2Vec2Model.from_pretrained(
                self.model_id, output_hidden_states=True
            )

        elif self._family == "hubert":
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_id
            )
            self._model = HubertModel.from_pretrained(
                self.model_id, output_hidden_states=True
            )

        elif self._family == "wavlm":
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_id
            )
            self._model = WavLMModel.from_pretrained(
                self.model_id, output_hidden_states=True
            )

        self._model = self._model.to(self.device)
        self._model.eval()
        print(f"  Loaded. Parameters: "
              f"{sum(p.numel() for p in self._model.parameters()):,}")

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features from a single audio array.

        Parameters
        ----------
        audio : np.ndarray
            Float32 mono audio at 16kHz.

        Returns
        -------
        np.ndarray, shape (T, D)
            One feature vector per 20ms frame.
            T ≈ len(audio) / 320.
            D = 512 for layer=0 (CNN), else model hidden size.
        """
        self._load_model()

        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            if self.layer == 0:
                # CNN feature extractor only — fast, no attention
                # All three families use the same CNN architecture
                feats = self._model.feature_extractor(input_values)
                # Shape: (1, 512, T) → (T, 512)
                feats = feats.squeeze(0).T
            else:
                # Full forward pass; extract hidden state at requested layer
                outputs = self._model(
                    input_values,
                    output_hidden_states=True,
                )
                # hidden_states is tuple of (batch, T, D):
                # index 0 = CNN projection, 1..N+1 = transformer layers
                # So transformer layer k → index k
                hidden_states = outputs.hidden_states  # tuple, len = N_layers+1
                max_layer = len(hidden_states) - 1
                if self.layer > max_layer:
                    raise ValueError(
                        f"Requested layer {self.layer} but model only has "
                        f"layers 0..{max_layer}."
                    )
                feats = hidden_states[self.layer]  # (1, T, D)
                feats = feats.squeeze(0)            # (T, D)

        return feats.cpu().numpy().astype(np.float32)

    def extract_file(self, path: str) -> np.ndarray:
        return self.extract(load_audio(str(path)))


# ─────────────────────────────────────────────────────────────────────────────
# K-means tokeniser — wraps feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class KMeansTokenizer:
    """
    Extracts features using SpeechFeatureExtractor and quantises with k-means.

    The k-means model MUST be fitted (or loaded from disk) on the same
    model + layer combination it will be used with.  Centroids from
    wav2vec2-base cannot be used with hubert-base, etc.

    Parameters
    ----------
    model_name : str
        E.g. 'hubert-base', 'wavlm-large', 'wav2vec2-base'.
    layer : int
        Feature layer (0 = CNN, 6/9 = transformer; see RECOMMENDED_LAYER).
    n_clusters : int
        Vocabulary size V.  100 is standard; 200 for finer granularity.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model_name: str = "hubert-base",
        layer: int = 6,
        n_clusters: int = 100,
        device: Optional[str] = None,
    ):
        self.model_name  = model_name
        self.layer       = layer
        self.n_clusters  = n_clusters
        self.extractor   = SpeechFeatureExtractor(model_name, layer, device)
        self.kmeans      = None

    # ── fitting ──────────────────────────────────────────────────────────────

    def fit(self, audio_paths: list, max_frames: int = 500_000):
        """
        Fit k-means on features from a list of audio files.

        Subsamples to max_frames to keep fitting time reasonable.
        With max_frames=500,000 and 20ms frames: ~10,000 seconds of audio.
        """
        from sklearn.cluster import MiniBatchKMeans

        self.extractor._load_model()
        all_feats = []
        total     = 0

        print(f"\nExtracting features for k-means fitting …")
        print(f"  Model: {self.model_name}  Layer: {self.layer}  "
              f"k: {self.n_clusters}")

        for path in tqdm(audio_paths, desc="features"):
            try:
                feats  = self.extractor.extract_file(str(path))
                all_feats.append(feats)
                total += len(feats)
                if total >= max_frames:
                    break
            except Exception as e:
                print(f"  WARNING: {Path(path).name}: {e}")

        all_feats = np.vstack(all_feats)

        # Subsample
        if len(all_feats) > max_frames:
            rng       = np.random.default_rng(42)
            idx       = rng.choice(len(all_feats), max_frames, replace=False)
            all_feats = all_feats[idx]

        print(f"\nFitting MiniBatchKMeans on {len(all_feats):,} frames "
              f"(dim={all_feats.shape[1]}) …")

        self.kmeans = MiniBatchKMeans(
            n_clusters  = self.n_clusters,
            batch_size  = 10_000,
            n_init      = 10,
            random_state= 42,
            verbose     = 1,
        )
        self.kmeans.fit(all_feats)
        print(f"  Inertia: {self.kmeans.inertia_:.2f}")

    def save_centroids(self, path: str):
        """Save centroids to .npy.  Saves metadata alongside as .json."""
        np.save(path, self.kmeans.cluster_centers_)
        meta_path = str(path).replace(".npy", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "model_name":  self.model_name,
                "layer":       self.layer,
                "n_clusters":  self.n_clusters,
            }, f, indent=2)
        print(f"  Centroids saved → {path}")
        print(f"  Metadata  saved → {meta_path}")

    def load_centroids(self, path: str):
        """Load centroids from .npy."""
        from sklearn.cluster import MiniBatchKMeans
        centers     = np.load(path)
        self.kmeans = MiniBatchKMeans(n_clusters=len(centers))
        self.kmeans.cluster_centers_ = centers
        # Warn if metadata suggests a different model/layer
        meta_path = str(path).replace(".npy", "_meta.json")
        if Path(meta_path).exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if (meta.get("model_name") != self.model_name or
                    meta.get("layer") != self.layer):
                print(f"  WARNING: centroids were fitted on "
                      f"{meta['model_name']} layer={meta['layer']} "
                      f"but you are using {self.model_name} layer={self.layer}.")
        print(f"  Loaded {len(centers)} centroids from {path}")

    # ── tokenisation ─────────────────────────────────────────────────────────

    def tokenize(self, audio: np.ndarray) -> list:
        """Tokenise a single audio array → list of int token ids."""
        assert self.kmeans is not None, \
            "Call fit() or load_centroids() before tokenizing."
        feats  = self.extractor.extract(audio)   # (T, D)
        tokens = self.kmeans.predict(feats)       # (T,) int
        return tokens.tolist()

    def tokenize_file(self, path: str) -> list:
        return self.tokenize(load_audio(str(path)))

    def tag(self) -> str:
        """Short string tag for file naming, e.g. 'hubert-base_l6_k100'."""
        return f"{self.model_name}_l{self.layer}_k{self.n_clusters}"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model runner — produce CSVs for all requested configs in one shot
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_model(
    corpus_dir: str,
    query_dir:  str,
    configs:    list,          # list of (model_name, layer, n_clusters)
    base_out:   str,
    max_frames: int = 500_000,
    load_dir:   Optional[str] = None,
):
    """
    Tokenise corpus + queries for multiple model/layer/cluster configs.

    Each config produces its own subdirectory under base_out:
      base_out/
        wav2vec2-base_l0_k100/corpus.csv
        wav2vec2-base_l0_k100/queries.csv
        wav2vec2-base_l0_k100/kmeans_centroids.npy
        hubert-base_l6_k100/corpus.csv
        ...

    Parameters
    ----------
    corpus_dir : path to corpus audio files
    query_dir  : path to query audio files
    configs    : list of (model_name, layer, n_clusters) tuples
    base_out   : parent output directory
    max_frames : max frames for k-means fitting
    load_dir   : if set, try to load centroids from this directory first
    """
    corpus_paths = sorted(
        list(Path(corpus_dir).glob("*.wav")) +
        list(Path(corpus_dir).glob("*.flac"))
    )
    query_paths  = sorted(
        list(Path(query_dir).glob("*.wav")) +
        list(Path(query_dir).glob("*.flac"))
    )

    if not corpus_paths:
        raise FileNotFoundError(f"No audio files in {corpus_dir}")
    print(f"\nCorpus: {len(corpus_paths)} files | Queries: {len(query_paths)} files")

    for model_name, layer, n_clusters in configs:
        tok = KMeansTokenizer(model_name, layer, n_clusters)
        tag = tok.tag()
        out_dir = Path(base_out) / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Config: {tag}")
        print(f"{'='*60}")

        # Try loading existing centroids
        centroid_path = out_dir / "kmeans_centroids.npy"
        if load_dir:
            centroid_path_load = Path(load_dir) / tag / "kmeans_centroids.npy"
            if centroid_path_load.exists():
                centroid_path = centroid_path_load

        if centroid_path.exists():
            print(f"  Loading existing centroids from {centroid_path}")
            tok.load_centroids(str(centroid_path))
        else:
            tok.fit([str(p) for p in corpus_paths], max_frames=max_frames)
            tok.save_centroids(str(out_dir / "kmeans_centroids.npy"))

        # Tokenise corpus
        corpus_csv = out_dir / "corpus.csv"
        if corpus_csv.exists():
            print(f"  corpus.csv already exists, skipping.")
        else:
            print(f"  Tokenising {len(corpus_paths)} corpus files …")
            _write_csv(
                _tokenize_files(corpus_paths, tok, "corpus"),
                str(corpus_csv),
            )

        # Tokenise queries
        query_csv = out_dir / "queries.csv"
        if query_csv.exists():
            print(f"  queries.csv already exists, skipping.")
        else:
            print(f"  Tokenising {len(query_paths)} query files …")
            _write_csv(
                _tokenize_files(query_paths, tok, "queries"),
                str(query_csv),
            )

        print(f"\n  ✓  {tag}: corpus.csv + queries.csv → {out_dir}")

    # Print run command for all configs
    print(f"\n{'='*60}")
    print("Run PMI-QuEST on all configs:")
    print(f"{'='*60}")
    for model_name, layer, n_clusters in configs:
        tag = f"{model_name}_l{layer}_k{n_clusters}"
        print(f"""
python run_pmiquest_comparison.py \\
    --corpus    {base_out}/{tag}/corpus.csv \\
    --queries   {base_out}/{tag}/queries.csv \\
    --relevance relevance.json \\
    --out       results/{tag}.csv \\
    --ablation""")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_files(paths: list, tok: KMeansTokenizer, desc: str) -> list:
    """Tokenise a list of audio paths → list of (filename, tokens) tuples."""
    rows = []
    for path in tqdm(paths, desc=desc):
        try:
            tokens = tok.tokenize_file(str(path))
            rows.append((Path(path).name, tokens))
        except Exception as e:
            print(f"  WARNING: {Path(path).name}: {e}")
    return rows


def _write_csv(rows: list, path: str):
    """Write (filename, tokens) rows to CSV in pmiquest format."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Filename", "Data"])
        for fname, tokens in rows:
            w.writerow([fname, ",".join(map(str, tokens))])
    print(f"  Written {len(rows)} rows → {path}")


def _print_stats(rows: list, tag: str):
    lengths = [len(r[1]) for r in rows]
    vocab   = set(t for _, toks in rows for t in toks)
    print(f"  {tag}: n={len(rows)}  len={min(lengths)}–{max(lengths)}"
          f"  mean={np.mean(lengths):.1f}  vocab={len(vocab)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenise audio with wav2vec2 / HuBERT / WavLM + k-means",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Model selection ───────────────────────────────────────────────────────
    parser.add_argument(
        "--model", default="wav2vec2-base",
        choices=list(MODEL_IDS.keys()),
        help="Model to use for feature extraction (default: wav2vec2-base).",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Transformer layer to extract (0=CNN, 6=standard for base models, "
             "9=standard for large). Defaults to RECOMMENDED_LAYER[model].",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=100,
        help="K-means vocabulary size V (default: 100).",
    )

    # ── Multi-model mode ──────────────────────────────────────────────────────
    parser.add_argument(
        "--all_models", action="store_true",
        help="Run all three base models (wav2vec2-base l6, hubert-base l6, "
             "wavlm-base l6) with k=100.  Ignores --model/--layer.",
    )

    # ── I/O ───────────────────────────────────────────────────────────────────
    parser.add_argument("--corpus_dir",  required=True,
                        help="Directory of corpus .wav/.flac files.")
    parser.add_argument("--query_dir",   required=True,
                        help="Directory of query .wav/.flac files.")
    parser.add_argument("--out_dir",     default="tokenised",
                        help="Output directory (default: tokenised/).")
    parser.add_argument("--load_kmeans", default=None,
                        help="Path to existing centroids .npy to skip fitting.")
    parser.add_argument("--save_kmeans", default=None,
                        help="Where to save fitted centroids (default: "
                             "out_dir/kmeans_centroids.npy).")
    parser.add_argument("--max_frames",  type=int, default=500_000,
                        help="Max frames for k-means fitting (default: 500k).")
    parser.add_argument("--device",      default=None,
                        help="'cuda' or 'cpu' (auto-detected if not set).")

    args = parser.parse_args()

    # Resolve layer default
    layer = args.layer if args.layer is not None else RECOMMENDED_LAYER.get(args.model, 6)

    if args.all_models:
        # Run the three standard base model configs
        configs = [
            ("wav2vec2-base", 6,  100),
            ("hubert-base",   6,  100),
            ("wavlm-base",    6,  100),
            ("indicwav2vec",  6,  100),
        ]
        run_multi_model(
            corpus_dir = args.corpus_dir,
            query_dir  = args.query_dir,
            configs    = configs,
            base_out   = args.out_dir,
            max_frames = args.max_frames,
        )
    else:
        # Single model/layer/k config
        tok = KMeansTokenizer(
            model_name  = args.model,
            layer       = layer,
            n_clusters  = args.n_clusters,
            device      = args.device,
        )
        out_dir = Path(args.out_dir) / tok.tag()
        out_dir.mkdir(parents=True, exist_ok=True)

        centroid_path = args.save_kmeans or str(out_dir / "kmeans_centroids.npy")

        if args.load_kmeans and Path(args.load_kmeans).exists():
            tok.load_centroids(args.load_kmeans)
        else:
            corpus_paths = (
                sorted(Path(args.corpus_dir).glob("*.wav")) +
                sorted(Path(args.corpus_dir).glob("*.flac"))
            )
            tok.fit([str(p) for p in corpus_paths], max_frames=args.max_frames)
            tok.save_centroids(centroid_path)

        # Tokenise corpus
        corpus_paths = (
            sorted(Path(args.corpus_dir).glob("*.wav")) +
            sorted(Path(args.corpus_dir).glob("*.flac"))
        )
        corpus_rows = _tokenize_files(corpus_paths, tok, "corpus")
        _write_csv(corpus_rows, str(out_dir / "corpus.csv"))
        _print_stats(corpus_rows, "corpus")

        # Tokenise queries
        query_paths = (
            sorted(Path(args.query_dir).glob("*.wav")) +
            sorted(Path(args.query_dir).glob("*.flac"))
        )
        query_rows = _tokenize_files(query_paths, tok, "queries")
        _write_csv(query_rows, str(out_dir / "queries.csv"))
        _print_stats(query_rows, "queries")

        print(f"\n✅  Done → {out_dir}/")
        print(f"\nNext step:")
        print(f"  python run_pmiquest_comparison.py \\")
        print(f"      --corpus    {out_dir}/corpus.csv \\")
        print(f"      --queries   {out_dir}/queries.csv \\")
        print(f"      --relevance relevance.json \\")
        print(f"      --out       results/{tok.tag()}.csv \\")
        print(f"      --ablation")
