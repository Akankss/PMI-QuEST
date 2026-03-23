"""
run_ssl_cosine_baseline.py
===========================
Mean-pooled SSL cosine similarity baseline — no discretisation.

For each utterance, extracts SSL frame-level features and averages them
into a single fixed-size vector. Retrieval is cosine similarity over
these mean-pooled vectors.

This baseline tests whether discrete tokenisation loses information
compared to retaining the continuous SSL representations.
If PMI-QuEST (discrete tokens) beats mean-pooled SSL cosine, it validates
the discrete-token retrieval paradigm.

Three pooling options:
  mean:   average all frames
  max:    element-wise max
  attn:   attention-weighted mean (softmax over L2 norms)

Usage
-----
python run_ssl_cosine_baseline.py \\
    --corpus_audio  qbe_librispeech/corpus/ \\
    --query_audio   qbe_librispeech/queries/ \\
    --relevance     qbe_librispeech/metadata/relevance.json \\
    --model         wav2vec2-base \\
    --layer         6 \\
    --pooling       mean max
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torchaudio


# ─────────────────────────────────────────────────────────────────────────────
# SSL feature extraction
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAMES = {
    'wav2vec2-base': 'facebook/wav2vec2-base',
    'hubert-base':   'facebook/hubert-base-ls960',
    'wavlm-base':    'microsoft/wavlm-base',
}

_model_cache = {}

def get_model(model_name, layer, device='cpu'):
    key = (model_name, layer)
    if key not in _model_cache:
        from transformers import (Wav2Vec2Model, HubertModel, WavLMModel,
                                  AutoFeatureExtractor)
        hf_name = MODEL_NAMES.get(model_name, model_name)
        print(f"  Loading {hf_name} (layer {layer})...")

        feat_ext = AutoFeatureExtractor.from_pretrained(hf_name)

        if 'wav2vec2' in model_name:
            model = Wav2Vec2Model.from_pretrained(hf_name, output_hidden_states=True)
        elif 'hubert' in model_name:
            model = HubertModel.from_pretrained(hf_name, output_hidden_states=True)
        elif 'wavlm' in model_name:
            model = WavLMModel.from_pretrained(hf_name, output_hidden_states=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.eval().to(device)
        _model_cache[key] = (model, feat_ext, layer, device)
        print(f"  Model loaded.")
    return _model_cache[key]


def extract_features(wav_path, model_name, layer, device='cpu'):
    """Extract SSL hidden states at `layer` for one audio file."""
    model, feat_ext, layer_idx, device = get_model(model_name, layer, device)

    try:
        waveform, sr = torchaudio.load(wav_path)
    except (Exception, ImportError):
        import soundfile as sf
        data, sr = sf.read(wav_path, dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        waveform = torch.tensor(data).unsqueeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    inputs = feat_ext(waveform.squeeze().numpy(),
                      sampling_rate=16000,
                      return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states[0] = embedding, [1..] = transformer layers
    hidden = outputs.hidden_states[layer_idx]   # (1, T, D)
    return hidden.squeeze(0).cpu().numpy()       # (T, D)


def pool_features(feats, mode='mean'):
    """Reduce (T, D) → (D,) by pooling."""
    if mode == 'mean':
        return feats.mean(axis=0)
    elif mode == 'max':
        return feats.max(axis=0)
    elif mode == 'attn':
        norms   = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        weights = np.exp(norms - norms.max())
        weights /= weights.sum()
        return (feats * weights).sum(axis=0)
    else:
        raise ValueError(f"Unknown pooling: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def ap(ranked, rel_set):
    if not rel_set:
        return 0.0
    hits = score = 0.0
    for rank, did in enumerate(ranked, 1):
        if did in rel_set:
            hits += 1; score += hits / rank
    return score / len(rel_set)


def precision_at_k(ranked, rel_set, k):
    return sum(1 for d in ranked[:k] if d in rel_set) / k


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus_audio', required=True,
                   help='Directory of corpus .wav files')
    p.add_argument('--query_audio',  required=True,
                   help='Directory of query .wav files')
    p.add_argument('--relevance',    required=True)
    p.add_argument('--model',        default='wav2vec2-base',
                   choices=list(MODEL_NAMES.keys()))
    p.add_argument('--layer',        type=int, default=6)
    p.add_argument('--pooling',      nargs='+', default=['mean'],
                   choices=['mean', 'max', 'attn'])
    p.add_argument('--device',       default='cpu')
    args = p.parse_args()

    with open(args.relevance) as f:
        relevance = json.load(f)

    corpus_dir = Path(args.corpus_audio)
    query_dir  = Path(args.query_audio)

    corpus_files = sorted(corpus_dir.glob('*.wav'))
    query_files  = [f for f in sorted(query_dir.glob('*.wav'))
                    if f.stem in relevance and relevance[f.stem]]

    print("=" * 60)
    print(f"Mean-pooled SSL Cosine Baseline")
    print(f"  Model: {args.model}  Layer: {args.layer}")
    print(f"  Corpus: {len(corpus_files)} files")
    print(f"  Queries: {len(query_files)} files")
    print("=" * 60)

    # ── Extract corpus features ───────────────────────────────────────────────
    print("\nExtracting corpus features...")
    t0 = time.time()
    corpus_vecs = {}
    for i, wav in enumerate(corpus_files):
        feats = extract_features(str(wav), args.model, args.layer, args.device)
        corpus_vecs[wav.stem] = feats
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(corpus_files)} done")
    print(f"  Corpus features extracted in {time.time()-t0:.1f}s")

    # ── Extract query features ────────────────────────────────────────────────
    print("\nExtracting query features...")
    query_vecs = {}
    for wav in query_files:
        feats = extract_features(str(wav), args.model, args.layer, args.device)
        query_vecs[wav.stem] = feats

    corpus_ids = list(corpus_vecs.keys())

    # ── Evaluate each pooling mode ────────────────────────────────────────────
    for pooling in args.pooling:
        print(f"\n  Pooling: {pooling}")

        # Pre-compute pooled corpus vectors
        pooled_corpus = {did: pool_features(v, pooling)
                         for did, v in corpus_vecs.items()}
        corpus_mat = np.stack([pooled_corpus[did] for did in corpus_ids])
        norms = np.linalg.norm(corpus_mat, axis=1, keepdims=True) + 1e-8
        corpus_mat_norm = corpus_mat / norms

        aps, p1s, p5s, p10s = [], [], [], []
        t1 = time.time()

        for qid, q_feats in query_vecs.items():
            rel_set = set(relevance.get(qid, []))
            if not rel_set:
                continue

            q_vec  = pool_features(q_feats, pooling)
            q_norm = np.linalg.norm(q_vec) + 1e-8
            q_vec  = q_vec / q_norm

            sims = corpus_mat_norm @ q_vec
            ranked_ids = [corpus_ids[i] for i in np.argsort(-sims)]

            hits = ap_score = 0.0
            for rank, did in enumerate(ranked_ids, 1):
                if did in rel_set:
                    hits += 1; ap_score += hits / rank
            aps.append(ap_score / len(rel_set))
            p1s.append(precision_at_k(ranked_ids, rel_set, 1))
            p5s.append(precision_at_k(ranked_ids, rel_set, 5))
            p10s.append(precision_at_k(ranked_ids, rel_set, 10))

        label = f"SSL-{args.model}-L{args.layer}-{pooling}-pool"
        print(f"  MAP={np.mean(aps):.4f}  P@1={np.mean(p1s):.4f}  "
              f"P@5={np.mean(p5s):.4f}  P@10={np.mean(p10s):.4f}  "
              f"N={len(aps)}  [{time.time()-t1:.1f}s]")
        print(f"  LaTeX: {label} & {np.mean(aps):.4f} & "
              f"{np.mean(p1s):.4f} & {np.mean(p5s):.4f} & "
              f"{np.mean(p10s):.4f} \\\\")


if __name__ == '__main__':
    main()
