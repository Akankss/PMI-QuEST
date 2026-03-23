"""
tokenisers/tokenise_best_std.py
==========================
Tokenises audio using the REAL BEST-STD model.
Requires mamba-ssm-macos (Apple Silicon MPS) or mamba-ssm (CUDA Linux).

Setup
-----
  pip install git+https://github.com/purohit10saurabh/mamba-ssm-macos.git
  python -c "from mamba_ssm import Mamba2; print('OK')"

Usage — run from INSIDE the BEST-STD repo root:
-----
  cd BEST-STD
  python ../tokenise_best_std_real.py \\
      --corpus_dir    ../qbe_librispeech/corpus/ \\
      --query_dir     ../qbe_librispeech/queries/ \\
      --checkpoint    codes_512/ckpt/epoch=781-valid_loss=0.00-train_loss=0.00.ckpt \\
      --config        config/main.yaml \\
      --out_dir       ../tokenised/best_std_real/

Output
------
  corpus.csv and queries.csv in the same (Filename, Data) format
  as all other tokenised/ CSVs — plug directly into
  run_bigram_selection_baselines.py with --k 512.
"""

import argparse, csv, sys, os, time, glob
from pathlib import Path

import torch
import soundfile as sf


# ─────────────────────────────────────────────────────────────────────────────
# Load BEST-STD model using their own code
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, config_path=None):
    """
    Load AudioTokenizer from BEST-STD's own train/trainer.py.
    Must be called from inside the BEST-STD repo root (or with src/ on path).
    """
    # Add BEST-STD source paths
    cwd = Path.cwd()
    for p in [str(cwd), str(cwd / 'src'), str(cwd / 'src' / 'train'),
              str(cwd / 'src' / 'models'), str(cwd / 'src' / 'utils')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Now import their class — mamba_ssm is available via mamba-ssm-macos
    try:
        from train.trainer import AudioTokenizer
    except ImportError as e:
        raise ImportError(
            f"Could not import AudioTokenizer. "
            f"Make sure you are running from inside the BEST-STD repo root.\n"
            f"Original error: {e}"
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    device = (
        'mps'  if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else
        'cpu'
    )
    print(f"Device: {device}")

    model = AudioTokenizer.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=True,     # real model — all weights should load
    )
    model.eval()
    if device != 'cpu':
        model = model.to(device)

    print(f"Model loaded. Device: {device}")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# Tokenise one audio file
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path, target_sr=16000):
    audio, sr = sf.read(str(path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wav = torch.tensor(audio).unsqueeze(0)   # (1, T)
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def get_tokens(model, wav_path, device):
    wav = load_wav(wav_path).to(device)

    # Always print methods (no caching flag)
    import traceback
    print(f"\n  --- Trying extract_tokens ---", flush=True)

    with torch.no_grad():
        for method in ['extract_tokens', 'get_predictions', 'tokenize',
                       'tokenise', 'encode', 'get_tokens', 'get_codes',
                       'quantize', 'encode_audio', 'inference',
                       'extract', 'forward', 'predict_step']:
            if not hasattr(model, method):
                continue
            print(f"  Trying {method}()...", flush=True)
            try:
                out = getattr(model, method)(wav)
            except Exception as e:
                print(f"  {method}() FAILED: {e}", flush=True)
                traceback.print_exc()
                continue

            # Flatten output
            if isinstance(out, torch.Tensor):
                tensors = [out]
            elif isinstance(out, (tuple, list)):
                tensors = [x for x in out if isinstance(x, torch.Tensor)]
            else:
                print(f"  {method}() returned non-tensor: {type(out)}", flush=True)
                continue

            print(f"  {method}() SUCCESS: {[(tuple(t.shape), str(t.dtype)) for t in tensors]}", flush=True)

            # Integer tensor = token IDs
            for t in tensors:
                if t.dtype in (torch.int32, torch.int64, torch.long):
                    toks = t.squeeze().cpu()
                    if toks.dim() == 0: toks = toks.unsqueeze(0)
                    print(f"  Found int tokens: shape={toks.shape} range=[{toks.min()},{toks.max()}]", flush=True)
                    return [int(x) for x in toks.tolist()]

            # Smallest float tensor fallback (VQ indices often stored as float)
            small = [t for t in tensors if t.numel() < 5000 and t.dim() <= 2]
            if small:
                t = min(small, key=lambda x: x.numel())
                toks = t.squeeze().long().cpu()
                if toks.dim() == 0: toks = toks.unsqueeze(0)
                print(f"  Using smallest tensor: shape={toks.shape}", flush=True)
                return [int(x) for x in toks.tolist()]

            print(f"  No usable tensor in output, trying next method...", flush=True)

    raise RuntimeError(
        "No method returned usable token IDs. "
        "See output above for shapes and errors."
    )





# ─────────────────────────────────────────────────────────────────────────────
# Process a directory
# ─────────────────────────────────────────────────────────────────────────────

def tokenise_dir(model, device, wav_dir, out_csv, label):
    wavs = sorted(glob.glob(str(Path(wav_dir) / '*.wav')))
    if not wavs:
        wavs = sorted(glob.glob(
            str(Path(wav_dir) / '**' / '*.wav'), recursive=True))
    assert wavs, f"No wav files in {wav_dir}"
    print(f"\n{label}: {len(wavs)} files → {out_csv}")

    t0 = time.time()
    errors = []

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Data'])
        for i, path in enumerate(wavs):
            try:
                toks = get_tokens(model, path, device)
                writer.writerow([
                    Path(path).name,
                    ','.join(str(t) for t in toks)
                ])
            except Exception as e:
                errors.append((path, str(e)))
                print(f"  ERROR {Path(path).name}: {e}")

            if (i + 1) % 100 == 0 or i + 1 == len(wavs):
                el  = time.time() - t0
                eta = el / (i + 1) * (len(wavs) - i - 1)
                print(f"  {i+1}/{len(wavs)}  "
                      f"{el/60:.1f}min  ETA {eta/60:.1f}min")

    if errors:
        print(f"  {len(errors)} errors — first: {errors[0][1][:80]}")
    return len(errors)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus_dir',  required=True)
    p.add_argument('--query_dir',   required=True)
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--config',      default=None)
    p.add_argument('--out_dir',     default='../tokenised/best_std_real')
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BEST-STD Real Tokeniser (mamba-ssm-macos / MPS)")
    print("=" * 60)

    model, device = load_model(args.checkpoint, args.config)

    # Quick test
    test = sorted(glob.glob(str(Path(args.corpus_dir) / '*.wav')))[:1]
    if test:
        print("\nTest inference...")
        toks = get_tokens(model, test[0], device)
        print(f"  {Path(test[0]).name}: {len(toks)} tokens")
        print(f"  First 15: {toks[:15]}")
        print(f"  Range: {min(toks)}–{max(toks)}  (k≈{max(toks)+1})")

    tokenise_dir(model, device, args.corpus_dir,
                 str(out / 'corpus.csv'),  'Corpus')
    tokenise_dir(model, device, args.query_dir,
                 str(out / 'queries.csv'), 'Queries')

    # Summary
    print("\nOutput summary:")
    for name in ['corpus', 'queries']:
        path = out / f'{name}.csv'
        rows, all_t = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                toks = [int(x) for x in row['Data'].split(',') if x.strip()]
                rows.append(toks); all_t.extend(toks)
        k = max(all_t) + 1
        print(f"  {name}.csv: {len(rows)} seqs  "
              f"mean_len={len(all_t)/len(rows):.0f}  k={k}")

    print(f"\nRun evaluation:")
    print(f"  python run_bigram_selection_baselines.py \\")
    print(f"      --corpus_csv {out}/corpus.csv \\")
    print(f"      --query_csv  {out}/queries.csv \\")
    print(f"      --relevance  qbe_librispeech/metadata/relevance.json \\")
    print(f"      --k 512")


if __name__ == '__main__':
    main()
