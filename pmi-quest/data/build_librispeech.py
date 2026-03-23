"""
LibriSpeech Query-by-Example Dataset Builder
=============================================
Two modes (--mode):

  word   (default) : queries are single word segments extracted via forced
                     alignment.  Relevance = utterances containing that word.
                     Query length ~ 5 tokens.  Good for STD benchmarking.

  utterance        : queries are FULL utterances sampled from the corpus.
                     Relevance = other utterances from the SAME SPEAKER.
                     Query length ~ 100 tokens.  Good for BPE / n-gram systems.
                     No forced alignment needed — instant dataset creation.

Produces:
  queries/          wav clips (word segments OR full utterances)
  corpus/           full utterance wav clips at 16 kHz mono
  metadata/
    queries.json    {query_id, speaker, audio_path, duration, [word]}
    corpus.json     {utt_id, speaker, transcript, audio_path, duration}
    relevance.json  {query_id -> [relevant corpus utt_ids]}
    qrel.tsv        TREC-style relevance file

Naming conventions:
  utterance mode  query:  utt_{UTT_ID}.wav          corpus: {UTT_ID}.wav
  word mode       query:  q_{WORD}_{UTT_ID}.wav      corpus: {UTT_ID}.wav

Usage
-----
pip install datasets torchaudio soundfile tqdm

# Utterance-level queries (longer, better for BPE) — no new download needed
python data/build_librispeech.py \\
    --split test-clean \\
    --mode utterance \\
    --n_queries 100 \\
    --min_utt_per_speaker 3 \\
    --out_dir ./qbe_utterance

# Word-level queries (original behaviour)
python build_qbe_dataset.py \\
    --split test-clean \\
    --mode word \\
    --n_queries 200 \\
    --min_query_words 10 \\
    --out_dir ./qbe_word
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O (soundfile-based — no torchcodec needed)
# ─────────────────────────────────────────────────────────────────────────────

def decode_flac_bytes(raw_bytes: bytes, target_sr: int = 16000):
    """Decode raw FLAC/WAV bytes using soundfile. No torchcodec needed."""
    import io, numpy as np, soundfile as sf
    with sf.SoundFile(io.BytesIO(raw_bytes)) as fh:
        audio = fh.read(dtype="float32")
        sr    = fh.samplerate
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wav = torch.tensor(audio).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr


def save_wav(path, wav: torch.Tensor, sr: int):
    import soundfile as sf, numpy as np
    sf.write(str(path), wav.squeeze().numpy().astype(np.float32), sr)


def load_wav(path: str, target_sr: int = 16000):
    import soundfile as sf
    with sf.SoundFile(str(path)) as fh:
        audio = fh.read(dtype="float32")
        sr    = fh.samplerate
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wav = torch.tensor(audio).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace loader  (single parquet shard — no full config download)
# ─────────────────────────────────────────────────────────────────────────────

def load_librispeech_hf(split: str, cache_dir=None):
    """
    Downloads ONLY the single parquet shard for the requested split.
    Sizes: test-clean ~340 MB, test-other ~350 MB, dev-* ~315-340 MB.
    Train splits (30-500 GB) → use --librispeech_root instead.
    """
    import datasets

    parquet_urls = {
        "test-clean": "hf://datasets/openslr/librispeech_asr/clean/test/0000.parquet",
        "test-other": "hf://datasets/openslr/librispeech_asr/other/test/0000.parquet",
        "dev-clean":  "hf://datasets/openslr/librispeech_asr/clean/validation/0000.parquet",
        "dev-other":  "hf://datasets/openslr/librispeech_asr/other/validation/0000.parquet",
    }
    if split not in parquet_urls:
        raise ValueError(
            f"Split '{split}' not supported for auto-download. "
            f"Choose from: {list(parquet_urls)}. "
            f"For train splits use --librispeech_root."
        )
    url = parquet_urls[split]
    print(f"Downloading single parquet shard: {url}")
    return datasets.load_dataset(
        "parquet",
        data_files={"data": url},
        split="data",
        cache_dir=cache_dir,
        features=datasets.Features({
            "id":         datasets.Value("string"),
            "speaker_id": datasets.Value("int64"),
            "text":       datasets.Value("string"),
            "audio":      datasets.features.Audio(decode=False),
            "file":       datasets.Value("string"),
            "chapter_id": datasets.Value("int64"),
        }),
    )


def load_librispeech_local(root: str, split: str):
    """Walk a local LibriSpeech directory tree."""
    split_dir = Path(root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"LibriSpeech split not found: {split_dir}")
    records = []
    for trans_file in split_dir.rglob("*.trans.txt"):
        speaker = trans_file.parts[-3]
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, *words = line.split()
                transcript = " ".join(words)
                audio_path = trans_file.parent / f"{utt_id}.flac"
                if audio_path.exists():
                    records.append(dict(
                        speaker=speaker, utt_id=utt_id,
                        audio_path=str(audio_path), transcript=transcript,
                    ))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Forced alignment (word mode only)
# ─────────────────────────────────────────────────────────────────────────────

def align_words_torchaudio(waveform, sample_rate, transcript):
    try:
        from torchaudio.pipelines import MMS_FA as bundle
        device    = torch.device("cpu")
        model     = bundle.get_model(with_star=False).to(device)
        tokenizer = bundle.get_tokenizer()
        aligner   = bundle.get_aligner()
        words     = [re.sub(r"[^A-Z]", "", w.upper()) for w in transcript.split()]
        words     = [w for w in words if w]
        if not words:
            return []
        with torch.inference_mode():
            emission, _ = model(waveform.to(device))
        spans = aligner(emission[0], tokenizer(words))
        ratio = waveform.shape[1] / emission.shape[1] / sample_rate
        return [(w, float(s[0].start * ratio), float(s[-1].end * ratio))
                for w, s in zip(words, spans)]
    except Exception:
        dur   = waveform.shape[1] / sample_rate
        words = [re.sub(r"[^A-Z]", "", w.upper()) for w in transcript.split()]
        words = [w for w in words if w]
        step  = dur / max(len(words), 1)
        return [(w, i * step, (i + 1) * step) for i, w in enumerate(words)]


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — Utterance-level queries
# ─────────────────────────────────────────────────────────────────────────────

def build_utterance_dataset(
    records,
    out_dir: str,
    n_queries: int = 100,
    min_utt_per_speaker: int = 3,
    seed: int = 42,
):
    """
    Build a QbE dataset where each QUERY is a full utterance and RELEVANCE
    is defined as other utterances from the SAME SPEAKER.

    This gives query lengths matching corpus lengths (~100 tokens each),
    which is ideal for BPE and n-gram TF-IDF systems.

    Relevance definition — a corpus utterance is relevant to query Q if:
      speaker(Q) == speaker(corpus_utt)  AND  corpus_utt != Q

    This is a realistic speaker-dependent retrieval task: "find me other
    utterances that sound like this one" — natural for voice search.

    Dataset split strategy:
      - Speakers with ≥ min_utt_per_speaker utterances are eligible
      - For each selected speaker, pick ONE utterance as the query
      - All OTHER utterances of that speaker stay in the corpus
      - Utterances from non-selected speakers also stay in corpus
      - The query utterance is REMOVED from the corpus (clean eval)
    """
    random.seed(seed)
    out_dir  = Path(out_dir)
    q_dir    = out_dir / "queries"
    c_dir    = out_dir / "corpus"
    meta_dir = out_dir / "metadata"
    for d in (q_dir, c_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Group utterances by speaker
    speaker_utts = defaultdict(list)
    for rec in records:
        speaker_utts[rec["speaker"]].append(rec)

    # Only speakers with enough utterances
    eligible_speakers = [
        spk for spk, utts in speaker_utts.items()
        if len(utts) >= min_utt_per_speaker
    ]
    if not eligible_speakers:
        raise ValueError(
            f"No speakers have >= {min_utt_per_speaker} utterances. "
            f"Lower --min_utt_per_speaker."
        )

    print(f"  {len(eligible_speakers)} eligible speakers "
          f"(>= {min_utt_per_speaker} utterances each)")

    # Sample speakers and pick one query utterance per speaker
    n = min(n_queries, len(eligible_speakers))
    selected_speakers = random.sample(eligible_speakers, n)

    query_utt_ids = set()   # utterances pulled out as queries
    queries_meta  = []

    for spk in selected_speakers:
        utts = speaker_utts[spk]
        # Pick the query utterance (prefer mid-length for stability)
        utts_sorted = sorted(utts, key=lambda r: len(r["transcript"].split()))
        chosen = utts_sorted[len(utts_sorted) // 2]   # median length

        query_id  = f"utt_{chosen['utt_id']}"
        dst_path  = q_dir / f"{query_id}.wav"
        if not dst_path.exists():
            wav, sr = load_wav(chosen["audio_path"])
            save_wav(dst_path, wav, sr)

        import soundfile as sf
        dur = sf.info(str(dst_path)).frames / 16000
        queries_meta.append(dict(
            query_id=query_id,
            utt_id=chosen["utt_id"],
            speaker=spk,
            transcript=chosen["transcript"],
            audio_path=str(dst_path),
            duration=round(dur, 3),
        ))
        query_utt_ids.add(chosen["utt_id"])

    # Build corpus = all utterances EXCEPT the selected query utterances
    print("Building corpus …")
    corpus_meta = []
    for rec in tqdm(records, desc="corpus"):
        if rec["utt_id"] in query_utt_ids:
            continue   # this utterance is a query — exclude from corpus

        dst_path = c_dir / f"{rec['utt_id']}.wav"
        if not dst_path.exists():
            wav, sr = load_wav(rec["audio_path"])
            save_wav(dst_path, wav, sr)

        import soundfile as sf
        dur = sf.info(str(dst_path)).frames / 16000
        corpus_meta.append(dict(
            utt_id=rec["utt_id"],
            speaker=rec["speaker"],
            transcript=rec["transcript"],
            audio_path=str(dst_path),
            duration=round(dur, 3),
        ))

    # Build relevance: query → all corpus utterances from same speaker
    # (excluding the query utterance itself, which is already removed)
    speaker_to_corpus_utts = defaultdict(list)
    for cm in corpus_meta:
        speaker_to_corpus_utts[cm["speaker"]].append(cm["utt_id"])

    relevance = {}
    for qm in queries_meta:
        rel_utts = speaker_to_corpus_utts.get(qm["speaker"], [])
        relevance[qm["query_id"]] = rel_utts

    # Print stats
    rel_sizes = [len(v) for v in relevance.values()]
    print(f"\n  Query length stats:")
    q_lens = [qm["duration"] for qm in queries_meta]
    print(f"    Duration — min: {min(q_lens):.1f}s, "
          f"max: {max(q_lens):.1f}s, mean: {sum(q_lens)/len(q_lens):.1f}s")
    print(f"  Relevant set size — min: {min(rel_sizes)}, "
          f"max: {max(rel_sizes)}, mean: {sum(rel_sizes)/len(rel_sizes):.1f}")

    # Write metadata
    _write_metadata(meta_dir, queries_meta, corpus_meta, relevance)

    print(f"\n✅  Utterance-level dataset written to: {out_dir}")
    print(f"   Queries : {len(queries_meta)}")
    print(f"   Corpus  : {len(corpus_meta)} utterances")
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — Word-level queries (original)
# ─────────────────────────────────────────────────────────────────────────────

def build_word_dataset(
    records,
    out_dir: str,
    n_queries: int = 200,
    min_query_words: int = 10,
    min_word_dur: float = 0.15,
    max_word_dur: float = 1.5,
    seed: int = 42,
):
    """
    Original word-segment mode. Queries are single spoken words extracted
    via forced alignment. Relevance = utterances containing that word.
    """
    random.seed(seed)
    out_dir  = Path(out_dir)
    q_dir    = out_dir / "queries"
    c_dir    = out_dir / "corpus"
    meta_dir = out_dir / "metadata"
    for d in (q_dir, c_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Building corpus …")
    corpus_meta  = []
    word_to_utts = defaultdict(list)

    for rec in tqdm(records, desc="corpus"):
        utt_id     = rec["utt_id"]
        transcript = rec["transcript"]
        speaker    = rec["speaker"]
        dst_path   = c_dir / f"{utt_id}.wav"
        if not dst_path.exists():
            wav, sr = load_wav(rec["audio_path"])
            save_wav(dst_path, wav, sr)
        import soundfile as sf
        dur = sf.info(str(dst_path)).frames / 16000
        corpus_meta.append(dict(
            utt_id=utt_id, speaker=speaker,
            transcript=transcript,
            audio_path=str(dst_path),
            duration=round(dur, 3),
        ))
        for w in set(re.sub(r"[^A-Z ]", "", transcript.upper()).split()):
            if w:
                word_to_utts[w].append(utt_id)

    candidates = [w for w, u in word_to_utts.items() if len(u) >= min_query_words]
    if not candidates:
        raise ValueError(f"No words in >= {min_query_words} utterances. Lower --min_query_words.")

    query_words  = random.sample(candidates, min(n_queries, len(candidates)))
    print(f"Selected {len(query_words)} query words from {len(candidates)} candidates.")

    print("Extracting word segments via forced alignment …")
    utt_id_map   = {r["utt_id"]: r for r in records}
    queries_meta = []
    relevance    = {}

    for word in tqdm(query_words, desc="queries"):
        candidate_utts = word_to_utts[word].copy()
        random.shuffle(candidate_utts)
        saved = False
        for utt_id in candidate_utts[:10]:
            rec = utt_id_map.get(utt_id)
            if not rec:
                continue
            wav, sr = load_wav(rec["audio_path"])
            for (w, start, end) in align_words_torchaudio(wav, sr, rec["transcript"]):
                if w != word:
                    continue
                dur = end - start
                if not (min_word_dur <= dur <= max_word_dur):
                    continue
                segment = wav[:, int(start * sr): int(end * sr)]
                if segment.shape[1] < int(min_word_dur * sr):
                    continue
                query_id = f"q_{word}_{utt_id}"
                save_wav(q_dir / f"{query_id}.wav", segment, sr)
                queries_meta.append(dict(
                    query_id=query_id, word=word,
                    source_utt=utt_id, speaker=rec["speaker"],
                    audio_path=str(q_dir / f"{query_id}.wav"),
                    duration=round(dur, 3),
                ))
                relevance[query_id] = list(word_to_utts[word])
                saved = True
                break
            if saved:
                break

    _write_metadata(meta_dir, queries_meta, corpus_meta, relevance)

    print(f"\n✅  Word-level dataset written to: {out_dir}")
    print(f"   Queries : {len(queries_meta)}")
    print(f"   Corpus  : {len(corpus_meta)} utterances")
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Shared metadata writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_metadata(meta_dir, queries_meta, corpus_meta, relevance):
    corpus_ids = {r["utt_id"] for r in corpus_meta}
    with open(meta_dir / "queries.json",   "w") as f: json.dump(queries_meta, f, indent=2)
    with open(meta_dir / "corpus.json",    "w") as f: json.dump(corpus_meta,  f, indent=2)
    with open(meta_dir / "relevance.json", "w") as f: json.dump(relevance,    f, indent=2)
    with open(meta_dir / "qrel.tsv",       "w") as f:
        f.write("query_id\titer\tdoc_id\trelevance\n")
        for qid, rel_ids in relevance.items():
            rel_set = set(rel_ids)
            for doc_id in corpus_ids:
                f.write(f"{qid}\t0\t{doc_id}\t{1 if doc_id in rel_set else 0}\n")
    print(f"  Metadata written to {meta_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def average_precision(relevant_set, ranked_list):
    hits = ap = 0.0
    for rank, doc_id in enumerate(ranked_list, 1):
        if doc_id in relevant_set:
            hits += 1
            ap   += hits / rank
    return ap / max(len(relevant_set), 1)

def precision_at_k(relevant_set, ranked_list, k):
    return sum(1 for d in ranked_list[:k] if d in relevant_set) / k

def recall_at_k(relevant_set, ranked_list, k):
    return sum(1 for d in ranked_list[:k] if d in relevant_set) / max(len(relevant_set), 1)

def compute_all_metrics(relevance, all_rankings, ks=(1, 5, 10)):
    aps  = []
    p_at = {k: [] for k in ks}
    r_at = {k: [] for k in ks}
    for qid, ranked in all_rankings.items():
        rel = set(relevance.get(qid, []))
        aps.append(average_precision(rel, ranked))
        for k in ks:
            p_at[k].append(precision_at_k(rel, ranked, k))
            r_at[k].append(recall_at_k(rel, ranked, k))
    results = {"MAP": round(sum(aps) / max(len(aps), 1), 4)}
    for k in ks:
        results[f"P@{k}"] = round(sum(p_at[k]) / max(len(p_at[k]), 1), 4)
        results[f"R@{k}"] = round(sum(r_at[k]) / max(len(r_at[k]), 1), 4)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_root", default=None,
                        help="Local LibriSpeech root. If omitted, streams from HuggingFace.")
    parser.add_argument("--split", default="test-clean",
                        choices=["test-clean", "test-other", "dev-clean", "dev-other"],
                        help="Split (~340 MB each). Train splits require --librispeech_root.")
    parser.add_argument("--mode", default="utterance",
                        choices=["utterance", "word"],
                        help=(
                            "utterance: full-utterance queries, relevance=same speaker. "
                            "Query length ~100 tokens. Best for BPE/n-gram systems. "
                            "word: single-word queries via forced alignment. "
                            "Query length ~5 tokens. Classic QbE-STD setup."
                        ))
    # Utterance mode args
    parser.add_argument("--n_queries",           type=int, default=100,
                        help="Number of query utterances to sample (utterance mode).")
    parser.add_argument("--min_utt_per_speaker", type=int, default=3,
                        help="Min utterances per speaker to be eligible as query source.")
    # Word mode args
    parser.add_argument("--n_word_queries",      type=int, default=200,
                        help="Number of query words to sample (word mode).")
    parser.add_argument("--min_query_words",     type=int, default=10,
                        help="Min corpus utterances a word must appear in (word mode).")
    parser.add_argument("--out_dir",  default="./qbe_librispeech")
    parser.add_argument("--hf_cache", default=None)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    # Load records
    if args.librispeech_root:
        records = load_librispeech_local(args.librispeech_root, args.split)
    else:
        ds  = load_librispeech_hf(args.split, cache_dir=args.hf_cache)
        tmp = Path(args.out_dir) / "_tmp_audio"
        tmp.mkdir(parents=True, exist_ok=True)
        records = []
        for item in tqdm(ds, desc="saving wavs"):
            utt_id = str(item.get("id", f"utt_{len(records)}"))
            fpath  = tmp / f"{utt_id}.wav"
            if not fpath.exists():
                wav_t, _ = decode_flac_bytes(item["audio"]["bytes"], target_sr=16000)
                save_wav(fpath, wav_t, 16000)
            records.append(dict(
                utt_id=utt_id,
                speaker=str(item.get("speaker_id", "unknown")),
                transcript=item.get("text", "").upper(),
                audio_path=str(fpath),
            ))

    print(f"Loaded {len(records)} utterances  (mode={args.mode})")

    if args.mode == "utterance":
        build_utterance_dataset(
            records,
            out_dir=args.out_dir,
            n_queries=args.n_queries,
            min_utt_per_speaker=args.min_utt_per_speaker,
            seed=args.seed,
        )
    else:
        build_word_dataset(
            records,
            out_dir=args.out_dir,
            n_queries=args.n_word_queries,
            min_query_words=args.min_query_words,
            seed=args.seed,
        )
