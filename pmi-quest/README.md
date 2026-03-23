# PMI-QuEST

**PMI-QuEST: Pointwise Mutual Information-Guided Token Bigrams for Query-by-Example Spoken Term Detection**

Akanksha Singh · Yi-Ping Phoebe Chen · Vipul Arora  
*IEEE/ACM Transactions on Audio, Speech, and Language Processing (submitted)*

---

## Overview

PMI-QuEST augments unigram TF-IDF retrieval with statistically selected token bigrams discovered via Pointwise Mutual Information (PMI), directly addressing the order-blindness of existing discrete-token QbE-STD systems. The system achieves consistent improvements over H-QuEST across all 16 SSL tokeniser configurations tested, with MAP 0.7857 on WavLM-base layer 7 and +13.9% average MAP over H-QuEST across 12 Indic languages.

```
Audio → SSL encoder → k-means tokens → PMI-TF-IDF vectors → HNSW → SW reranking → Ranked list
```

---

## Results

### LibriSpeech (English, word-mode QbE-STD)

| System | MAP | P@1 |
|---|---|---|
| TF-IDF Baseline | 0.5300 | 0.5300 |
| H-QuEST | 0.6333 | 0.6750 |
| **PMI-QuEST** (WavLM-base L7) | **0.7857** | **0.7800** |

### IndicSUPERB Kathbath (12 Indic languages, XLS-R-300M)

PMI-QuEST achieves the highest MAP on all 12 languages, with +13.9% average MAP over H-QuEST without any language-specific adaptation.

---

## Installation

```bash
git clone https://github.com/akanksha-singh/pmi-quest.git
cd pmi-quest
pip install -r requirements.txt
```

**Optional — BEST-STD tokeniser (Apple Silicon):**
```bash
pip install git+https://github.com/purohit10saurabh/mamba-ssm-macos.git
```

**Optional — BEST-STD tokeniser (Linux/CUDA):**
```bash
pip install mamba-ssm causal-conv1d
```

---

## Repository Structure

```
pmi-quest/
├── pmiquest/
│   ├── __init__.py          # public API
│   ├── system.py            # core: PMIQuest, HQuest, TFIDFBaseline
│   └── tokeniser.py         # SSL encoder + k-means tokenisation
├── data/
│   ├── build_librispeech.py # build LibriSpeech QbE dataset
│   └── build_kathbath.py    # build IndicSUPERB Kathbath dataset
├── experiments/
│   ├── run_main_comparison.py    # Table I: TF-IDF / H-QuEST / PMI-QuEST
│   ├── run_ablation.py           # Table II: bigram selection ablation
│   ├── run_multi_tokeniser.py    # Table III: multi-tokeniser comparison
│   ├── run_layer_sweep.py        # Table IV: layer sweep
│   ├── run_cross_lingual.py      # Table V: Kathbath 12 languages
│   ├── run_baselines.py          # Token-DTW + SSL mean-pool baselines
│   ├── run_significance.py       # Wilcoxon signed-rank tests
│   └── run_bestsd_comparison.py  # BEST-STD retrieval vs PMI-QuEST
└── tokenisers/
    └── tokenise_best_std.py      # BEST-STD token extraction (MPS/CUDA)
```

---

## Quick Start

### Step 1 — Build dataset

```bash
python data/build_librispeech.py \
    --librispeech_root /path/to/LibriSpeech/test-clean \
    --out_dir          data/qbe_librispeech

python data/build_kathbath.py \
    --kathbath_root /path/to/Kathbath \
    --out_dir       data/qbe_kathbath
```

### Step 2 — Tokenise

```bash
python pmiquest/tokeniser.py \
    --corpus_dir data/qbe_librispeech/corpus/ \
    --query_dir  data/qbe_librispeech/queries/ \
    --model      wavlm-base \
    --layer      7 \
    --k          100 \
    --out_dir    tokenised/wavlm_l7_k100/
```

### Step 3 — Run main comparison

```bash
python experiments/run_main_comparison.py \
    --corpus_csv  tokenised/wavlm_l7_k100/corpus.csv \
    --query_csv   tokenised/wavlm_l7_k100/queries.csv \
    --relevance   data/qbe_librispeech/metadata/relevance.json \
    --tau         0.5 \
    --alpha       0.5
```

### Step 4 — Run full ablation

```bash
python experiments/run_ablation.py \
    --corpus_csv  tokenised/wavlm_l7_k100/corpus.csv \
    --query_csv   tokenised/wavlm_l7_k100/queries.csv \
    --relevance   data/qbe_librispeech/metadata/relevance.json
```

### Step 5 — Cross-lingual evaluation

```bash
# First tokenise all 12 languages with XLS-R-300M
python pmiquest/tokeniser.py \
    --corpus_dir data/qbe_kathbath/hindi/corpus/ \
    --query_dir  data/qbe_kathbath/hindi/queries/ \
    --model      xlsr-300m \
    --layer      7 \
    --k          100 \
    --out_dir    tokenised/kathbath/hindi/

python experiments/run_cross_lingual.py \
    --tokenised_root tokenised/kathbath/ \
    --relevance_root data/qbe_kathbath/
```

---

## Token CSV Format

All scripts consume and produce token CSVs in this format:

```
Filename,Data
1089-134686-0000.wav,23,61,47,61,23,12,45,...
1089-134686-0001.wav,47,12,33,61,23,...
```

`Filename` is the basename of the audio file. `Data` is a comma-separated sequence of integer token IDs.

---

## Reproducing Paper Tables

| Table | Script |
|---|---|
| Table I (main comparison) | `experiments/run_main_comparison.py` |
| Table II (ablation) | `experiments/run_ablation.py` |
| Table III (multi-tokeniser) | `experiments/run_multi_tokeniser.py` |
| Table IV (layer sweep) | `experiments/run_layer_sweep.py` |
| Table V (Kathbath) | `experiments/run_cross_lingual.py` |
| Table VI (significance) | `experiments/run_significance.py` |
| BEST-STD comparison | `experiments/run_bestsd_comparison.py` |

---

## Citation

```bibtex
@article{singh2025pmiquest,
  title   = {{PMI-QuEST}: Pointwise Mutual Information-Guided Token Bigrams
             for Query-by-Example Spoken Term Detection},
  author  = {Singh, Akanksha and Chen, Yi-Ping Phoebe and Arora, Vipul},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2025},
  note    = {Under review}
}
```

---

## Related Work

- **H-QuEST** (Interspeech 2025): [link]
- **BEST-STD** (ICASSP 2025): https://doi.org/10.1109/ICASSP49357.2025.10889633
- **IndicSUPERB / Kathbath** (Interspeech 2023): https://doi.org/10.21437/Interspeech.2023-536

---

## License

MIT License. See `LICENSE` for details.
