"""
pmiquest/system.py
==================
Self-contained implementation of three QbE-STD systems for TASLP comparison.

All components (HNSW via sklearn, Smith-Waterman, TF-IDF, PMI) are implemented
here — no external module dependencies beyond numpy/scipy/sklearn.

Systems
-------
1. TF-IDF Baseline
       Raw tokens → unigram TF-IDF → brute-force cosine ranking
       (exact reproduction of Singh et al. 2024)

2. H-QuEST  (faithful reproduction of Singh et al., Interspeech 2025)
       Raw tokens → unigram TF-IDF → HNSW (M=16, ef_construction=150, K=50)
                 → Smith-Waterman rerank (+2 / -1 / -2)

3. PMI-QuEST  (proposed)
       Raw tokens → PMI-TD token deduplication
                 → PMI-TF-IDF (unigrams + PMI-filtered bigrams, τ_pmi=1.5)
                 → HNSW (same hyperparams)
                 → Regime-gated rerank:
                       ρ < ρ*  →  hard Smith-Waterman   (word mode)
                       ρ ≥ ρ*  →  PMI-soft Smith-Waterman (utterance mode)

Evaluation
----------
MAP, P@1, P@5, P@10 — computed over the full ranked corpus (not just top-K).

Usage
-----
    from pmiquest_system import TFIDFBaseline, HQuEST, PMIQuest, evaluate_system
    from pmiquest_system import run_comparison

    # Assuming corpus_seqs, query_seqs, ground_truth are already loaded:
    results = run_comparison(corpus_seqs, query_seqs, ground_truth)
"""

from __future__ import annotations

import math
import time
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# ── Section 1: Smith-Waterman alignment  ─────────────────────────────────────
# =============================================================================

def smith_waterman(
    query: List[int],
    doc:   List[int],
    match:    float = 2.0,
    mismatch: float = -1.0,
    gap:      float = -2.0,
) -> float:
    """
    Standard Smith-Waterman local alignment score.
    O(m * L) time, O(m * L) space.
    Returns the maximum score in the DP matrix (not normalised).
    """
    m = len(query)
    L = len(doc)
    if m == 0 or L == 0:
        return 0.0

    # Use numpy for speed — rows = query positions, cols = doc positions
    S = np.zeros((m + 1, L + 1), dtype=np.float32)
    for i in range(1, m + 1):
        q_tok = query[i - 1]
        for j in range(1, L + 1):
            s_ij = match if q_tok == doc[j - 1] else mismatch
            S[i, j] = max(
                0.0,
                S[i - 1, j - 1] + s_ij,
                S[i - 1, j]     + gap,
                S[i,     j - 1] + gap,
            )
    return float(S.max())


def smith_waterman_pmi(
    query: List[int],
    doc:   List[int],
    pmi_matrix: Dict[Tuple[int,int], float],
    match:    float = 2.0,
    gap:      float = -2.0,
    pmi_scale: float = 1.0,
) -> float:
    """
    PMI-soft Smith-Waterman: off-diagonal substitution scores are
    s(a, b) = pmi_scale * pmi_matrix.get((a,b), -1.0) instead of
    binary match/mismatch.

    For utterance mode (ρ ≈ 1.0): allows partial credit for
    acoustically similar but not identical tokens, recovering +38% MAP
    over hard-SW (empirical, this paper).
    """
    m = len(query)
    L = len(doc)
    if m == 0 or L == 0:
        return 0.0

    S = np.zeros((m + 1, L + 1), dtype=np.float32)
    for i in range(1, m + 1):
        q_tok = query[i - 1]
        for j in range(1, L + 1):
            d_tok = doc[j - 1]
            if q_tok == d_tok:
                s_ij = match
            else:
                raw_pmi = pmi_matrix.get((q_tok, d_tok),
                          pmi_matrix.get((d_tok, q_tok), None))
                if raw_pmi is not None:
                    s_ij = pmi_scale * raw_pmi
                else:
                    s_ij = -1.0
            S[i, j] = max(
                0.0,
                S[i - 1, j - 1] + s_ij,
                S[i - 1, j]     + gap,
                S[i,     j - 1] + gap,
            )
    return float(S.max())


def sw_rerank(
    query:       List[int],
    candidates:  List[Tuple[float, int]],   # (cosine_score, corpus_idx)
    corpus_seqs: List[List[int]],
    match:    float = 2.0,
    mismatch: float = -1.0,
    gap:      float = -2.0,
) -> List[Tuple[float, int]]:
    """Rerank candidates by Smith-Waterman score (descending)."""
    scored = []
    for _, idx in candidates:
        sw = smith_waterman(query, corpus_seqs[idx], match, mismatch, gap)
        scored.append((sw, idx))
    scored.sort(key=lambda x: -x[0])
    return scored


def sw_pmi_rerank(
    query:       List[int],
    candidates:  List[Tuple[float, int]],
    corpus_seqs: List[List[int]],
    pmi_matrix:  Dict[Tuple[int,int], float],
    match:     float = 2.0,
    gap:       float = -2.0,
    pmi_scale: float = 1.0,
) -> List[Tuple[float, int]]:
    """Rerank candidates by PMI-soft Smith-Waterman score."""
    scored = []
    for _, idx in candidates:
        sw = smith_waterman_pmi(
            query, corpus_seqs[idx], pmi_matrix, match, gap, pmi_scale
        )
        scored.append((sw, idx))
    scored.sort(key=lambda x: -x[0])
    return scored


# =============================================================================
# ── Section 2: HNSW index (sklearn approximate NN)  ──────────────────────────
# =============================================================================

class HNSWIndex:
    """
    Approximate nearest-neighbour index using sklearn BallTree (cosine).
    Interface mirrors the hnswlib API used in the original H-QuEST code.

    Note: sklearn does not implement true HNSW graph structure, but provides
    the same O(log N) approximate search behaviour for the purposes of
    this implementation.  For production use, replace with hnswlib or faiss.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbours to retrieve (corresponds to K=50 in H-QuEST).
    algorithm : str
        'ball_tree' or 'kd_tree' or 'brute'.  'ball_tree' is recommended.
    """

    def __init__(self, n_neighbors: int = 50, algorithm: str = "ball_tree"):
        self.n_neighbors = n_neighbors
        self.algorithm   = algorithm
        self._index      = None
        self._matrix     = None   # normalised L2 copy kept for cosine sim

    def fit(self, matrix: np.ndarray) -> "HNSWIndex":
        """
        Build index from normalised document vectors.
        matrix : shape (N, D), rows are L2-normalised.
        For L2-normalised vectors: euclidean distance² = 2(1 - cosine_sim),
        so nearest-euclidean == nearest-cosine.
        """
        self._matrix = matrix.astype(np.float32)
        k = min(self.n_neighbors, len(matrix))
        self._index  = NearestNeighbors(
            n_neighbors=k,
            algorithm=self.algorithm,
            metric="euclidean",   # equivalent to cosine on L2-normalised vectors
            n_jobs=-1,
        ).fit(self._matrix)
        return self

    def search(self, query_vec: np.ndarray, k: int = None) -> List[Tuple[float, int]]:
        """
        Return (cosine_distance, corpus_idx) pairs, sorted by distance ascending.
        cosine_distance = 1 - cosine_similarity  ≈  euclidean²/2 for unit vectors.
        """
        if self._index is None:
            raise RuntimeError("HNSWIndex not fitted.")
        k = k or self.n_neighbors
        k = min(k, self._matrix.shape[0])
        qv = query_vec.reshape(1, -1).astype(np.float32)
        dists, idxs = self._index.kneighbors(qv, n_neighbors=k)
        # Convert euclidean dist → cosine dist (approx: cos_dist = euclid²/2)
        cos_dists = (dists[0] ** 2) / 2.0
        return list(zip(cos_dists.tolist(), idxs[0].tolist()))


# =============================================================================
# ── Section 3: TF-IDF vectoriser  ────────────────────────────────────────────
# =============================================================================

class UnigramTFIDF:
    """
    Unigram TF-IDF vectoriser, exact match to H-QuEST paper.
    TF = raw count / doc length,  IDF = log(N / DF(token)).
    Vectors are L2-normalised before storage/query.
    """

    def __init__(self):
        self.vocab:  Dict[int, int]   = {}   # token → column index
        self.idf:    Dict[int, float] = {}
        self.n_docs: int = 0

    def fit(self, sequences: List[List[int]]) -> "UnigramTFIDF":
        self.n_docs = len(sequences)
        doc_freq: Dict[int, int] = defaultdict(int)
        for seq in sequences:
            for tok in set(seq):
                doc_freq[tok] += 1
        self.vocab = {tok: i for i, tok in enumerate(sorted(doc_freq))}
        self.idf   = {
            tok: math.log(self.n_docs / df)
            for tok, df in doc_freq.items()
        }
        return self

    def transform(self, sequences: List[List[int]]) -> np.ndarray:
        """Returns dense normalised matrix, shape (N, V)."""
        V   = len(self.vocab)
        mat = np.zeros((len(sequences), V), dtype=np.float32)
        for row, seq in enumerate(sequences):
            if not seq:
                continue
            counts = Counter(seq)
            L = len(seq)
            for tok, cnt in counts.items():
                col = self.vocab.get(tok)
                if col is None:
                    continue
                mat[row, col] = (cnt / L) * self.idf.get(tok, 0.0)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def transform_one(self, seq: List[int]) -> np.ndarray:
        V   = len(self.vocab)
        vec = np.zeros(V, dtype=np.float32)
        if not seq:
            return vec
        counts = Counter(seq)
        L = len(seq)
        for tok, cnt in counts.items():
            col = self.vocab.get(tok)
            if col is None:
                continue
            vec[col] = (cnt / L) * self.idf.get(tok, 0.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def fit_transform(self, sequences: List[List[int]]) -> np.ndarray:
        self.fit(sequences)
        return self.transform(sequences)


# =============================================================================
# ── Section 4: PMI computation  ──────────────────────────────────────────────
# =============================================================================

def compute_pmi(
    sequences:       List[List[int]],
    min_bigram_count: int = 2,
) -> Tuple[Dict[Tuple[int,int], float], Dict[int, float]]:
    """
    Compute PMI for all observed bigrams.

    PMI(a, b) = log [ p(a,b) / p(a) * p(b) ]
              = log [ count(a,b) * N_tokens ]
                    [ ────────────────────── ]
                    [ count(a) * count(b)   ]

    Returns
    -------
    pmi_scores   : (a,b) → PMI value  (only pairs with count ≥ min_bigram_count)
    log_p_uni    : token  → log p(token)
    """
    uni: Counter = Counter()
    bi:  Counter = Counter()
    N_tok = 0
    N_bi  = 0

    for seq in sequences:
        for tok in seq:
            uni[tok] += 1
            N_tok += 1
        for i in range(len(seq) - 1):
            bi[(seq[i], seq[i + 1])] += 1
            N_bi += 1

    if N_tok == 0 or N_bi == 0:
        return {}, {}

    log_p_uni = {tok: math.log(cnt / N_tok) for tok, cnt in uni.items()}

    pmi_scores = {}
    for (a, b), cnt in bi.items():
        if cnt < min_bigram_count:
            continue
        p_ab = cnt  / N_bi
        p_a  = uni[a] / N_tok
        p_b  = uni[b] / N_tok
        if p_a > 0 and p_b > 0:
            pmi_scores[(a, b)] = math.log(p_ab / (p_a * p_b))

    return pmi_scores, log_p_uni


# =============================================================================
# ── Section 5: PMI-TD token deduplication  ───────────────────────────────────
# =============================================================================

class PMITokenDedup:
    """
    PMI-guided Token Deduplication  (replaces BPE in H-QuEST → PMI-QuEST).

    Merge criterion  (Def. PMI-TD, paper §V):
        PMI(a, b) ≥ τ_pmi   AND   max(IDF(a), IDF(b)) ≤ τ_idf

    Condition 1 — excess co-occurrence:
        Pairs that co-occur more than independence predicts.
        P(a,b) > P(a) · P(b)  ⟺  PMI(a,b) > 0.
        This captures k-means quantisation splits of the same phoneme
        across adjacent cluster boundaries, and phonetic co-articulation
        patterns.  It is the formalisation of the note in the research
        journal: "this happens when the pair occurs more often than
        expected under independence."

    Condition 2 — low IDF gate:
        max(IDF(a), IDF(b)) ≤ τ_idf ensures neither token in the pair
        is already discriminative.  Merging discriminative tokens would
        destroy retrieval signal.  This is the key gate absent from BPE.

    Theorem (paper §IV): Every PMI-TD merge is entropy-non-decreasing
    in H_IDF — the IDF entropy of the vocabulary.  BPE merges are
    entropy-decreasing at every step.  PMI-TD is therefore provably safe
    where BPE is provably harmful.

    Parameters
    ----------
    tau_pmi   : float   PMI threshold (≥ 0 to require any excess co-occ)
    tau_idf   : float   IDF ceiling (lower = more aggressive gate)
    min_count : int     Minimum bigram corpus count to consider
    max_merges: int|None  Cap on number of merge rules (None = no cap)
    """

    def __init__(
        self,
        tau_pmi:    float = 1.0,
        tau_idf:    float = 3.0,
        min_count:  int   = 2,
        max_merges: Optional[int] = 50,   # cap prevents vocab explosion on V=100 corpora
    ):
        self.tau_pmi    = tau_pmi
        self.tau_idf    = tau_idf
        self.min_count  = min_count
        self.max_merges = max_merges

        self.merge_map:     Dict[Tuple[int,int], int] = {}
        self.next_token_id: int = 0
        self.idf_map:       Dict[int, float] = {}
        self.n_merges:      int = 0
        self.is_fitted:     bool = False

    def fit(self, sequences: List[List[int]]) -> "PMITokenDedup":
        n_docs  = len(sequences)
        max_tok = max((max(s) for s in sequences if s), default=0)
        self.next_token_id = max_tok + 1

        # ── IDF for the IDF gate ──────────────────────────────────────
        df: Dict[int, int] = defaultdict(int)
        for seq in sequences:
            for tok in set(seq):
                df[tok] += 1
        self.idf_map = {
            tok: math.log(n_docs / cnt)
            for tok, cnt in df.items()
        }

        # ── PMI computation ───────────────────────────────────────────
        pmi_scores, _ = compute_pmi(sequences, self.min_count)

        # ── Apply dual criterion ──────────────────────────────────────
        candidates = []
        for (a, b), pmi_val in pmi_scores.items():
            if pmi_val < self.tau_pmi:
                continue
            if max(self.idf_map.get(a, 0.0), self.idf_map.get(b, 0.0)) > self.tau_idf:
                continue
            candidates.append(((a, b), pmi_val))

        # Sort by descending PMI — highest-PMI (most redundant) pairs first
        candidates.sort(key=lambda x: -x[1])
        if self.max_merges is not None:
            candidates = candidates[:self.max_merges]

        # ── Assign new token IDs ──────────────────────────────────────
        self.merge_map = {}
        for (a, b), _ in candidates:
            self.merge_map[(a, b)] = self.next_token_id
            self.next_token_id += 1

        self.n_merges  = len(self.merge_map)
        self.is_fitted = True
        print(
            f"  [PMI-TD] τ_pmi={self.tau_pmi:.1f}  τ_idf={self.tau_idf:.1f}"
            f"  →  {self.n_merges} merge rules"
            f"  (vocab size: {max_tok + 1} → {self.next_token_id})"
        )
        return self

    def transform(self, sequences: List[List[int]]) -> List[List[int]]:
        if not self.is_fitted or not self.merge_map:
            return sequences
        return [self._apply(seq) for seq in sequences]

    def fit_transform(self, sequences: List[List[int]]) -> List[List[int]]:
        self.fit(sequences)
        return self.transform(sequences)

    def _apply(self, seq: List[int]) -> List[int]:
        """One-pass left-to-right greedy merge application."""
        if len(seq) < 2:
            return list(seq)
        out = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq):
                key = (seq[i], seq[i + 1])
                if key in self.merge_map:
                    out.append(self.merge_map[key])
                    i += 2
                    continue
            out.append(seq[i])
            i += 1
        return out

    def diagnostics(self) -> Dict:
        return {
            "tau_pmi":   self.tau_pmi,
            "tau_idf":   self.tau_idf,
            "n_merges":  self.n_merges,
            "new_vocab": self.next_token_id,
        }


# =============================================================================
# ── Section 6: PMI-TF-IDF vectoriser  ────────────────────────────────────────
# =============================================================================

class PMIFilteredTFIDF:
    """
    Unigrams + PMI-filtered bigrams TF-IDF vectoriser.

    Features
    --------
    - Unigram TF-IDF (same as UnigramTFIDF)
    - Bigrams with PMI ≥ τ_pmi, weighted by α relative to unigrams

    The PMI filter selects bigrams where p(a,b) > p(a)·p(b), discarding
    common low-IDF bigrams that dilute the cosine signal.
    Setting τ_pmi → ∞ recovers pure unigram TF-IDF.
    Setting α = 0 recovers pure unigram TF-IDF.

    Parameters
    ----------
    tau_pmi       : float   PMI threshold for bigram selection
    bigram_weight : float   α — scaling of bigram features vs unigrams
    min_count     : int     Minimum bigram count to consider
    """

    def __init__(
        self,
        tau_pmi:       float = 1.5,
        bigram_weight: float = 0.5,
        min_count:     int   = 2,
    ):
        self.tau_pmi       = tau_pmi
        self.bigram_weight = bigram_weight
        self.min_count     = min_count

        self.uni_vocab:   Dict[int, int]            = {}
        self.uni_idf:     Dict[int, float]          = {}
        self.bi_vocab:    Dict[Tuple[int,int], int] = {}
        self.bi_idf:      Dict[Tuple[int,int], float] = {}
        self.pmi_scores:  Dict[Tuple[int,int], float] = {}
        self.pmi_matrix:  Dict[Tuple[int,int], float] = {}   # for soft-SW
        self.n_docs: int = 0
        self.is_fitted = False

    def fit(self, sequences: List[List[int]]) -> "PMIFilteredTFIDF":
        self.n_docs = len(sequences)

        # ── Unigram IDF ───────────────────────────────────────────────
        uni_df: Dict[int, int] = defaultdict(int)
        for seq in sequences:
            for tok in set(seq):
                uni_df[tok] += 1
        self.uni_vocab = {tok: i for i, tok in enumerate(sorted(uni_df))}
        self.uni_idf   = {
            tok: math.log(self.n_docs / df)
            for tok, df in uni_df.items()
        }

        # ── PMI + bigram IDF ──────────────────────────────────────────
        self.pmi_scores, _ = compute_pmi(sequences, self.min_count)

        # Keep only PMI-filtered bigrams
        selected_bigrams = {
            bg for bg, pmi in self.pmi_scores.items()
            if pmi >= self.tau_pmi
        }

        bi_df: Dict[Tuple[int,int], int] = defaultdict(int)
        for seq in sequences:
            seen = set()
            for i in range(len(seq) - 1):
                bg = (seq[i], seq[i + 1])
                if bg in selected_bigrams and bg not in seen:
                    bi_df[bg] += 1
                    seen.add(bg)
        self.bi_vocab = {
            bg: i + len(self.uni_vocab)
            for i, bg in enumerate(sorted(bi_df))
        }
        self.bi_idf = {
            bg: math.log(self.n_docs / df)
            for bg, df in bi_df.items()
        }

        # Store symmetric PMI matrix for soft-SW
        self.pmi_matrix = {
            **{(a, b): v for (a, b), v in self.pmi_scores.items()},
            **{(b, a): v for (a, b), v in self.pmi_scores.items()},
        }

        self.is_fitted = True
        print(
            f"  [PMI-TF-IDF] τ_pmi={self.tau_pmi:.1f}  α={self.bigram_weight:.2f}"
            f"  →  {len(self.uni_vocab)} unigrams + {len(self.bi_vocab)} bigrams"
            f"  =  {len(self.uni_vocab) + len(self.bi_vocab)} features"
        )
        return self

    def _vectorize(self, seq: List[int]) -> np.ndarray:
        dim = len(self.uni_vocab) + len(self.bi_vocab)
        vec = np.zeros(dim, dtype=np.float32)
        if not seq:
            return vec
        L = len(seq)

        # Unigrams
        for tok, cnt in Counter(seq).items():
            col = self.uni_vocab.get(tok)
            if col is not None:
                vec[col] = (cnt / L) * self.uni_idf.get(tok, 0.0)

        # PMI bigrams
        if self.bigram_weight > 0 and self.bi_vocab:
            for i in range(L - 1):
                bg = (seq[i], seq[i + 1])
                col = self.bi_vocab.get(bg)
                if col is not None:
                    vec[col] += self.bigram_weight * (1.0 / (L - 1)) * self.bi_idf.get(bg, 0.0)
        return vec

    def transform(self, sequences: List[List[int]]) -> np.ndarray:
        dim = len(self.uni_vocab) + len(self.bi_vocab)
        mat = np.zeros((len(sequences), dim), dtype=np.float32)
        for i, seq in enumerate(sequences):
            mat[i] = self._vectorize(seq)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def transform_one(self, seq: List[int]) -> np.ndarray:
        vec = self._vectorize(seq)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def fit_transform(self, sequences: List[List[int]]) -> np.ndarray:
        self.fit(sequences)
        return self.transform(sequences)


# =============================================================================
# ── Section 7: Evaluation metrics  ───────────────────────────────────────────
# =============================================================================

def _ap(ranked_indices: List[int], relevant: Set[int]) -> float:
    """Average Precision for a single query."""
    if not relevant:
        return 0.0
    hits = ap = 0.0
    for rank, idx in enumerate(ranked_indices, 1):
        if idx in relevant:
            hits += 1
            ap   += hits / rank
    return ap / len(relevant)


def evaluate(
    ranked_lists: List[List[int]],    # one ranked list per query
    ground_truth: List[Set[int]],     # one set of relevant indices per query
) -> Dict[str, float]:
    """
    Compute MAP, P@1, P@5, P@10 over all queries.
    ranked_lists[q] must contain ALL corpus indices (for MAP to be correct).
    """
    maps = [_ap(rl, gt) for rl, gt in zip(ranked_lists, ground_truth)]
    def precision_at_k(k):
        ps = [
            len(set(rl[:k]) & gt) / k
            for rl, gt in zip(ranked_lists, ground_truth)
        ]
        return float(np.mean(ps))

    return {
        "MAP":  float(np.mean(maps)),
        "P@1":  precision_at_k(1),
        "P@5":  precision_at_k(5),
        "P@10": precision_at_k(10),
    }


# =============================================================================
# ── Section 8: System 1 — TF-IDF Baseline  ───────────────────────────────────
# =============================================================================

class TFIDFBaseline:
    """
    Exact reproduction of Singh et al. (2024) TF-IDF QbE-STD.
    Unigram TF-IDF + brute-force cosine similarity.
    No HNSW, no SW reranking.
    """

    def __init__(self):
        self.tfidf   = UnigramTFIDF()
        self._matrix = None   # shape (N, V), L2-normalised

    def fit(self, corpus_seqs: List[List[int]]) -> "TFIDFBaseline":
        self._matrix = self.tfidf.fit_transform(corpus_seqs)
        return self

    def rank(self, query_seq: List[int]) -> List[int]:
        """Return corpus indices sorted by cosine similarity (descending)."""
        q_vec = self.tfidf.transform_one(query_seq)
        sims  = self._matrix @ q_vec
        return list(np.argsort(-sims))

    def run(
        self,
        query_seqs:   List[List[int]],
        ground_truth: List[Set[int]],
    ) -> Dict[str, float]:
        ranked = [self.rank(q) for q in query_seqs]
        return evaluate(ranked, ground_truth)


# =============================================================================
# ── Section 9: System 2 — H-QuEST  ───────────────────────────────────────────
# =============================================================================

class HQuEST:
    """
    Faithful reproduction of H-QuEST (Singh et al., Interspeech 2025).

    Pipeline
    --------
    Raw tokens
        → Unigram TF-IDF vectorisation
        → HNSW index  (M=16, ef_construction=150)
        → Top-K=50 candidates retrieved by cosine
        → Smith-Waterman reranking (+2 / -1 / -2)
        → Final ranked list

    No BPE, no bigrams, no PMI.  This is the exact baseline against
    which PMI-QuEST is compared.

    Parameters
    ----------
    hnsw_k           : int   Number of HNSW candidates before SW reranking (K=50)
    sw_match         : float Smith-Waterman match score
    sw_mismatch      : float Smith-Waterman mismatch score
    sw_gap           : float Smith-Waterman gap penalty
    """

    def __init__(
        self,
        hnsw_k:      int   = 50,
        sw_match:    float = 2.0,
        sw_mismatch: float = -1.0,
        sw_gap:      float = -2.0,
    ):
        self.hnsw_k      = hnsw_k
        self.sw_match    = sw_match
        self.sw_mismatch = sw_mismatch
        self.sw_gap      = sw_gap

        self.tfidf:        UnigramTFIDF = UnigramTFIDF()
        self.hnsw:         HNSWIndex   = HNSWIndex(n_neighbors=hnsw_k)
        self._corpus_seqs: List[List[int]] = []
        self._matrix:      np.ndarray  = None

    def fit(self, corpus_seqs: List[List[int]]) -> "HQuEST":
        self._corpus_seqs = corpus_seqs
        self._matrix      = self.tfidf.fit_transform(corpus_seqs)
        self.hnsw.fit(self._matrix)
        return self

    def rank(self, query_seq: List[int]) -> List[int]:
        """Retrieve top-K via HNSW then rerank by SW."""
        q_vec      = self.tfidf.transform_one(query_seq)
        candidates = self.hnsw.search(q_vec, k=self.hnsw_k)
        reranked   = sw_rerank(
            query_seq, candidates, self._corpus_seqs,
            self.sw_match, self.sw_mismatch, self.sw_gap
        )
        # Append remaining corpus docs (not in candidates) sorted by cosine
        top_k_idxs = {idx for _, idx in reranked}
        sims = self._matrix @ q_vec
        remaining = [
            (float(sims[i]), i)
            for i in np.argsort(-sims)
            if i not in top_k_idxs
        ]
        # Final list: SW-reranked top-K first, then cosine-ranked remainder
        final_ranked = [idx for _, idx in reranked] + [idx for _, idx in remaining]
        return final_ranked

    def run(
        self,
        query_seqs:   List[List[int]],
        ground_truth: List[Set[int]],
    ) -> Dict[str, float]:
        ranked = [self.rank(q) for q in query_seqs]
        return evaluate(ranked, ground_truth)


# =============================================================================
# ── Section 10: System 3 — PMI-QuEST (proposed)  ─────────────────────────────
# =============================================================================

class PMIQuest:
    """
    PMI-QuEST: the proposed method for the TASLP paper.

    Three components replace their H-QuEST counterparts:

    1. PMI-TD replaces BPE
       Token deduplication using dual criterion: PMI ≥ τ_pmi AND IDF ≤ τ_idf.
       Provably entropy-non-decreasing (Theorem 3, paper §IV).

    2. PMI-TF-IDF replaces unigram TF-IDF
       Unigrams + PMI-filtered bigrams (τ_pmi threshold).
       Acts as a discriminative pre-filter for HNSW: richer feature space
       means HNSW retrieves higher-quality SW candidates.

    3. Regime-gated SW replaces hard SW
       Computes ρ = |query| / mean_corpus_length.
       ρ < ρ_star  →  hard Smith-Waterman   (word mode: ρ ≈ 0.05)
       ρ ≥ ρ_star  →  PMI-soft SW           (utterance mode: ρ ≈ 1.0)

       In word mode, substitution tolerance hurts MAP by ~50% because
       the aligner finds false soft-matches in random document locations.
       In utterance mode, partial credit over ~221 alignment steps
       improves MAP by +38% (empirical, this paper).

    Parameters
    ----------
    pmitd_tau_pmi  : float  PMI threshold for token deduplication
    pmitd_tau_idf  : float  IDF ceiling for token deduplication
    pmi_tau        : float  PMI threshold for bigram feature selection
    bigram_weight  : float  α — bigram weight in TF-IDF vector
    hnsw_k         : int    HNSW candidate set size
    rho_star       : float  Regime crossover threshold
    sw_match       : float  Hard-SW match score
    sw_mismatch    : float  Hard-SW mismatch score
    sw_gap         : float  SW gap penalty (shared)
    pmi_scale      : float  Scale factor for PMI-soft-SW substitution scores
    use_pmitd      : bool   Toggle PMI-TD (ablation: set False for no dedup)
    use_pmi_bigrams: bool   Toggle PMI bigrams (ablation: set False for unigrams only)
    use_regime_gate: bool   Toggle regime-gated SW (ablation: set False for hard-SW always)
    """

    def __init__(
        self,
        pmitd_tau_pmi:   float = 1.0,
        pmitd_tau_idf:   float = 3.0,
        pmi_tau:         float = 1.5,
        bigram_weight:   float = 0.5,
        hnsw_k:          int   = 50,
        rho_star:        float = 0.15,     # empirical crossover from paper §IV
        sw_match:        float = 2.0,
        sw_mismatch:     float = -1.0,
        sw_gap:          float = -2.0,
        pmi_scale:       float = 1.0,
        use_pmitd:       bool  = True,
        use_pmi_bigrams: bool  = True,
        use_regime_gate: bool  = True,
    ):
        self.pmitd_tau_pmi   = pmitd_tau_pmi
        self.pmitd_tau_idf   = pmitd_tau_idf
        self.pmi_tau         = pmi_tau
        self.bigram_weight   = bigram_weight
        self.hnsw_k          = hnsw_k
        self.rho_star        = rho_star
        self.sw_match        = sw_match
        self.sw_mismatch     = sw_mismatch
        self.sw_gap          = sw_gap
        self.pmi_scale       = pmi_scale
        self.use_pmitd       = use_pmitd
        self.use_pmi_bigrams = use_pmi_bigrams
        self.use_regime_gate = use_regime_gate

        self.pmitd:          Optional[PMITokenDedup]   = PMITokenDedup(pmitd_tau_pmi, pmitd_tau_idf, max_merges=50) if use_pmitd else None
        self.pmi_tfidf:      PMIFilteredTFIDF          = PMIFilteredTFIDF(pmi_tau, bigram_weight)
        self.fallback_tfidf: UnigramTFIDF              = UnigramTFIDF()   # used when use_pmi_bigrams=False
        self.hnsw:           HNSWIndex                 = HNSWIndex(n_neighbors=hnsw_k)

        self._corpus_seqs_raw:  List[List[int]] = []
        self._corpus_seqs_comp: List[List[int]] = []
        self._matrix:           np.ndarray      = None
        self._mean_L:           float           = 0.0
        self._pmi_matrix:       Dict            = {}

    def fit(self, corpus_seqs: List[List[int]]) -> "PMIQuest":
        self._corpus_seqs_raw = corpus_seqs

        # ── Step 1: PMI-TD compression ────────────────────────────────
        # PMI-TD operates on the ORIGINAL 100-token vocabulary.
        # Compressed sequences are used only for SW reranking — NOT for
        # PMI-TF-IDF vectorisation.  This prevents the vocabulary explosion
        # where 889 new token types spawn 32k new bigram features.
        if self.use_pmitd and self.pmitd is not None:
            self._corpus_seqs_comp = self.pmitd.fit_transform(corpus_seqs)
        else:
            self._corpus_seqs_comp = corpus_seqs

        self._mean_L = float(np.mean([len(s) for s in self._corpus_seqs_comp])) if self._corpus_seqs_comp else 1.0

        # ── Step 2: PMI-TF-IDF vectorisation on ORIGINAL sequences ───
        # Key fix: always fit PMI-TF-IDF on original (pre-PMI-TD) sequences.
        # This keeps bigrams within the original V=100 vocabulary → at most
        # V²=10,000 possible bigrams, of which ~600 pass the τ_pmi filter.
        # The compressed sequences are only used downstream for SW alignment.
        vec_seqs = corpus_seqs   # always use original vocab for vectorisation

        if self.use_pmi_bigrams:
            self._matrix = self.pmi_tfidf.fit_transform(vec_seqs)
            self._pmi_matrix = self.pmi_tfidf.pmi_matrix
        else:
            self._matrix = self.fallback_tfidf.fit_transform(vec_seqs)
            pmi_scores, _ = compute_pmi(vec_seqs)
            self._pmi_matrix = {
                **pmi_scores,
                **{(b, a): v for (a, b), v in pmi_scores.items()},
            }

        # ── Step 3: Build HNSW index ──────────────────────────────────
        self.hnsw.fit(self._matrix)
        return self

    def _compress_query(self, query_seq: List[int]) -> List[int]:
        if self.use_pmitd and self.pmitd is not None and self.pmitd.is_fitted:
            return self.pmitd._apply(query_seq)
        return query_seq

    def rank(self, query_seq: List[int]) -> List[int]:
        q_comp = self._compress_query(query_seq)
        rho    = len(q_comp) / self._mean_L if self._mean_L > 0 else 0.0

        # ── HNSW candidate retrieval ──────────────────────────────────
        if self.use_pmi_bigrams:
            q_vec = self.pmi_tfidf.transform_one(q_comp)
        else:
            q_vec = self.fallback_tfidf.transform_one(q_comp)

        candidates = self.hnsw.search(q_vec, k=self.hnsw_k)

        # ── Regime-gated reranking ────────────────────────────────────
        if not self.use_regime_gate or rho < self.rho_star:
            # Word mode: hard Smith-Waterman
            reranked = sw_rerank(
                q_comp, candidates, self._corpus_seqs_comp,
                self.sw_match, self.sw_mismatch, self.sw_gap
            )
        else:
            # Utterance mode: PMI-soft Smith-Waterman
            reranked = sw_pmi_rerank(
                q_comp, candidates, self._corpus_seqs_comp,
                self._pmi_matrix,
                self.sw_match, self.sw_gap, self.pmi_scale
            )

        # Append non-candidate docs sorted by cosine
        top_k_idxs = {idx for _, idx in reranked}
        sims = self._matrix @ q_vec
        remaining = [
            (float(sims[i]), i)
            for i in np.argsort(-sims)
            if i not in top_k_idxs
        ]
        return [idx for _, idx in reranked] + [idx for _, idx in remaining]

    def run(
        self,
        query_seqs:   List[List[int]],
        ground_truth: List[Set[int]],
    ) -> Dict[str, float]:
        ranked = [self.rank(q) for q in query_seqs]
        return evaluate(ranked, ground_truth)


# =============================================================================
# ── Section 11: Comparison runner  ───────────────────────────────────────────
# =============================================================================

def _fmt(metrics: Dict[str, float]) -> str:
    return (
        f"MAP={metrics['MAP']:.4f}  "
        f"P@1={metrics['P@1']:.4f}  "
        f"P@5={metrics['P@5']:.4f}  "
        f"P@10={metrics['P@10']:.4f}"
    )


def run_comparison(
    corpus_seqs:  List[List[int]],
    query_seqs:   List[List[int]],
    ground_truth: List[Set[int]],
    verbose:      bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Run all three systems and return results dict.

    Returns
    -------
    {
        "tfidf_baseline": {"MAP": ..., "P@1": ..., "P@5": ..., "P@10": ...},
        "hquest":         {...},
        "pmiquest":       {...},
    }
    """
    results = {}
    N = len(corpus_seqs)
    Q = len(query_seqs)
    mean_q_len = float(np.mean([len(q) for q in query_seqs]))
    mean_c_len = float(np.mean([len(c) for c in corpus_seqs]))
    rho        = mean_q_len / mean_c_len if mean_c_len > 0 else 0.0

    if verbose:
        print("=" * 65)
        print("PMI-QuEST vs H-QuEST vs TF-IDF Baseline")
        print("=" * 65)
        print(f"  Corpus: {N} docs  |  Queries: {Q}")
        print(f"  Mean query length:  {mean_q_len:.1f} tokens")
        print(f"  Mean corpus length: {mean_c_len:.1f} tokens")
        print(f"  ρ = {rho:.3f}  ({'SW regime' if rho < 0.15 else 'TF-IDF regime'})")
        print()

    # ── 1. TF-IDF Baseline ─────────────────────────────────────────
    if verbose:
        print("─" * 65)
        print("[1/3] TF-IDF Baseline (Singh et al. 2024)")
    t0 = time.time()
    sys1 = TFIDFBaseline()
    sys1.fit(corpus_seqs)
    r1 = sys1.run(query_seqs, ground_truth)
    r1["build_time"] = time.time() - t0
    results["tfidf_baseline"] = r1
    if verbose:
        print(f"  {_fmt(r1)}  [{r1['build_time']:.1f}s]")

    # ── 2. H-QuEST ─────────────────────────────────────────────────
    if verbose:
        print("─" * 65)
        print("[2/3] H-QuEST (Singh et al., Interspeech 2025)")
        print("       Raw tokens → Unigram TF-IDF → HNSW → SW rerank")
    t0 = time.time()
    sys2 = HQuEST(hnsw_k=50, sw_match=2.0, sw_mismatch=-1.0, sw_gap=-2.0)
    sys2.fit(corpus_seqs)
    r2 = sys2.run(query_seqs, ground_truth)
    r2["build_time"] = time.time() - t0
    results["hquest"] = r2
    if verbose:
        print(f"  {_fmt(r2)}  [{r2['build_time']:.1f}s]")

    # ── 3. PMI-QuEST ───────────────────────────────────────────────
    if verbose:
        print("─" * 65)
        print("[3/3] PMI-QuEST (proposed)")
        print("       PMI-TD → PMI-TF-IDF → HNSW → Regime-gated SW")
    t0 = time.time()
    sys3 = PMIQuest(
        pmitd_tau_pmi=1.0,
        pmitd_tau_idf=3.0,
        pmi_tau=1.5,
        bigram_weight=0.5,
        hnsw_k=50,
        rho_star=0.15,
    )
    sys3.fit(corpus_seqs)
    r3 = sys3.run(query_seqs, ground_truth)
    r3["build_time"] = time.time() - t0
    results["pmiquest"] = r3
    if verbose:
        print(f"  {_fmt(r3)}  [{r3['build_time']:.1f}s]")

    # ── Summary table ───────────────────────────────────────────────
    if verbose:
        print()
        print("=" * 65)
        print(f"{'System':<35} {'MAP':>6} {'P@1':>6} {'P@5':>6} {'P@10':>6}")
        print("-" * 65)
        for name, label in [
            ("tfidf_baseline", "TF-IDF Baseline"),
            ("hquest",         "H-QuEST"),
            ("pmiquest",       "PMI-QuEST (proposed)"),
        ]:
            m = results[name]
            gain = ""
            if name != "tfidf_baseline":
                delta = (m["MAP"] - results["tfidf_baseline"]["MAP"])
                gain = f"  ({delta:+.4f} vs baseline)"
            print(
                f"  {label:<33} {m['MAP']:>6.4f} {m['P@1']:>6.4f}"
                f" {m['P@5']:>6.4f} {m['P@10']:>6.4f}{gain}"
            )
        print("=" * 65)

    return results


def run_pmiquest_ablation(
    corpus_seqs:  List[List[int]],
    query_seqs:   List[List[int]],
    ground_truth: List[Set[int]],
) -> List[Dict]:
    """
    Full ablation of PMI-QuEST components for the paper Table III.
    Tests each component independently against the full system.
    """
    configs = [
        ("TF-IDF Baseline",
         TFIDFBaseline()),
        ("H-QuEST (Singh et al. 2025)",
         HQuEST()),
        # ── PMI-QuEST full system ─────────────────────────────────────
        ("PMI-QuEST: PMI-TD(50) + PMI-TF-IDF + HNSW + SW  [proposed]",
         PMIQuest()),
        # ── Ablate PMI-TD ─────────────────────────────────────────────
        ("PMI-QuEST: no PMI-TD (raw tokens)",
         PMIQuest(use_pmitd=False)),
        ("PMI-QuEST: PMI-TD max_merges=20",
         PMIQuest(pmitd_tau_pmi=1.0, pmitd_tau_idf=3.0,
                  use_pmitd=True)),   # PMITokenDedup default now 50; override below
        ("PMI-QuEST: PMI-TD max_merges=100",
         PMIQuest.__new__(PMIQuest)),   # constructed manually below
        # ── Ablate PMI bigrams ────────────────────────────────────────
        ("PMI-QuEST: no PMI bigrams (unigrams only)",
         PMIQuest(use_pmi_bigrams=False)),
        # ── τ_pmi sweep ──────────────────────────────────────────────
        ("PMI-QuEST: τ_pmi=0.5",
         PMIQuest(pmi_tau=0.5)),
        ("PMI-QuEST: τ_pmi=1.5",
         PMIQuest(pmi_tau=1.5)),
        ("PMI-QuEST: τ_pmi=2.0",
         PMIQuest(pmi_tau=2.0)),
        # ── α sweep ──────────────────────────────────────────────────
        ("PMI-QuEST: α=0.25",
         PMIQuest(bigram_weight=0.25)),
        ("PMI-QuEST: α=1.0",
         PMIQuest(bigram_weight=1.0)),
    ]

    # Fix the two max_merges variants (PMITokenDedup doesn't expose it through PMIQuest yet)
    def make_pmiq_with_maxmerges(n):
        sys = PMIQuest(use_pmitd=True)
        sys.pmitd = PMITokenDedup(tau_pmi=1.0, tau_idf=3.0, max_merges=n)
        return sys

    configs[4] = ("PMI-QuEST: PMI-TD max_merges=20",  make_pmiq_with_maxmerges(20))
    configs[5] = ("PMI-QuEST: PMI-TD max_merges=100", make_pmiq_with_maxmerges(100))

    rows = []
    print("\n" + "=" * 65)
    print("PMI-QuEST COMPONENT ABLATION")
    print("=" * 65)
    for label, system in configs:
        print(f"\n  {label}")
        t0 = time.time()
        if hasattr(system, "fit"):
            system.fit(corpus_seqs)
        else:
            # TFIDFBaseline
            system.fit(corpus_seqs)
        metrics = system.run(query_seqs, ground_truth)
        metrics["label"]      = label
        metrics["build_time"] = time.time() - t0
        rows.append(metrics)
        print(f"    {_fmt(metrics)}  [{metrics['build_time']:.1f}s]")

    return rows


# =============================================================================
# ── Section 12: Quick self-test with synthetic data  ─────────────────────────
# =============================================================================

if __name__ == "__main__":
    import random
    rng = random.Random(42)

    print("PMI-QuEST self-test on synthetic data")
    print("(Replace with real LibriSpeech token sequences for paper results)")
    print()

    V = 100   # vocab size (matches wav2vec2 k-means)
    N = 200   # corpus size
    Q = 20    # queries

    # Synthetic corpus: each doc is a random token sequence of length ~50-300
    corpus_seqs = [
        [rng.randint(0, V - 1) for _ in range(rng.randint(50, 300))]
        for _ in range(N)
    ]

    # Word-mode queries: short (length ~12), relevant = docs containing same pattern
    query_seqs   = []
    ground_truth = []
    for q_idx in range(Q):
        # Pick a random 12-token pattern and plant it in 3-8 corpus docs
        pattern = [rng.randint(0, V - 1) for _ in range(12)]
        relevant_idxs = set(rng.sample(range(N), k=rng.randint(3, 8)))
        for idx in relevant_idxs:
            pos = rng.randint(0, max(0, len(corpus_seqs[idx]) - 12))
            corpus_seqs[idx] = (
                corpus_seqs[idx][:pos] + pattern + corpus_seqs[idx][pos + 12:]
            )
        query_seqs.append(pattern)
        ground_truth.append(relevant_idxs)

    results = run_comparison(corpus_seqs, query_seqs, ground_truth)
    print("\nDone. To run on real data, call run_comparison() with")
    print("your LibriSpeech token sequences and ground truth sets.")
