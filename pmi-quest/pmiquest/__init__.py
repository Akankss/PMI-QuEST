"""
PMI-QuEST: Pointwise Mutual Information-Guided Token Bigrams
for Query-by-Example Spoken Term Detection.

Singh, Chen, Arora — IEEE/ACM TASLP (submitted 2025)

Quick usage:
    from pmiquest import PMIQuest, evaluate

    system = PMIQuest(tau=0.5, alpha=0.5, K=50)
    system.index(corpus_seqs)
    results = system.query(query_seq)
    metrics = evaluate(results, relevance)
"""

from pmiquest.system import (
    PMIQuest,
    HQuest,
    TFIDFBaseline,
    evaluate,
    average_precision,
    precision_at_k,
)

__all__ = [
    "PMIQuest",
    "HQuest",
    "TFIDFBaseline",
    "evaluate",
    "average_precision",
    "precision_at_k",
]

__version__ = "1.0.0"
__author__  = "Akanksha Singh, Yi-Ping Phoebe Chen, Vipul Arora"
