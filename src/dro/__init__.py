"""Neuro-GRAPE Distributionally Robust Optimization & Evaluation (Member 3).

This package provides:
- GroupDRO: Distributionally robust group-weighted loss computation and exponentiated weight updating.
- EmbeddingClustering: Unsupervised regime discovery on training embeddings via K-Means.
- GroupEvaluator: Comprehensive evaluation computing overall, per-group, and worst-group metrics.
- Pipeline: End-to-end reproducible pipeline orchestrator.
"""

from __future__ import annotations

from src.dro.group_dro import GroupDRO
from src.dro.clustering import EmbeddingClustering
from src.dro.evaluation import GroupEvaluator, EvaluationResult

__all__ = [
    "GroupDRO",
    "EmbeddingClustering",
    "GroupEvaluator",
    "EvaluationResult",
]
