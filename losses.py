"""Neuro-GRAPE Losses Entrypoint (Member 2 — Phase 5 Handoff).

Re-exports contrastive loss functions and per-example loss vector utilities
for Member 3's evaluation and GroupDRO integration.

API:
- per_sample_trimodal_contrastive_loss: Returns (N,) per-example loss vector + stimulus IDs.
- trimodal_contrastive_loss: Supports reduction='mean', 'none', and sample_weights.
- multimodal_contrastive_loss: Pairwise contrastive loss with reduction options.
- compute_group_weighted_loss: Computes group-weighted objective for GroupDRO.
"""

from __future__ import annotations

from src.models.losses import (
    compute_group_weighted_loss,
    multimodal_contrastive_loss,
    per_sample_trimodal_contrastive_loss,
    trimodal_contrastive_loss,
)

__all__ = [
    "multimodal_contrastive_loss",
    "trimodal_contrastive_loss",
    "per_sample_trimodal_contrastive_loss",
    "compute_group_weighted_loss",
]
