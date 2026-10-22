"""Neuro-GRAPE Model Entrypoint (Member 2 — Phase 5 Handoff).

Re-exports TrimodalNeuroGrapeModel and NeuroGrapeModel from src.models.model
with full interface and tensor dimension documentation.

Tensor Dimensions
-----------------
EEG:  (B, 63, 225) float32       -> (B, 128) L2-normalized
MEG:  (B, 272, 225) float32      -> (B, 128) L2-normalized
fMRI: (B, 1, 96, 96, 60) float32 -> (B, 128) L2-normalized
"""

from __future__ import annotations

from src.models.encoders import EEGEncoder, FMRIEncoder, MEGEncoder
from src.models.model import NeuroGrapeModel, TrimodalNeuroGrapeModel

__all__ = [
    "TrimodalNeuroGrapeModel",
    "NeuroGrapeModel",
    "EEGEncoder",
    "MEGEncoder",
    "FMRIEncoder",
]
