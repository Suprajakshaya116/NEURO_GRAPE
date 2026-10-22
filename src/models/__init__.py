"""Neuro-GRAPE model package.

Public API
----------
EEGEncoder                  : Compact 1-D CNN encoder for EEG signals.
MEGEncoder                  : Compact 1-D CNN encoder for MEG signals.
FMRIEncoder                 : Compact 3-D CNN encoder for fMRI beta volumes.
multimodal_contrastive_loss : Symmetric pairwise NT-Xent loss (two modalities).
trimodal_contrastive_loss   : Mean of three pairwise losses (EEG, MEG, fMRI).
"""

from src.models.encoders import EEGEncoder, FMRIEncoder, MEGEncoder
from src.models.model import NeuroGrapeModel, TrimodalNeuroGrapeModel
from src.models.contrastive_loss import (
    multimodal_contrastive_loss,
    trimodal_contrastive_loss,
)
from src.models.normalization import TrimodalNormalizer
from src.models.training import (
    evaluate,
    prepare_cross_modal_batch,
    set_seed,
    train_epoch,
    train_model,
)

__all__ = [
    "EEGEncoder",
    "MEGEncoder",
    "FMRIEncoder",
    "TrimodalNeuroGrapeModel",
    "NeuroGrapeModel",
    "multimodal_contrastive_loss",
    "trimodal_contrastive_loss",
    "TrimodalNormalizer",
    "set_seed",
    "prepare_cross_modal_batch",
    "train_epoch",
    "evaluate",
    "train_model",
]

