"""Neuro-GRAPE Contrastive Alignment Loss (Member 2 — Phase 3).

Standalone, pure-PyTorch loss functions for cross-modal stimulus alignment.
Does NOT import from src.data or depend on any dataset class.

Public API
----------
multimodal_contrastive_loss : Pairwise symmetric NT-Xent loss for two modalities.
trimodal_contrastive_loss   : Averages three pairwise losses (EEG, MEG, fMRI).

Design notes
------------
* Positive-pair masks are built from stimulus_id string equality, so neither
  function assumes row-order alignment between the two modality batches.
* Multi-positive: a row may have more than one matching positive (e.g. if the
  same stimulus_id appears multiple times). The objective uses a soft
  log-sum-exp over all positives in the denominator, following
  "Supervised Contrastive Learning" (Khosla et al., 2020).
* Temperature is a plain float; no learnable parameter in Phase 3.
* All operations stay on the device and dtype of the input tensors.
* Numerically stable via logsumexp formulation.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_positive_mask(ids_a: Sequence[str], ids_b: Sequence[str]) -> torch.Tensor:
    """Return a boolean mask of shape (N, M) where mask[i, j] is True iff
    ids_a[i] == ids_b[j].

    Args:
        ids_a: Length-N list of stimulus IDs.
        ids_b: Length-M list of stimulus IDs.

    Returns:
        Boolean tensor of shape (N, M).
    """
    n, m = len(ids_a), len(ids_b)
    mask = torch.zeros(n, m, dtype=torch.bool)
    for i, a in enumerate(ids_a):
        for j, b in enumerate(ids_b):
            if a == b:
                mask[i, j] = True
    return mask


def _validate_inputs(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    ids_a: Sequence[str],
    ids_b: Sequence[str],
    temperature: float,
) -> None:
    """Raise ValueError for any invalid input combination."""
    if z_a.ndim != 2:
        raise ValueError(
            f"z_a must be a 2-D tensor (N, D); got ndim={z_a.ndim}."
        )
    if z_b.ndim != 2:
        raise ValueError(
            f"z_b must be a 2-D tensor (M, D); got ndim={z_b.ndim}."
        )
    if z_a.shape[1] != z_b.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: z_a has D={z_a.shape[1]}, "
            f"z_b has D={z_b.shape[1]}. Both must share the same embed_dim."
        )
    if len(ids_a) != z_a.shape[0]:
        raise ValueError(
            f"ids_a length ({len(ids_a)}) must equal z_a.shape[0] ({z_a.shape[0]})."
        )
    if len(ids_b) != z_b.shape[0]:
        raise ValueError(
            f"ids_b length ({len(ids_b)}) must equal z_b.shape[0] ({z_b.shape[0]})."
        )
    if temperature <= 0.0:
        raise ValueError(
            f"temperature must be > 0; got {temperature}."
        )


# ---------------------------------------------------------------------------
# Pairwise symmetric contrastive loss
# ---------------------------------------------------------------------------

def multimodal_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    ids_a: Sequence[str],
    ids_b: Sequence[str],
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric cross-modal contrastive (NT-Xent-style) loss.

    Positive pairs are identified by matching stimulus_id strings — no
    row-order alignment is assumed between the two modality batches.
    Multi-positive rows are handled via a supervised contrastive objective:
    the loss for row i is the mean negative log-likelihood over all its
    positive targets.

    Args:
        z_a:         (N, D) L2-normalised embeddings for modality A.
        z_b:         (M, D) L2-normalised embeddings for modality B.
        ids_a:       Length-N sequence of stimulus IDs for z_a rows.
        ids_b:       Length-M sequence of stimulus IDs for z_b rows.
        temperature: Softmax temperature scalar (> 0). Default 0.07.

    Returns:
        Scalar contrastive loss tensor, suitable for loss.backward().

    Raises:
        ValueError: On invalid shapes, dimension mismatch, ID length
            mismatch, non-positive temperature, or no valid positive matches.
    """
    _validate_inputs(z_a, z_b, ids_a, ids_b, temperature)

    # Build positive mask on CPU (string comparison), then move to device.
    pos_mask = _build_positive_mask(ids_a, ids_b).to(device=z_a.device)

    if not pos_mask.any():
        raise ValueError(
            "No positive stimulus matches found between ids_a and ids_b. "
            "Ensure both sequences share at least one common stimulus_id."
        )

    # NOTE: inputs are expected to be L2-normalised by the caller (e.g. the
    # encoder's F.normalize output). Re-normalising here would detach the
    # computational graph from the original leaf tensors and break .backward().

    # Similarity matrix: (N, M), scaled by temperature
    logits_ab = torch.mm(z_a, z_b.t()) / temperature  # (N, M)
    logits_ba = logits_ab.t()                           # (M, N)

    # --- A -> B direction: for each row in A that has >= 1 positive -------
    loss_ab = _row_contrastive_loss(logits_ab, pos_mask)

    # --- B -> A direction: for each row in B that has >= 1 positive -------
    loss_ba = _row_contrastive_loss(logits_ba, pos_mask.t())

    return (loss_ab + loss_ba) / 2.0


def _row_contrastive_loss(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute mean supervised contrastive loss over rows that have positives.

    For row i, the loss is:
        L_i = -1/|P_i| * sum_{j in P_i}[ logits[i,j] - logsumexp(logits[i,:]) ]

    Args:
        logits:   (N, M) similarity matrix.
        pos_mask: (N, M) boolean tensor; True where (i,j) is a positive pair.

    Returns:
        Scalar mean loss over rows that have at least one positive.
    """
    # Numerically stable log-partition via logsumexp over all columns.
    log_Z = torch.logsumexp(logits, dim=1, keepdim=True)  # (N, 1)
    log_softmax = logits - log_Z                            # (N, M)

    # Sum log-probs over positives for each row.
    n_pos_per_row = pos_mask.float().sum(dim=1)             # (N,)
    has_positive = n_pos_per_row > 0                        # (N,) bool

    # For rows without a positive, contribution is zero (excluded from mean).
    pos_log_prob_sum = (log_softmax * pos_mask.float()).sum(dim=1)  # (N,)

    # Per-row mean loss: -mean_{j in P_i} log p(j|i)
    # Safe division: rows without positives have n_pos_per_row == 0; their
    # pos_log_prob_sum is also 0, so the division is 0/1 after clamping.
    safe_n_pos = n_pos_per_row.clamp(min=1)
    per_row_loss = -pos_log_prob_sum / safe_n_pos           # (N,)

    # Average only over rows that have at least one positive.
    return per_row_loss[has_positive].mean()


# ---------------------------------------------------------------------------
# Three-modality helper
# ---------------------------------------------------------------------------

_REQUIRED_MODALITIES = frozenset({"eeg", "meg", "fmri"})


def trimodal_contrastive_loss(
    embeddings: Mapping[str, torch.Tensor],
    stimulus_ids: Mapping[str, Sequence[str]],
    temperature: float = 0.07,
) -> torch.Tensor:
    """Average pairwise contrastive loss across all three modality pairs.

    Computes EEG-MEG, EEG-fMRI, and MEG-fMRI losses and returns their mean.
    Row order within each modality's embedding tensor need not match.

    Args:
        embeddings:   Mapping with keys "eeg", "meg", "fmri"; values are
                      (N_mod, D) L2-normalised tensors.
        stimulus_ids: Mapping with the same keys; values are sequences of
                      stimulus ID strings, one per embedding row.
        temperature:  Softmax temperature (> 0). Default 0.07.

    Returns:
        Scalar mean of the three pairwise contrastive losses.

    Raises:
        ValueError: If any required modality key is missing, or if
            temperature is invalid.
    """
    missing = _REQUIRED_MODALITIES - set(embeddings.keys())
    if missing:
        raise ValueError(
            f"embeddings is missing required modality key(s): {sorted(missing)}. "
            f"Expected all of {sorted(_REQUIRED_MODALITIES)}."
        )
    missing_ids = _REQUIRED_MODALITIES - set(stimulus_ids.keys())
    if missing_ids:
        raise ValueError(
            f"stimulus_ids is missing required modality key(s): {sorted(missing_ids)}. "
            f"Expected all of {sorted(_REQUIRED_MODALITIES)}."
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0; got {temperature}.")

    pairs = [
        ("eeg", "meg"),
        ("eeg", "fmri"),
        ("meg", "fmri"),
    ]

    total_loss = None
    for mod_a, mod_b in pairs:
        pair_loss = multimodal_contrastive_loss(
            z_a=embeddings[mod_a],
            z_b=embeddings[mod_b],
            ids_a=stimulus_ids[mod_a],
            ids_b=stimulus_ids[mod_b],
            temperature=temperature,
        )
        total_loss = pair_loss if total_loss is None else total_loss + pair_loss

    return total_loss / 3.0
