"""Neuro-GRAPE Contrastive Loss Functions & Per-Example Vectors (Member 2 — Phase 5 Handoff).

Provides scalar, weighted, and per-example loss formulations for cross-modal
alignment across EEG, MEG, and fMRI. Exposes per-example loss vectors and
sample/group weighting interfaces specifically required for Member 3's
GroupDRO (Distributionally Robust Optimization) integration.

Public API
----------
multimodal_contrastive_loss        : Pairwise contrastive loss (scalar or per-example).
trimodal_contrastive_loss          : Trimodal contrastive loss (scalar or per-example).
per_sample_trimodal_contrastive_loss : Explicit per-example loss vector (N,) + stimulus IDs.
compute_group_weighted_loss        : Applies GroupDRO group weights to per-example losses.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from src.models.contrastive_loss import _build_positive_mask, _validate_inputs

_REQUIRED_MODALITIES = frozenset({"eeg", "meg", "fmri"})


def _row_contrastive_loss_vector(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-row supervised contrastive loss vector of shape (N,).

    For row i:
        L_i = -1/|P_i| * sum_{j in P_i}[ logits[i,j] - logsumexp(logits[i,:]) ]

    Args:
        logits: (N, M) similarity matrix scaled by temperature.
        pos_mask: (N, M) boolean mask of positive pairs.

    Returns:
        (N,) tensor of loss values per row.
    """
    log_Z = torch.logsumexp(logits, dim=1, keepdim=True)  # (N, 1)
    log_softmax = logits - log_Z                            # (N, M)

    n_pos_per_row = pos_mask.float().sum(dim=1)             # (N,)
    pos_log_prob_sum = (log_softmax * pos_mask.float()).sum(dim=1)  # (N,)

    safe_n_pos = n_pos_per_row.clamp(min=1)
    per_row_loss = -pos_log_prob_sum / safe_n_pos           # (N,)
    return per_row_loss


def multimodal_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    ids_a: Sequence[str],
    ids_b: Sequence[str],
    temperature: float = 0.07,
    reduction: str = "mean",
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pairwise symmetric cross-modal contrastive loss with reduction options.

    Args:
        z_a: (N, D) L2-normalized embeddings for modality A.
        z_b: (M, D) L2-normalized embeddings for modality B.
        ids_a: Sequence of stimulus IDs for z_a rows (length N).
        ids_b: Sequence of stimulus IDs for z_b rows (length M).
        temperature: Softmax scaling temperature (> 0). Default 0.07.
        reduction: 'mean' for scalar loss, 'none' or 'per_example' for (N,) vector.
        sample_weights: Optional (N,) tensor of sample/group weights.

    Returns:
        Scalar tensor if reduction == 'mean', or (N,) vector if reduction in ('none', 'per_example').
    """
    _validate_inputs(z_a, z_b, ids_a, ids_b, temperature)

    pos_mask = _build_positive_mask(ids_a, ids_b).to(device=z_a.device)
    if not pos_mask.any():
        raise ValueError("No positive stimulus matches found between ids_a and ids_b.")

    logits_ab = torch.mm(z_a, z_b.t()) / temperature  # (N, M)
    logits_ba = logits_ab.t()                           # (M, N)

    loss_ab_vec = _row_contrastive_loss_vector(logits_ab, pos_mask)     # (N,)
    loss_ba_vec = _row_contrastive_loss_vector(logits_ba, pos_mask.t()) # (M,)

    # When both modalities have the same aligned stimuli (standard batch),
    # map B->A losses back to A's row order via stimulus ID lookup
    id_to_b_idx = {sid: idx for idx, sid in enumerate(ids_b)}
    aligned_ba_losses = torch.stack([loss_ba_vec[id_to_b_idx[sid]] for sid in ids_a])
    symmetric_per_sample = (loss_ab_vec + aligned_ba_losses) / 2.0  # (N,)

    if reduction in ("none", "per_example"):
        return symmetric_per_sample

    if sample_weights is not None:
        weights = sample_weights.to(device=symmetric_per_sample.device, dtype=symmetric_per_sample.dtype)
        if weights.shape != symmetric_per_sample.shape:
            raise ValueError(
                f"sample_weights shape {weights.shape} does not match loss shape {symmetric_per_sample.shape}."
            )
        w_sum = weights.sum().clamp(min=1e-8)
        return (symmetric_per_sample * weights).sum() / w_sum

    return symmetric_per_sample.mean()


def per_sample_trimodal_contrastive_loss(
    embeddings: Mapping[str, torch.Tensor],
    stimulus_ids: Mapping[str, Sequence[str]],
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, List[str]]:
    """Compute exact per-example trimodal contrastive loss vector for Member 3.

    Averages the three pairwise losses (EEG-MEG, EEG-fMRI, MEG-fMRI) for each
    individual stimulus in the cohort.

    Args:
        embeddings: Mapping with keys 'eeg', 'meg', 'fmri' to (N, D) tensors.
        stimulus_ids: Mapping with keys 'eeg', 'meg', 'fmri' to stimulus ID sequences.
        temperature: Softmax temperature (> 0). Default 0.07.

    Returns:
        per_example_loss: 1-D tensor of shape (N,) containing loss for each stimulus.
        ordered_stimulus_ids: List of N stimulus ID strings corresponding to the loss vector.
    """
    missing = _REQUIRED_MODALITIES - set(embeddings.keys())
    if missing:
        raise ValueError(f"Missing required modality keys: {sorted(missing)}.")

    ref_ids = list(stimulus_ids["eeg"])
    pairs = [("eeg", "meg"), ("eeg", "fmri"), ("meg", "fmri")]

    loss_vectors = []
    for mod_a, mod_b in pairs:
        vec = multimodal_contrastive_loss(
            z_a=embeddings[mod_a],
            z_b=embeddings[mod_b],
            ids_a=stimulus_ids[mod_a],
            ids_b=stimulus_ids[mod_b],
            temperature=temperature,
            reduction="none",
        )
        loss_vectors.append(vec)

    # Average over the three pairs: shape (N,)
    trimodal_per_sample = torch.stack(loss_vectors, dim=0).mean(dim=0)
    return trimodal_per_sample, ref_ids


def trimodal_contrastive_loss(
    embeddings: Mapping[str, torch.Tensor],
    stimulus_ids: Mapping[str, Sequence[str]],
    temperature: float = 0.07,
    reduction: str = "mean",
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Trimodal contrastive loss supporting scalar, weighted, or per-example output.

    Args:
        embeddings: Dict with keys 'eeg', 'meg', 'fmri'.
        stimulus_ids: Dict with keys 'eeg', 'meg', 'fmri'.
        temperature: Softmax temperature scalar (> 0). Default 0.07.
        reduction: 'mean' for scalar loss, 'none' or 'per_example' for (N,) vector.
        sample_weights: Optional (N,) tensor of sample/group weights.

    Returns:
        Scalar loss tensor if reduction == 'mean', or (N,) vector if reduction in ('none', 'per_example').
    """
    per_sample_loss, _ = per_sample_trimodal_contrastive_loss(
        embeddings=embeddings,
        stimulus_ids=stimulus_ids,
        temperature=temperature,
    )

    if reduction in ("none", "per_example"):
        return per_sample_loss

    if sample_weights is not None:
        weights = sample_weights.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
        if weights.shape != per_sample_loss.shape:
            raise ValueError(
                f"sample_weights shape {weights.shape} does not match per_sample_loss shape {per_sample_loss.shape}."
            )
        w_sum = weights.sum().clamp(min=1e-8)
        return (per_sample_loss * weights).sum() / w_sum

    return per_sample_loss.mean()


def compute_group_weighted_loss(
    per_example_loss: torch.Tensor,
    group_indices: torch.Tensor,
    group_weights: torch.Tensor,
) -> torch.Tensor:
    """Helper for Member 3's GroupDRO: applies group weights to per-example losses.

    Args:
        per_example_loss: (N,) tensor of loss values per stimulus.
        group_indices: (N,) integer tensor assigning each stimulus to a group [0, K-1].
        group_weights: (K,) float tensor containing GroupDRO group weights (summing to 1).

    Returns:
        Scalar GroupDRO objective: sum_k q_k * E_{i in group k}[L_i].
    """
    device = per_example_loss.device
    group_indices = group_indices.to(device)
    group_weights = group_weights.to(device)

    num_groups = len(group_weights)
    group_losses = []
    for g in range(num_groups):
        mask = group_indices == g
        if mask.any():
            group_loss = per_example_loss[mask].mean()
        else:
            group_loss = torch.tensor(0.0, device=device)
        group_losses.append(group_loss)

    stacked_group_losses = torch.stack(group_losses)
    return (group_weights * stacked_group_losses).sum()
