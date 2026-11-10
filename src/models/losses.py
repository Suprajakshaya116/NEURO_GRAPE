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

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
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
    present_mods = [m for m in ("eeg", "meg", "fmri") if m in embeddings and m in stimulus_ids]
    if len(present_mods) < 2:
        # If fewer than 2 modalities are present, cross-modal alignment cannot be computed
        ref_mod = present_mods[0] if present_mods else next(iter(stimulus_ids.keys()))
        ref_ids = list(stimulus_ids[ref_mod])
        device = embeddings[ref_mod].device if present_mods else torch.device("cpu")
        return torch.zeros(len(ref_ids), device=device), ref_ids

    ref_ids = list(stimulus_ids[present_mods[0]])
    pairs = [
        (present_mods[i], present_mods[j])
        for i in range(len(present_mods))
        for j in range(i + 1, len(present_mods))
    ]

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

    # Average over available pairs: shape (N,)
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


# ---------------------------------------------------------------------------
# Shared / Private Separation & Preservation Objectives (Phase 2 & 3)
# ---------------------------------------------------------------------------

def orthogonal_separation_loss(
    z_shared: torch.Tensor,
    z_private: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute orthogonality separation penalty between shared and private subspaces.

    Loss = (z_shared^T z_private)^2. For unit-norm vectors, this is the squared
    cosine similarity, penalizing linear correlation between representations.

    Args:
        z_shared: (B, D) L2-normalized shared embeddings.
        z_private: (B, D) L2-normalized private embeddings.
        reduction: 'mean' for scalar loss, 'none' for (B,) per-sample vector.

    Returns:
        Scalar penalty or (B,) vector.
    """
    if z_shared.shape != z_private.shape:
        raise ValueError(
            f"Shape mismatch: z_shared {z_shared.shape} vs z_private {z_private.shape}."
        )

    # Dot product per sample: (B,)
    cos_sim = (z_shared * z_private).sum(dim=1)
    sq_penalty = cos_sim**2

    if reduction == "none":
        return sq_penalty
    return sq_penalty.mean()


def multimodal_separation_loss(
    shared_embeddings: Mapping[str, torch.Tensor],
    private_embeddings: Mapping[str, torch.Tensor],
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute average orthogonality separation loss across all common modalities.

    Args:
        shared_embeddings: Dict mapping modality name to (B, D) shared embeddings.
        private_embeddings: Dict mapping modality name to (B, D) private embeddings.
        reduction: 'mean' for scalar loss, 'none' for (B,) average per-sample loss.

    Returns:
        Scalar loss or (B,) vector.
    """
    common_mods = set(shared_embeddings.keys()) & set(private_embeddings.keys())
    if not common_mods:
        raise ValueError("No common modalities between shared and private dictionaries.")

    losses = [
        orthogonal_separation_loss(
            shared_embeddings[m], private_embeddings[m], reduction=reduction
        )
        for m in sorted(common_mods)
    ]

    stacked = torch.stack(losses, dim=0)  # (M, ...)
    return stacked.mean(dim=0)


def compute_sensor_summary_targets(
    batch: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Extract standard summary statistics from raw inputs to serve as reconstruction targets.

    - EEG (B, 63, 225) -> Mean squared sensor energy per channel: (B, 63)
    - MEG (B, 272, 225) -> Mean squared sensor energy per channel: (B, 272)
    - fMRI (B, 1, 96, 96, 60) -> Mean spatial slice intensity along Z: (B, 60)

    Args:
        batch: Dict mapping modality names to input tensors.

    Returns:
        Dict mapping modality names to (B, C_target) normalized targets.
    """
    targets: Dict[str, torch.Tensor] = {}

    if "eeg" in batch:
        x_eeg = batch["eeg"]  # (B, 63, 225)
        energy = torch.mean(x_eeg**2, dim=-1)  # (B, 63)
        targets["eeg"] = F.normalize(energy, p=2, dim=-1)

    if "meg" in batch:
        x_meg = batch["meg"]  # (B, 272, 225)
        energy = torch.mean(x_meg**2, dim=-1)  # (B, 272)
        targets["meg"] = F.normalize(energy, p=2, dim=-1)

    if "fmri" in batch:
        x_fmri = batch["fmri"]  # (B, 1, 96, 96, 60)
        # Average across spatial X and Y to produce 60-dimensional slice profile
        slice_profile = torch.mean(x_fmri.squeeze(1), dim=(1, 2))  # (B, 60)
        targets["fmri"] = F.normalize(slice_profile, p=2, dim=-1)

    return targets


def multimodal_reconstruction_loss(
    private_embeddings: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    decoders: nn.ModuleDict,
) -> torch.Tensor:
    """Compute MSE reconstruction loss between private representations and sensor targets.

    Ensures the private pathway retains useful modality-specific information rather
    than simply collapsing to a trivial vector orthogonal to shared representations.

    Args:
        private_embeddings: Dict mapping modality names to (B, D) private embeddings.
        targets: Dict mapping modality names to (B, C_target) target vectors.
        decoders: nn.ModuleDict mapping modality names to linear decoders (D -> C_target).

    Returns:
        Scalar MSE reconstruction loss averaged across modalities.
    """
    losses = []
    for mod in sorted(private_embeddings.keys()):
        if mod in targets and mod in decoders:
            pred = decoders[mod](private_embeddings[mod])  # (B, C_target)
            loss_m = F.mse_loss(pred, targets[mod])
            losses.append(loss_m)

    if not losses:
        return torch.tensor(0.0, device=next(iter(private_embeddings.values())).device)

    return torch.stack(losses).mean()


def neuro_grape_total_loss(
    alignment_loss: torch.Tensor,
    separation_loss: torch.Tensor,
    reconstruction_loss: torch.Tensor,
    dro_loss: Optional[torch.Tensor] = None,
    lambda_align: float = 1.0,
    lambda_private: float = 0.5,
    lambda_recon: float = 0.2,
    lambda_robust: float = 1.0,
    interpolate_dro: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute unified total training objective for Neuro-GRAPE:

        L_total = lambda_recon * L_recon
                + lambda_private * L_sep
                + L_align_effective

    When dro_loss is provided:
        To prevent double-counting of the cross-modal alignment objective (since
        L_DRO = sum_g q_g L_g is the group-weighted alignment loss and L_align is the
        empirical average alignment loss), L_align_effective interpolates:
            L_align_effective = (1 - lambda_robust) * lambda_align * L_align
                              + lambda_robust * L_DRO
        (when interpolate_dro=True). If interpolate_dro=False, adds both directly.

    Args:
        alignment_loss: Symmetric InfoNCE alignment loss.
        separation_loss: Orthogonality separation penalty.
        reconstruction_loss: Private information reconstruction loss.
        dro_loss: Optional GroupDRO robust loss (if None, omitted).
        lambda_align: Weight for alignment loss. Default 1.0.
        lambda_private: Weight for separation loss. Default 0.5.
        lambda_recon: Weight for reconstruction loss. Default 0.2.
        lambda_robust: Weight for GroupDRO loss. Default 1.0.
        interpolate_dro: Whether to interpolate between ERM and DRO to prevent double-counting.

    Returns:
        total_loss: Differentiable scalar PyTorch loss.
        metrics: Dictionary of unweighted and weighted scalar float values for all components.
    """
    weighted_sep = lambda_private * separation_loss
    weighted_recon = lambda_recon * reconstruction_loss

    if dro_loss is not None and lambda_robust > 0.0:
        if interpolate_dro:
            weighted_align = (1.0 - lambda_robust) * lambda_align * alignment_loss
            weighted_dro = lambda_robust * dro_loss
        else:
            weighted_align = lambda_align * alignment_loss
            weighted_dro = lambda_robust * dro_loss
        total_loss = weighted_align + weighted_sep + weighted_recon + weighted_dro
        dro_unweighted = float(dro_loss.item())
        dro_weighted = float(weighted_dro.item())
    else:
        weighted_align = lambda_align * alignment_loss
        total_loss = weighted_align + weighted_sep + weighted_recon
        dro_unweighted = 0.0
        dro_weighted = 0.0

    metrics = {
        # Unweighted terms
        "unweighted_loss_align": float(alignment_loss.item()),
        "unweighted_loss_separation": float(separation_loss.item()),
        "unweighted_loss_reconstruction": float(reconstruction_loss.item()),
        "unweighted_loss_dro": dro_unweighted,
        # Weighted terms
        "weighted_loss_align": float(weighted_align.item()),
        "weighted_loss_separation": float(weighted_sep.item()),
        "weighted_loss_reconstruction": float(weighted_recon.item()),
        "weighted_loss_dro": dro_weighted,
        # Backward compatibility aliases
        "loss_align": float(alignment_loss.item()),
        "loss_separation": float(separation_loss.item()),
        "loss_reconstruction": float(reconstruction_loss.item()),
        "loss_dro": dro_unweighted,
        "loss_total": float(total_loss.item()),
    }

    return total_loss, metrics

