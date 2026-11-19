"""Neuro-GRAPE Training Pipeline (Member 2 — Phase 4).

Full-batch training loop and evaluation routines for cross-modal stimulus alignment
across EEG, MEG, and fMRI encoders.

Public API
----------
set_seed                  : Sets deterministic random seeds across PyTorch and NumPy.
prepare_cross_modal_batch : Assembles a stimulus-aligned cross-modal batch from a dataset.
train_epoch               : Executes a single training step with optimizer update.
evaluate                  : Evaluates trimodal loss without gradients or weight updates.
train_model               : Coordinates full training loop with in-memory validation checkpointing.

Design Rules
------------
* Alignment Key: Cross-modal samples are assembled strictly by matching stimulus_id,
  never by relying on CSV row order.
* Split Isolation: Only the training split is used for gradient updates. The validation
  split is used strictly for evaluation. The test split is NEVER loaded or accessed.
* fMRI Geometry: Stacks fMRI beta maps and unsqueezes channel dimension to (B, 1, 96, 96, 60).
* Mode Safety: evaluate() records and restores original encoder .training states via try...finally.
* Gradient Hygiene: optimizer.zero_grad() is executed strictly before backward propagation.
* Finite Loss Guard: Checks that losses are finite before backward propagation or step.
* Checkpoints: Best validation weights are cloned to CPU and retained in-memory only.
  No checkpoint files are written to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import NeuroGrapeDataset
from src.models.encoders import EEGEncoder, FMRIEncoder, MEGEncoder
from src.models.losses import trimodal_contrastive_loss
from src.models.normalization import TrimodalNormalizer


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across PyTorch and NumPy.

    Args:
        seed: Integer seed value. Default 42.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def prepare_cross_modal_batch(
    dataset: NeuroGrapeDataset,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
    """Assemble a stimulus-aligned cross-modal batch from a NeuroGrapeDataset instance.

    Iterates through unique stimulus IDs and retrieves matched cross-modal records,
    ensuring alignment is keyed on stimulus identity rather than CSV row order.
    Dynamically supports available modalities (e.g., 'eeg', 'meg', and optional 'fmri').
    """
    if device is None:
        device = torch.device("cpu")

    meta_df = dataset.get_metadata_table()
    if len(meta_df) == 0:
        raise ValueError("Dataset contains no valid samples.")

    stim_ids = sorted(list(meta_df["stimulus_id"].unique()))
    if len(stim_ids) == 0:
        raise ValueError("Dataset contains no unique stimulus IDs.")

    active_modalities = sorted(list(meta_df["modality"].str.lower().unique()))

    modal_tensors: Dict[str, List[torch.Tensor]] = {m: [] for m in active_modalities}

    for sid in stim_ids:
        sample_dict = dataset.get_cross_modal_sample(sid)
        for mod in active_modalities:
            if mod not in sample_dict:
                raise KeyError(f"Modality '{mod}' missing for stimulus '{sid}'.")
            if sample_dict[mod]["stimulus_id"] != sid:
                raise ValueError(
                    f"Stimulus ID mismatch in {mod}: {sample_dict[mod]['stimulus_id']} != {sid}"
                )
            feat = sample_dict[mod]["features"]
            if not isinstance(feat, torch.Tensor):
                feat = torch.as_tensor(feat, dtype=torch.float32)
            else:
                feat = feat.to(dtype=torch.float32)
            modal_tensors[mod].append(feat)

    batch: Dict[str, torch.Tensor] = {}
    stimulus_ids_map: Dict[str, List[str]] = {}

    for mod in active_modalities:
        stacked = torch.stack(modal_tensors[mod], dim=0).to(device)
        if mod == "fmri" and stacked.ndim == 4:
            # (B, X, Y, Z) -> (B, 1, X, Y, Z)
            stacked = stacked.unsqueeze(1)
        batch[mod] = stacked
        stimulus_ids_map[mod] = list(stim_ids)

    return batch, stimulus_ids_map


def train_epoch(
    encoders: Mapping[str, nn.Module],
    batch: Mapping[str, torch.Tensor],
    stimulus_ids: Mapping[str, Sequence[str]],
    optimizer: torch.optim.Optimizer,
    temperature: float = 0.07,
    sample_weights: Optional[torch.Tensor] = None,
) -> float:
    """Execute a single full-batch training step with optimizer update."""
    for enc in encoders.values():
        enc.train()

    optimizer.zero_grad()

    embeddings = {
        mod: encoders[mod](batch[mod])
        for mod in encoders
        if mod in batch
    }

    loss = trimodal_contrastive_loss(
        embeddings,
        stimulus_ids,
        temperature=temperature,
        sample_weights=sample_weights,
    )

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite loss ({loss.item()}) encountered during train_epoch."
        )

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(
    encoders: Mapping[str, nn.Module],
    batch: Mapping[str, torch.Tensor],
    stimulus_ids: Mapping[str, Sequence[str]],
    temperature: float = 0.07,
) -> float:
    """Evaluate trimodal contrastive loss on validation batch without gradient updates."""
    was_training = {k: enc.training for k, enc in encoders.items()}
    try:
        for enc in encoders.values():
            enc.eval()

        with torch.no_grad():
            embeddings = {
                mod: encoders[mod](batch[mod])
                for mod in encoders
                if mod in batch
            }

            val_loss = trimodal_contrastive_loss(
                embeddings, stimulus_ids, temperature=temperature
            )

            if not torch.isfinite(val_loss):
                raise FloatingPointError(
                    f"Non-finite loss ({val_loss.item()}) encountered during evaluate."
                )

            return val_loss.item()
    finally:
        for k, enc in encoders.items():
            enc.train(was_training[k])


def train_model(
    manifest_path: Union[str, Path] = "data_inventory.csv",
    num_epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    temperature: float = 0.07,
    seed: int = 42,
    device: Optional[torch.device] = None,
    restore_best_weights: bool = True,
    normalize_inputs: bool = True,
    dropout: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Execute full cross-modal contrastive training loop with in-memory validation checkpointing."""
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")

    set_seed(seed)

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_ds = NeuroGrapeDataset(manifest=manifest_path, split="train", as_torch=True)
    val_ds = NeuroGrapeDataset(manifest=manifest_path, split="val", as_torch=True)

    if len(train_ds) == 0:
        raise ValueError("Training split yielded no valid samples.")
    if len(val_ds) == 0:
        raise ValueError("Validation split yielded no valid samples.")

    train_batch, train_ids = prepare_cross_modal_batch(train_ds, device=device)
    val_batch, val_ids = prepare_cross_modal_batch(val_ds, device=device)

    normalizer: Optional[TrimodalNormalizer] = None
    if normalize_inputs:
        normalizer = TrimodalNormalizer().to(device)
        train_batch = normalizer.fit_transform(train_batch)
        val_batch = normalizer.transform(val_batch)

    encoders: Dict[str, nn.Module] = {}
    if "eeg" in train_batch:
        encoders["eeg"] = EEGEncoder(dropout=dropout).to(device)
    if "meg" in train_batch:
        encoders["meg"] = MEGEncoder(dropout=dropout).to(device)
    if "fmri" in train_batch:
        encoders["fmri"] = FMRIEncoder(dropout=dropout).to(device)

    all_params = []
    for enc in encoders.values():
        all_params.extend(list(enc.parameters()))

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_states: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(
            encoders=encoders,
            batch=train_batch,
            stimulus_ids=train_ids,
            optimizer=optimizer,
            temperature=temperature,
            sample_weights=sample_weights,
        )
        val_loss = evaluate(
            encoders=encoders,
            batch=val_batch,
            stimulus_ids=val_ids,
            temperature=temperature,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Retain independent CPU clones in-memory only (no disk checkpoints)
            best_states = {
                mod: {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
                for mod, enc in encoders.items()
            }

    if restore_best_weights and best_states is not None:
        for mod, enc in encoders.items():
            enc.load_state_dict({k: v.to(device) for k, v in best_states[mod].items()})

    return {
        "encoders": encoders,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_state_dicts": best_states,
        "normalizer": normalizer,
    }
