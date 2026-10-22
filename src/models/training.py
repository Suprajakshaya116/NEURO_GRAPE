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

    Args:
        dataset: NeuroGrapeDataset instance (filtered to train or val split).
        device: Target torch.device for stacked tensors. Defaults to CPU if None.

    Returns:
        batch: Dict with keys 'eeg', 'meg', 'fmri' mapping to stacked feature tensors:
            - 'eeg':  (B, 63, 225) float32 tensor
            - 'meg':  (B, 272, 225) float32 tensor
            - 'fmri': (B, 1, 96, 96, 60) float32 tensor
        stimulus_ids_map: Dict with keys 'eeg', 'meg', 'fmri' mapping to the ordered
            list of stimulus IDs.

    Raises:
        ValueError: If dataset contains no samples or no unique stimulus IDs.
    """
    if device is None:
        device = torch.device("cpu")

    meta_df = dataset.get_metadata_table()
    if len(meta_df) == 0:
        raise ValueError("Dataset contains no valid samples.")

    stim_ids = sorted(list(meta_df["stimulus_id"].unique()))
    if len(stim_ids) == 0:
        raise ValueError("Dataset contains no unique stimulus IDs.")

    eeg_list: List[torch.Tensor] = []
    meg_list: List[torch.Tensor] = []
    fmri_list: List[torch.Tensor] = []

    for sid in stim_ids:
        sample_dict = dataset.get_cross_modal_sample(sid)
        for mod in ("eeg", "meg", "fmri"):
            if mod not in sample_dict:
                raise KeyError(f"Modality '{mod}' missing for stimulus '{sid}'.")
            if sample_dict[mod]["stimulus_id"] != sid:
                raise ValueError(
                    f"Stimulus ID mismatch in {mod}: {sample_dict[mod]['stimulus_id']} != {sid}"
                )

        eeg_feat = sample_dict["eeg"]["features"]
        meg_feat = sample_dict["meg"]["features"]
        fmri_feat = sample_dict["fmri"]["features"]

        if not isinstance(eeg_feat, torch.Tensor):
            eeg_feat = torch.as_tensor(eeg_feat, dtype=torch.float32)
        else:
            eeg_feat = eeg_feat.to(dtype=torch.float32)

        if not isinstance(meg_feat, torch.Tensor):
            meg_feat = torch.as_tensor(meg_feat, dtype=torch.float32)
        else:
            meg_feat = meg_feat.to(dtype=torch.float32)

        if not isinstance(fmri_feat, torch.Tensor):
            fmri_feat = torch.as_tensor(fmri_feat, dtype=torch.float32)
        else:
            fmri_feat = fmri_feat.to(dtype=torch.float32)

        eeg_list.append(eeg_feat)
        meg_list.append(meg_feat)
        fmri_list.append(fmri_feat)

    x_eeg = torch.stack(eeg_list, dim=0).to(device)
    x_meg = torch.stack(meg_list, dim=0).to(device)
    # Add channel dimension to fMRI: (B, 96, 96, 60) -> (B, 1, 96, 96, 60)
    x_fmri = torch.stack(fmri_list, dim=0).unsqueeze(1).to(device)

    batch = {
        "eeg": x_eeg,
        "meg": x_meg,
        "fmri": x_fmri,
    }
    stimulus_ids_map = {
        "eeg": list(stim_ids),
        "meg": list(stim_ids),
        "fmri": list(stim_ids),
    }
    return batch, stimulus_ids_map


def train_epoch(
    encoders: Mapping[str, nn.Module],
    batch: Mapping[str, torch.Tensor],
    stimulus_ids: Mapping[str, Sequence[str]],
    optimizer: torch.optim.Optimizer,
    temperature: float = 0.07,
    sample_weights: Optional[torch.Tensor] = None,
) -> float:
    """Execute a single full-batch training step with optimizer update.

    Ensures zero_grad() is executed before backward propagation, checks loss
    finiteness, backpropagates, and updates parameters. Supports sample/group
    weighting for Member 3's GroupDRO integration.

    Args:
        encoders: Mapping with keys 'eeg', 'meg', 'fmri' to encoder nn.Modules.
        batch: Mapping with keys 'eeg', 'meg', 'fmri' to input feature tensors.
        stimulus_ids: Mapping with keys 'eeg', 'meg', 'fmri' to stimulus ID sequences.
        optimizer: PyTorch optimizer containing parameters of all three encoders.
        temperature: Softmax scaling temperature for contrastive loss. Default 0.07.
        sample_weights: Optional (N,) tensor of weights per training sample/stimulus.

    Returns:
        Scalar training loss value (float).

    Raises:
        FloatingPointError: If non-finite (NaN or Inf) loss is encountered.
    """
    for enc in encoders.values():
        enc.train()

    optimizer.zero_grad()

    z_eeg = encoders["eeg"](batch["eeg"])
    z_meg = encoders["meg"](batch["meg"])
    z_fmri = encoders["fmri"](batch["fmri"])

    embeddings = {
        "eeg": z_eeg,
        "meg": z_meg,
        "fmri": z_fmri,
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
    """Evaluate trimodal contrastive loss on validation batch without gradient updates.

    Captures original .training modes and restores them in a finally block even if
    an exception occurs during evaluation.

    Args:
        encoders: Mapping with keys 'eeg', 'meg', 'fmri' to encoder nn.Modules.
        batch: Mapping with keys 'eeg', 'meg', 'fmri' to input feature tensors.
        stimulus_ids: Mapping with keys 'eeg', 'meg', 'fmri' to stimulus ID sequences.
        temperature: Softmax scaling temperature for contrastive loss. Default 0.07.

    Returns:
        Scalar validation loss value (float).

    Raises:
        FloatingPointError: If non-finite (NaN or Inf) loss is encountered.
    """
    was_training = {k: enc.training for k, enc in encoders.items()}
    try:
        for enc in encoders.values():
            enc.eval()

        with torch.no_grad():
            z_eeg = encoders["eeg"](batch["eeg"])
            z_meg = encoders["meg"](batch["meg"])
            z_fmri = encoders["fmri"](batch["fmri"])

            embeddings = {
                "eeg": z_eeg,
                "meg": z_meg,
                "fmri": z_fmri,
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
    """Execute full cross-modal contrastive training loop with in-memory validation checkpointing.

    Uses full-batch training on the 12 training stimuli and evaluates after each epoch
    on the 4 validation stimuli. The test split is NEVER loaded or evaluated.
    Supports sample/group weighting for Member 3's GroupDRO integration.

    Args:
        manifest_path: Path to data inventory CSV. Default 'data_inventory.csv'.
        num_epochs: Number of training epochs (>= 1). Default 50.
        lr: AdamW learning rate (> 0). Default 1e-4.
        weight_decay: AdamW weight decay (>= 0). Default 1e-2.
        temperature: Softmax temperature (> 0). Default 0.07.
        seed: Random seed for reproducibility. Default 42.
        device: Target torch device (MPS or CPU). Auto-detected if None.
        restore_best_weights: If True, restores models to state with lowest val_loss. Default True.
        normalize_inputs: If True, standardizes inputs using training statistics only. Default True.
        dropout: Dropout rate for encoder backbones (>= 0.0 and < 1.0). Default 0.0.
        sample_weights: Optional (N,) tensor of weights per training sample for GroupDRO.

    Returns:
        result: Dict containing:
            - 'encoders': Dict of trained encoder modules ('eeg', 'meg', 'fmri')
            - 'history': Dict with lists for 'epoch', 'train_loss', 'val_loss'
            - 'best_epoch': Epoch index with minimum validation loss (1-indexed)
            - 'best_val_loss': Minimum validation loss achieved
            - 'best_state_dicts': Dict of detached CPU state dicts at best_epoch
            - 'normalizer': Fitted TrimodalNormalizer instance (if normalize_inputs else None)

    Raises:
        ValueError: If hyperparameter arguments are invalid or if dataset splits are empty.
    """
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

    # Load only train and val splits. The test split is NEVER loaded or referenced.
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

    encoders: Dict[str, nn.Module] = {
        "eeg": EEGEncoder(dropout=dropout).to(device),
        "meg": MEGEncoder(dropout=dropout).to(device),
        "fmri": FMRIEncoder(dropout=dropout).to(device),
    }

    all_params = (
        list(encoders["eeg"].parameters())
        + list(encoders["meg"].parameters())
        + list(encoders["fmri"].parameters())
    )
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
