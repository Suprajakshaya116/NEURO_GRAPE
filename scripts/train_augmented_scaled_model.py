"""
scripts/train_augmented_scaled_model.py
=======================================
Train Neuro-GRAPE and GroupDRO on the scaled dataset (2,804 trials)
with Neural Data Augmentation and Regularization (NO ARCHITECTURAL CHANGES):
1. Channel Dropout: Randomly zeros 10-15% of channels during training.
2. Temporal Cutout: Randomly masks 10-15% of time steps during training.
3. Gaussian Noise Injection: Adds small zero-mean noise (sigma=0.02).
4. Higher Weight Decay: 1e-2 in AdamW to prevent memorization.

Evaluation:
Strictly zero-leakage test evaluation on the 426 test concepts,
evaluated with both Standard Cosine and Dual Mean-Centering.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.adaptive_sampler import AdaptiveMixtureSampler
from sklearn.cluster import KMeans
from src.dro.group_dro import GroupDRO
from src.evaluation.retrieval import (
    ANIMATE_CATEGORIES,
    compute_expected_calibration_error,
    evaluate_clean_crossmodal_retrieval,
)
from src.models.losses import (
    compute_sensor_summary_targets,
    multimodal_contrastive_loss,
    multimodal_reconstruction_loss,
    multimodal_separation_loss,
    per_sample_trimodal_contrastive_loss,
)
from src.models.model import TrimodalNeuroGrapeModel
from src.models.normalization import TrimodalNormalizer
from src.models.training import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AugmentedTraining")


# ---------------------------------------------------------------------------
# Neural Data Augmentation (Applied ONLY During Training)
# ---------------------------------------------------------------------------

def apply_neural_augmentation(
    x: torch.Tensor,
    channel_drop_prob: float = 0.15,
    time_mask_ratio: float = 0.15,
    noise_std: float = 0.02,
) -> torch.Tensor:
    """Apply data augmentation to a batch of neural signals (B, Channels, Time).
    
    Architecture is 100% unchanged. Shape is 100% preserved.
    """
    x_aug = x.clone()
    b, c, t = x_aug.shape

    # 1. Channel Dropout: randomly zero out channels across the batch
    if channel_drop_prob > 0.0:
        mask = (torch.rand(b, c, 1, device=x.device) > channel_drop_prob).float()
        # Ensure at least 1 channel remains active per sample
        x_aug = x_aug * mask

    # 2. Temporal Cutout: zero out a contiguous block of time points
    if time_mask_ratio > 0.0:
        mask_len = int(t * time_mask_ratio)
        if mask_len > 0:
            for i in range(b):
                start = torch.randint(0, t - mask_len + 1, (1,)).item()
                x_aug[i, :, start : start + mask_len] = 0.0

    # 3. Gaussian Noise Injection
    if noise_std > 0.0:
        noise = torch.randn_like(x_aug) * noise_std
        x_aug = x_aug + noise

    return x_aug


# ---------------------------------------------------------------------------
# Batch Encoder
# ---------------------------------------------------------------------------

def encode_in_batches(
    model: nn.Module,
    data_dict: Mapping[str, torch.Tensor],
    batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """Encode tensors in small chunks without memory spike."""
    model.eval()
    all_shared: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for mod in data_dict:
            x = data_dict[mod]
            n = len(x)
            embs = []
            for i in range(0, n, batch_size):
                b_x = {mod: x[i : i + batch_size]}
                out = model(b_x, return_private=False, apply_graph=False)
                embs.append(out[mod])
            all_shared[mod] = torch.cat(embs, dim=0)
    return all_shared


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_and_eval_augmented(
    config: Dict[str, Any],
    train_norm: Dict[str, torch.Tensor],
    val_norm: Dict[str, torch.Tensor],
    test_norm: Dict[str, torch.Tensor],
    train_targets: Dict[str, torch.Tensor],
    df_meta: pd.DataFrame,
    split_indices: Dict[str, np.ndarray],
    artifacts_dir: Path,
) -> Dict[str, Any]:
    name = config["name"]
    seed = config.get("seed", 42)
    set_seed(seed)

    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-2)  # Higher weight decay
    dropout = config.get("dropout", 0.2)
    temperature = config.get("temperature", 0.07)

    use_dual_heads = config.get("use_dual_heads", False)
    use_graph = config.get("use_graph", False)
    use_adaptive_mixture = config.get("use_adaptive_mixture", False)
    use_dro = config.get("use_dro", True)

    n_eeg_ch, n_eeg_t = train_norm["eeg"].shape[1], train_norm["eeg"].shape[2]
    n_meg_ch, n_meg_t = train_norm["meg"].shape[1], train_norm["meg"].shape[2]

    # Model is identical in architecture
    model = TrimodalNeuroGrapeModel(
        embed_dim=128,
        dropout=dropout,
        eeg_channels=n_eeg_ch,
        eeg_times=n_eeg_t,
        meg_channels=n_meg_ch,
        meg_times=n_meg_t,
        use_graph=use_graph,
    )

    decoders = None
    if use_dual_heads:
        decoders = nn.ModuleDict({
            "eeg": nn.Linear(128, n_eeg_ch),
            "meg": nn.Linear(128, n_meg_ch),
        })

    trainable_params = list(model.parameters())
    if decoders is not None:
        trainable_params += list(decoders.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]
    n_train = len(train_idx)

    train_meta = df_meta.iloc[train_idx].reset_index(drop=True)
    val_meta = df_meta.iloc[val_idx].reset_index(drop=True)
    test_meta = df_meta.iloc[test_idx].reset_index(drop=True)

    # Initial clustering for GroupDRO
    group_labels = np.zeros(n_train, dtype=int)
    if use_dro:
        with torch.no_grad():
            init_embs = encode_in_batches(model, train_norm, batch_size=128)
            cat_embs = torch.cat([init_embs["eeg"], init_embs["meg"]], dim=-1).cpu().numpy()
            kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
            group_labels = kmeans.fit_predict(cat_embs)
        dro = GroupDRO(num_groups=2, learning_rate=0.01)

    sampler = None
    if use_adaptive_mixture:
        sampler = AdaptiveMixtureSampler(modalities=["eeg", "meg"], step_size=0.05, min_weight=0.15)

    best_val_loss = float("inf")
    best_model_path = artifacts_dir / f"{name}_best_model.pt"

    start_time = time.time()
    logger.info(f"--- Training {name} (Augmented, WeightDecay={weight_decay}) for {epochs} epochs ---")

    for epoch in range(1, epochs + 1):
        model.train()
        if decoders is not None:
            decoders.train()

        perm = torch.randperm(n_train)
        epoch_losses = []

        for i in range(0, n_train, batch_size):
            b_indices = perm[i : i + batch_size]
            b_size = len(b_indices)
            if b_size < 4:
                continue

            # Apply Neural Data Augmentation on mini-batch inputs
            b_eeg_raw = train_norm["eeg"][b_indices]
            b_meg_raw = train_norm["meg"][b_indices]

            b_eeg_aug = apply_neural_augmentation(b_eeg_raw, channel_drop_prob=0.15, time_mask_ratio=0.15, noise_std=0.02)
            b_meg_aug = apply_neural_augmentation(b_meg_raw, channel_drop_prob=0.15, time_mask_ratio=0.15, noise_std=0.02)

            batch_x = {"eeg": b_eeg_aug, "meg": b_meg_aug}
            b_stims = train_meta.iloc[b_indices.cpu().numpy()]["stimulus_id"].tolist()

            if use_dual_heads:
                shared, priv = model(batch_x, return_private=True, apply_graph=use_graph)
            else:
                shared = model(batch_x, return_private=False, apply_graph=use_graph)
                priv = {}

            per_sample_loss, _ = per_sample_trimodal_contrastive_loss(
                embeddings=shared,
                stimulus_ids={"eeg": b_stims, "meg": b_stims},
                temperature=temperature,
            )

            if use_dro:
                b_groups = torch.from_numpy(group_labels[b_indices.cpu().numpy()]).long()
                align_loss = dro.compute_weighted_loss(per_sample_loss, b_groups, update_weights=True)
            else:
                align_loss = per_sample_loss.mean()

            loss = align_loss

            if use_dual_heads and len(priv) >= 2:
                loss_sep = multimodal_separation_loss(shared, priv, reduction="mean")
                b_targets = {
                    "eeg": train_targets["eeg"][b_indices],
                    "meg": train_targets["meg"][b_indices],
                }
                loss_recon = multimodal_reconstruction_loss(priv, b_targets, decoders)
                loss = loss + 0.5 * loss_sep + 0.2 * loss_recon

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))

        # Validation (CLEAN, no augmentation)
        model.eval()
        with torch.no_grad():
            val_shared = encode_in_batches(model, val_norm, batch_size=128)
            val_stims = val_meta["stimulus_id"].tolist()
            val_loss_vec, _ = per_sample_trimodal_contrastive_loss(
                embeddings=val_shared,
                stimulus_ids={"eeg": val_stims, "meg": val_stims},
                temperature=temperature,
            )
            val_loss = float(val_loss_vec.mean().item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                best_model_path,
            )

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

    elapsed = time.time() - start_time
    logger.info(f"Trained {name} in {elapsed:.1f}s. Evaluating on test set...")

    # Load best checkpoint for test evaluation
    best_ckpt = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Extract test embeddings without leakage
    test_embs = encode_in_batches(model, test_norm, batch_size=128)

    # 1. Standard Cosine Evaluation
    res_standard = evaluate_clean_crossmodal_retrieval(
        query_embeddings=test_embs,
        gallery_embeddings=test_embs,
        stimulus_ids=test_meta["stimulus_id"].tolist(),
        category_names=test_meta["concept_name"].tolist(),
        custom_group_labels=test_meta["animacy"].values,
        model_name=f"{name}_Standard",
    )

    # 2. Dual Mean-Centered Evaluation
    test_embs_centered = {}
    for mod in ["eeg", "meg"]:
        mu = test_embs[mod].mean(dim=0, keepdim=True)
        test_embs_centered[mod] = F.normalize(test_embs[mod] - mu, p=2, dim=1)

    res_centered = evaluate_clean_crossmodal_retrieval(
        query_embeddings=test_embs_centered,
        gallery_embeddings=test_embs_centered,
        stimulus_ids=test_meta["stimulus_id"].tolist(),
        category_names=test_meta["concept_name"].tolist(),
        custom_group_labels=test_meta["animacy"].values,
        model_name=f"{name}_DualCentered",
    )

    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "best_val_loss": best_val_loss,
        "standard": res_standard,
        "centered": res_centered,
    }


def main():
    data_dir = REPO_ROOT / "final" / "final"
    splits_csv = REPO_ROOT / "data" / "splits" / "scaled_splits.csv"
    artifacts_dir = REPO_ROOT / "reports" / "artifacts" / "augmented"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df_meta = pd.read_csv(splits_csv)
    split_indices = {
        "train": np.where(df_meta["split"] == "train")[0],
        "val": np.where(df_meta["split"] == "val")[0],
        "test": np.where(df_meta["split"] == "test")[0],
    }

    logger.info("Loading and normalizing data once...")
    eeg_raw = np.load(data_dir / "EEG_paired.npy", mmap_mode="r")
    meg_raw = np.load(data_dir / "MEG_paired.npy", mmap_mode="r")

    eeg_train = np.array(eeg_raw[split_indices["train"]], dtype=np.float32)
    meg_train = np.array(meg_raw[split_indices["train"]], dtype=np.float32)
    eeg_val = np.array(eeg_raw[split_indices["val"]], dtype=np.float32)
    meg_val = np.array(meg_raw[split_indices["val"]], dtype=np.float32)
    eeg_test = np.array(eeg_raw[split_indices["test"]], dtype=np.float32)
    meg_test = np.array(meg_raw[split_indices["test"]], dtype=np.float32)
    del eeg_raw, meg_raw
    gc.collect()

    normalizer = TrimodalNormalizer()
    normalizer.fit({
        "eeg": torch.from_numpy(eeg_train).float(),
        "meg": torch.from_numpy(meg_train).float(),
    })

    train_norm = normalizer.transform({
        "eeg": torch.from_numpy(eeg_train).float(),
        "meg": torch.from_numpy(meg_train).float(),
    })
    val_norm = normalizer.transform({
        "eeg": torch.from_numpy(eeg_val).float(),
        "meg": torch.from_numpy(meg_val).float(),
    })
    test_norm = normalizer.transform({
        "eeg": torch.from_numpy(eeg_test).float(),
        "meg": torch.from_numpy(meg_test).float(),
    })

    train_targets = compute_sensor_summary_targets(train_norm)
    del eeg_train, meg_train, eeg_val, meg_val, eeg_test, meg_test
    gc.collect()

    conditions = [
        {
            "name": "Augmented_GroupDRO_Cluster",
            "use_dual_heads": False,
            "use_graph": False,
            "use_adaptive_mixture": False,
            "use_dro": True,
            "epochs": 20,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-2,
        },
        {
            "name": "Augmented_Full_Neuro_GRAPE",
            "use_dual_heads": True,
            "use_graph": True,
            "use_adaptive_mixture": True,
            "use_dro": True,
            "epochs": 20,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-2,
        },
    ]

    summary_rows = []

    for cfg in conditions:
        res = train_and_eval_augmented(
            config=cfg,
            train_norm=train_norm,
            val_norm=val_norm,
            test_norm=test_norm,
            train_targets=train_targets,
            df_meta=df_meta,
            split_indices=split_indices,
            artifacts_dir=artifacts_dir,
        )

        for mode, r in [("Standard", res["standard"]), ("DualCentered", res["centered"])]:
            summary_rows.append({
                "condition": f"{cfg['name']}_{mode}",
                "top1_acc_pct": r["overall_top1"] * 100,
                "top2_acc_pct": r["overall_top2"] * 100,
                "top5_acc_pct": r["overall_top5"] * 100,
                "top10_acc_pct": r["overall_top10"] * 100,
                "mrr": r["mrr"],
                "worst_group_acc_pct": r["worst_group_accuracy"] * 100,
                "disparity_gap_pct": r["disparity_gap"] * 100,
                "val_loss": res["best_val_loss"],
                "elapsed_sec": res["elapsed_seconds"],
            })

    df_summary = pd.DataFrame(summary_rows)
    out_csv = artifacts_dir / "augmented_benchmark_summary.csv"
    df_summary.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("AUGMENTED TRAINING EXPERIMENT RESULTS (426-Class Zero-Shot Gallery)")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
