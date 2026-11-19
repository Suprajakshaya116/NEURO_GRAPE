"""
scripts/train_scaled_model.py
=============================
Train and benchmark Neuro-GRAPE and baseline models on the scaled dataset (2,804 paired trials).

Data Source:
- final/final/EEG_paired.npy  (2804, 63, 257)
- final/final/MEG_paired.npy  (2804, 310, 257)
- final/final/paired_metadata.csv (2804 rows)

Optimized Memory & Speed Architecture:
- Data normalized ONCE at startup; memory freed immediately.
- Batch-wise feature encoding (batch_size=128) for clustering, val, and test evaluation.
- Fully vectorized mini-batch contrastive training with CosineAnnealingLR.

Experimental Conditions:
1. Single_EEG: EEG-only encoder baseline
2. Single_MEG: MEG-only encoder baseline
3. Bimodal_ERM: Standard cross-modal InfoNCE baseline (no graph, no GroupDRO, no adaptive mixture)
4. GroupDRO_Cluster: GroupDRO with learned-regime KMeans clustering
5. Full_Neuro_GRAPE: Dual heads + Orthogonal Separation + Sensor Reconstruction +
                     Cross-Modal Graph Training + Adaptive Mixture + Learned-Regime GroupDRO
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
from src.dro.clustering import EmbeddingClustering
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
    orthogonal_separation_loss,
    per_sample_trimodal_contrastive_loss,
)
from src.models.model import TrimodalNeuroGrapeModel
from src.models.normalization import TrimodalNormalizer
from src.models.training import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ScaledTraining")


# ---------------------------------------------------------------------------
# Helper: Batch-wise Feature Encoding (Memory-Safe)
# ---------------------------------------------------------------------------

def encode_in_batches(
    model: nn.Module,
    data_dict: Mapping[str, torch.Tensor],
    batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """Encode tensors in small chunks to prevent CPU memory spike."""
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
# Data Preparation & Leakage-Proof Splitting
# ---------------------------------------------------------------------------

def prepare_scaled_splits(
    data_dir: Path,
    splits_csv_path: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Create concept-disjoint leakage-proof train/val/test splits."""
    metadata_path = data_dir / "paired_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df_meta)} records from {metadata_path}")

    # Extract concept name (word prefix before '_')
    df_meta["concept_name"] = df_meta["stimulus_id"].apply(
        lambda x: x.split("_")[0] if "_" in str(x) else str(x)
    )

    # WordNet Animacy labeling for semantic groups
    try:
        from nltk.corpus import wordnet as wn
        def get_animacy(word: str) -> int:
            synsets = wn.synsets(word.replace("-", "_"), pos=wn.NOUN)
            for s in synsets[:3]:
                for path in s.hypernym_paths():
                    names = [h.name().split(".")[0] for h in path]
                    if "animal" in names or "person" in names or "fauna" in names or "organism" in names:
                        return 1
            return 0
        df_meta["animacy"] = df_meta["concept_name"].apply(get_animacy)
    except Exception as e:
        logger.warning(f"WordNet not available, fallback to pilot animacy set: {e}")
        df_meta["animacy"] = df_meta["concept_name"].apply(
            lambda x: 1 if x in ANIMATE_CATEGORIES else 0
        )

    # Concept-disjoint splitting
    rng = np.random.RandomState(seed)
    unique_concepts = np.array(sorted(df_meta["concept_name"].unique()))
    rng.shuffle(unique_concepts)

    n_concepts = len(unique_concepts)
    n_train = int(n_concepts * train_ratio)
    n_val = int(n_concepts * val_ratio)

    train_concepts = set(unique_concepts[:n_train])
    val_concepts = set(unique_concepts[n_train : n_train + n_val])
    test_concepts = set(unique_concepts[n_train + n_val :])

    def assign_split(c: str) -> str:
        if c in train_concepts:
            return "train"
        elif c in val_concepts:
            return "val"
        else:
            return "test"

    df_meta["split"] = df_meta["concept_name"].apply(assign_split)
    splits_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_meta.to_csv(splits_csv_path, index=False)
    logger.info(f"Saved split manifest to {splits_csv_path}")

    split_counts = df_meta["split"].value_counts().to_dict()
    logger.info(f"Split trial counts: {split_counts}")

    split_indices = {
        "train": np.where(df_meta["split"] == "train")[0],
        "val": np.where(df_meta["split"] == "val")[0],
        "test": np.where(df_meta["split"] == "test")[0],
    }

    return df_meta, split_indices


# ---------------------------------------------------------------------------
# Training & Evaluation Engine
# ---------------------------------------------------------------------------

def train_and_eval_scaled_condition(
    config: Dict[str, Any],
    train_norm: Dict[str, torch.Tensor],
    val_norm: Dict[str, torch.Tensor],
    test_norm: Dict[str, torch.Tensor],
    train_targets: Dict[str, torch.Tensor],
    df_meta: pd.DataFrame,
    split_indices: Dict[str, np.ndarray],
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """Train and evaluate a single condition on pre-normalized scaled splits."""
    name = config["name"]
    seed = config.get("seed", 42)
    set_seed(seed)

    epochs = config.get("epochs", 25)
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 1e-3)
    temperature = config.get("temperature", 0.07)
    weight_decay = config.get("weight_decay", 1e-4)
    dropout = config.get("dropout", 0.2)

    use_dual_heads = config.get("use_dual_heads", True)
    use_graph = config.get("use_graph", True)
    use_adaptive_mixture = config.get("use_adaptive_mixture", False)
    use_dro = config.get("use_dro", False)
    group_type = config.get("group_type", "cluster")  # 'semantic', 'cluster', 'subject'
    is_unimodal = config.get("is_unimodal", False)
    unimodal_mod = config.get("unimodal_modality", "eeg")

    lambda_align = config.get("lambda_align", 1.0)
    lambda_private = config.get("lambda_private", 0.5) if use_dual_heads else 0.0
    lambda_recon = config.get("lambda_recon", 0.2) if use_dual_heads else 0.0

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    n_train = len(train_idx)
    n_val = len(val_idx)
    n_test = len(test_idx)

    train_meta = df_meta.iloc[train_idx].reset_index(drop=True)
    val_meta = df_meta.iloc[val_idx].reset_index(drop=True)
    test_meta = df_meta.iloc[test_idx].reset_index(drop=True)

    # Initialize model with exact scaled dimensions
    n_eeg_ch, n_eeg_t = train_norm["eeg"].shape[1], train_norm["eeg"].shape[2]
    n_meg_ch, n_meg_t = train_norm["meg"].shape[1], train_norm["meg"].shape[2]

    model = TrimodalNeuroGrapeModel(
        embed_dim=128,
        dropout=dropout,
        eeg_channels=n_eeg_ch,
        eeg_times=n_eeg_t,
        meg_channels=n_meg_ch,
        meg_times=n_meg_t,
        use_graph=use_graph,
    )

    # Decoders for sensor summary reconstruction if dual heads are used
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

    # Adaptive Mixture Sampler
    sampler = None
    if use_adaptive_mixture:
        sampler = AdaptiveMixtureSampler(
            modalities=["eeg", "meg"],
            step_size=0.05,
            min_weight=0.15,
            seed=seed,
        )

    # Group Partitioning
    dro = None
    group_indices_train = None
    group_indices_test = None
    num_groups = 2
    group_names_map = {0: "Inanimate", 1: "Animate"}

    if use_dro or group_type == "semantic":
        if group_type == "semantic":
            group_indices_train = torch.tensor(train_meta["animacy"].values, dtype=torch.long)
            group_indices_test = torch.tensor(test_meta["animacy"].values, dtype=torch.long)
            num_groups = 2
            group_names_map = {0: "Inanimate", 1: "Animate"}
        elif group_type == "cluster":
            # Fit KMeans strictly on initial training embeddings via batch encoding
            init_embs = encode_in_batches(model, train_norm, batch_size=128)
            train_concat = torch.cat([init_embs["eeg"], init_embs["meg"]], dim=1).numpy()
            clustering = EmbeddingClustering(n_clusters=2, random_state=seed)
            cluster_meta = pd.DataFrame({"sample_id": train_meta["stimulus_id"], "split": ["train"] * n_train})
            clustering.fit({sid: train_concat[i] for i, sid in enumerate(train_meta["stimulus_id"])}, cluster_meta)

            train_assigned = clustering.assign_clusters(
                {sid: train_concat[i] for i, sid in enumerate(train_meta["stimulus_id"])}, cluster_meta
            )
            group_indices_train = torch.tensor(train_assigned["cluster_id"].values, dtype=torch.long)

            test_embs = encode_in_batches(model, test_norm, batch_size=128)
            test_concat = torch.cat([test_embs["eeg"], test_embs["meg"]], dim=1).numpy()
            test_cluster_meta = pd.DataFrame({"sample_id": test_meta["stimulus_id"], "split": ["test"] * n_test})
            test_assigned = clustering.assign_clusters(
                {sid: test_concat[i] for i, sid in enumerate(test_meta["stimulus_id"])}, test_cluster_meta
            )
            group_indices_test = torch.tensor(test_assigned["cluster_id"].values, dtype=torch.long)
            num_groups = 2
            group_names_map = {0: "Regime_0", 1: "Regime_1"}

        if use_dro:
            dro = GroupDRO(num_groups=num_groups, learning_rate=0.01)

    # Mini-batch Training Loop
    logger.info(f"[{name}] Starting training ({epochs} epochs, batch_size={batch_size}, N_train={n_train})...")
    best_val_loss = float("inf")
    best_checkpoint_path = artifacts_dir / f"{name}_best_model.pt"
    history: List[Dict[str, Any]] = []

    start_time = time.time()
    n_batches = math.ceil(n_train / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        if decoders is not None:
            decoders.train()

        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        epoch_align_loss = 0.0
        epoch_sep_loss = 0.0
        epoch_recon_loss = 0.0

        for b in range(n_batches):
            b_idx = perm[b * batch_size : (b + 1) * batch_size]
            b_size = len(b_idx)
            batch_eeg = train_norm["eeg"][b_idx]
            batch_meg = train_norm["meg"][b_idx]
            batch_stims = train_meta["stimulus_id"].iloc[b_idx.numpy()].tolist()

            batch_in = {"eeg": batch_eeg, "meg": batch_meg}
            optimizer.zero_grad()

            if is_unimodal:
                # Unimodal baseline training
                u_mod = unimodal_mod
                u_in = {u_mod: batch_in[u_mod]}
                z_u = model(u_in, return_private=False, apply_graph=False)[u_mod]
                perm_u = torch.randperm(b_size)
                loss = -F.cosine_similarity(z_u, z_u[perm_u]).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * b_size
                continue

            # Forward pass with graph (if enabled)
            if use_dual_heads:
                shared, private = model(batch_in, return_private=True, apply_graph=use_graph)
            else:
                shared = model(batch_in, return_private=False, apply_graph=use_graph)
                private = {}

            # Alignment loss
            stims_dict = {"eeg": batch_stims, "meg": batch_stims}
            per_sample_loss, _ = per_sample_trimodal_contrastive_loss(
                embeddings=shared,
                stimulus_ids=stims_dict,
                temperature=temperature,
            )

            # Adaptive mixture weighting
            if sampler is not None:
                with torch.no_grad():
                    l_eeg = multimodal_contrastive_loss(
                        shared["eeg"], shared["meg"], batch_stims, batch_stims,
                        temperature=temperature, reduction="mean"
                    )
                    sampler.update_from_losses({"eeg": float(l_eeg.item()), "meg": float(l_eeg.item())}, is_train=True)
                weights = sampler.current_weights
                per_sample_loss = per_sample_loss * (weights["eeg"] + weights["meg"])

            # GroupDRO objective
            if dro is not None and group_indices_train is not None:
                b_groups = group_indices_train[b_idx]
                align_loss = dro.compute_weighted_loss(
                    per_example_losses=per_sample_loss,
                    group_indices=b_groups,
                    update_weights=True,
                )
            else:
                align_loss = per_sample_loss.mean()

            # Orthogonal separation & Reconstruction loss
            sep_loss = torch.tensor(0.0)
            recon_loss = torch.tensor(0.0)

            if use_dual_heads and len(private) >= 2:
                sep_loss = multimodal_separation_loss(shared, private, reduction="mean")
                b_targets = {
                    "eeg": train_targets["eeg"][b_idx],
                    "meg": train_targets["meg"][b_idx],
                }
                recon_loss = multimodal_reconstruction_loss(private, b_targets, decoders)

            total_loss = (
                lambda_align * align_loss
                + lambda_private * sep_loss
                + lambda_recon * recon_loss
            )

            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item()) * b_size
            epoch_align_loss += float(align_loss.item()) * b_size
            epoch_sep_loss += float(sep_loss.item()) * b_size
            epoch_recon_loss += float(recon_loss.item()) * b_size

        scheduler.step()

        epoch_loss /= n_train
        epoch_align_loss /= n_train
        epoch_sep_loss /= n_train
        epoch_recon_loss /= n_train

        # Validation evaluation in safe batches (zero leakage: apply_graph=False)
        model.eval()
        with torch.no_grad():
            if is_unimodal:
                val_loss = epoch_loss
            else:
                val_shared = encode_in_batches(model, val_norm, batch_size=128)
                val_stims = val_meta["stimulus_id"].tolist()
                val_loss_vec, _ = per_sample_trimodal_contrastive_loss(
                    embeddings=val_shared,
                    stimulus_ids={"eeg": val_stims, "meg": val_stims},
                    temperature=temperature,
                )
                val_loss = float(val_loss_vec.mean().item())

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_align_loss": epoch_align_loss,
            "train_sep_loss": epoch_sep_loss,
            "train_recon_loss": epoch_recon_loss,
            "val_loss": val_loss,
            "lr": float(scheduler.get_last_lr()[0]),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "decoders_state_dict": decoders.state_dict() if decoders is not None else None,
                    "val_loss": val_loss,
                    "config": config,
                },
                best_checkpoint_path,
            )

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                f"[{name}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

    elapsed_time = time.time() - start_time
    logger.info(f"[{name}] Training finished in {elapsed_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # Load best checkpoint for final zero-leakage test evaluation
    best_ckpt = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Zero-leakage retrieval evaluation on test set (N=426)
    test_q = encode_in_batches(model, test_norm, batch_size=128)
    test_g = encode_in_batches(model, test_norm, batch_size=128)

    test_stims = test_meta["stimulus_id"].tolist()
    test_concepts = test_meta["concept_name"].tolist()

    eval_results = evaluate_clean_crossmodal_retrieval(
        query_embeddings=test_q,
        gallery_embeddings=test_g,
        stimulus_ids=test_stims,
        category_names=test_concepts,
        custom_group_labels=group_indices_test.numpy() if group_indices_test is not None else None,
        custom_group_names=group_names_map,
        temperature=temperature,
        seed=seed,
        model_name=name,
    )

    eval_results["elapsed_time"] = elapsed_time
    eval_results["best_val_loss"] = best_val_loss
    eval_results["history"] = history
    eval_results["checkpoint_path"] = str(best_checkpoint_path)

    logger.info(
        f"[{name}] TEST RESULTS: Top-1: {eval_results['overall_top1']:.4f} | "
        f"Top-2: {eval_results['overall_top2']:.4f} | "
        f"Top-5: {eval_results['overall_top5']:.4f} | "
        f"Top-10: {eval_results['overall_top10']:.4f} | "
        f"MRR: {eval_results['mrr']:.4f} | "
        f"Worst-Group: {eval_results['worst_group_accuracy']:.4f} | "
        f"Disparity Gap: {eval_results['disparity_gap']:.4f} | "
        f"ECE: {eval_results['ece']:.4f}"
    )

    return eval_results


# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------

def generate_scaled_plots(
    results_dict: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate high-resolution publication figures for scaled experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({"font.size": 11, "font.sans-serif": "Arial"})

    # 1. Performance comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    names = list(results_dict.keys())
    top1s = [results_dict[m]["overall_top1"] * 100 for m in names]
    top5s = [results_dict[m]["overall_top5"] * 100 for m in names]
    worst_groups = [results_dict[m]["worst_group_accuracy"] * 100 for m in names]

    x = np.arange(len(names))
    width = 0.25

    rects1 = ax.bar(x - width, top1s, width, label="Top-1 Retrieval Accuracy (%)", color="#1f77b4")
    rects2 = ax.bar(x, top5s, width, label="Top-5 Retrieval Accuracy (%)", color="#2ca02c")
    rects3 = ax.bar(x + width, worst_groups, width, label="Worst-Group Accuracy (%)", color="#d62728")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Scaled Benchmark Performance Comparison (2,804 Trials, N_test=426)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    fig.savefig(output_dir / "scaled_benchmark_performance.png")
    plt.close(fig)

    # 2. Loss & calibration curves for Neuro-GRAPE
    if "Full_Neuro_GRAPE" in results_dict:
        ng = results_dict["Full_Neuro_GRAPE"]
        hist = ng["history"]
        epochs = [h["epoch"] for h in hist]
        train_losses = [h["train_loss"] for h in hist]
        val_losses = [h["val_loss"] for h in hist]

        fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax1.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", lw=2)
        ax1.plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Full Neuro-GRAPE Training Convergence on Scaled Dataset (Best Val: {ng['best_val_loss']:.4f})")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(output_dir / "scaled_loss_and_retrieval_curves.png")
        plt.close(fig)

    logger.info(f"Generated all scaled benchmark plots in {output_dir}")


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main():
    data_dir = REPO_ROOT / "final" / "final"
    artifacts_dir = REPO_ROOT / "reports" / "artifacts" / "scaled"
    plots_dir = artifacts_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Neuro-GRAPE Scaled Training Pipeline (2,804 Trials) ===")

    # 1. Load scaled dataset arrays
    eeg_path = data_dir / "EEG_paired.npy"
    meg_path = data_dir / "MEG_paired.npy"
    logger.info(f"Loading {eeg_path} and {meg_path}...")
    eeg_data = np.load(eeg_path)
    meg_data = np.load(meg_path)
    logger.info(f"Loaded EEG shape: {eeg_data.shape}, MEG shape: {meg_data.shape}")

    # 2. Leakage-Proof Splits
    splits_csv = REPO_ROOT / "data" / "splits" / "scaled_splits.csv"
    df_meta, split_indices = prepare_scaled_splits(data_dir, splits_csv, seed=42)

    # 3. Fit Normalizer strictly on training data
    train_idx = split_indices["train"]
    normalizer = TrimodalNormalizer()
    normalizer.fit({
        "eeg": torch.from_numpy(eeg_data[train_idx]).float(),
        "meg": torch.from_numpy(meg_data[train_idx]).float(),
    })
    logger.info("Fitted TrimodalNormalizer strictly on training split.")

    # 4. Transform all splits ONCE and extract sensor targets
    logger.info("Transforming train, val, and test splits with frozen training statistics...")
    train_norm = normalizer.transform({
        "eeg": torch.from_numpy(eeg_data[split_indices["train"]]).float(),
        "meg": torch.from_numpy(meg_data[split_indices["train"]]).float(),
    })
    val_norm = normalizer.transform({
        "eeg": torch.from_numpy(eeg_data[split_indices["val"]]).float(),
        "meg": torch.from_numpy(meg_data[split_indices["val"]]).float(),
    })
    test_norm = normalizer.transform({
        "eeg": torch.from_numpy(eeg_data[split_indices["test"]]).float(),
        "meg": torch.from_numpy(meg_data[split_indices["test"]]).float(),
    })
    train_targets = compute_sensor_summary_targets(train_norm)

    # Free raw numpy arrays to release ~1.0 GB RAM
    del eeg_data, meg_data
    gc.collect()
    logger.info("Freed raw arrays from RAM. Pre-normalized splits ready.")

    # 5. Experimental Configurations
    configs = [
        {
            "name": "Single_EEG",
            "is_unimodal": True,
            "unimodal_modality": "eeg",
            "use_dual_heads": False,
            "use_graph": False,
            "use_adaptive_mixture": False,
            "use_dro": False,
            "epochs": 15,
            "batch_size": 64,
            "lr": 1e-3,
        },
        {
            "name": "Single_MEG",
            "is_unimodal": True,
            "unimodal_modality": "meg",
            "use_dual_heads": False,
            "use_graph": False,
            "use_adaptive_mixture": False,
            "use_dro": False,
            "epochs": 15,
            "batch_size": 64,
            "lr": 1e-3,
        },
        {
            "name": "Bimodal_ERM",
            "is_unimodal": False,
            "use_dual_heads": False,
            "use_graph": False,
            "use_adaptive_mixture": False,
            "use_dro": False,
            "epochs": 20,
            "batch_size": 64,
            "lr": 1e-3,
        },
        {
            "name": "GroupDRO_Cluster",
            "is_unimodal": False,
            "use_dual_heads": False,
            "use_graph": False,
            "use_adaptive_mixture": False,
            "use_dro": True,
            "group_type": "cluster",
            "epochs": 20,
            "batch_size": 64,
            "lr": 1e-3,
        },
        {
            "name": "Full_Neuro_GRAPE",
            "is_unimodal": False,
            "use_dual_heads": True,
            "use_graph": True,
            "use_adaptive_mixture": True,
            "use_dro": True,
            "group_type": "cluster",
            "epochs": 25,
            "batch_size": 64,
            "lr": 1e-3,
        },
    ]

    all_results: Dict[str, Dict[str, Any]] = {}
    all_predictions: List[Dict[str, Any]] = []

    for cfg in configs:
        c_name = cfg["name"]
        logger.info(f"\n========================================\nRunning {c_name}\n========================================")
        res = train_and_eval_scaled_condition(
            config=cfg,
            train_norm=train_norm,
            val_norm=val_norm,
            test_norm=test_norm,
            train_targets=train_targets,
            df_meta=df_meta,
            split_indices=split_indices,
            artifacts_dir=artifacts_dir,
        )
        all_results[c_name] = res
        all_predictions.extend(res["prediction_records"])

    # 6. Save Artifacts
    # Predictions CSV
    pred_df = pd.DataFrame(all_predictions)
    pred_csv_path = artifacts_dir / "scaled_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    logger.info(f"Saved {len(pred_df)} prediction records to {pred_csv_path}")

    # Benchmark comparison CSV
    summary_rows = []
    for c_name, res in all_results.items():
        summary_rows.append({
            "model_name": c_name,
            "overall_top1": res["overall_top1"],
            "overall_top2": res["overall_top2"],
            "overall_top5": res["overall_top5"],
            "overall_top10": res["overall_top10"],
            "mrr": res["mrr"],
            "worst_group_accuracy": res["worst_group_accuracy"],
            "mean_group_accuracy": res["mean_group_accuracy"],
            "disparity_gap": res["disparity_gap"],
            "ece": res["ece"],
            "balanced_accuracy": res["balanced_accuracy"],
            "macro_f1": res["macro_f1"],
            "best_val_loss": res["best_val_loss"],
            "elapsed_time_s": res["elapsed_time"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = artifacts_dir / "scaled_benchmark_comparison.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved benchmark summary table to {summary_csv_path}")

    # JSON summary
    clean_summary = {
        c_name: {k: v for k, v in res.items() if k not in ("prediction_records", "confusion_matrix", "history")}
        for c_name, res in all_results.items()
    }
    summary_json_path = artifacts_dir / "scaled_benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)
    logger.info(f"Saved benchmark JSON summary to {summary_json_path}")

    # 7. Copy best Neuro-GRAPE checkpoint to checkpoints/
    best_ng_ckpt = artifacts_dir / "Full_Neuro_GRAPE_best_model.pt"
    final_dest_ckpt = REPO_ROOT / "checkpoints" / "best_scaled_neurogrape_model.pt"
    if best_ng_ckpt.exists():
        shutil.copy2(best_ng_ckpt, final_dest_ckpt)
        logger.info(f"Copied best model to {final_dest_ckpt}")

    # 8. Generate plots
    generate_scaled_plots(all_results, plots_dir)

    # 9. Print formatted markdown table
    print("\n" + "="*80)
    print("SCALED DATASET BENCHMARK RESULTS (2,804 Trials, N_test=426)")
    print("="*80)
    print(summary_df.to_markdown(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
