"""
Neuro-GRAPE K-Regime Ablation Experiment.
Evaluates GroupDRO across varying numbers of neural difficulty regimes: K in {2, 3, 4, 5}.
Trained and evaluated on the complete scaled dataset (2,804 trials, 426-class zero-shot gallery).
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dro.clustering import EmbeddingClustering
from src.dro.group_dro import GroupDRO
from src.evaluation.retrieval import evaluate_clean_crossmodal_retrieval
from src.models.losses import (
    compute_sensor_summary_targets,
    multimodal_reconstruction_loss,
    multimodal_separation_loss,
    per_sample_trimodal_contrastive_loss,
)
from src.models.model import TrimodalNeuroGrapeModel
from src.models.normalization import TrimodalNormalizer
from src.models.training import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("KRegimeAblation")


def encode_in_batches(
    model: nn.Module,
    data_dict: Mapping[str, torch.Tensor],
    batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """Encode tensors in batches to prevent memory spikes."""
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


def prepare_scaled_splits(
    data_dir: Path,
    splits_csv_path: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    metadata_path = data_dir / "paired_metadata.csv"
    df_meta = pd.read_csv(metadata_path)
    df_meta["concept_name"] = df_meta["stimulus_id"].apply(
        lambda x: x.split("_")[0] if "_" in str(x) else str(x)
    )

    rng = np.random.RandomState(seed)
    unique_concepts = np.array(sorted(df_meta["concept_name"].unique()))
    rng.shuffle(unique_concepts)

    n_concepts = len(unique_concepts)
    n_train = int(n_concepts * train_ratio)
    n_val = int(n_concepts * val_ratio)

    train_concepts = set(unique_concepts[:n_train])
    val_concepts = set(unique_concepts[n_train : n_train + n_val])

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

    split_indices = {
        "train": np.where(df_meta["split"] == "train")[0],
        "val": np.where(df_meta["split"] == "val")[0],
        "test": np.where(df_meta["split"] == "test")[0],
    }
    return df_meta, split_indices


def evaluate_dual_mean_centering(
    query_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
    train_q_embs: torch.Tensor,
    train_g_embs: torch.Tensor,
    stims: List[str],
    concepts: List[str],
    group_labels: Optional[np.ndarray],
    group_names: Optional[Dict[int, str]],
) -> Dict[str, Any]:
    """Evaluate retrieval with Inductive Dual Mean-Centering using training centroids."""
    mu_q = train_q_embs.mean(dim=0, keepdim=True)
    mu_g = train_g_embs.mean(dim=0, keepdim=True)

    q_centered = F.normalize(query_embs - mu_q, p=2, dim=-1)
    g_centered = F.normalize(gallery_embs - mu_g, p=2, dim=-1)

    sim_matrix = torch.mm(q_centered, g_centered.t()).numpy()
    n = len(q_centered)
    top1, top2, top5, top10 = 0, 0, 0, 0
    reciprocal_ranks = []
    per_sample_correct = []
    hub_counts = np.zeros(n, dtype=int)

    for i in range(n):
        scores = sim_matrix[i]
        ranked_indices = np.argsort(-scores)
        predicted_top1 = ranked_indices[0]
        hub_counts[predicted_top1] += 1

        is_top1 = (predicted_top1 == i)
        is_top2 = (i in ranked_indices[:2])
        is_top5 = (i in ranked_indices[:5])
        is_top10 = (i in ranked_indices[:10])

        rank = np.where(ranked_indices == i)[0][0] + 1
        reciprocal_ranks.append(1.0 / rank)

        if is_top1: top1 += 1
        if is_top2: top2 += 1
        if is_top5: top5 += 1
        if is_top10: top10 += 1
        per_sample_correct.append(int(is_top1))

    worst_group_acc = 0.0
    disparity_gap = 0.0
    if group_labels is not None:
        unique_groups = np.unique(group_labels)
        grp_accs = {}
        for g in unique_groups:
            g_mask = (group_labels == g)
            if g_mask.sum() > 0:
                grp_accs[g] = np.mean(np.array(per_sample_correct)[g_mask])
        if grp_accs:
            worst_group_acc = min(grp_accs.values())
            disparity_gap = max(grp_accs.values()) - worst_group_acc

    return {
        "top1": top1 / n,
        "top2": top2 / n,
        "top5": top5 / n,
        "top10": top10 / n,
        "mrr": float(np.mean(reciprocal_ranks)),
        "worst_group_acc": float(worst_group_acc),
        "disparity_gap": float(disparity_gap),
        "max_hub_count": int(np.max(hub_counts)),
    }


def train_k_condition(
    k: int,
    train_norm: Dict[str, torch.Tensor],
    val_norm: Dict[str, torch.Tensor],
    test_norm: Dict[str, torch.Tensor],
    train_targets: Dict[str, torch.Tensor],
    df_meta: pd.DataFrame,
    split_indices: Dict[str, np.ndarray],
    artifacts_dir: Path,
    seed: int = 42,
    epochs: int = 15,
) -> Dict[str, Any]:
    set_seed(seed)
    name = f"GroupDRO_K{k}"
    logger.info(f"\n==========================================")
    logger.info(f"   STARTING EXPERIMENT: {name} (K={k})   ")
    logger.info(f"==========================================")

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    n_train = len(train_idx)
    n_val = len(val_idx)
    n_test = len(test_idx)

    train_meta = df_meta.iloc[train_idx].reset_index(drop=True)
    val_meta = df_meta.iloc[val_idx].reset_index(drop=True)
    test_meta = df_meta.iloc[test_idx].reset_index(drop=True)

    n_eeg_ch, n_eeg_t = train_norm["eeg"].shape[1], train_norm["eeg"].shape[2]
    n_meg_ch, n_meg_t = train_norm["meg"].shape[1], train_norm["meg"].shape[2]

    # Initialize model
    model = TrimodalNeuroGrapeModel(
        embed_dim=128,
        dropout=0.2,
        eeg_channels=n_eeg_ch,
        eeg_times=n_eeg_t,
        meg_channels=n_meg_ch,
        meg_times=n_meg_t,
        use_graph=True,
    )

    decoders = nn.ModuleDict({
        "eeg": nn.Linear(128, n_eeg_ch),
        "meg": nn.Linear(128, n_meg_ch),
    })

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoders.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # 1. Fit KMeans strictly on initial training representations
    logger.info(f"Extracting initial training embeddings to discover K={k} latent regimes...")
    init_embs = encode_in_batches(model, train_norm, batch_size=128)
    train_concat = torch.cat([init_embs["eeg"], init_embs["meg"]], dim=1).numpy()

    clustering = EmbeddingClustering(n_clusters=k, random_state=seed)
    cluster_meta = pd.DataFrame({"sample_id": train_meta["stimulus_id"], "split": ["train"] * n_train})
    clustering.fit({sid: train_concat[i] for i, sid in enumerate(train_meta["stimulus_id"])}, cluster_meta)

    # Assign training clusters
    train_assigned = clustering.assign_clusters(
        {sid: train_concat[i] for i, sid in enumerate(train_meta["stimulus_id"])}, cluster_meta
    )
    group_indices_train = torch.tensor(train_assigned["cluster_id"].values, dtype=torch.long)
    train_counts = np.bincount(group_indices_train.numpy(), minlength=k)
    sil = silhouette_score(train_concat, group_indices_train.numpy())
    logger.info(f"[K={k}] Training cluster sizes: {train_counts.tolist()} | Silhouette Score: {sil:.4f}")

    # Assign test clusters strictly via frozen training centroids
    test_embs_init = encode_in_batches(model, test_norm, batch_size=128)
    test_concat = torch.cat([test_embs_init["eeg"], test_embs_init["meg"]], dim=1).numpy()
    test_cluster_meta = pd.DataFrame({"sample_id": test_meta["stimulus_id"], "split": ["test"] * n_test})
    test_assigned = clustering.assign_clusters(
        {sid: test_concat[i] for i, sid in enumerate(test_meta["stimulus_id"])}, test_cluster_meta
    )
    group_indices_test = torch.tensor(test_assigned["cluster_id"].values, dtype=torch.long)
    test_counts = np.bincount(group_indices_test.numpy(), minlength=k)
    logger.info(f"[K={k}] Test cluster sizes: {test_counts.tolist()}")

    group_names_map = {i: f"Regime_{i}" for i in range(k)}
    dro = GroupDRO(num_groups=k, learning_rate=0.01)

    # Training loop
    t0 = time.time()
    batch_size = 64
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        decoders.train()
        perm = torch.randperm(n_train)
        epoch_losses = []

        for b_start in range(0, n_train, batch_size):
            b_idx = perm[b_start : b_start + batch_size]
            b_data = {
                "eeg": train_norm["eeg"][b_idx],
                "meg": train_norm["meg"][b_idx],
            }
            b_stims = train_meta["stimulus_id"].iloc[b_idx.numpy()].tolist()
            b_groups = group_indices_train[b_idx]

            # Model forward: returns (shared, private)
            shared, private = model(b_data, return_private=True, apply_graph=True)

            # Alignment loss
            stims_dict = {"eeg": b_stims, "meg": b_stims}
            per_sample_loss, _ = per_sample_trimodal_contrastive_loss(
                embeddings=shared,
                stimulus_ids=stims_dict,
                temperature=0.07,
            )

            # GroupDRO weighted alignment loss
            align_loss = dro.compute_weighted_loss(
                per_example_losses=per_sample_loss,
                group_indices=b_groups,
                update_weights=True,
            )

            # Separation & Reconstruction
            sep_loss = multimodal_separation_loss(shared, private, reduction="mean")
            b_targets = {
                "eeg": train_targets["eeg"][b_idx],
                "meg": train_targets["meg"][b_idx],
            }
            recon_loss = multimodal_reconstruction_loss(private, b_targets, decoders)

            total_loss = align_loss + 0.5 * sep_loss + 0.2 * recon_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(decoders.parameters()), 1.0)
            optimizer.step()
            epoch_losses.append(total_loss.item())

        # Validation (apply_graph=False)
        model.eval()
        with torch.no_grad():
            val_shared = encode_in_batches(model, val_norm, batch_size=128)
            val_stims = val_meta["stimulus_id"].tolist()
            val_loss_vec, _ = per_sample_trimodal_contrastive_loss(
                embeddings=val_shared,
                stimulus_ids={"eeg": val_stims, "meg": val_stims},
                temperature=0.07,
            )
            v_loss = float(val_loss_vec.mean().item())

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k_param: v_param.cpu().clone() for k_param, v_param in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            q_weights = [round(float(w), 3) for w in dro.group_weights.tolist()]
            logger.info(f"[{name}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {np.mean(epoch_losses):.4f} | Val Loss: {v_loss:.4f} | Group weights: {q_weights}")

    elapsed = time.time() - t0
    logger.info(f"[{name}] Finished training in {elapsed:.1f}s. Evaluating zero-shot test retrieval...")

    # Load best model for evaluation
    model.load_state_dict(best_state)
    model.eval()

    # Zero-leakage encoding
    test_encoded = encode_in_batches(model, test_norm, batch_size=128)
    train_encoded = encode_in_batches(model, train_norm, batch_size=128)

    test_q = test_encoded["eeg"]
    test_g = test_encoded["meg"]
    train_q = train_encoded["eeg"]
    train_g = train_encoded["meg"]

    test_stims = test_meta["stimulus_id"].tolist()
    test_concepts = test_meta["concept_name"].tolist()

    # 1. Standard Raw Cosine Evaluation
    raw_eval = evaluate_clean_crossmodal_retrieval(
        query_embeddings=test_encoded,
        gallery_embeddings=test_encoded,
        stimulus_ids=test_stims,
        category_names=test_concepts,
        custom_group_labels=group_indices_test.numpy(),
        custom_group_names=group_names_map,
        temperature=0.07,
        seed=seed,
        model_name=name,
    )

    # 2. Inductive Dual Mean-Centering Evaluation
    centered_eval = evaluate_dual_mean_centering(
        query_embs=test_q,
        gallery_embs=test_g,
        train_q_embs=train_q,
        train_g_embs=train_g,
        stims=test_stims,
        concepts=test_concepts,
        group_labels=group_indices_test.numpy(),
        group_names=group_names_map,
    )

    # Save checkpoint
    ckpt_path = artifacts_dir / f"checkpoint_{name}.pt"
    torch.save({
        "k": k,
        "name": name,
        "model_state_dict": best_state,
        "kmeans": clustering.model,
        "raw_results": raw_eval,
        "centered_results": centered_eval,
    }, ckpt_path)

    return {
        "k": k,
        "name": name,
        "elapsed_time": elapsed,
        "best_val_loss": best_val_loss,
        "silhouette": float(sil),
        "cluster_sizes_train": train_counts.tolist(),
        "cluster_sizes_test": test_counts.tolist(),
        "raw_top1": raw_eval["overall_top1"],
        "raw_top2": raw_eval["overall_top2"],
        "raw_top5": raw_eval["overall_top5"],
        "raw_top10": raw_eval["overall_top10"],
        "raw_mrr": raw_eval["mrr"],
        "raw_worst_group": raw_eval["worst_group_accuracy"],
        "raw_disparity_gap": raw_eval["disparity_gap"],
        "raw_ece": raw_eval["ece"],
        "centered_top1": centered_eval["top1"],
        "centered_top2": centered_eval["top2"],
        "centered_top5": centered_eval["top5"],
        "centered_top10": centered_eval["top10"],
        "centered_mrr": centered_eval["mrr"],
        "centered_worst_group": centered_eval["worst_group_acc"],
        "centered_disparity_gap": centered_eval["disparity_gap"],
        "max_hub_count": centered_eval["max_hub_count"],
    }


def main():
    data_dir = REPO_ROOT / "final" / "final"
    artifacts_dir = REPO_ROOT / "reports" / "artifacts" / "k_regime_study"
    plots_dir = artifacts_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=============================================================")
    logger.info("   NEURO-GRAPE: K-REGIME ABLATION BENCHMARK (K in {2,3,4,5}) ")
    logger.info("=============================================================")

    # 1. Load scaled dataset arrays
    eeg_data = np.load(data_dir / "EEG_paired.npy")
    meg_data = np.load(data_dir / "MEG_paired.npy")
    splits_csv = REPO_ROOT / "data" / "splits" / "scaled_splits.csv"
    df_meta, split_indices = prepare_scaled_splits(data_dir, splits_csv, seed=42)

    # 2. Normalize strictly with frozen training statistics
    normalizer = TrimodalNormalizer()
    normalizer.fit({
        "eeg": torch.from_numpy(eeg_data[split_indices["train"]]).float(),
        "meg": torch.from_numpy(meg_data[split_indices["train"]]).float(),
    })

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

    del eeg_data, meg_data
    gc.collect()

    # 3. Run all K conditions
    all_results = []
    k_values = [2, 3, 4, 5]

    for k in k_values:
        res = train_k_condition(
            k=k,
            train_norm=train_norm,
            val_norm=val_norm,
            test_norm=test_norm,
            train_targets=train_targets,
            df_meta=df_meta,
            split_indices=split_indices,
            artifacts_dir=artifacts_dir,
            seed=42,
            epochs=15,
        )
        all_results.append(res)

    # 4. Save structured results JSON and CSV
    results_df = pd.DataFrame(all_results)
    results_csv = artifacts_dir / "k_regime_benchmark_results.csv"
    results_json = artifacts_dir / "k_regime_benchmark_results.json"
    results_df.to_csv(results_csv, index=False)
    results_json.write_text(json.dumps(all_results, indent=2))
    logger.info(f"Saved benchmark summary to {results_csv} and {results_json}")

    # 5. Generate high-resolution comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    ks = [r["k"] for r in all_results]

    # Panel A: Top-1 Retrieval (Raw vs Centered)
    ax = axes[0, 0]
    w = 0.35
    x = np.arange(len(ks))
    raw_t1 = [r["raw_top1"] * 100 for r in all_results]
    cent_t1 = [r["centered_top1"] * 100 for r in all_results]
    ax.bar(x - w/2, raw_t1, width=w, label="Raw Cosine Top-1", color="#4a7bb0")
    ax.bar(x + w/2, cent_t1, width=w, label="Centered Top-1", color="#2a9d8f")
    ax.axhline(100.0/426.0, color="crimson", linestyle="--", label="Random Chance (0.235%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A. Top-1 Zero-Shot Retrieval vs. K")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel B: Top-5 Candidate Retrieval (Raw vs Centered)
    ax = axes[0, 1]
    raw_t5 = [r["raw_top5"] * 100 for r in all_results]
    cent_t5 = [r["centered_top5"] * 100 for r in all_results]
    ax.bar(x - w/2, raw_t5, width=w, label="Raw Cosine Top-5", color="#e76f51")
    ax.bar(x + w/2, cent_t5, width=w, label="Centered Top-5", color="#f4a261")
    ax.axhline(500.0/426.0, color="crimson", linestyle="--", label="Random Chance (1.174%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("B. Top-5 Zero-Shot Candidate Retrieval vs. K")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel C: Worst-Group Accuracy vs. K
    ax = axes[1, 0]
    raw_wg = [r["raw_worst_group"] * 100 for r in all_results]
    cent_wg = [r["centered_worst_group"] * 100 for r in all_results]
    ax.plot(ks, raw_wg, marker="o", lw=2, label="Raw Cosine Worst-Group", color="#1d3557")
    ax.plot(ks, cent_wg, marker="s", lw=2, label="Centered Worst-Group", color="#2a9d8f")
    ax.set_xticks(ks)
    ax.set_xlabel("Number of Latent Regimes (K)")
    ax.set_ylabel("Worst-Group Accuracy (%)")
    ax.set_title("C. Worst-Group Accuracy Across Regimes")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel D: Maximum Hub Count
    ax = axes[1, 1]
    max_hubs = [r["max_hub_count"] for r in all_results]
    ax.bar(x, max_hubs, width=0.5, color="#8338ec", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Maximum Frequency of a Single Concept")
    ax.set_title("D. Hubness Mitigation (Max Hub Frequency)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = plots_dir / "k_regime_ablation_comparison.png"
    fig.savefig(plot_path)
    plt.close(fig)
    logger.info(f"Generated comparison plot at {plot_path}")

    # Copy plot to paper figures
    paper_fig_path = REPO_ROOT / "paper" / "figures" / "k_regime_ablation_comparison.png"
    import shutil
    shutil.copy2(plot_path, paper_fig_path)
    logger.info(f"Copied figure to {paper_fig_path}")

    # Print summary table to console
    print("\n" + "="*95)
    print("                       K-REGIME ABLATION BENCHMARK SUMMARY")
    print("="*95)
    print(f"{'Condition':<14} | {'Silhouette':<10} | {'Raw Top-1':<10} | {'Raw Top-5':<10} | {'Cent Top-1':<10} | {'Cent Top-5':<10} | {'Cent Worst-G':<12} | {'Max Hub':<8}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['name']:<14} | {r['silhouette']:<10.4f} | {r['raw_top1']*100:<9.3f}% | {r['raw_top5']*100:<9.3f}% | {r['centered_top1']*100:<9.3f}% | {r['centered_top5']*100:<9.3f}% | {r['centered_worst_group']*100:<11.3f}% | {r['max_hub_count']:<8}")
    print("="*95 + "\n")

if __name__ == "__main__":
    main()
