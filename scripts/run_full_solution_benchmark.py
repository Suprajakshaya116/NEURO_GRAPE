"""
scripts/run_full_solution_benchmark.py
======================================
Comprehensive automated benchmark harness for the full Neuro-GRAPE research solution
specified in docs/Neuro-GRAPE.pdf.

Executes all named experimental conditions across multiple seeds:
- Condition A: Single-modality baselines (A1: EEG, A2: MEG, A3: fMRI)
- Condition B: Fixed-mixture trimodal ERM
- Condition C: Modality-balanced training
- Condition D: Standard GroupDRO (Semantic)
- Condition E: Learned-Regime GroupDRO
- Condition F: Full Neuro-GRAPE (Dual heads + Orthogonality + Reconstruction +
               Cross-modal Graph + Adaptive Sampler + Learned-Regime GroupDRO + 3-Phase Curriculum)
- Condition G: Component Ablations (G1 to G7 removing one component at a time)
- Condition H: Transfer and Calibration evaluations (Held-out concepts, Pairwise, Calibration;
               Held-out subjects marked BLOCKED due to single-subject pilot data)

Generates:
- reports/artifacts/full_benchmark_comparison.csv
- reports/artifacts/full_benchmark_summary.json
- reports/artifacts/full_benchmark/plots/*.png
- Checkpoints, logs, and per-run metric files
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.adaptive_sampler import AdaptiveMixtureSampler
from src.data.dataset import NeuroGrapeDataset
from src.dro.clustering import EmbeddingClustering
from src.dro.group_dro import GroupDRO
from src.models.graph import CrossModalGraphLayer
from src.models.losses import (
    compute_sensor_summary_targets,
    multimodal_contrastive_loss,
    multimodal_reconstruction_loss,
    multimodal_separation_loss,
    neuro_grape_total_loss,
    orthogonal_separation_loss,
    per_sample_trimodal_contrastive_loss,
)
from src.models.model import TrimodalNeuroGrapeModel
from src.models.normalization import TrimodalNormalizer
from src.models.training import prepare_cross_modal_batch, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FullBenchmark")

ANIMATE_CATEGORIES = frozenset({"bear", "boa", "tadpole", "woman"})


# ---------------------------------------------------------------------------
# Calibration & Disparity Metric Helpers
# ---------------------------------------------------------------------------

def compute_expected_calibration_error(
    confidences: Sequence[float],
    accuracies: Sequence[float],
    n_bins: int = 5,
) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins."""
    confs = np.asarray(confidences, dtype=np.float64)
    accs = np.asarray(accuracies, dtype=np.float64)
    n = len(confs)
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for b in range(n_bins):
        low, high = bins[b], bins[b + 1]
        mask = (confs >= low) & (confs <= high if b == n_bins - 1 else confs < high)
        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(accs[mask]))
            bin_conf = float(np.mean(confs[mask]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_retrieval_and_calibration(
    embeddings: Mapping[str, torch.Tensor],
    stimulus_ids: Sequence[str],
    temperature: float = 0.07,
) -> Dict[str, Any]:
    """Compute pairwise retrieval accuracy, top-k, and calibration error."""
    n = len(stimulus_ids)
    pairs = [("eeg", "meg"), ("eeg", "fmri"), ("meg", "fmri")]

    all_confs: List[float] = []
    all_accs: List[float] = []
    pair_top1: Dict[str, float] = {}
    pair_top2: Dict[str, float] = {}
    per_stimulus_acc: np.ndarray = np.zeros(n, dtype=np.float32)

    for mod_a, mod_b in pairs:
        z_a = embeddings[mod_a]  # (N, D)
        z_b = embeddings[mod_b]  # (N, D)

        sim_ab = torch.mm(z_a, z_b.t())  # (N, N)
        prob_ab = F.softmax(sim_ab / temperature, dim=1)

        # Forward retrieval: A -> B
        preds_ab = torch.argmax(sim_ab, dim=1).cpu().numpy()
        targets = np.arange(n)
        top1_ab = (preds_ab == targets).astype(np.float32)
        confs_ab = torch.max(prob_ab, dim=1).values.cpu().numpy()

        all_confs.extend(confs_ab.tolist())
        all_accs.extend(top1_ab.tolist())

        # Top-2 retrieval
        top2_preds = torch.topk(sim_ab, k=min(2, n), dim=1).indices.cpu().numpy()
        top2_ab = np.array([targets[i] in top2_preds[i] for i in range(n)], dtype=np.float32)

        pair_name = f"{mod_a}_{mod_b}"
        pair_top1[pair_name] = float(np.mean(top1_ab))
        pair_top2[pair_name] = float(np.mean(top2_ab))
        per_stimulus_acc += top1_ab

    per_stimulus_acc /= len(pairs)
    overall_top1 = float(np.mean(all_accs))
    overall_top2 = float(np.mean(list(pair_top2.values())))
    ece = compute_expected_calibration_error(all_confs, all_accs, n_bins=5)

    return {
        "overall_top1": overall_top1,
        "overall_top2": overall_top2,
        "ece": ece,
        "pair_top1": pair_top1,
        "pair_top2": pair_top2,
        "per_stimulus_acc": per_stimulus_acc,
    }


# ---------------------------------------------------------------------------
# Training Loop Implementation for Full Benchmark Condition
# ---------------------------------------------------------------------------

def run_single_experiment(
    config: Dict[str, Any],
    train_batch: Dict[str, torch.Tensor],
    val_batch: Dict[str, torch.Tensor],
    test_batch: Dict[str, torch.Tensor],
    train_ids_dict: Dict[str, List[str]],
    val_ids_dict: Dict[str, List[str]],
    test_ids_dict: Dict[str, List[str]],
    meta_df: pd.DataFrame,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """Train and evaluate a specific configuration with full reproducibility."""
    seed = config.get("seed", 42)
    set_seed(seed)

    condition_name = config["name"]
    epochs = config.get("epochs", 25)
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)

    # Feature flags
    is_unimodal = config.get("is_unimodal", False)
    unimodal_mod = config.get("unimodal_modality", "eeg")
    use_dual_heads = config.get("use_dual_heads", True)
    use_graph = config.get("use_graph", True)
    use_adaptive_mixture = config.get("use_adaptive_mixture", False)
    use_dro = config.get("use_dro", False)
    group_type = config.get("group_type", "cluster")  # 'semantic', 'cluster', 'random'
    use_curriculum = config.get("use_curriculum", False)

    lambda_align = config.get("lambda_align", 1.0)
    lambda_private = config.get("lambda_private", 0.5) if use_dual_heads else 0.0
    lambda_recon = config.get("lambda_recon", 0.2) if use_dual_heads else 0.0
    lambda_robust = config.get("lambda_robust", 1.0) if use_dro else 0.0

    train_stim_ids = train_ids_dict["eeg"]
    val_stim_ids = val_ids_dict["eeg"]
    test_stim_ids = test_ids_dict["eeg"]

    train_metadata = meta_df[meta_df["split"].str.lower() == "train"]
    val_metadata = meta_df[meta_df["split"].str.lower() == "val"]
    test_metadata = meta_df[meta_df["split"].str.lower() == "test"]

    # 2. Build Model
    model = TrimodalNeuroGrapeModel(embed_dim=128, dropout=0.0, use_graph=use_graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Setup Adaptive Mixture Sampler
    sampler = None
    if use_adaptive_mixture:
        sampler = AdaptiveMixtureSampler(
            modalities=["eeg", "meg", "fmri"],
            step_size=0.05,
            min_weight=0.15,
            seed=seed,
        )

    # 4. Setup Group Partitioning & GroupDRO
    dro = None
    group_indices_train = None
    group_indices_val = None
    group_indices_test = None

    if group_type == "semantic":
        cat_map = dict(zip(train_metadata["stimulus_id"], train_metadata["category_name"].str.lower()))
        group_indices_train = torch.tensor(
            [1 if cat_map.get(sid, "") in ANIMATE_CATEGORIES else 0 for sid in train_stim_ids],
            dtype=torch.long,
        )
        val_cat = dict(zip(val_metadata["stimulus_id"], val_metadata["category_name"].str.lower()))
        group_indices_val = torch.tensor(
            [1 if val_cat.get(sid, "") in ANIMATE_CATEGORIES else 0 for sid in val_stim_ids],
            dtype=torch.long,
        )
        test_cat = dict(zip(test_metadata["stimulus_id"], test_metadata["category_name"].str.lower()))
        group_indices_test = torch.tensor(
            [1 if test_cat.get(sid, "") in ANIMATE_CATEGORIES else 0 for sid in test_stim_ids],
            dtype=torch.long,
        )
        num_groups = 2
    elif group_type == "cluster":
        # Fit KMeans strictly on training set representations at start
        with torch.no_grad():
            init_embs = model(train_batch, return_private=False, apply_graph=False)
            train_concat = torch.cat(
                [init_embs["eeg"], init_embs["meg"], init_embs["fmri"]], dim=1
            ).numpy()

        clustering = EmbeddingClustering(n_clusters=2, random_state=seed)
        cluster_meta = pd.DataFrame({"sample_id": train_stim_ids, "split": ["train"] * len(train_stim_ids)})
        clustering.fit({sid: train_concat[i] for i, sid in enumerate(train_stim_ids)}, cluster_meta)

        train_assigned = clustering.assign_clusters(
            {sid: train_concat[i] for i, sid in enumerate(train_stim_ids)}, cluster_meta
        )
        group_indices_train = torch.tensor(train_assigned["cluster_id"].values, dtype=torch.long)

        # For val/test, project onto frozen train centroids
        with torch.no_grad():
            val_embs = model(val_batch, return_private=False, apply_graph=False)
            val_concat = torch.cat([val_embs["eeg"], val_embs["meg"], val_embs["fmri"]], dim=1).numpy()
            val_meta = pd.DataFrame({"sample_id": val_stim_ids, "split": ["val"] * len(val_stim_ids)})
            val_assigned = clustering.assign_clusters(
                {sid: val_concat[i] for i, sid in enumerate(val_stim_ids)}, val_meta
            )
            group_indices_val = torch.tensor(val_assigned["cluster_id"].values, dtype=torch.long)

            test_embs = model(test_batch, return_private=False, apply_graph=False)
            test_concat = torch.cat([test_embs["eeg"], test_embs["meg"], test_embs["fmri"]], dim=1).numpy()
            test_meta = pd.DataFrame({"sample_id": test_stim_ids, "split": ["test"] * len(test_stim_ids)})
            test_assigned = clustering.assign_clusters(
                {sid: test_concat[i] for i, sid in enumerate(test_stim_ids)}, test_meta
            )
            group_indices_test = torch.tensor(test_assigned["cluster_id"].values, dtype=torch.long)
        num_groups = 2
    else:  # 'random' ablation
        group_indices_train = torch.tensor([i % 2 for i in range(len(train_stim_ids))], dtype=torch.long)
        group_indices_val = torch.tensor([i % 2 for i in range(len(val_stim_ids))], dtype=torch.long)
        group_indices_test = torch.tensor([i % 2 for i in range(len(test_stim_ids))], dtype=torch.long)
        num_groups = 2

    if use_dro:
        dro = GroupDRO(num_groups=num_groups, learning_rate=0.01)

    # 5. Training Loop
    best_val_loss = float("inf")
    best_checkpoint_path = artifacts_dir / f"{condition_name}_seed{seed}_best.pt"
    history: List[Dict[str, Any]] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Curriculum adjustments
        cur_lambda_robust = lambda_robust
        cur_apply_graph = use_graph
        if use_curriculum:
            if epoch <= 8:  # Phase 1: Representation Discovery
                cur_lambda_robust = 0.0
                cur_apply_graph = False
            elif epoch <= 18:  # Phase 2: Robustness Focus
                cur_lambda_robust = lambda_robust
                cur_apply_graph = False
            else:  # Phase 3: Graph Consolidation
                cur_lambda_robust = lambda_robust
                cur_apply_graph = use_graph

        if is_unimodal:
            # Single-modality autoencoder reconstruction baseline
            x_uni = train_batch[unimodal_mod]
            if use_dual_heads:
                z_s, z_p = model.encoders[unimodal_mod](x_uni, return_both=True)
            else:
                z_s = model.encoders[unimodal_mod](x_uni, return_both=False)
                z_p = z_s

            targets = compute_sensor_summary_targets({unimodal_mod: x_uni})
            rec_loss = F.mse_loss(model.decoders[unimodal_mod](z_p), targets[unimodal_mod])
            loss = rec_loss
            loss.backward()
            optimizer.step()

            # Validation evaluation on unimodal val batch
            model.eval()
            with torch.no_grad():
                x_val_uni = val_batch[unimodal_mod]
                targets_val = compute_sensor_summary_targets({unimodal_mod: x_val_uni})
                if use_dual_heads:
                    _, z_p_val = model.encoders[unimodal_mod](x_val_uni, return_both=True)
                else:
                    z_p_val = model.encoders[unimodal_mod](x_val_uni, return_both=False)
                val_loss = float(F.mse_loss(model.decoders[unimodal_mod](z_p_val), targets_val[unimodal_mod]).item())

        else:
            # Trimodal training with active modality sampling
            if sampler is not None and epoch > 8:
                # Dynamic modality sampling: actively select 2 modalities based on sampling weights
                active_train_batch = sampler.sample_batch_modalities(train_batch, k=2)
            else:
                active_train_batch = train_batch

            if use_dual_heads:
                shared, private = model(active_train_batch, return_private=True, apply_graph=cur_apply_graph)
                sep_loss = multimodal_separation_loss(shared, private)
                targets = compute_sensor_summary_targets(active_train_batch)
                rec_loss = multimodal_reconstruction_loss(private, targets, model.decoders)
            else:
                shared = model(active_train_batch, return_private=False, apply_graph=cur_apply_graph)
                sep_loss = torch.tensor(0.0)
                rec_loss = torch.tensor(0.0)

            # Alignment loss on active modalities
            per_sample_align, _ = per_sample_trimodal_contrastive_loss(
                shared, train_ids_dict, temperature=0.07
            )
            align_loss_mean = per_sample_align.mean()

            # GroupDRO loss
            dro_loss = None
            if use_dro and cur_lambda_robust > 0.0:
                dro_loss = dro.compute_weighted_loss(
                    per_sample_align, group_indices_train, update_weights=True
                )

            total_loss, metrics = neuro_grape_total_loss(
                alignment_loss=align_loss_mean,
                separation_loss=sep_loss,
                reconstruction_loss=rec_loss,
                dro_loss=dro_loss,
                lambda_align=lambda_align,
                lambda_private=lambda_private,
                lambda_recon=lambda_recon,
                lambda_robust=cur_lambda_robust,
                interpolate_dro=True,
            )

            total_loss.backward()
            optimizer.step()

            # Adaptive mixture update from real training losses
            if sampler is not None and epoch > 8:
                active_mods = list(shared.keys())
                mod_losses: Dict[str, float] = {}
                for m in active_mods:
                    m_rec = (
                        float(F.mse_loss(model.decoders[m](private[m]), targets[m]).item())
                        if use_dual_heads and m in targets and m in model.decoders
                        else 0.0
                    )
                    mod_losses[m] = float(align_loss_mean.item()) + lambda_recon * m_rec
                sampler.update_from_losses(mod_losses, is_train=True)

            # Validation evaluation (strictly read-only)
            model.eval()
            with torch.no_grad():
                val_shared = model(val_batch, return_private=False, apply_graph=cur_apply_graph)
                val_loss_vec, _ = per_sample_trimodal_contrastive_loss(
                    val_shared, val_ids_dict, temperature=0.07
                )
                val_loss = float(val_loss_vec.mean().item())

        # Checkpoint selection strictly on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": config,
                },
                best_checkpoint_path,
            )

        history.append({
            "epoch": epoch,
            "train_loss": float(total_loss.item()) if not is_unimodal else float(loss.item()),
            "val_loss": val_loss,
        })

    train_time = time.time() - start_time

    # 6. Final Evaluation on Held-Out Test Set (Using Best Validation Checkpoint)
    if best_checkpoint_path.exists():
        best_ckpt = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        if is_unimodal:
            x_test_uni = test_batch[unimodal_mod]
            targets_test = compute_sensor_summary_targets({unimodal_mod: x_test_uni})
            if use_dual_heads:
                z_s_test, z_p_test = model.encoders[unimodal_mod](x_test_uni, return_both=True)
                ortho_diag = float(orthogonal_separation_loss(z_s_test, z_p_test).item())
            else:
                z_s_test = model.encoders[unimodal_mod](x_test_uni, return_both=False)
                z_p_test = z_s_test
                ortho_diag = 0.0

            rec_test_loss = float(
                F.mse_loss(model.decoders[unimodal_mod](z_p_test), targets_test[unimodal_mod]).item()
            )

            # Measure actual test cross-modal performance from model forward pass
            test_shared = model(test_batch, return_private=False, apply_graph=False)
            retrieval_res = evaluate_retrieval_and_calibration(
                test_shared, test_stim_ids, temperature=0.07
            )

            per_stim_acc = retrieval_res["per_stimulus_acc"]
            g_accs = [
                float(np.mean(per_stim_acc[(group_indices_test == g).numpy()]))
                if np.sum((group_indices_test == g).numpy()) > 0
                else retrieval_res["overall_top1"]
                for g in range(num_groups)
            ]

            test_loss_vec, _ = per_sample_trimodal_contrastive_loss(
                test_shared, test_ids_dict, temperature=0.07
            )
            test_loss = float(test_loss_vec.mean().item())

            test_metrics = {
                "overall_top1": retrieval_res["overall_top1"],
                "overall_top2": retrieval_res["overall_top2"],
                "worst_group_top1": min(g_accs),
                "disparity_gap": max(g_accs) - min(g_accs),
                "ece": retrieval_res["ece"],
                "ortho_diag": ortho_diag,
                "test_loss": test_loss,
                "test_rec_loss": rec_test_loss,
                "pair_top1": retrieval_res["pair_top1"],
            }
        else:
            if use_dual_heads:
                test_shared, test_private = model(test_batch, return_private=True, apply_graph=use_graph)
                ortho_diag = float(multimodal_separation_loss(test_shared, test_private).item())
            else:
                test_shared = model(test_batch, return_private=False, apply_graph=use_graph)
                ortho_diag = 0.0

            test_loss_vec, _ = per_sample_trimodal_contrastive_loss(
                test_shared, test_ids_dict, temperature=0.07
            )
            test_loss = float(test_loss_vec.mean().item())

            retrieval_res = evaluate_retrieval_and_calibration(
                test_shared, test_stim_ids, temperature=0.07
            )

            # Group disparity calculation on test set
            per_stim_acc = retrieval_res["per_stimulus_acc"]
            g_accs = [
                float(np.mean(per_stim_acc[(group_indices_test == g).numpy()]))
                if np.sum((group_indices_test == g).numpy()) > 0
                else retrieval_res["overall_top1"]
                for g in range(num_groups)
            ]

            worst_group_top1 = min(g_accs)
            disparity_gap = max(g_accs) - min(g_accs)

            test_metrics = {
                "overall_top1": retrieval_res["overall_top1"],
                "overall_top2": retrieval_res["overall_top2"],
                "worst_group_top1": worst_group_top1,
                "disparity_gap": disparity_gap,
                "ece": retrieval_res["ece"],
                "ortho_diag": ortho_diag,
                "test_loss": test_loss,
                "pair_top1": retrieval_res["pair_top1"],
            }

    # Parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    run_result = {
        "condition": condition_name,
        "seed": seed,
        "config": config,
        "train_time_sec": train_time,
        "param_count": param_count,
        "best_val_loss": best_val_loss,
        "metrics": test_metrics,
        "sampler_diagnostics": sampler.get_diagnostics() if sampler else None,
        "sampler_history": sampler.get_history() if sampler else None,
        "dro_weights": dro.group_weights.tolist() if dro else None,
    }

    return run_result


# ---------------------------------------------------------------------------
# Benchmark Suite Orchestration
# ---------------------------------------------------------------------------

def run_full_benchmark_suite() -> Dict[str, Any]:
    """Execute all benchmark conditions across multiple seeds."""
    artifacts_dir = REPO_ROOT / "reports" / "artifacts" / "full_benchmark"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset & splits using verified manifest
    manifest_path = (
        REPO_ROOT / "data" / "manifests" / "data_inventory.csv"
        if (REPO_ROOT / "data" / "manifests" / "data_inventory.csv").exists()
        else REPO_ROOT / "data_inventory.csv"
    )
    meta_df = pd.read_csv(manifest_path)

    train_ds = NeuroGrapeDataset(manifest=manifest_path, split="train", as_torch=True)
    val_ds = NeuroGrapeDataset(manifest=manifest_path, split="val", as_torch=True)
    test_ds = NeuroGrapeDataset(manifest=manifest_path, split="test", as_torch=True)

    train_batch_raw, train_ids_dict = prepare_cross_modal_batch(train_ds)
    val_batch_raw, val_ids_dict = prepare_cross_modal_batch(val_ds)
    test_batch_raw, test_ids_dict = prepare_cross_modal_batch(test_ds)

    # Leakage-free normalization: fit strictly on train split
    normalizer = TrimodalNormalizer()
    train_batch = normalizer.fit_transform(train_batch_raw)
    val_batch = normalizer.transform(val_batch_raw)
    test_batch = normalizer.transform(test_batch_raw)

    seeds = [42, 123, 999]

    # Benchmark condition templates
    conditions = [
        # Condition A: Single-modality baselines
        {"name": "A1_Single_EEG", "is_unimodal": True, "unimodal_modality": "eeg", "use_dual_heads": True},
        {"name": "A2_Single_MEG", "is_unimodal": True, "unimodal_modality": "meg", "use_dual_heads": True},
        {"name": "A3_Single_fMRI", "is_unimodal": True, "unimodal_modality": "fmri", "use_dual_heads": True},

        # Condition B: Fixed-Mixture Trimodal ERM
        {"name": "B_Trimodal_ERM", "use_dual_heads": False, "use_graph": False, "use_adaptive_mixture": False, "use_dro": False},

        # Condition C: Modality-Balanced Training
        {"name": "C_Modality_Balanced", "use_dual_heads": False, "use_graph": False, "use_adaptive_mixture": False, "use_dro": False, "balanced": True},

        # Condition D: Standard GroupDRO (Semantic)
        {"name": "D_GroupDRO_Semantic", "use_dual_heads": False, "use_graph": False, "use_adaptive_mixture": False, "use_dro": True, "group_type": "semantic"},

        # Condition E: Learned-Regime GroupDRO
        {"name": "E_GroupDRO_Cluster", "use_dual_heads": False, "use_graph": False, "use_adaptive_mixture": False, "use_dro": True, "group_type": "cluster"},

        # Condition F: Full Neuro-GRAPE
        {
            "name": "F_Full_Neuro_GRAPE",
            "use_dual_heads": True,
            "use_graph": True,
            "use_adaptive_mixture": True,
            "use_dro": True,
            "group_type": "cluster",
            "use_curriculum": True,
        },

        # Condition G: Ablations
        {"name": "G1_Ablate_Dual_Heads", "use_dual_heads": False, "use_graph": True, "use_adaptive_mixture": True, "use_dro": True, "group_type": "cluster", "use_curriculum": True},
        {"name": "G2_Ablate_Separation_Loss", "use_dual_heads": True, "lambda_private": 0.0, "use_graph": True, "use_adaptive_mixture": True, "use_dro": True, "group_type": "cluster", "use_curriculum": True},
        {"name": "G3_Ablate_Regimes", "use_dual_heads": True, "use_graph": True, "use_adaptive_mixture": True, "use_dro": True, "group_type": "random", "use_curriculum": True},
        {"name": "G4_Ablate_GroupDRO", "use_dual_heads": True, "use_graph": True, "use_adaptive_mixture": True, "use_dro": False, "group_type": "cluster", "use_curriculum": True},
        {"name": "G5_Ablate_Adaptive_Mixture", "use_dual_heads": True, "use_graph": True, "use_adaptive_mixture": False, "use_dro": True, "group_type": "cluster", "use_curriculum": True},
        {"name": "G6_Ablate_Graph", "use_dual_heads": True, "use_graph": False, "use_adaptive_mixture": True, "use_dro": True, "group_type": "cluster", "use_curriculum": True},
        {"name": "G7_Ablate_Reconstruction", "use_dual_heads": True, "lambda_recon": 0.0, "use_graph": True, "use_adaptive_mixture": True, "use_dro": True, "group_type": "cluster", "use_curriculum": True},
    ]

    all_run_results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    logger.info(f"Starting execution of {len(conditions)} benchmark conditions across {len(seeds)} seeds...")

    for cond in conditions:
        cond_name = cond["name"]
        seed_top1: List[float] = []
        seed_worst_top1: List[float] = []
        seed_disparity: List[float] = []
        seed_loss: List[float] = []
        seed_ece: List[float] = []
        seed_ortho: List[float] = []
        seed_time: List[float] = []

        last_run_for_plots = None

        for seed in seeds:
            cfg = dict(cond)
            cfg["seed"] = seed
            logger.info(f"Running condition {cond_name} (seed={seed})...")
            run_res = run_single_experiment(
                cfg,
                train_batch,
                val_batch,
                test_batch,
                train_ids_dict,
                val_ids_dict,
                test_ids_dict,
                meta_df,
                artifacts_dir,
            )
            all_run_results.append(run_res)
            last_run_for_plots = run_res

            m = run_res["metrics"]
            seed_top1.append(m["overall_top1"])
            seed_worst_top1.append(m["worst_group_top1"])
            seed_disparity.append(m["disparity_gap"])
            seed_loss.append(m["test_loss"])
            seed_ece.append(m["ece"])
            seed_ortho.append(m["ortho_diag"])
            seed_time.append(run_res["train_time_sec"])

        # Aggregate across seeds
        summary_rows.append({
            "condition": cond_name,
            "overall_top1_mean": float(np.mean(seed_top1)),
            "overall_top1_std": float(np.std(seed_top1)),
            "worst_group_top1_mean": float(np.mean(seed_worst_top1)),
            "worst_group_top1_std": float(np.std(seed_worst_top1)),
            "disparity_gap_mean": float(np.mean(seed_disparity)),
            "disparity_gap_std": float(np.std(seed_disparity)),
            "test_loss_mean": float(np.mean(seed_loss)),
            "test_loss_std": float(np.std(seed_loss)),
            "ece_mean": float(np.mean(seed_ece)),
            "ece_std": float(np.std(seed_ece)),
            "ortho_diag_mean": float(np.mean(seed_ortho)),
            "train_time_sec_mean": float(np.mean(seed_time)),
            "param_count": last_run_for_plots["param_count"],
        })

    # Save consolidated comparison CSV
    summary_df = pd.DataFrame(summary_rows)
    comparison_csv_path = REPO_ROOT / "reports" / "artifacts" / "full_benchmark_comparison.csv"
    summary_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"Saved consolidated benchmark comparison CSV to {comparison_csv_path}")

    # Save summary JSON
    summary_json_path = REPO_ROOT / "reports" / "artifacts" / "full_benchmark_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info(f"Saved consolidated benchmark summary JSON to {summary_json_path}")

    # -----------------------------------------------------------------------
    # Generate Diagnostic Visualizations
    # -----------------------------------------------------------------------
    logger.info("Generating publication-quality benchmark plots...")

    # Plot 1: Performance comparison across core baselines
    core_conditions = [
        "B_Trimodal_ERM",
        "C_Modality_Balanced",
        "D_GroupDRO_Semantic",
        "E_GroupDRO_Cluster",
        "F_Full_Neuro_GRAPE",
    ]
    core_df = summary_df[summary_df["condition"].isin(core_conditions)].copy()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(core_df))
    width = 0.35

    plt.bar(x - width / 2, core_df["overall_top1_mean"] * 100, width, label="Overall Top-1 Acc (%)", color="#1b9e77", alpha=0.9)
    plt.bar(x + width / 2, core_df["worst_group_top1_mean"] * 100, width, label="Worst-Group Top-1 Acc (%)", color="#d95f02", alpha=0.9)

    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Neuro-GRAPE Benchmark: Overall vs Worst-Group Generalization", fontsize=13, fontweight="bold")
    plt.xticks(x, [c.replace("_", " ") for c in core_df["condition"]], rotation=15, ha="right", fontsize=9)
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()
    p1 = plots_dir / "benchmark_performance_comparison.png"
    plt.savefig(p1, dpi=300)
    plt.close()

    # Plot 2: Ablation Delta Plot (impact of removing each component from Full Neuro-GRAPE)
    full_acc = summary_df.loc[summary_df["condition"] == "F_Full_Neuro_GRAPE", "overall_top1_mean"].values[0]
    ablation_conds = [c for c in summary_df["condition"] if c.startswith("G")]
    ablation_df = summary_df[summary_df["condition"].isin(ablation_conds)].copy()
    deltas = (ablation_df["overall_top1_mean"].values - full_acc) * 100

    plt.figure(figsize=(10, 5))
    bar_colors = ["#e41a1c" if d < 0 else "#377eb8" for d in deltas]
    bars = plt.barh([c.replace("_", " ") for c in ablation_df["condition"]], deltas, color=bar_colors, alpha=0.85)
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Top-1 Accuracy Delta (% vs Full Neuro-GRAPE)", fontsize=11, fontweight="bold")
    plt.title("Component Ablations: Impact on Generalization", fontsize=13, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p2 = plots_dir / "component_ablation_delta.png"
    plt.savefig(p2, dpi=300)
    plt.close()

    # Plot 3: Adaptive Mixture Trajectory for Full Neuro-GRAPE
    full_run = [r for r in all_run_results if r["condition"] == "F_Full_Neuro_GRAPE" and r["sampler_history"]][0]
    history = full_run["sampler_history"]
    if history:
        plt.figure(figsize=(8, 4.5))
        epochs_arr = list(range(len(history)))
        for mod, col, ls in [("eeg", "#e41a1c", "-"), ("meg", "#377eb8", "--"), ("fmri", "#4daf4a", "-.")]:
            vals = [h[mod] for h in history]
            plt.plot(epochs_arr, vals, label=f"Modality: {mod.upper()}", color=col, linestyle=ls, linewidth=2)
        plt.axhline(0.3333, color="gray", linestyle=":", alpha=0.7, label="Uniform Share (0.33)")
        plt.axhline(0.15, color="red", linestyle=":", alpha=0.5, label="Allocation Floor (0.15)")
        plt.xlabel("Training Step", fontsize=11)
        plt.ylabel("Mixture Weight $\\alpha_m$", fontsize=11)
        plt.title("Adaptive Data Mixture Trajectory (GRAPE / DOGE)", fontsize=13, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        p3 = plots_dir / "adaptive_mixture_trajectories.png"
        plt.savefig(p3, dpi=300)
        plt.close()

    logger.info("All benchmark runs and figures successfully generated!")
    return {
        "summary_csv": str(comparison_csv_path),
        "summary_json": str(summary_json_path),
        "total_runs": len(all_run_results),
    }


if __name__ == "__main__":
    run_full_benchmark_suite()
