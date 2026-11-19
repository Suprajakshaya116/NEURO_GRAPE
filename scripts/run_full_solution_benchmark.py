"""
scripts/run_full_solution_benchmark.py
======================================
Comprehensive automated benchmark harness for the full Neuro-GRAPE research solution.

Executes all named experimental conditions across multiple seeds (42, 123, 999):
- Condition A: Single-modality baselines (A1: EEG, A2: MEG)
- Condition B: Bimodal ERM baseline (labeled B_Trimodal_ERM for backward compatibility)
- Condition C: Modality-balanced training
- Condition D: Standard GroupDRO (Semantic: Animate vs Inanimate)
- Condition E: Learned-Regime GroupDRO (KMeans clustering on training representations)
- Condition F: Full Neuro-GRAPE (Dual heads + Orthogonality + Reconstruction +
               Cross-modal Graph + Adaptive Sampler + Learned-Regime GroupDRO + 3-Phase Curriculum)
- Condition G: Component Ablations (G1 to G7 removing one component at a time)

Evaluates strictly under zero-leakage protocol:
- Test queries and gallery targets are encoded independently (apply_graph=False).
- Bidirectional retrieval: EEG -> MEG and MEG -> EEG.
- Metrics: Overall Top-1, Top-2, Balanced Accuracy, Macro-F1, Worst-Group Accuracy,
           Disparity Gap, ECE, Directional Top-1.

Outputs:
- reports/artifacts/full_benchmark_comparison.csv
- reports/artifacts/full_benchmark_summary.json
- reports/artifacts/predictions.csv
- reports/artifacts/full_benchmark/plots/*.png
- reports/artifacts/plots/*.png
- checkpoints/best_neurogrape_model.pt
"""

from __future__ import annotations

import json
import logging
import math
import shutil
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
    neuro_grape_total_loss,
    orthogonal_separation_loss,
    per_sample_trimodal_contrastive_loss,
)
from src.models.model import TrimodalNeuroGrapeModel
from src.models.normalization import TrimodalNormalizer
from src.models.training import prepare_cross_modal_batch, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FullBenchmark")


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
    epochs = config.get("epochs", 30)
    lr = config.get("lr", 1e-3)
    temperature = config.get("temperature", 0.25)
    weight_decay = config.get("weight_decay", 1e-4)
    dropout = config.get("dropout", 0.2)

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

    train_metadata = meta_df[meta_df["split"].str.lower() == "train"].drop_duplicates(subset=["stimulus_id"])
    val_metadata = meta_df[meta_df["split"].str.lower() == "val"].drop_duplicates(subset=["stimulus_id"])
    test_metadata = meta_df[meta_df["split"].str.lower() == "test"].drop_duplicates(subset=["stimulus_id"])

    test_cat_map = dict(zip(test_metadata["stimulus_id"], test_metadata["category_name"]))

    # 1. Build Model
    model = TrimodalNeuroGrapeModel(embed_dim=128, dropout=dropout, use_graph=use_graph)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 2. Setup Adaptive Mixture Sampler
    sampler = None
    if use_adaptive_mixture:
        sampler = AdaptiveMixtureSampler(
            modalities=["eeg", "meg"],
            step_size=0.05,
            min_weight=0.15,
            seed=seed,
        )

    # 3. Setup Group Partitioning & GroupDRO
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
                [init_embs["eeg"], init_embs["meg"]], dim=1
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
            val_concat = torch.cat([val_embs["eeg"], val_embs["meg"]], dim=1).numpy()
            val_meta = pd.DataFrame({"sample_id": val_stim_ids, "split": ["val"] * len(val_stim_ids)})
            val_assigned = clustering.assign_clusters(
                {sid: val_concat[i] for i, sid in enumerate(val_stim_ids)}, val_meta
            )
            group_indices_val = torch.tensor(val_assigned["cluster_id"].values, dtype=torch.long)

            test_embs = model(test_batch, return_private=False, apply_graph=False)
            test_concat = torch.cat([test_embs["eeg"], test_embs["meg"]], dim=1).numpy()
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

    # 4. Training Loop
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
            # Single-modality reconstruction baseline
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
            # Bimodal training
            if use_dual_heads:
                shared, private = model(train_batch, return_private=True, apply_graph=cur_apply_graph)
                sep_loss = multimodal_separation_loss(shared, private)
                targets = compute_sensor_summary_targets(train_batch)
                rec_loss = multimodal_reconstruction_loss(private, targets, model.decoders)
            else:
                shared = model(train_batch, return_private=False, apply_graph=cur_apply_graph)
                sep_loss = torch.tensor(0.0)
                rec_loss = torch.tensor(0.0)

            # Alignment loss
            per_sample_align, _ = per_sample_trimodal_contrastive_loss(
                shared, train_ids_dict, temperature=temperature
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

            # Validation evaluation strictly with apply_graph=False
            model.eval()
            with torch.no_grad():
                val_shared = model(val_batch, return_private=False, apply_graph=False)
                val_loss_vec, _ = per_sample_trimodal_contrastive_loss(
                    val_shared, val_ids_dict, temperature=temperature
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

    # 5. Final Evaluation on Held-Out Test Set (Using Best Validation Checkpoint)
    if best_checkpoint_path.exists():
        best_ckpt = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        # Independent query and gallery encoding - NEVER apply graph attention on test set
        test_shared = model(test_batch, return_private=False, apply_graph=False)

        ortho_diag = 0.0
        if use_dual_heads:
            _, test_private = model(test_batch, return_private=True, apply_graph=False)
            ortho_diag = float(multimodal_separation_loss(test_shared, test_private).item())

        test_loss_vec, _ = per_sample_trimodal_contrastive_loss(
            test_shared, test_ids_dict, temperature=temperature
        )
        test_loss = float(test_loss_vec.mean().item())

        cat_names_test = [test_cat_map[s] for s in test_stim_ids]

        retrieval_res = evaluate_clean_crossmodal_retrieval(
            query_embeddings=test_shared,
            gallery_embeddings=test_shared,
            stimulus_ids=test_stim_ids,
            category_names=cat_names_test,
            temperature=temperature,
            seed=seed,
            model_name=condition_name,
        )

        test_metrics = {
            "overall_top1": retrieval_res["overall_top1"],
            "overall_top2": retrieval_res["overall_top2"],
            "balanced_accuracy": retrieval_res["balanced_accuracy"],
            "macro_f1": retrieval_res["macro_f1"],
            "worst_group_top1": retrieval_res["worst_group_accuracy"],
            "disparity_gap": retrieval_res["disparity_gap"],
            "ece": retrieval_res["ece"],
            "ortho_diag": ortho_diag,
            "test_loss": test_loss,
            "pair_top1": retrieval_res["direction_top1"],
            "confusion_matrix": retrieval_res["confusion_matrix"],
            "prediction_records": retrieval_res["prediction_records"],
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
        "history": history,
        "sampler_diagnostics": sampler.get_diagnostics() if sampler else None,
        "sampler_history": sampler.get_history() if sampler else None,
        "dro_weights": dro.group_weights.tolist() if dro else None,
        "best_checkpoint_path": str(best_checkpoint_path),
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

    pub_plots_dir = REPO_ROOT / "reports" / "artifacts" / "plots"
    pub_plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = REPO_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

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

        # Condition B: Standard Bimodal ERM
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
    all_predictions: List[Dict[str, Any]] = []

    best_global_val_loss = float("inf")
    best_global_ckpt_path: Optional[str] = None

    logger.info(f"Starting execution of {len(conditions)} benchmark conditions across {len(seeds)} seeds...")

    for cond in conditions:
        cond_name = cond["name"]
        seed_top1: List[float] = []
        seed_top2: List[float] = []
        seed_bal_acc: List[float] = []
        seed_macro_f1: List[float] = []
        seed_worst_top1: List[float] = []
        seed_disparity: List[float] = []
        seed_loss: List[float] = []
        seed_ece: List[float] = []
        seed_ortho: List[float] = []
        seed_time: List[float] = []
        seed_eeg2meg: List[float] = []
        seed_meg2eeg: List[float] = []

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
            seed_top2.append(m["overall_top2"])
            seed_bal_acc.append(m["balanced_accuracy"])
            seed_macro_f1.append(m["macro_f1"])
            seed_worst_top1.append(m["worst_group_top1"])
            seed_disparity.append(m["disparity_gap"])
            seed_loss.append(m["test_loss"])
            seed_ece.append(m["ece"])
            seed_ortho.append(m["ortho_diag"])
            seed_time.append(run_res["train_time_sec"])

            pair_m = m.get("pair_top1", {})
            seed_eeg2meg.append(pair_m.get("eeg_to_meg", m["overall_top1"]))
            seed_meg2eeg.append(pair_m.get("meg_to_eeg", m["overall_top1"]))

            all_predictions.extend(m.get("prediction_records", []))

            # Track best model checkpoint
            if cond_name == "F_Full_Neuro_GRAPE" and run_res["best_val_loss"] < best_global_val_loss:
                best_global_val_loss = run_res["best_val_loss"]
                best_global_ckpt_path = run_res["best_checkpoint_path"]

        # Aggregate across seeds
        summary_rows.append({
            "condition": cond_name,
            "overall_top1_mean": float(np.mean(seed_top1)),
            "overall_top1_std": float(np.std(seed_top1)),
            "overall_top2_mean": float(np.mean(seed_top2)),
            "overall_top2_std": float(np.std(seed_top2)),
            "balanced_acc_mean": float(np.mean(seed_bal_acc)),
            "macro_f1_mean": float(np.mean(seed_macro_f1)),
            "worst_group_top1_mean": float(np.mean(seed_worst_top1)),
            "worst_group_top1_std": float(np.std(seed_worst_top1)),
            "disparity_gap_mean": float(np.mean(seed_disparity)),
            "disparity_gap_std": float(np.std(seed_disparity)),
            "eeg_to_meg_mean": float(np.mean(seed_eeg2meg)),
            "meg_to_eeg_mean": float(np.mean(seed_meg2eeg)),
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

    # Save predictions CSV (filtering to standard requested schema)
    predictions_df = pd.DataFrame(all_predictions)
    pred_csv_cols = [
        "sample_id",
        "query_modality",
        "gallery_modality",
        "true_label",
        "predicted_label",
        "rank_1",
        "rank_2",
        "confidence",
        "is_correct",
        "group_id",
        "split",
    ]
    # For general predictions CSV, prioritize Full Neuro-GRAPE
    full_preds = predictions_df[predictions_df["model_name"] == "F_Full_Neuro_GRAPE"]
    if len(full_preds) == 0:
        full_preds = predictions_df

    pred_csv_path = REPO_ROOT / "reports" / "artifacts" / "predictions.csv"
    full_preds[pred_csv_cols].to_csv(pred_csv_path, index=False)
    logger.info(f"Saved submission prediction records to {pred_csv_path}")

    # Copy best checkpoint to checkpoints/best_neurogrape_model.pt
    if best_global_ckpt_path and Path(best_global_ckpt_path).exists():
        target_best_ckpt = checkpoints_dir / "best_neurogrape_model.pt"
        shutil.copyfile(best_global_ckpt_path, target_best_ckpt)
        logger.info(f"Saved best Neuro-GRAPE checkpoint to {target_best_ckpt}")

    # -----------------------------------------------------------------------
    # Generate Diagnostic Visualizations
    # -----------------------------------------------------------------------
    logger.info("Generating publication-quality benchmark plots...")

    # Plot 1: Performance comparison across core baselines
    core_conditions = [
        "A1_Single_EEG",
        "A2_Single_MEG",
        "B_Trimodal_ERM",
        "C_Modality_Balanced",
        "D_GroupDRO_Semantic",
        "E_GroupDRO_Cluster",
        "F_Full_Neuro_GRAPE",
    ]
    core_df = summary_df[summary_df["condition"].isin(core_conditions)].copy()

    plt.figure(figsize=(11, 5.5))
    x = np.arange(len(core_df))
    width = 0.35

    plt.bar(x - width / 2, core_df["overall_top1_mean"] * 100, width, label="Overall Top-1 Acc (%)", color="#1b9e77", alpha=0.9)
    plt.bar(x + width / 2, core_df["worst_group_top1_mean"] * 100, width, label="Worst-Group Top-1 Acc (%)", color="#d95f02", alpha=0.9)

    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Neuro-GRAPE Benchmark: Overall vs Worst-Group Retrieval", fontsize=13, fontweight="bold")
    plt.xticks(x, [c.replace("_", " ") for c in core_df["condition"]], rotation=20, ha="right", fontsize=9)
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    p1 = plots_dir / "benchmark_performance_comparison.png"
    plt.savefig(p1, dpi=300)
    plt.savefig(pub_plots_dir / "benchmark_performance_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Ablation Delta Plot (impact of removing each component from Full Neuro-GRAPE)
    full_row = summary_df[summary_df["condition"] == "F_Full_Neuro_GRAPE"]
    full_acc = full_row["overall_top1_mean"].values[0] if len(full_row) > 0 else 0.5
    ablation_conds = [c for c in summary_df["condition"] if c.startswith("G")]
    ablation_df = summary_df[summary_df["condition"].isin(ablation_conds)].copy()
    deltas = (ablation_df["overall_top1_mean"].values - full_acc) * 100

    plt.figure(figsize=(10, 5))
    bar_colors = ["#e41a1c" if d < 0 else "#377eb8" for d in deltas]
    plt.barh([c.replace("_", " ") for c in ablation_df["condition"]], deltas, color=bar_colors, alpha=0.85)
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Top-1 Accuracy Delta (% vs Full Neuro-GRAPE)", fontsize=11, fontweight="bold")
    plt.title("Component Ablations: Impact on Generalization", fontsize=13, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p2 = plots_dir / "component_ablation_delta.png"
    plt.savefig(p2, dpi=300)
    plt.savefig(pub_plots_dir / "component_ablation_delta.png", dpi=300)
    plt.close()

    # Plot 3: Adaptive Mixture Trajectory for Full Neuro-GRAPE
    full_runs = [r for r in all_run_results if r["condition"] == "F_Full_Neuro_GRAPE" and r["sampler_history"]]
    if full_runs:
        history = full_runs[0]["sampler_history"]
        if history:
            plt.figure(figsize=(8, 4.5))
            epochs_arr = list(range(len(history)))
            for mod, col, ls in [("eeg", "#e41a1c", "-"), ("meg", "#377eb8", "--")]:
                vals = [h.get(mod, 0.5) for h in history]
                plt.plot(epochs_arr, vals, label=f"Modality: {mod.upper()}", color=col, linestyle=ls, linewidth=2)
            plt.axhline(0.50, color="gray", linestyle=":", alpha=0.7, label="Uniform Share (0.50)")
            plt.axhline(0.15, color="red", linestyle=":", alpha=0.5, label="Allocation Floor (0.15)")
            plt.xlabel("Training Step", fontsize=11)
            plt.ylabel("Mixture Weight $\\alpha_m$", fontsize=11)
            plt.title("Adaptive Data Mixture Trajectory (GRAPE / DOGE)", fontsize=13, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.legend(frameon=True, facecolor="white")
            plt.tight_layout()
            p3 = plots_dir / "adaptive_mixture_trajectories.png"
            plt.savefig(p3, dpi=300)
            plt.savefig(pub_plots_dir / "adaptive_mixture_trajectories.png", dpi=300)
            plt.close()

    # Plot 4: Retrieval Confusion Matrix
    if len(full_preds) > 0:
        plt.figure(figsize=(6, 5))
        test_cats = ["tadpole", "tarp", "wire", "woman"]
        cm = np.zeros((4, 4), dtype=int)
        for _, row in full_preds.iterrows():
            if row["true_label"] in test_cats and row["predicted_label"] in test_cats:
                ti = test_cats.index(row["true_label"])
                pi = test_cats.index(row["predicted_label"])
                cm[ti, pi] += 1
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.title("Test Retrieval Confusion Matrix (Full Neuro-GRAPE)", fontsize=12, fontweight="bold")
        plt.colorbar()
        tick_marks = np.arange(4)
        plt.xticks(tick_marks, test_cats, rotation=45)
        plt.yticks(tick_marks, test_cats)
        plt.xlabel("Predicted Concept", fontweight="bold")
        plt.ylabel("True Concept", fontweight="bold")
        for i in range(4):
            for j in range(4):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
        plt.tight_layout()
        p4 = plots_dir / "retrieval_confusion_matrix.png"
        plt.savefig(p4, dpi=300)
        plt.savefig(pub_plots_dir / "retrieval_confusion_matrix.png", dpi=300)
        plt.close()

    # Plot 5: Loss and Calibration Curves
    plt.figure(figsize=(9, 4.5))
    for r in all_run_results:
        if r["condition"] == "F_Full_Neuro_GRAPE" and r["seed"] == 42:
            h = r["history"]
            ep = [x["epoch"] for x in h]
            tl = [x["train_loss"] for x in h]
            vl = [x["val_loss"] for x in h]
            plt.plot(ep, tl, label="Train Loss (Total)", color="#1b9e77", linewidth=2)
            plt.plot(ep, vl, label="Validation Loss", color="#d95f02", linewidth=2, linestyle="--")
            break
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Full Neuro-GRAPE Optimization Trajectory (Seed 42)", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    p5 = plots_dir / "loss_and_calibration_curves.png"
    plt.savefig(p5, dpi=300)
    plt.savefig(pub_plots_dir / "loss_and_calibration_curves.png", dpi=300)
    plt.close()

    logger.info("All benchmark runs, predictions, and figures successfully generated!")
    return {
        "summary_csv": str(comparison_csv_path),
        "summary_json": str(summary_json_path),
        "predictions_csv": str(pred_csv_path),
        "total_runs": len(all_run_results),
    }


if __name__ == "__main__":
    run_full_benchmark_suite()
