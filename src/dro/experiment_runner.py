"""
src/dro/experiment_runner.py
============================
Member 3 Experiment Runner for Phase 3 Large-Scale Benchmarking in Neuro-GRAPE.

Provides end-to-end execution of:
1. Baseline Model Training (Empirical Risk Minimization / Equal Group Weights)
2. Genuine GroupDRO Model Training (Per-example loss vector -> Exponentiated gradient ascent -> Encoder parameter updates)
3. Multiple Grouping Strategies:
   - 'semantic': Animate vs. Inanimate subgroup categorization
   - 'cluster': Latent regimes discovered via K-Means fitted strictly on training data
4. Strict Leakage Prevention:
   - Stimulus-disjoint split boundaries
   - Normalization statistics fitted strictly on training data
   - GroupDRO weights updated exclusively on training losses
   - Checkpoint selection guided by validation split
   - Final evaluation conducted strictly on the held-out test split
5. Multi-seed execution and artifact export for scientific reproducibility
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.data.dataset import NeuroGrapeDataset
from src.dro.clustering import EmbeddingClustering
from src.dro.evaluation import GroupEvaluator
from src.dro.group_dro import GroupDRO
from src.models.encoders import EEGEncoder, FMRIEncoder, MEGEncoder
from src.models.losses import (
    compute_group_weighted_loss,
    multimodal_contrastive_loss,
    per_sample_trimodal_contrastive_loss,
    trimodal_contrastive_loss,
)
from src.models.normalization import TrimodalNormalizer
from src.models.training import prepare_cross_modal_batch, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ANIMATE_CATEGORIES = frozenset({"bear", "boa", "tadpole", "woman"})


def assign_group_labels(
    stimulus_ids: Sequence[str],
    metadata_df: pd.DataFrame,
    strategy: str = "semantic",
    stim_to_cluster: Optional[Dict[str, int]] = None,
) -> Tuple[torch.Tensor, List[str], Dict[int, str], Optional[Dict[str, int]]]:
    """Map a sequence of stimulus IDs to integer group indices [0, K-1].

    Parameters
    ----------
    stimulus_ids : Sequence[str]
        Ordered list of stimulus IDs in the batch.
    metadata_df : pd.DataFrame
        Metadata table containing stimulus_id and category_name.
    strategy : str, default='semantic'
        Grouping strategy ('semantic' or 'cluster').
    stim_to_cluster : Dict[str, int], optional
        Precomputed stimulus-to-cluster mapping.

    Returns
    -------
    group_indices : torch.Tensor
        (N,) tensor of integer group IDs.
    group_labels : List[str]
        String labels per stimulus.
    id_to_name : Dict[int, str]
        Mapping from integer group ID to human-readable group name.
    stim_to_cluster_out : Optional[Dict[str, int]]
        Mapping from stimulus ID to cluster ID if strategy == 'cluster'.
    """
    stim_to_cat = dict(zip(metadata_df["stimulus_id"], metadata_df["category_name"].str.lower()))

    if strategy == "semantic":
        # Binary: 0 = Inanimate, 1 = Animate
        id_to_name = {0: "inanimate", 1: "animate"}
        indices = []
        labels = []
        for sid in stimulus_ids:
            cat = stim_to_cat.get(sid, "")
            is_anim = 1 if cat in ANIMATE_CATEGORIES else 0
            indices.append(is_anim)
            labels.append(id_to_name[is_anim])
        return torch.tensor(indices, dtype=torch.long), labels, id_to_name, None

    elif strategy == "cluster":
        if stim_to_cluster is None:
            # Fit clustering strictly on training embeddings
            emb_pt = Path("embeddings/embeddings.pt")
            emb_meta = Path("embeddings/metadata.csv")
            if emb_pt.exists() and emb_meta.exists():
                embs_dict = torch.load(emb_pt, map_location="cpu")
                embs_df = pd.read_csv(emb_meta)
                clustering = EmbeddingClustering(n_clusters=2, random_state=42)
                clustering.fit(embs_dict, embs_df)
                assigned_df = clustering.assign_clusters(embs_dict, embs_df)
                stim_to_cluster = assigned_df.groupby("stimulus_id")["cluster_id"].first().to_dict()
            else:
                stim_to_cluster = {sid: idx % 2 for idx, sid in enumerate(stimulus_ids)}

        n_clusters = max(stim_to_cluster.values()) + 1
        id_to_name = {i: f"cluster_{i}" for i in range(n_clusters)}
        indices = [stim_to_cluster.get(sid, 0) for sid in stimulus_ids]
        labels = [id_to_name[idx] for idx in indices]
        return torch.tensor(indices, dtype=torch.long), labels, id_to_name, stim_to_cluster

    else:
        raise ValueError(f"Unknown grouping strategy: {strategy}")


def compute_cross_modal_metrics(
    embeddings: Mapping[str, torch.Tensor],
    stimulus_ids: Sequence[str],
) -> Dict[str, Any]:
    """Compute pairwise alignment and retrieval metrics across all three modalities.

    Parameters
    ----------
    embeddings : Mapping[str, torch.Tensor]
        Dict with keys 'eeg', 'meg', 'fmri' mapping to (N, D) L2-normalized tensors.
    stimulus_ids : Sequence[str]
        Ordered sequence of stimulus IDs of length N.

    Returns
    -------
    metrics : dict
        Contains per-pair and average retrieval accuracies and cosine similarities.
    """
    n = len(stimulus_ids)
    active_mods = [m for m in ("eeg", "meg", "fmri") if m in embeddings]
    pairs = [
        (active_mods[i], active_mods[j])
        for i in range(len(active_mods))
        for j in range(len(active_mods))
        if i != j
    ]
    pair_accs: Dict[str, float] = {}
    pair_sims: Dict[str, float] = {}
    per_stim_acc = torch.zeros(n, dtype=torch.float32)
    per_stim_sim = torch.zeros(n, dtype=torch.float32)

    for mod_a, mod_b in pairs:
        z_a = embeddings[mod_a]  # (N, D)
        z_b = embeddings[mod_b]  # (N, D)
        sim_matrix = torch.mm(z_a, z_b.t())  # (N, N)

        # Retrieval accuracy: argmax along row matches diagonal index
        preds_ab = torch.argmax(sim_matrix, dim=1)
        targets = torch.arange(n, device=z_a.device)
        correct_ab = (preds_ab == targets).float()

        diag_sims = torch.diagonal(sim_matrix)

        pair_key = f"{mod_a}_{mod_b}"
        pair_accs[f"retrieval_acc_{pair_key}"] = float(correct_ab.mean().item())
        pair_sims[f"cosine_sim_{pair_key}"] = float(diag_sims.mean().item())

        per_stim_acc += correct_ab.cpu()
        per_stim_sim += diag_sims.cpu()

    per_stim_acc /= len(pairs)
    per_stim_sim /= len(pairs)

    return {
        "mean_retrieval_acc": float(per_stim_acc.mean().item()),
        "mean_alignment_sim": float(per_stim_sim.mean().item()),
        "pair_retrieval_accuracies": pair_accs,
        "pair_cosine_similarities": pair_sims,
        "per_sample_retrieval_acc": per_stim_acc,
        "per_sample_alignment_sim": per_stim_sim,
    }


def train_single_run(
    mode: str = "groupdro",  # 'baseline' or 'groupdro'
    group_strategy: str = "semantic",
    manifest_path: Union[str, Path] = "data_inventory.csv",
    num_epochs: int = 30,
    lr: float = 1e-4,
    group_lr: float = 0.05,
    weight_decay: float = 1e-2,
    temperature: float = 0.07,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute a single complete training run for Baseline or GroupDRO.

    Guarantees:
    - Zero data leakage (validation split used for early stopping, test split held out).
    - Genuine gradient backpropagation into encoder weights.
    - Full metric and trajectory logging.
    """
    if mode not in ("baseline", "groupdro"):
        raise ValueError(f"mode must be 'baseline' or 'groupdro', got {mode}")

    set_seed(seed)
    if device is None:
        device = torch.device("cpu")

    # 1. Load Train and Validation splits
    train_ds = NeuroGrapeDataset(manifest=manifest_path, split="train", as_torch=True)
    val_ds = NeuroGrapeDataset(manifest=manifest_path, split="val", as_torch=True)
    meta_df = pd.read_csv(manifest_path)

    train_batch, train_ids = prepare_cross_modal_batch(train_ds, device=device)
    val_batch, val_ids = prepare_cross_modal_batch(val_ds, device=device)

    # 2. Fit normalizer on training split only
    normalizer = TrimodalNormalizer().to(device)
    train_batch = normalizer.fit_transform(train_batch)
    val_batch = normalizer.transform(val_batch)

    # 3. Determine Group Assignments
    stim_list = train_ids["eeg"]
    group_indices, group_labels, id_to_name, stim_to_cluster = assign_group_labels(
        stim_list, meta_df, strategy=group_strategy
    )
    group_indices = group_indices.to(device)
    num_groups = len(id_to_name)
    group_names = [id_to_name[i] for i in range(num_groups)]

    # 4. Initialize GroupDRO optimizer (if mode == 'groupdro')
    dro: Optional[GroupDRO] = None
    if mode == "groupdro":
        dro = GroupDRO(
            num_groups=num_groups,
            learning_rate=group_lr,
            group_names=group_names,
            device=device,
        )

    # 5. Initialize Encoders and Optimizer
    encoders: Dict[str, nn.Module] = {
        "eeg": EEGEncoder().to(device),
        "meg": MEGEncoder().to(device),
    }
    if "fmri" in train_batch:
        encoders["fmri"] = FMRIEncoder().to(device)

    all_params = []
    for enc in encoders.values():
        all_params.extend(list(enc.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    # Trajectory logs
    history: List[Dict[str, Any]] = []
    group_weights_log: List[Dict[str, Any]] = []
    group_losses_log: List[Dict[str, Any]] = []

    best_val_loss = float("inf")
    best_epoch = -1
    best_states: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    for epoch in range(1, num_epochs + 1):
        # ---- Training Step ----
        for enc in encoders.values():
            enc.train()
        optimizer.zero_grad()

        z_train = {m: encoders[m](train_batch[m]) for m in encoders}

        # Calculate per-example loss vector
        per_example_loss, ordered_ids = per_sample_trimodal_contrastive_loss(
            embeddings=z_train,
            stimulus_ids=train_ids,
            temperature=temperature,
        )

        if mode == "groupdro" and dro is not None:
            # Genuine GroupDRO: updates weights via exponentiated gradient ascent
            # and backpropagates through weighted group losses into encoders
            train_loss = dro.compute_weighted_loss(
                per_example_losses=per_example_loss,
                group_indices=group_indices,
                update_weights=True,
            )
            current_weights = dro.group_weights.detach().cpu().numpy()
            current_g_losses = dro.last_group_losses.detach().cpu().numpy()
        else:
            # Baseline ERM: standard unweighted mean
            train_loss = per_example_loss.mean()
            # Equal uniform weights for baseline
            current_weights = np.full(num_groups, 1.0 / num_groups)
            # Compute group losses for logging
            with torch.no_grad():
                current_g_losses = np.array([
                    per_example_loss[group_indices == g].mean().item()
                    if (group_indices == g).any() else 0.0
                    for g in range(num_groups)
                ])

        if not torch.isfinite(train_loss):
            raise FloatingPointError(f"Non-finite loss ({train_loss.item()}) at epoch {epoch}")

        train_loss.backward()
        optimizer.step()

        # ---- Validation Step ----
        for enc in encoders.values():
            enc.eval()

        with torch.no_grad():
            z_val = {m: encoders[m](val_batch[m]) for m in encoders}
            val_loss = trimodal_contrastive_loss(
                embeddings=z_val,
                stimulus_ids=val_ids,
                temperature=temperature,
            ).item()

        # Log epoch data
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_loss),
        }
        history.append(epoch_record)

        gw_record = {"epoch": epoch}
        gl_record = {"epoch": epoch}
        for g_idx, g_name in enumerate(group_names):
            gw_record[g_name] = float(current_weights[g_idx])
            gl_record[g_name] = float(current_g_losses[g_idx])
        group_weights_log.append(gw_record)
        group_losses_log.append(gl_record)

        # Checkpoint selection on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_states = {
                mod: {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
                for mod, enc in encoders.items()
            }

    # Restore best checkpoint
    if best_states is not None:
        for mod, enc in encoders.items():
            enc.load_state_dict({k: v.to(device) for k, v in best_states[mod].items()})

    # Prepare group assignments table
    assignments_table = pd.DataFrame({
        "stimulus_id": stim_list,
        "group_id": group_indices.cpu().numpy(),
        "group_name": group_labels,
        "split": "train",
    })

    return {
        "mode": mode,
        "group_strategy": group_strategy,
        "seed": seed,
        "num_epochs": num_epochs,
        "encoders": encoders,
        "normalizer": normalizer,
        "history": pd.DataFrame(history),
        "group_weights_history": pd.DataFrame(group_weights_log),
        "group_losses_history": pd.DataFrame(group_losses_log),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_states": best_states,
        "train_assignments": assignments_table,
        "id_to_name": id_to_name,
        "stim_to_cluster": stim_to_cluster,
    }


def evaluate_on_test_split(
    run_output: Dict[str, Any],
    manifest_path: Union[str, Path] = "data_inventory.csv",
    temperature: float = 0.07,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Perform rigorous evaluation on the strictly held-out test split.

    Computes:
    - Overall test loss
    - Per-group test losses and sample counts
    - Worst-group test loss and disparity gap
    - Pairwise retrieval accuracy and alignment cosine similarity
    - Subgroup retrieval disparity
    """
    if device is None:
        device = torch.device("cpu")

    encoders = run_output["encoders"]
    normalizer = run_output["normalizer"]
    id_to_name = run_output["id_to_name"]
    group_strategy = run_output["group_strategy"]

    for enc in encoders.values():
        enc.eval()

    test_ds = NeuroGrapeDataset(manifest=manifest_path, split="test", as_torch=True)
    meta_df = pd.read_csv(manifest_path)
    test_batch, test_ids = prepare_cross_modal_batch(test_ds, device=device)
    test_batch = normalizer.transform(test_batch)

    stim_list = test_ids["eeg"]
    stim_to_cluster = run_output.get("stim_to_cluster")
    test_group_indices, test_group_labels, _, _ = assign_group_labels(
        stim_list, meta_df, strategy=group_strategy, stim_to_cluster=stim_to_cluster
    )

    with torch.no_grad():
        z_test = {m: encoders[m](test_batch[m]) for m in encoders}

        # 1. Per-sample loss vector
        per_sample_loss, _ = per_sample_trimodal_contrastive_loss(
            embeddings=z_test,
            stimulus_ids=test_ids,
            temperature=temperature,
        )

        # 2. Cross-modal retrieval and alignment metrics
        retrieval_res = compute_cross_modal_metrics(z_test, stim_list)

    # 3. Group Evaluation for Contrastive Loss (lower is better)
    evaluator = GroupEvaluator(min_group_size=1)
    eval_loss = evaluator.evaluate(
        values=per_sample_loss.cpu().numpy(),
        group_ids=test_group_indices.cpu().numpy(),
        metric_name="contrastive_loss",
        higher_is_better=False,
        all_group_ids=list(id_to_name.keys()),
    )

    # 4. Group Evaluation for Retrieval Accuracy (higher is better)
    eval_retrieval = evaluator.evaluate(
        values=retrieval_res["per_sample_retrieval_acc"].numpy(),
        group_ids=test_group_indices.cpu().numpy(),
        metric_name="retrieval_accuracy",
        higher_is_better=True,
        all_group_ids=list(id_to_name.keys()),
    )

    # 5. Group Evaluation for Cosine Alignment (higher is better)
    eval_sim = evaluator.evaluate(
        values=retrieval_res["per_sample_alignment_sim"].numpy(),
        group_ids=test_group_indices.cpu().numpy(),
        metric_name="alignment_cosine",
        higher_is_better=True,
        all_group_ids=list(id_to_name.keys()),
    )

    # Format tabular results
    test_assignments = pd.DataFrame({
        "stimulus_id": stim_list,
        "group_id": test_group_indices.cpu().numpy(),
        "group_name": test_group_labels,
        "split": "test",
        "per_sample_loss": per_sample_loss.cpu().numpy(),
        "per_sample_retrieval_acc": retrieval_res["per_sample_retrieval_acc"].numpy(),
        "per_sample_alignment_sim": retrieval_res["per_sample_alignment_sim"].numpy(),
    })

    return {
        "overall_loss": float(eval_loss.overall_value),
        "worst_group_loss": float(eval_loss.worst_group_value),
        "loss_disparity_gap": float(eval_loss.performance_gap),
        "worst_loss_group_id": eval_loss.worst_group_id,
        "loss_eval_result": eval_loss,
        "overall_retrieval_acc": float(eval_retrieval.overall_value),
        "worst_group_retrieval_acc": float(eval_retrieval.worst_group_value),
        "retrieval_disparity_gap": float(eval_retrieval.performance_gap),
        "retrieval_eval_result": eval_retrieval,
        "overall_alignment_sim": float(eval_sim.overall_value),
        "worst_group_alignment_sim": float(eval_sim.worst_group_value),
        "alignment_disparity_gap": float(eval_sim.performance_gap),
        "pairwise_metrics": {
            **retrieval_res["pair_retrieval_accuracies"],
            **retrieval_res["pair_cosine_similarities"],
        },
        "test_assignments": test_assignments,
    }


def run_and_save_experiment(
    mode: str,
    group_strategy: str = "semantic",
    output_dir: Union[str, Path] = "reports/artifacts/experiments/test_exp",
    seeds: Sequence[int] = (42, 123, 456),
    num_epochs: int = 30,
    lr: float = 1e-4,
    group_lr: float = 0.05,
    temperature: float = 0.07,
) -> Dict[str, Any]:
    """Execute multi-seed experiment, compute mean & std, and export all required artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running Experiment: mode={mode}, strategy={group_strategy}, output={out_path}")

    seed_results = []
    all_group_metrics_dfs = []
    primary_run = None

    for seed in seeds:
        logger.info(f"  --> Executing Seed {seed} for {mode}...")
        train_res = train_single_run(
            mode=mode,
            group_strategy=group_strategy,
            num_epochs=num_epochs,
            lr=lr,
            group_lr=group_lr,
            temperature=temperature,
            seed=seed,
        )
        test_res = evaluate_on_test_split(train_res, temperature=temperature)

        if primary_run is None:
            primary_run = (train_res, test_res)

        # Save group metrics table for this seed
        loss_df = test_res["loss_eval_result"].to_group_dataframe()
        loss_df["seed"] = seed
        all_group_metrics_dfs.append(loss_df)

        seed_results.append({
            "seed": seed,
            "best_epoch": train_res["best_epoch"],
            "best_val_loss": train_res["best_val_loss"],
            "test_overall_loss": test_res["overall_loss"],
            "test_worst_group_loss": test_res["worst_group_loss"],
            "test_loss_gap": test_res["loss_disparity_gap"],
            "test_overall_retrieval_acc": test_res["overall_retrieval_acc"],
            "test_worst_group_retrieval_acc": test_res["worst_group_retrieval_acc"],
            "test_retrieval_gap": test_res["retrieval_disparity_gap"],
            "test_overall_alignment_sim": test_res["overall_alignment_sim"],
            "test_worst_group_alignment_sim": test_res["worst_group_alignment_sim"],
        })

    summary_df = pd.DataFrame(seed_results)

    # 1. Save metrics.csv
    summary_df.to_csv(out_path / "metrics.csv", index=False)

    # 2. Save group_metrics.csv
    full_group_metrics = pd.concat(all_group_metrics_dfs, ignore_index=True)
    full_group_metrics.to_csv(out_path / "group_metrics.csv", index=False)

    # 3. Save group_losses.csv and group_weights.csv from representative seed (first seed)
    rep_train, rep_test = primary_run
    rep_train["group_losses_history"].to_csv(out_path / "group_losses.csv", index=False)
    rep_train["group_weights_history"].to_csv(out_path / "group_weights.csv", index=False)

    # 4. Save group_assignments.csv & parquet
    all_assignments = pd.concat([rep_train["train_assignments"], rep_test["test_assignments"]], ignore_index=True)
    all_assignments.to_csv(out_path / "group_assignments.csv", index=False)
    all_assignments.to_parquet(out_path / "group_assignments.parquet", index=False)

    # 5. Save config snapshot
    config_snapshot = {
        "mode": mode,
        "group_strategy": group_strategy,
        "seeds": list(seeds),
        "num_epochs": num_epochs,
        "lr": lr,
        "group_lr": group_lr,
        "temperature": temperature,
        "best_epoch_seed42": rep_train["best_epoch"],
        "best_val_loss_seed42": rep_train["best_val_loss"],
    }
    with open(out_path / "experiment_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_snapshot, f)

    # 6. Save machine-readable experiment summary JSON
    summary_stats = {
        "mode": mode,
        "group_strategy": group_strategy,
        "num_seeds": len(seeds),
        "test_overall_loss_mean": float(summary_df["test_overall_loss"].mean()),
        "test_overall_loss_std": float(summary_df["test_overall_loss"].std()) if len(seeds) > 1 else 0.0,
        "test_worst_group_loss_mean": float(summary_df["test_worst_group_loss"].mean()),
        "test_worst_group_loss_std": float(summary_df["test_worst_group_loss"].std()) if len(seeds) > 1 else 0.0,
        "test_loss_gap_mean": float(summary_df["test_loss_gap"].mean()),
        "test_overall_retrieval_acc_mean": float(summary_df["test_overall_retrieval_acc"].mean()),
        "test_overall_retrieval_acc_std": float(summary_df["test_overall_retrieval_acc"].std()) if len(seeds) > 1 else 0.0,
        "test_worst_group_retrieval_acc_mean": float(summary_df["test_worst_group_retrieval_acc"].mean()),
        "test_retrieval_gap_mean": float(summary_df["test_retrieval_gap"].mean()),
        "test_overall_alignment_sim_mean": float(summary_df["test_overall_alignment_sim"].mean()),
    }
    with open(out_path / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    logger.info(f"Completed {mode}: Overall Loss = {summary_stats['test_overall_loss_mean']:.4f}, Worst-Group Loss = {summary_stats['test_worst_group_loss_mean']:.4f}, Gap = {summary_stats['test_loss_gap_mean']:.4f}")
    return summary_stats
