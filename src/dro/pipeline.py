"""Reproducible End-to-End Pipeline for Neuro-GRAPE Member 3.

Coordinates:
1. Loading exported embeddings & metadata (with dynamic dimensionality and sample count).
2. Group assignment via flexible strategy:
   - "cluster": Unsupervised K-Means clustering fitted strictly on training split.
   - "modality": Grouping by neural recording modality (EEG, MEG, fMRI).
   - "metadata": Predefined metadata groupings (e.g., category_name or category_id).
3. Validating trimodal stimulus alignment with safe handling of missing modalities.
4. Executing GroupDRO weight updates and tracking group losses (simulation on fixed embeddings).
5. Computing overall, per-group, and worst-group evaluation metrics across splits.
6. Exporting reproducible artifacts:
   - metrics.csv (summary metrics across splits)
   - group_metrics.csv (granular long-format group breakdown)
   - group_losses.csv
   - group_weights.csv
   - group_assignments.parquet & group_assignments.csv
   - clustering_diagnostics.json
   - experiment_config.yaml & pipeline_summary.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml

from src.dro.clustering import EmbeddingClustering, fit_regime_clustering
from src.dro.evaluation import GroupEvaluator
from src.dro.group_dro import GroupDRO
from src.models.losses import multimodal_contrastive_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_pipeline_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load declarative pipeline YAML configuration."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def run_member3_pipeline(
    config_or_path: Union[str, Path, Dict[str, Any]] = "configs/groupdro_config.yaml",
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute the complete Member 3 GroupDRO, clustering, and evaluation pipeline.

    Parameters
    ----------
    config_or_path : str, Path, or dict
        Path to YAML configuration or dictionary containing parameters.
    output_dir : str or Path, optional
        Directory where outputs and artifacts are saved. Overrides config value if set.

    Returns
    -------
    pipeline_outputs : dict
        Paths to generated artifacts and key summary metrics.
    """
    if isinstance(config_or_path, (str, Path)):
        cfg = load_pipeline_config(config_or_path)
    elif isinstance(config_or_path, dict):
        cfg = config_or_path.copy()
    else:
        raise TypeError(f"Invalid configuration input type: {type(config_or_path)}")

    # 1. Resolve output directory
    paths_cfg = cfg.get("paths", {})
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(paths_cfg.get("output_dir", "reports/artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initialized output directory: {out_dir}")

    # 2. Load embeddings and metadata
    metadata_path = Path(paths_cfg.get("metadata_csv", "embeddings/metadata.csv"))
    embeddings_pt = Path(paths_cfg.get("embeddings_pt", "embeddings/embeddings.pt"))

    if not metadata_path.exists():
        raise FileNotFoundError(f"Embeddings metadata not found at: {metadata_path}")
    if not embeddings_pt.exists():
        raise FileNotFoundError(f"Embeddings PyTorch tensor not found at: {embeddings_pt}")

    meta_df = pd.read_csv(metadata_path)
    embeddings_dict = torch.load(embeddings_pt, map_location="cpu")
    logger.info(f"Loaded {len(meta_df)} metadata records and {len(embeddings_dict)} embedding vectors.")

    # Auto-detect embedding dimension
    sample_key = next(iter(embeddings_dict.keys()))
    sample_vec = embeddings_dict[sample_key]
    detected_dim = sample_vec.shape[-1] if hasattr(sample_vec, "shape") else len(sample_vec)
    logger.info(f"Detected embedding dimensionality: {detected_dim}")

    # 3. Determine Grouping Strategy
    grouping_cfg = cfg.get("grouping", {})
    grouping_strategy = grouping_cfg.get("strategy", "cluster").lower()
    clust_cfg = cfg.get("clustering", {})
    n_clusters = int(clust_cfg.get("n_clusters", 2))
    random_state = int(clust_cfg.get("random_state", 42))
    expected_dim = clust_cfg.get("expected_dim", detected_dim)

    diagnostics_path: Optional[Path] = None
    group_names: List[str] = []

    if grouping_strategy == "cluster":
        logger.info(f"Fitting KMeans clustering (K={n_clusters}) strictly on training split...")
        clustering = EmbeddingClustering(
            n_clusters=n_clusters,
            random_state=random_state,
            expected_dim=expected_dim,
        )
        clustering.fit(embeddings_dict, meta_df)
        assignments_df = clustering.assign_clusters(embeddings_dict, meta_df)
        assignments_df["group_name"] = assignments_df["cluster_id"].apply(lambda c: f"Cluster_{c}")
        group_names = [f"Cluster_{c}" for c in range(n_clusters)]
        num_groups = n_clusters

        # Save clustering diagnostics
        diagnostics_path = out_dir / "clustering_diagnostics.json"
        clustering.save_diagnostics(diagnostics_path)

    elif grouping_strategy == "modality":
        logger.info("Using modality groupings (eeg, meg, fmri)...")
        assignments_df = meta_df.copy()
        unique_mods = sorted(list(assignments_df["modality"].str.lower().unique()))
        mod_to_id = {m: i for i, m in enumerate(unique_mods)}
        assignments_df["group_id"] = assignments_df["modality"].str.lower().map(mod_to_id)
        assignments_df["cluster_id"] = assignments_df["group_id"]
        assignments_df["group_name"] = assignments_df["modality"].str.lower()
        group_names = unique_mods
        num_groups = len(unique_mods)

    elif grouping_strategy == "metadata":
        meta_col = grouping_cfg.get("metadata_column", "category_name")
        if meta_col not in meta_df.columns:
            raise ValueError(f"Requested grouping metadata column '{meta_col}' not found in metadata.")
        logger.info(f"Using metadata column '{meta_col}' for grouping...")
        assignments_df = meta_df.copy()
        unique_vals = sorted(list(assignments_df[meta_col].astype(str).unique()))
        val_to_id = {v: i for i, v in enumerate(unique_vals)}
        assignments_df["group_id"] = assignments_df[meta_col].astype(str).map(val_to_id)
        assignments_df["cluster_id"] = assignments_df["group_id"]
        assignments_df["group_name"] = assignments_df[meta_col].astype(str)
        group_names = unique_vals
        num_groups = len(unique_vals)
    else:
        raise ValueError(
            f"Unsupported grouping strategy: '{grouping_strategy}'. Choose 'cluster', 'modality', or 'metadata'."
        )

    # 4. Save group assignments (parquet and csv)
    parquet_path = out_dir / "group_assignments.parquet"
    csv_path = out_dir / "group_assignments.csv"
    try:
        assignments_df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved group assignments to {parquet_path}")
    except Exception as e:
        logger.warning(f"Could not write parquet format ({e}). Proceeding with CSV only.")
        parquet_path = None

    assignments_df.to_csv(csv_path, index=False)
    logger.info(f"Saved group assignments to {csv_path}")

    # 5. Compute per-sample crossmodal contrastive losses and cosine similarities
    dro_cfg = cfg.get("group_dro", {})
    dro_lr = float(dro_cfg.get("learning_rate", 0.05))
    dro_smoothing = float(dro_cfg.get("smoothing", 0.01))
    dro_steps = int(dro_cfg.get("num_steps", 10))
    temperature = float(cfg.get("evaluation", {}).get("temperature", 0.07))

    splits = ["train", "val", "test"]
    split_sample_data: Dict[str, Dict[str, Any]] = {}

    for sp in splits:
        sp_meta = assignments_df[assignments_df["split"].str.lower() == sp].copy()
        if len(sp_meta) == 0:
            continue

        stims = sorted(list(sp_meta["stimulus_id"].unique()))
        valid_stims, eeg_sids, meg_sids, fmri_sids = [], [], [], []

        has_fmri = "fmri" in sp_meta["modality"].str.lower().values

        for s in stims:
            e_sub = sp_meta[(sp_meta["stimulus_id"] == s) & (sp_meta["modality"].str.lower() == "eeg")]
            m_sub = sp_meta[(sp_meta["stimulus_id"] == s) & (sp_meta["modality"].str.lower() == "meg")]
            f_sub = sp_meta[(sp_meta["stimulus_id"] == s) & (sp_meta["modality"].str.lower() == "fmri")]

            if len(e_sub) > 0 and len(m_sub) > 0:
                if has_fmri and len(f_sub) == 0:
                    continue
                valid_stims.append(s)
                eeg_sids.append(e_sub.iloc[0]["sample_id"])
                meg_sids.append(m_sub.iloc[0]["sample_id"])
                if has_fmri:
                    fmri_sids.append(f_sub.iloc[0]["sample_id"])

        if not valid_stims:
            logger.warning(f"Split '{sp}' has no stimuli with complete modal coverage; skipping.")
            continue

        z_eeg = torch.stack([embeddings_dict[sid] for sid in eeg_sids])
        z_meg = torch.stack([embeddings_dict[sid] for sid in meg_sids])

        # Pairwise symmetric contrastive losses
        l_em = multimodal_contrastive_loss(z_eeg, z_meg, valid_stims, valid_stims, temperature=temperature, reduction="per_example")
        cos_em = (z_eeg * z_meg).sum(dim=1)

        if has_fmri and len(fmri_sids) == len(valid_stims):
            z_fmri = torch.stack([embeddings_dict[sid] for sid in fmri_sids])
            l_ef = multimodal_contrastive_loss(z_eeg, z_fmri, valid_stims, valid_stims, temperature=temperature, reduction="per_example")
            l_mf = multimodal_contrastive_loss(z_meg, z_fmri, valid_stims, valid_stims, temperature=temperature, reduction="per_example")
            l_sample_eeg = 0.5 * (l_em + l_ef)
            l_sample_meg = 0.5 * (l_em + l_mf)
            l_sample_fmri = 0.5 * (l_ef + l_mf)

            cos_ef = (z_eeg * z_fmri).sum(dim=1)
            cos_mf = (z_meg * z_fmri).sum(dim=1)
            cos_sample_eeg = 0.5 * (cos_em + cos_ef)
            cos_sample_meg = 0.5 * (cos_em + cos_mf)
            cos_sample_fmri = 0.5 * (cos_ef + cos_mf)

            ordered_sids = eeg_sids + meg_sids + fmri_sids
            all_losses = torch.cat([l_sample_eeg, l_sample_meg, l_sample_fmri]).detach().cpu()
            all_cosines = torch.cat([cos_sample_eeg, cos_sample_meg, cos_sample_fmri]).detach().cpu()
        else:
            ordered_sids = eeg_sids + meg_sids
            all_losses = torch.cat([l_em, l_em]).detach().cpu()
            all_cosines = torch.cat([cos_em, cos_em]).detach().cpu()

        sid_to_group = dict(zip(sp_meta["sample_id"], sp_meta["group_id"]))
        all_groups = torch.tensor([sid_to_group[sid] for sid in ordered_sids], dtype=torch.long)

        split_sample_data[sp] = {
            "sample_ids": ordered_sids,
            "losses": all_losses,
            "cosines": all_cosines,
            "group_ids": all_groups,
            "stimuli_count": len(valid_stims),
        }

    # 6. Execute GroupDRO weight updates on Training Split (Simulation Mode on Fixed Embeddings)
    logger.info("Executing GroupDRO weight updates on training split (Simulation Mode)...")
    group_dro = GroupDRO(
        num_groups=num_groups,
        learning_rate=dro_lr,
        smoothing=dro_smoothing,
        group_names=group_names,
    )

    train_data = split_sample_data.get("train")
    if train_data is None:
        raise RuntimeError("No valid training samples found for GroupDRO execution.")

    train_losses = train_data["losses"].float()
    train_groups = train_data["group_ids"].long()

    weights_history = []
    losses_history = []

    for step in range(1, dro_steps + 1):
        g_losses, g_counts = group_dro.compute_group_losses(train_losses, train_groups)
        weights = group_dro.update_weights(g_losses, g_counts)

        for g_idx in range(num_groups):
            g_name = group_names[g_idx]
            weights_history.append({
                "step": step,
                "group_id": g_idx,
                "group_name": g_name,
                "weight": float(weights[g_idx].item()),
            })
            losses_history.append({
                "step": step,
                "group_id": g_idx,
                "group_name": g_name,
                "mean_loss": float(g_losses[g_idx].item()),
                "sample_count": int(g_counts[g_idx].item()),
            })

    group_weights_df = pd.DataFrame(weights_history)
    group_losses_df = pd.DataFrame(losses_history)

    group_weights_path = out_dir / "group_weights.csv"
    group_losses_path = out_dir / "group_losses.csv"
    group_weights_df.to_csv(group_weights_path, index=False)
    group_losses_df.to_csv(group_losses_path, index=False)
    logger.info(f"Saved GroupDRO weight logs to {group_weights_path}")
    logger.info(f"Saved GroupDRO loss logs to {group_losses_path}")

    # 7. Comprehensive Evaluation across Splits
    logger.info("Computing evaluation metrics across train/val/test splits...")
    evaluator = GroupEvaluator(min_group_size=1)
    all_metrics_rows = []
    granular_group_rows = []

    for sp in splits:
        if sp not in split_sample_data:
            continue
        sp_data = split_sample_data[sp]
        sp_losses = sp_data["losses"].numpy()
        sp_cos = sp_data["cosines"].numpy()
        sp_groups = sp_data["group_ids"].numpy()

        # Evaluate Contrastive Loss (lower is better -> higher_is_better=False)
        loss_res = evaluator.evaluate(
            values=sp_losses,
            group_ids=sp_groups,
            metric_name="crossmodal_contrastive_loss",
            higher_is_better=False,
            all_group_ids=list(range(num_groups)),
        )

        loss_row: Dict[str, Any] = {
            "split": sp,
            "metric_name": "crossmodal_contrastive_loss",
            "higher_is_better": False,
            "overall_value": loss_res.overall_value,
            "worst_group_id": loss_res.worst_group_id,
            "worst_group_name": group_names[loss_res.worst_group_id] if loss_res.worst_group_id is not None else None,
            "worst_group_value": loss_res.worst_group_value,
            "performance_gap": loss_res.performance_gap,
            "num_active_groups": len(loss_res.active_groups),
        }
        for g_idx in range(num_groups):
            loss_row[f"group_{g_idx}_value"] = loss_res.per_group_values.get(g_idx, np.nan)
            loss_row[f"group_{g_idx}_count"] = loss_res.group_counts.get(g_idx, 0)
        all_metrics_rows.append(loss_row)

        for g_df_row in loss_res.to_group_dataframe().to_dict(orient="records"):
            g_df_row["split"] = sp
            g_df_row["higher_is_better"] = False
            g_df_row["group_name"] = group_names[g_df_row["group_id"]]
            granular_group_rows.append(g_df_row)

        # Evaluate Cosine Similarity (higher is better -> higher_is_better=True)
        cos_res = evaluator.evaluate(
            values=sp_cos,
            group_ids=sp_groups,
            metric_name="crossmodal_cosine_similarity",
            higher_is_better=True,
            all_group_ids=list(range(num_groups)),
        )

        cos_row: Dict[str, Any] = {
            "split": sp,
            "metric_name": "crossmodal_cosine_similarity",
            "higher_is_better": True,
            "overall_value": cos_res.overall_value,
            "worst_group_id": cos_res.worst_group_id,
            "worst_group_name": group_names[cos_res.worst_group_id] if cos_res.worst_group_id is not None else None,
            "worst_group_value": cos_res.worst_group_value,
            "performance_gap": cos_res.performance_gap,
            "num_active_groups": len(cos_res.active_groups),
        }
        for g_idx in range(num_groups):
            cos_row[f"group_{g_idx}_value"] = cos_res.per_group_values.get(g_idx, np.nan)
            cos_row[f"group_{g_idx}_count"] = cos_res.group_counts.get(g_idx, 0)
        all_metrics_rows.append(cos_row)

        for g_df_row in cos_res.to_group_dataframe().to_dict(orient="records"):
            g_df_row["split"] = sp
            g_df_row["higher_is_better"] = True
            g_df_row["group_name"] = group_names[g_df_row["group_id"]]
            granular_group_rows.append(g_df_row)

    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    # Granular per-group long table
    group_metrics_df = pd.DataFrame(granular_group_rows)
    group_metrics_path = out_dir / "group_metrics.csv"
    group_metrics_df.to_csv(group_metrics_path, index=False)
    logger.info(f"Saved detailed group breakdown to {group_metrics_path}")

    # 8. Save configuration and run summary
    config_out_path = out_dir / "experiment_config.yaml"
    with open(config_out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    summary = {
        "status": "SUCCESS",
        "output_dir": str(out_dir),
        "execution_mode": "groupdro_weight_simulation_on_fixed_embeddings",
        "is_trained_model": False,
        "methodological_notice": (
            "GroupDRO weights were iteratively updated on fixed baseline embeddings "
            "to evaluate group risk and distributional adaptation without modifying model weights. "
            "This is a simulation benchmark on representations, not a newly trained model."
        ),
        "grouping_strategy": grouping_strategy,
        "num_groups": num_groups,
        "group_names": group_names,
        "detected_embedding_dim": detected_dim,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "group_metrics_csv": str(group_metrics_path),
            "group_losses_csv": str(group_losses_path),
            "group_weights_csv": str(group_weights_path),
            "group_assignments_parquet": str(parquet_path) if parquet_path else None,
            "group_assignments_csv": str(csv_path),
            "clustering_diagnostics_json": str(diagnostics_path) if diagnostics_path else None,
            "experiment_config_yaml": str(config_out_path),
        },
        "train_sample_count": len(train_data["losses"]),
        "final_group_weights": {
            name: float(w) for name, w in zip(group_names, group_dro.get_weights().tolist())
        },
    }

    summary_path = out_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Member 3 Pipeline completed successfully. Summary saved to {summary_path}")
    return summary
