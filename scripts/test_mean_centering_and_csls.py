"""
scripts/test_mean_centering_and_csls.py
=======================================
Evaluate post-processing retrieval strategies on pre-trained scaled models:
1. Baseline: Standard Cosine Similarity
2. Gallery Mean-Centering (subtract gallery centroid, re-normalize)
3. Dual Mean-Centering (subtract query & gallery centroids, re-normalize)
4. Train-Set Inductive Centering (subtract train centroids, re-normalize)
5. CSLS (Cross-domain Similarity Local Scaling / Hubness mitigation, k=5, k=10)

Evaluates on:
- GroupDRO_Cluster (best scaled model)
- Full_Neuro_GRAPE
- Bimodal_ERM
"""

import gc
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MeanCenteringTest")

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.model import TrimodalNeuroGrapeModel
from src.models.normalization import TrimodalNormalizer


def compute_csls_sim(sim_matrix: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Compute Cross-domain Similarity Local Scaling (CSLS) matrix.
    
    sim_matrix: (N, N) cosine similarity between queries (rows) and gallery (cols)
    CSLS(q, g) = 2 * cos(q, g) - mean_topk(q -> gallery) - mean_topk(g -> queries)
    """
    n_q, n_g = sim_matrix.shape
    k_g = min(k, n_g)
    k_q = min(k, n_q)
    
    # Average similarity of each query to its k nearest gallery neighbors
    topk_q_to_g, _ = torch.topk(sim_matrix, k_g, dim=1)
    r_q = topk_q_to_g.mean(dim=1, keepdim=True)  # (N, 1)
    
    # Average similarity of each gallery item to its k nearest query neighbors
    topk_g_to_q, _ = torch.topk(sim_matrix.t(), k_q, dim=1)
    r_g = topk_g_to_q.mean(dim=1, keepdim=True).t()  # (1, N)
    
    return 2.0 * sim_matrix - r_q - r_g


def evaluate_sim_matrix(sim_matrix: torch.Tensor, category_names: list, animacy_labels: np.ndarray, temperature: float = 0.07):
    """Compute retrieval metrics from a similarity matrix."""
    n = sim_matrix.shape[0]
    sim_np = sim_matrix.detach().cpu().numpy()
    
    targets = np.arange(n)
    preds = np.argmax(sim_np, axis=1)
    top1_correct = (preds == targets).astype(int)
    
    sorted_indices = np.argsort(-sim_np, axis=1)
    top2_correct = np.array([targets[i] in sorted_indices[i, :min(2, n)] for i in range(n)], dtype=int)
    top5_correct = np.array([targets[i] in sorted_indices[i, :min(5, n)] for i in range(n)], dtype=int)
    top10_correct = np.array([targets[i] in sorted_indices[i, :min(10, n)] for i in range(n)], dtype=int)
    
    ranks = np.array([int(np.where(sorted_indices[i] == targets[i])[0][0]) + 1 for i in range(n)])
    mrr = float(np.mean(1.0 / ranks))
    
    # Hubness metric: Robin Hood index or Skewness of k-occurrences (k=1)
    occ = np.bincount(preds, minlength=n)
    hubness_skew = float(np.mean(((occ - np.mean(occ)) / (np.std(occ) + 1e-8)) ** 3))
    max_hub = int(np.max(occ))  # highest number of times a single gallery item was retrieved
    
    # Group disparity
    unique_groups = np.unique(animacy_labels)
    group_accs = {}
    for g in unique_groups:
        mask = animacy_labels == g
        group_accs[int(g)] = float(np.mean(top1_correct[mask])) if np.sum(mask) > 0 else 0.0
        
    worst_group_acc = min(group_accs.values()) if group_accs else 0.0
    best_group_acc = max(group_accs.values()) if group_accs else 0.0
    disparity = best_group_acc - worst_group_acc
    
    return {
        "top1": float(np.mean(top1_correct)),
        "top2": float(np.mean(top2_correct)),
        "top5": float(np.mean(top5_correct)),
        "top10": float(np.mean(top10_correct)),
        "mrr": mrr,
        "worst_group_acc": worst_group_acc,
        "disparity_gap": disparity,
        "hubness_skew": hubness_skew,
        "max_hub_occurrences": max_hub,
    }


def main():
    data_dir = REPO_ROOT / "final" / "final"
    splits_csv = REPO_ROOT / "data" / "splits" / "scaled_splits.csv"
    
    if not splits_csv.exists() or not data_dir.exists():
        logger.error("Data or splits missing.")
        return

    logger.info("Loading metadata and splits...")
    df_meta = pd.read_csv(splits_csv)
    train_idx = np.where(df_meta["split"] == "train")[0]
    test_idx = np.where(df_meta["split"] == "test")[0]
    
    logger.info(f"Train trials: {len(train_idx)}, Test trials: {len(test_idx)}")
    
    test_meta = df_meta.iloc[test_idx].reset_index(drop=True)
    test_categories = test_meta["concept_name"].tolist()
    test_animacy = test_meta["animacy"].values
    
    logger.info("Loading preprocessed arrays...")
    eeg_raw = np.load(data_dir / "EEG_paired.npy", mmap_mode="r")
    meg_raw = np.load(data_dir / "MEG_paired.npy", mmap_mode="r")
    
    eeg_train = np.array(eeg_raw[train_idx], dtype=np.float32)
    meg_train = np.array(meg_raw[train_idx], dtype=np.float32)
    eeg_test = np.array(eeg_raw[test_idx], dtype=np.float32)
    meg_test = np.array(meg_raw[test_idx], dtype=np.float32)
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
    test_norm = normalizer.transform({
        "eeg": torch.from_numpy(eeg_test).float(),
        "meg": torch.from_numpy(meg_test).float(),
    })
    del eeg_train, meg_train, eeg_test, meg_test
    gc.collect()

    n_eeg_ch, n_eeg_t = test_norm["eeg"].shape[1], test_norm["eeg"].shape[2]
    n_meg_ch, n_meg_t = test_norm["meg"].shape[1], test_norm["meg"].shape[2]

    models_to_test = [
        ("GroupDRO_Cluster", REPO_ROOT / "reports" / "artifacts" / "scaled" / "GroupDRO_Cluster_best_model.pt"),
        ("Full_Neuro_GRAPE", REPO_ROOT / "checkpoints" / "best_scaled_neurogrape_model.pt"),
        ("Bimodal_ERM", REPO_ROOT / "reports" / "artifacts" / "scaled" / "Bimodal_ERM_best_model.pt"),
    ]

    results = []

    for model_name, ckpt_path in models_to_test:
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue
            
        logger.info(f"=== Evaluating {model_name} from {ckpt_path.name} ===")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_config = ckpt.get("config", {})
        use_graph = model_config.get("use_graph", True)
        dropout = model_config.get("dropout", 0.2)

        model = TrimodalNeuroGrapeModel(
            embed_dim=128,
            dropout=dropout,
            eeg_channels=n_eeg_ch,
            eeg_times=n_eeg_t,
            meg_channels=n_meg_ch,
            meg_times=n_meg_t,
            use_graph=use_graph,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Extract normalized embeddings in batches with apply_graph=False (zero-leakage)
        def encode_mod(tensors):
            model.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(tensors), 128):
                    b = tensors[i : i + 128]
                    # return_private=False, apply_graph=False
                    m_out = model.encode_modality(b, mod_name) if hasattr(model, "encode_modality") else None
            return out

        # Direct forward call per modality
        with torch.no_grad():
            train_embs = {}
            test_embs = {}
            for mod in ["eeg", "meg"]:
                # train
                tr_list = []
                for i in range(0, len(train_norm[mod]), 128):
                    b_x = {mod: train_norm[mod][i : i + 128]}
                    m_out = model(b_x, return_private=False, apply_graph=False)
                    tr_list.append(m_out[mod])
                train_embs[mod] = torch.cat(tr_list, dim=0)
                
                # test
                te_list = []
                for i in range(0, len(test_norm[mod]), 128):
                    b_x = {mod: test_norm[mod][i : i + 128]}
                    m_out = model(b_x, return_private=False, apply_graph=False)
                    te_list.append(m_out[mod])
                test_embs[mod] = torch.cat(te_list, dim=0)

        # For both retrieval directions: EEG -> MEG and MEG -> EEG
        directions = [("eeg", "meg"), ("meg", "eeg")]

        strategies = [
            "Baseline_Standard_Cosine",
            "Gallery_Mean_Centered",
            "Dual_Mean_Centered",
            "Train_Centering_Inductive",
            "CSLS_k5",
            "CSLS_k10",
        ]

        for strat in strategies:
            strat_metrics = []
            for q_mod, g_mod in directions:
                z_q = test_embs[q_mod]  # (N, D)
                z_g = test_embs[g_mod]  # (N, D)
                
                if strat == "Baseline_Standard_Cosine":
                    sim = torch.mm(z_q, z_g.t())
                elif strat == "Gallery_Mean_Centered":
                    mu_g = z_g.mean(dim=0, keepdim=True)
                    z_g_centered = F.normalize(z_g - mu_g, p=2, dim=1)
                    sim = torch.mm(z_q, z_g_centered.t())
                elif strat == "Dual_Mean_Centered":
                    mu_q = z_q.mean(dim=0, keepdim=True)
                    mu_g = z_g.mean(dim=0, keepdim=True)
                    z_q_centered = F.normalize(z_q - mu_q, p=2, dim=1)
                    z_g_centered = F.normalize(z_g - mu_g, p=2, dim=1)
                    sim = torch.mm(z_q_centered, z_g_centered.t())
                elif strat == "Train_Centering_Inductive":
                    mu_q_tr = train_embs[q_mod].mean(dim=0, keepdim=True)
                    mu_g_tr = train_embs[g_mod].mean(dim=0, keepdim=True)
                    z_q_centered = F.normalize(z_q - mu_q_tr, p=2, dim=1)
                    z_g_centered = F.normalize(z_g - mu_g_tr, p=2, dim=1)
                    sim = torch.mm(z_q_centered, z_g_centered.t())
                elif strat == "CSLS_k5":
                    base_sim = torch.mm(z_q, z_g.t())
                    sim = compute_csls_sim(base_sim, k=5)
                elif strat == "CSLS_k10":
                    base_sim = torch.mm(z_q, z_g.t())
                    sim = compute_csls_sim(base_sim, k=10)
                else:
                    raise ValueError(strat)

                m = evaluate_sim_matrix(sim, test_categories, test_animacy)
                strat_metrics.append(m)

            # Average over bidirectional directions
            avg_m = {
                "model": model_name,
                "strategy": strat,
                "top1_acc_pct": float(np.mean([x["top1"] for x in strat_metrics]) * 100),
                "top2_acc_pct": float(np.mean([x["top2"] for x in strat_metrics]) * 100),
                "top5_acc_pct": float(np.mean([x["top5"] for x in strat_metrics]) * 100),
                "top10_acc_pct": float(np.mean([x["top10"] for x in strat_metrics]) * 100),
                "mrr": float(np.mean([x["mrr"] for x in strat_metrics])),
                "worst_group_acc_pct": float(np.mean([x["worst_group_acc"] for x in strat_metrics]) * 100),
                "disparity_gap_pct": float(np.mean([x["disparity_gap"] for x in strat_metrics]) * 100),
                "hubness_skew": float(np.mean([x["hubness_skew"] for x in strat_metrics])),
                "max_hub_count": float(np.mean([x["max_hub_occurrences"] for x in strat_metrics])),
            }
            results.append(avg_m)
            logger.info(
                f"[{model_name}] {strat:<28} -> Top-1: {avg_m['top1_acc_pct']:.3f}%, "
                f"Top-5: {avg_m['top5_acc_pct']:.3f}%, MRR: {avg_m['mrr']:.4f}, "
                f"HubSkew: {avg_m['hubness_skew']:.2f}, MaxHub: {avg_m['max_hub_count']:.1f}"
            )

    df_res = pd.DataFrame(results)
    out_csv = REPO_ROOT / "reports" / "artifacts" / "scaled" / "mean_centering_and_csls_benchmark.csv"
    out_json = REPO_ROOT / "reports" / "artifacts" / "scaled" / "mean_centering_and_csls_benchmark.json"
    df_res.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {out_csv}")
    print("\n" + "=" * 95)
    print(df_res.to_string(index=False))
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
