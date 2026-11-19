"""Centralized, Leakage-Free Cross-Modal Retrieval Evaluation Suite.

Provides unified, rigorously validated evaluation routines for all models and baselines:
1. Strict query-gallery independence: query embeddings and gallery embeddings
   must be encoded separately without any cross-modal graph attention or joint mixing.
2. Supports bidirectional retrieval:
   - EEG -> MEG (Query: EEG, Gallery: MEG)
   - MEG -> EEG (Query: MEG, Gallery: EEG)
3. Computes standard evaluation metrics:
   - Top-1 retrieval accuracy
   - Top-2 retrieval accuracy
   - Balanced accuracy
   - Macro-F1 score
   - Worst-group accuracy and performance disparity gap across semantic groups
   - Expected Calibration Error (ECE)
   - Confusion matrices
4. Generates granular, traceable prediction-level records for audit and submission.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

ANIMATE_CATEGORIES = frozenset({"bear", "boa", "tadpole", "woman"})


def compute_expected_calibration_error(
    confidences: Sequence[float],
    accuracies: Sequence[float],
    n_bins: int = 5,
) -> float:
    """Compute binned Expected Calibration Error (ECE)."""
    confs = np.asarray(confidences, dtype=np.float32)
    accs = np.asarray(accuracies, dtype=np.float32)
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


def evaluate_clean_crossmodal_retrieval(
    query_embeddings: Mapping[str, torch.Tensor],
    gallery_embeddings: Mapping[str, torch.Tensor],
    stimulus_ids: Sequence[str],
    category_names: Sequence[str],
    custom_group_labels: Optional[Sequence[int]] = None,
    custom_group_names: Optional[Dict[int, str]] = None,
    temperature: float = 0.07,
    seed: int = 42,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Execute clean, leakage-free cross-modal retrieval evaluation.

    Args:
        query_embeddings: Dict mapping modality name ('eeg', 'meg') to (N, D) query tensors.
        gallery_embeddings: Dict mapping modality name ('eeg', 'meg') to (N, D) gallery tensors.
        stimulus_ids: Ordered sequence of N ground-truth stimulus IDs.
        category_names: Ordered sequence of N concept category strings.
        custom_group_labels: Optional (N,) integer array of group assignments.
        custom_group_names: Optional dict mapping group int to name string.
        temperature: Scaling temperature for softmax confidence computation.
        seed: Random seed for prediction tracing.
        model_name: Identifier for model logging.

    Returns:
        Dict containing comprehensive summary metrics and detailed prediction rows.
    """
    n = len(stimulus_ids)
    if n == 0:
        raise ValueError("Cannot evaluate empty stimulus sequence.")

    # Group assignment: custom or fallback to binary animacy
    if custom_group_labels is not None:
        group_labels = np.asarray(custom_group_labels, dtype=int)
    else:
        group_labels = np.array(
            [1 if str(cat).lower() in ANIMATE_CATEGORIES else 0 for cat in category_names],
            dtype=int,
        )

    directions = [("eeg", "meg"), ("meg", "eeg")]
    all_confs: List[float] = []
    all_correct: List[int] = []
    all_top5: List[int] = []
    all_top10: List[int] = []
    all_reciprocal_ranks: List[float] = []
    direction_top1: Dict[str, float] = {}
    direction_top2: Dict[str, float] = {}

    prediction_records: List[Dict[str, Any]] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for q_mod, g_mod in directions:
        z_q = query_embeddings[q_mod]  # (N, D)
        z_g = gallery_embeddings[g_mod]  # (N, D)

        if z_q.ndim != 2 or z_g.ndim != 2:
            raise ValueError(f"Embeddings must be 2-D (N, D); got {z_q.shape} and {z_g.shape}.")
        if z_q.shape[0] != n or z_g.shape[0] != n:
            raise ValueError(f"Batch dimension mismatch: expected {n}, got {z_q.shape[0]}, {z_g.shape[0]}.")

        # Cosine similarity matrix: query i vs gallery j
        sim_matrix = torch.mm(z_q, z_g.t())  # (N, N)
        prob_matrix = F.softmax(sim_matrix / temperature, dim=1)

        sim_np = sim_matrix.detach().cpu().numpy()
        prob_np = prob_matrix.detach().cpu().numpy()

        preds = np.argmax(sim_np, axis=1)
        targets = np.arange(n)
        top1_correct = (preds == targets).astype(int)

        # Top-k and MRR retrieval
        sorted_indices = np.argsort(-sim_np, axis=1)
        top2_k = min(2, n)
        top5_k = min(5, n)
        top10_k = min(10, n)
        top2_preds = sorted_indices[:, :top2_k]

        top2_correct = np.array([targets[i] in sorted_indices[i, :top2_k] for i in range(n)], dtype=int)
        top5_correct = np.array([targets[i] in sorted_indices[i, :top5_k] for i in range(n)], dtype=int)
        top10_correct = np.array([targets[i] in sorted_indices[i, :top10_k] for i in range(n)], dtype=int)
        ranks = np.array([int(np.where(sorted_indices[i] == targets[i])[0][0]) + 1 for i in range(n)])
        reciprocal_ranks = 1.0 / ranks

        dir_key = f"{q_mod}_to_{g_mod}"
        direction_top1[dir_key] = float(np.mean(top1_correct))
        direction_top2[dir_key] = float(np.mean(top2_correct))

        confs = np.max(prob_np, axis=1)
        all_confs.extend(confs.tolist())
        all_correct.extend(top1_correct.tolist())
        all_top5.extend(top5_correct.tolist())
        all_top10.extend(top10_correct.tolist())
        all_reciprocal_ranks.extend(reciprocal_ranks.tolist())

        y_true_all.extend(targets.tolist())
        y_pred_all.extend(preds.tolist())

        for i in range(n):
            sorted_sims = np.sort(sim_np[i])[::-1]
            top1_sim = float(sorted_sims[0])
            top2_sim = float(sorted_sims[1]) if len(sorted_sims) > 1 else top1_sim
            correct_flag = int(top1_correct[i])

            r1_idx = int(sorted_indices[i, 0])
            r2_idx = int(sorted_indices[i, 1]) if n > 1 else r1_idx

            g_val = int(group_labels[i])
            g_name = custom_group_names.get(g_val, f"Group_{g_val}") if custom_group_names else ("Animate" if g_val == 1 else "Inanimate")

            prediction_records.append({
                "model_name": model_name,
                "seed": seed,
                "sample_id": f"{q_mod}_{stimulus_ids[i]}",
                "query_id": f"query_{q_mod}_{i}",
                "query_modality": q_mod,
                "gallery_modality": g_mod,
                "target_modality": g_mod,
                "true_label": category_names[i],
                "predicted_label": category_names[preds[i]],
                "true_concept": category_names[i],
                "predicted_concept": category_names[preds[i]],
                "rank_1": category_names[r1_idx],
                "rank_2": category_names[r2_idx],
                "true_target_idx": int(targets[i]),
                "predicted_target_idx": int(preds[i]),
                "retrieval_rank": int(ranks[i]),
                "is_correct": correct_flag,
                "correct": correct_flag,
                "confidence": float(confs[i]),
                "top1_similarity": top1_sim,
                "top2_similarity": top2_sim,
                "group_id": g_val,
                "group": g_val,
                "group_name": g_name,
                "split": "test",
            })

    overall_top1 = float(np.mean(all_correct))
    overall_top2 = float(np.mean(list(direction_top2.values())))
    overall_top5 = float(np.mean(all_top5))
    overall_top10 = float(np.mean(all_top10))
    mrr = float(np.mean(all_reciprocal_ranks))
    ece = compute_expected_calibration_error(all_confs, all_correct, n_bins=5)

    # Balanced accuracy and Macro-F1
    balanced_acc = float(balanced_accuracy_score(y_true_all, y_pred_all))
    macro_f1 = float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    conf_mat = confusion_matrix(y_true_all, y_pred_all, labels=list(range(n))).tolist()

    # Per-group accuracies across all evaluated directional queries
    pred_df = pd.DataFrame(prediction_records)
    group_accs: Dict[str, float] = {}
    unique_groups = sorted(list(np.unique(group_labels)))
    for g_val in unique_groups:
        g_sub = pred_df[pred_df["group"] == g_val]
        g_name = custom_group_names.get(g_val, f"Group_{g_val}") if custom_group_names else ("Animate" if g_val == 1 else "Inanimate")
        if len(g_sub) > 0:
            group_accs[g_name] = float(g_sub["correct"].mean())
        else:
            group_accs[g_name] = overall_top1

    worst_group_acc = min(group_accs.values())
    mean_group_acc = float(np.mean(list(group_accs.values())))
    disparity_gap = max(group_accs.values()) - min(group_accs.values())

    return {
        "model_name": model_name,
        "seed": seed,
        "overall_top1": overall_top1,
        "overall_top2": overall_top2,
        "overall_top5": overall_top5,
        "overall_top10": overall_top10,
        "mrr": mrr,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "worst_group_accuracy": worst_group_acc,
        "mean_group_accuracy": mean_group_acc,
        "disparity_gap": disparity_gap,
        "group_accuracies": group_accs,
        "ece": ece,
        "direction_top1": direction_top1,
        "direction_top2": direction_top2,
        "confusion_matrix": conf_mat,
        "prediction_records": prediction_records,
    }
