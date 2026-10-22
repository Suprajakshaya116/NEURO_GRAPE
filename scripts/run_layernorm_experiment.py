"""Neuro-GRAPE Controlled LayerNorm Experiment (Member 2 -- Phase 4).

50-epoch controlled experiment using the current LayerNorm encoders (dropout=0.0).

Baseline for comparison
-----------------------
The primary baseline is the run with the SAME hyperparameters but WITHOUT
LayerNorm (pre-fix encoders): seed=42, lr=1e-4, weight_decay=1e-2,
temperature=0.07, normalize_inputs=True, dropout=0.0.

The dropout=0.2 run (run_dropout_experiment.py) is a secondary comparison
point with a different regularisation setting.

Settings
--------
seed             = 42
lr               = 1e-4
weight_decay     = 1e-2
temperature      = 0.07
normalize_inputs = True
dropout          = 0.0
num_epochs       = 50
restore_best     = True

Data splits
-----------
Train  12 stimuli  gradient updates and normalizer fitting only
Val     4 stimuli  evaluation and best-checkpoint selection only
Test    4 stimuli  NOT loaded, NOT accessed

Output
------
Best val loss and epoch, per-epoch history, all six retrieval directions,
per-modality mean off-diagonal cosine similarity, similarity-matrix row
variance. No disk checkpoints written. Source data not modified.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.dataset import NeuroGrapeDataset
from src.models.training import prepare_cross_modal_batch, train_model


# ---------------------------------------------------------------------------
# Pre-flight guard: verify test split is absent from training pipeline
# ---------------------------------------------------------------------------

def _assert_no_test_split_in_training() -> None:
    """Read training.py and assert split='test' is never referenced."""
    src = (PROJECT_ROOT / "src" / "models" / "training.py").read_text(encoding="utf-8")
    forbidden = ['split="test"', "split='test'"]
    for token in forbidden:
        if token in src:
            raise RuntimeError(
                f"ABORT: training.py contains '{token}'. "
                "Test split must never be loaded during training or validation."
            )
    print("[PRE-CHECK] training.py: test split confirmed NOT accessed.")


# ---------------------------------------------------------------------------
# Retrieval evaluation (stimulus-ID-keyed)
# ---------------------------------------------------------------------------

def compute_retrieval_accuracy(
    z_query: torch.Tensor,
    z_target: torch.Tensor,
    query_ids: List[str],
    target_ids: List[str],
) -> Tuple[float, List[int], List[int]]:
    """Top-1 cross-modal retrieval accuracy keyed on stimulus IDs.

    Rows of z_query correspond to query_ids; rows of z_target to target_ids.
    The ground-truth for query i is the row in z_target whose stimulus_id
    matches query_ids[i]. This is safe even if the two ID lists are not in
    the same order (they should be, but we verify).

    Returns
    -------
    accuracy   : fraction of queries correctly retrieved (0.0 -- 1.0)
    predictions: list of predicted target row indices (argmax of sim row)
    ground_truth: list of correct target row indices
    """
    # Build ground-truth index mapping
    target_index = {sid: idx for idx, sid in enumerate(target_ids)}
    ground_truth = [target_index[sid] for sid in query_ids]

    sim = torch.mm(z_query, z_target.t())
    preds = torch.argmax(sim, dim=1).cpu().tolist()
    correct = sum(p == g for p, g in zip(preds, ground_truth))
    return correct / len(ground_truth), preds, ground_truth


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> None:
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "data_inventory.csv"
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "data_inventory.csv"

    _assert_no_test_split_in_training()

    print("=" * 72)
    print("NEURO-GRAPE CONTROLLED LAYERNORM EXPERIMENT (dropout=0.0)")
    print("=" * 72)
    print("Configuration:")
    print("  Epochs:             50")
    print("  Learning Rate:      1e-4")
    print("  Weight Decay:       1e-2")
    print("  Temperature:        0.07")
    print("  Random Seed:        42")
    print("  Normalization:      Enabled (TrimodalNormalizer, train statistics only)")
    print("  Dropout:            0.0 (no Dropout layers active)")
    print("  LayerNorm:          nn.LayerNorm(128) after proj, before F.normalize")
    print("  Best-weight restore: In-memory only (no files written)")
    print("  Test split:         NOT accessed")
    print("  Manifest:          ", manifest_path)
    print()
    print("Primary baseline:   dropout=0.0, NO LayerNorm (pre-fix encoders),")
    print("                    otherwise identical settings.")
    print("Secondary baseline: dropout=0.2, NO LayerNorm (run_dropout_experiment.py).")
    print("=" * 72)

    start_time = time.time()
    res = train_model(
        manifest_path=manifest_path,
        num_epochs=50,
        lr=1e-4,
        weight_decay=1e-2,
        temperature=0.07,
        seed=42,
        restore_best_weights=True,
        normalize_inputs=True,
        dropout=0.0,
    )
    runtime = time.time() - start_time

    history: Dict[str, List] = res["history"]
    best_epoch: int = res["best_epoch"]
    best_val_loss: float = res["best_val_loss"]
    encoders = res["encoders"]
    normalizer = res["normalizer"]

    # Per-epoch table
    print(f"\nTraining complete -- runtime {runtime:.1f} s")
    print(f"Best Validation Epoch:  {best_epoch}")
    print(f"Best Validation Loss:   {best_val_loss:.6f}")
    print("-" * 72)
    print(f"{'Epoch':>5} | {'Train Loss':>14} | {'Val Loss':>14} | Notes")
    print("-" * 72)
    for ep, tr, va in zip(
        history["epoch"], history["train_loss"], history["val_loss"]
    ):
        mark = " <-- Best Val (Restored)" if ep == best_epoch else ""
        print(f"{ep:>5} | {tr:>14.6f} | {va:>14.6f} |{mark}")
    print("-" * 72)

    # Build validation batch with restored best weights
    val_ds = NeuroGrapeDataset(manifest=manifest_path, split="val", as_torch=True)
    device = next(encoders["eeg"].parameters()).device
    val_batch, val_ids = prepare_cross_modal_batch(val_ds, device=device)
    if normalizer is not None:
        val_batch = normalizer.transform(val_batch)

    for enc in encoders.values():
        enc.eval()

    with torch.no_grad():
        z = {
            "eeg": encoders["eeg"](val_batch["eeg"]),
            "meg": encoders["meg"](val_batch["meg"]),
            "fmri": encoders["fmri"](val_batch["fmri"]),
        }

    print("\nValidation Cohort (N=4):", val_ids["eeg"])

    # Embedding collapse diagnostics
    print("\n" + "=" * 72)
    print("EMBEDDING COLLAPSE DIAGNOSTICS (RESTORED BEST MODEL)")
    print("=" * 72)
    collapse_metrics: Dict[str, Dict[str, float]] = {}
    for mod in ["eeg", "meg", "fmri"]:
        emb = z[mod]
        sim_mat = torch.mm(emb, emb.t()).cpu().numpy()
        norms = torch.norm(emb, p=2, dim=1).cpu().numpy()
        n = emb.shape[0]
        off_diag = [sim_mat[i, j] for i in range(n) for j in range(n) if i != j]
        mean_off_diag = float(np.mean(off_diag))
        row_var = float(np.mean([np.var(sim_mat[i]) for i in range(n)]))
        collapse_metrics[mod] = {
            "mean_off_diag": mean_off_diag,
            "row_var": row_var,
        }
        max_diff = float(
            torch.max(torch.abs(emb.unsqueeze(1) - emb.unsqueeze(0))).item()
        )
        print(f"\nModality: {mod.upper()}")
        print(f"  L2 Norms:                       {norms}")
        print(f"  Mean Off-Diagonal Cosine Sim:   {mean_off_diag:.6f}")
        print(f"  Similarity Matrix Row Variance: {row_var:.6f}")
        print(f"  Max Elementwise Difference:     {max_diff:.6e}")
        print("  Cosine Similarity Matrix:")
        for row in sim_mat:
            print("   ", [f"{v:+.4f}" for v in row])

    # Cross-modal retrieval (stimulus-ID-keyed)
    print("\n" + "=" * 72)
    print("CROSS-MODAL RETRIEVAL ACCURACY (RESTORED BEST MODEL)")
    print("(Retrieval target keyed by stimulus_id, not row index)")
    print("=" * 72)
    directions: List[Tuple[str, str]] = [
        ("eeg", "meg"),
        ("meg", "eeg"),
        ("eeg", "fmri"),
        ("fmri", "eeg"),
        ("meg", "fmri"),
        ("fmri", "meg"),
    ]
    retrieval_results: Dict[str, float] = {}
    for q_mod, t_mod in directions:
        acc, preds, gt = compute_retrieval_accuracy(
            z_query=z[q_mod],
            z_target=z[t_mod],
            query_ids=val_ids[q_mod],
            target_ids=val_ids[t_mod],
        )
        label = f"{q_mod.upper()} -> {t_mod.upper()}"
        retrieval_results[label] = acc
        print(
            f"  {q_mod.upper():>4} -> {t_mod.upper():<4}: "
            f"Top-1 = {acc * 100:>5.1f}% | "
            f"Preds={preds}  GT={gt}"
        )
    avg_acc = float(np.mean(list(retrieval_results.values())))
    print(f"\nMean Cross-Modal Retrieval Accuracy: {avg_acc * 100:.1f}%")
    print("=" * 72)

    # Primary baseline numbers (historical 50-epoch run: dropout=0.0, no LayerNorm)
    baseline_primary = {
        "best_epoch": 7,
        "best_val_loss": 1.387684,
        "mean_acc": 0.250,
        "retrieval": {
            "EEG -> MEG": 0.250,
            "MEG -> EEG": 0.250,
            "EEG -> FMRI": 0.500,
            "FMRI -> EEG": 0.250,
            "MEG -> FMRI": 0.000,
            "FMRI -> MEG": 0.250,
        },
        "off_diag": {
            "eeg": 0.968049,
            "meg": 0.933092,
            "fmri": 0.997968,
        },
        "row_var": {
            "eeg": 0.000311,
            "meg": 0.001555,
            "fmri": 0.000003,
        },
    }

    # Summary table comparing against primary baseline
    print("\n" + "=" * 72)
    print("COMPARISON: Current LayerNorm vs Primary Baseline (dropout=0.0, No LayerNorm)")
    print("=" * 72)
    print(f"{'Metric':<36} | {'Baseline (No LN)':<16} | {'LayerNorm Run':<16} | {'Delta':<10}")
    print("-" * 85)
    print(f"{'Best Val Loss':<36} | {baseline_primary['best_val_loss']:<16.6f} | {best_val_loss:<16.6f} | {best_val_loss - baseline_primary['best_val_loss']:+.6f}")
    print(f"{'Best Val Epoch':<36} | {baseline_primary['best_epoch']:<16} | {best_epoch:<16} | {best_epoch - baseline_primary['best_epoch']:+d}")
    print(f"{'Mean Retrieval Accuracy':<36} | {baseline_primary['mean_acc']*100:<15.1f}% | {avg_acc*100:<15.1f}% | {(avg_acc - baseline_primary['mean_acc'])*100:+.1f}%")
    print("-" * 85)
    for label, acc in retrieval_results.items():
        base_acc = baseline_primary["retrieval"][label]
        print(f"  {label:<34} | {base_acc*100:<15.1f}% | {acc*100:<15.1f}% | {(acc - base_acc)*100:+.1f}%")
    print("-" * 85)
    for mod in ["eeg", "meg", "fmri"]:
        base_od = baseline_primary["off_diag"][mod]
        cur_od = collapse_metrics[mod]["mean_off_diag"]
        print(f"  {mod.upper() + ' Mean Off-Diag Cosine':<34} | {base_od:<16.6f} | {cur_od:<16.6f} | {cur_od - base_od:+.6f}")
    print("-" * 85)
    for mod in ["eeg", "meg", "fmri"]:
        base_rv = baseline_primary["row_var"][mod]
        cur_rv = collapse_metrics[mod]["row_var"]
        print(f"  {mod.upper() + ' Sim Row Variance':<34} | {base_rv:<16.6f} | {cur_rv:<16.6f} | {cur_rv - base_rv:+.6f}")
    print("=" * 72)
    print("Note: The dropout=0.2 experiment serves as a separate comparison point.")
    print("Improvement requires: lower best val loss AND higher retrieval accuracy.")
    print("Do not claim improvement unless the measurements support it.")
    print("=" * 72)


if __name__ == "__main__":
    run_experiment()
