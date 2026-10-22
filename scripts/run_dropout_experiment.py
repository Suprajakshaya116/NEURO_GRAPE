"""Neuro-GRAPE Controlled Dropout Experiment (Member 2 — Phase 4).

Executes a single controlled 50-epoch experiment testing spatial/channel
dropout (p=0.2) against the normalized baseline on EEG, MEG, and fMRI encoders.

Evaluates in-memory restored best validation weights on the 4 validation stimuli
across all six cross-modal retrieval directions and embedding-collapse metrics.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import NeuroGrapeDataset
from src.models.training import prepare_cross_modal_batch, train_model


def run_experiment() -> None:
    manifest_path = Path("data/manifests/data_inventory.csv")
    if not manifest_path.exists():
        manifest_path = Path("data_inventory.csv")

    print("=" * 72)
    print("NEURO-GRAPE CONTROLLED DROPOUT EXPERIMENT (dropout=0.2)")
    print("=" * 72)
    print("Configuration:")
    print("  Epochs:             50")
    print("  Learning Rate:      1e-4")
    print("  Weight Decay:       1e-2")
    print("  Temperature:        0.07")
    print("  Random Seed:        42")
    print("  Normalization:      Enabled (TrimodalNormalizer, train statistics only)")
    print("  Dropout:            0.2 (nn.Dropout1d for EEG/MEG, nn.Dropout3d for fMRI)")
    print("  Validation Restore: In-memory checkpointing of lowest val_loss state")
    print("  Manifest:          ", manifest_path)
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
        dropout=0.2,
    )
    runtime = time.time() - start_time

    history: Dict[str, List] = res["history"]
    best_epoch: int = res["best_epoch"]
    best_val_loss: float = res["best_val_loss"]
    encoders = res["encoders"]
    normalizer = res["normalizer"]

    print(f"\nExecution Complete:")
    print(f"  Runtime:                {runtime:.2f} s")
    print(f"  Best Validation Epoch:  {best_epoch}")
    print(f"  Best Validation Loss:   {best_val_loss:.6f}")
    print("-" * 72)
    print(f"{'Epoch':>5} | {'Train Loss':>14} | {'Val Loss':>14} | {'Notes'}")
    print("-" * 72)
    for ep, tr, va in zip(history["epoch"], history["train_loss"], history["val_loss"]):
        marker = " <-- Best Val (Restored)" if ep == best_epoch else ""
        print(f"{ep:>5} | {tr:>14.6f} | {va:>14.6f} |{marker}")
    print("-" * 72)

    # In-memory evaluation on validation split (N=4) only
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

    stimuli = val_ids["eeg"]
    print("\nValidation Cohort (N=4 stimuli):", stimuli)

    print("\n" + "=" * 72)
    print("EMBEDDING COLLAPSE DIAGNOSTICS (RESTORED BEST MODEL)")
    print("=" * 72)
    for mod in ["eeg", "meg", "fmri"]:
        emb = z[mod]
        sim_mat = torch.mm(emb, emb.t()).cpu().numpy()
        norms = torch.norm(emb, p=2, dim=1).cpu().numpy()

        n = emb.shape[0]
        off_diag_vals = [sim_mat[i, j] for i in range(n) for j in range(n) if i != j]
        mean_off_diag = sum(off_diag_vals) / len(off_diag_vals)

        diff_mat = torch.abs(emb.unsqueeze(1) - emb.unsqueeze(0))
        max_elem_diff = torch.max(diff_mat).item()

        print(f"\nModality: {mod.upper()}")
        print(f"  L2 Norms:                     {norms}")
        print(f"  Mean Off-Diagonal Cosine Sim: {mean_off_diag:.6f}")
        print(f"  Max Elementwise Difference:   {max_elem_diff:.6e}")
        print("  Cosine Similarity Matrix:")
        for row in sim_mat:
            print("   ", [f"{v:+.4f}" for v in row])

    print("\n" + "=" * 72)
    print("CROSS-MODAL RETRIEVAL ACCURACY (RESTORED BEST MODEL)")
    print("=" * 72)
    directions: List[Tuple[str, str]] = [
        ("eeg", "meg"),
        ("meg", "eeg"),
        ("eeg", "fmri"),
        ("fmri", "eeg"),
        ("meg", "fmri"),
        ("fmri", "meg"),
    ]

    retrieval_results = {}
    for q_mod, t_mod in directions:
        sim = torch.mm(z[q_mod], z[t_mod].t())
        preds = torch.argmax(sim, dim=1).cpu().tolist()
        correct = [p == i for i, p in enumerate(preds)]
        acc = sum(correct) / len(correct)
        retrieval_results[f"{q_mod.upper()} -> {t_mod.upper()}"] = (acc, preds)
        print(
            f"  {q_mod.upper():>4} -> {t_mod.upper():<4}: "
            f"Top-1 Accuracy = {acc * 100:>5.1f}% | "
            f"Predictions = {preds} (Targets: [0, 1, 2, 3])"
        )

    avg_acc = sum(acc for acc, _ in retrieval_results.values()) / len(retrieval_results)
    print(f"\nMean Cross-Modal Retrieval Accuracy: {avg_acc * 100:.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    run_experiment()
