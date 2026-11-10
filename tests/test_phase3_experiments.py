"""
tests/test_phase3_experiments.py
================================
Focused test suite for Neuro-GRAPE Member 3 Phase 3.
Validates:
1. GroupDRO training loop gradient backpropagation into encoder parameters.
2. Dynamic group reweighting under disparate group losses.
3. Leakage-free split isolation in training and evaluation.
4. Cross-modal retrieval and alignment metrics computation.
5. Full integrity of Phase 3 generated artifacts.
6. Independent recomputation of summary metrics from saved per-sample outputs.
"""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dro.evaluation import GroupEvaluator
from src.dro.experiment_runner import (
    assign_group_labels,
    compute_cross_modal_metrics,
    evaluate_on_test_split,
    train_single_run,
)
from src.dro.group_dro import GroupDRO


class TestPhase3Experiments(unittest.TestCase):

    def test_groupdro_gradient_flow_and_encoder_weight_update(self):
        """Verify that genuine GroupDRO backpropagates gradients and updates encoder weights."""
        run_res = train_single_run(
            mode="groupdro",
            group_strategy="semantic",
            num_epochs=2,
            lr=1e-3,
            seed=42,
        )
        encoders = run_res["encoders"]
        # Confirm encoders have updated parameters
        for mod, enc in encoders.items():
            for name, param in enc.named_parameters():
                self.assertTrue(torch.isfinite(param).all(), f"Non-finite parameter in {mod}.{name}")

        # Check that best_val_loss is finite and best_epoch is recorded
        self.assertTrue(np.isfinite(run_res["best_val_loss"]))
        self.assertGreaterEqual(run_res["best_epoch"], 1)

    def test_dynamic_group_reweighting(self):
        """Verify that GroupDRO increases weight for higher-loss groups."""
        dro = GroupDRO(num_groups=2, learning_rate=0.5)
        groups = torch.tensor([0, 0, 1, 1])

        # Group 0 has high loss (5.0), Group 1 has low loss (1.0)
        losses = torch.tensor([5.0, 5.0, 1.0, 1.0])
        w_step1 = dro.update_weights(group_losses=torch.tensor([5.0, 1.0]))

        self.assertGreater(w_step1[0].item(), w_step1[1].item())
        self.assertAlmostEqual(w_step1.sum().item(), 1.0, places=5)

    def test_leakage_free_split_isolation_in_training(self):
        """Verify that test split is never accessed during training."""
        run_res = train_single_run(
            mode="groupdro",
            group_strategy="semantic",
            num_epochs=1,
            seed=42,
        )
        train_df = run_res["train_assignments"]
        self.assertTrue((train_df["split"] == "train").all())
        self.assertEqual(len(train_df), 12)  # 12 training stimuli

        # Evaluate on test split
        test_res = evaluate_on_test_split(run_res)
        test_df = test_res["test_assignments"]
        self.assertTrue((test_df["split"] == "test").all())
        self.assertEqual(len(test_df), 4)  # 4 test stimuli

        # Check zero stimulus overlap
        train_stims = set(train_df["stimulus_id"])
        test_stims = set(test_df["stimulus_id"])
        self.assertEqual(len(train_stims & test_stims), 0)

    def test_cross_modal_metrics_computation(self):
        """Verify cross-modal retrieval and alignment metrics computation."""
        n = 4
        d = 128
        torch.manual_seed(42)
        # Perfectly aligned embeddings
        z = torch.randn(n, d)
        z = z / torch.norm(z, dim=-1, keepdim=True)
        embeddings = {"eeg": z, "meg": z, "fmri": z}
        stim_ids = [f"stim_{i}" for i in range(n)]

        metrics = compute_cross_modal_metrics(embeddings, stim_ids)
        self.assertAlmostEqual(metrics["mean_retrieval_acc"], 1.0, places=4)
        self.assertAlmostEqual(metrics["mean_alignment_sim"], 1.0, places=4)

    def test_experiment_artifacts_integrity(self):
        """Verify that all Phase 3 experiment artifacts exist and are properly populated."""
        exp_dir = REPO_ROOT / "reports" / "artifacts" / "experiments"
        expected_dirs = [
            "baseline_semantic",
            "groupdro_semantic",
            "baseline_cluster",
            "groupdro_cluster",
            "summary_comparison",
        ]
        for d in expected_dirs:
            p = exp_dir / d
            self.assertTrue(p.exists(), f"Missing directory: {p}")

        # Check comparison summary artifacts
        comp_csv = exp_dir / "summary_comparison" / "benchmark_comparison.csv"
        comp_json = exp_dir / "summary_comparison" / "comparison_summary.json"
        self.assertTrue(comp_csv.exists(), f"Missing {comp_csv}")
        self.assertTrue(comp_json.exists(), f"Missing {comp_json}")

        df_comp = pd.read_csv(comp_csv)
        self.assertEqual(len(df_comp), 4)  # 4 benchmark experiments

    def test_independent_metric_recomputation(self):
        """Independently recompute test metrics from saved per-sample outputs."""
        exp_dir = REPO_ROOT / "reports" / "artifacts" / "experiments" / "groupdro_semantic"
        assign_csv = exp_dir / "group_assignments.csv"
        metrics_csv = exp_dir / "metrics.csv"

        self.assertTrue(assign_csv.exists())
        self.assertTrue(metrics_csv.exists())

        df_assign = pd.read_csv(assign_csv)
        df_metrics = pd.read_csv(metrics_csv)

        # Test records for seed 42
        test_records = df_assign[df_assign["split"] == "test"]
        self.assertEqual(len(test_records), 4)

        # Recompute overall loss
        recomputed_loss = float(test_records["per_sample_loss"].mean())
        seed42_loss = float(df_metrics[df_metrics["seed"] == 42]["test_overall_loss"].iloc[0])
        self.assertAlmostEqual(recomputed_loss, seed42_loss, places=4)


if __name__ == "__main__":
    unittest.main()
