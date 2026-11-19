"""Comprehensive Unit & Regression Tests for Member 3 (GroupDRO, Clustering, Evaluation).

Tests:
1. GroupDRO weight normalization, non-negativity, and log-space numerical stability.
2. Higher-loss groups receiving higher relative weights when comparable groups are eligible.
3. Empty, missing, and undersized groups handling without NaN or zero division.
4. Weighted-loss calculations and differentiability.
5. Metric direction and worst-group aggregation (lower-is-better vs higher-is-better).
6. Clustering fit strictly on training data only (test/val splits excluded from fit).
7. Stable-ID joins, duplicate IDs, invalid embeddings, and mismatched inputs.
8. Artifact generation (metrics.csv, group_losses.csv, group_weights.csv, group_assignments.parquet).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.dro.clustering import EmbeddingClustering, fit_regime_clustering
from src.dro.evaluation import GroupEvaluator, evaluate_per_group
from src.dro.group_dro import GroupDRO
from src.dro.pipeline import run_member3_pipeline


class TestGroupDROMustPass(unittest.TestCase):
    """Test suite validating GroupDRO mathematical properties and interfaces."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.num_groups = 3
        self.dro = GroupDRO(
            num_groups=self.num_groups,
            learning_rate=0.1,
            smoothing=0.0,
            device=self.device,
        )

    # -------------------------------------------------------------------------
    # 1. Weight Normalization & Numerical Stability
    # -------------------------------------------------------------------------
    def test_weight_normalization_and_stability(self):
        """Weights must remain non-negative, finite, and sum to 1.0 even with large losses."""
        weights = self.dro.get_weights()
        self.assertEqual(len(weights), self.num_groups)
        self.assertTrue((weights >= 0.0).all())
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0, device=self.device))

        # Test stability under extreme loss values (e.g. 1000.0 or -1000.0)
        extreme_losses = torch.tensor([1000.0, 500.0, 100.0], device=self.device)
        updated_weights = self.dro.update_weights(extreme_losses)

        self.assertTrue(torch.isfinite(updated_weights).all())
        self.assertTrue((updated_weights >= 0.0).all())
        torch.testing.assert_close(updated_weights.sum(), torch.tensor(1.0, device=self.device))
        # With extreme difference, group 0 must have dominant weight
        self.assertGreater(updated_weights[0].item(), 0.99)

    # -------------------------------------------------------------------------
    # 2. Higher-Loss Groups Receive Higher Weights
    # -------------------------------------------------------------------------
    def test_higher_loss_group_receives_higher_weight(self):
        """Group with persistently higher loss must receive higher weight."""
        self.dro.reset()
        initial_weights = self.dro.get_weights().clone()

        # Group 1 has higher loss than Group 0 and Group 2
        losses = torch.tensor([1.0, 5.0, 2.0], device=self.device)
        new_weights = self.dro.update_weights(losses)

        self.assertGreater(new_weights[1].item(), initial_weights[1].item())
        self.assertGreater(new_weights[1].item(), new_weights[0].item())
        self.assertGreater(new_weights[1].item(), new_weights[2].item())
        torch.testing.assert_close(new_weights.sum(), torch.tensor(1.0, device=self.device))

    # -------------------------------------------------------------------------
    # 3. Empty, Missing, and Undersized Groups Handled Safely
    # -------------------------------------------------------------------------
    def test_empty_and_undersized_groups(self):
        """Groups with 0 samples or fewer than min_group_size must not produce NaN."""
        dro_min2 = GroupDRO(num_groups=3, min_group_size=2, device=self.device)

        # Batch with 0 samples for group 1, 1 sample for group 2 (undersized), 3 for group 0
        per_example_losses = torch.tensor([2.0, 3.0, 4.0, 1.5])
        group_indices = torch.tensor([0, 0, 0, 2])  # group 1 has 0 samples; group 2 has 1 (<2)

        g_losses, g_counts = dro_min2.compute_group_losses(per_example_losses, group_indices)

        self.assertEqual(g_counts[0].item(), 3)
        self.assertEqual(g_counts[1].item(), 0)
        self.assertEqual(g_counts[2].item(), 1)

        # Undersized and empty group losses default safely to 0.0 without crashing
        self.assertEqual(g_losses[1].item(), 0.0)
        self.assertEqual(g_losses[2].item(), 0.0)
        self.assertAlmostEqual(g_losses[0].item(), 3.0, places=5)

        # Update weights with counts: inactive groups should not cause NaN or crash
        weights = dro_min2.update_weights(g_losses, g_counts)
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0, device=self.device))

    # -------------------------------------------------------------------------
    # 4. Weighted Loss Computation & Differentiability
    # -------------------------------------------------------------------------
    def test_weighted_loss_computation_and_gradient(self):
        """compute_weighted_loss must be differentiable and reflect group weights."""
        losses = torch.tensor([1.0, 2.0, 10.0, 20.0], requires_grad=True)
        groups = torch.tensor([0, 0, 1, 1])

        # Set unequal weights: group 1 has 80%, group 0 has 20%
        dro_2g = GroupDRO(num_groups=2, device=self.device)
        dro_2g.set_weights([0.2, 0.8])

        weighted_loss = dro_2g.compute_weighted_loss(losses, groups)
        self.assertTrue(weighted_loss.ndim == 0)
        # Expected: 0.2 * mean(1, 2) + 0.8 * mean(10, 20) = 0.2 * 1.5 + 0.8 * 15 = 0.3 + 12 = 12.3
        self.assertAlmostEqual(weighted_loss.item(), 12.3, places=5)

        # Verify gradient flow
        weighted_loss.backward()
        self.assertIsNotNone(losses.grad)
        self.assertTrue(torch.isfinite(losses.grad).all())
        # Gradients for group 1 should be higher due to 0.8 weight
        self.assertGreater(losses.grad[2].item(), losses.grad[0].item())


class TestClusteringMustPass(unittest.TestCase):
    """Test suite validating unsupervised regime clustering requirements."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.n_samples = 30
        self.dim = 128

        # Create synthetic metadata
        sample_ids = [f"sample_{i:03d}" for i in range(self.n_samples)]
        splits = ["train"] * 18 + ["val"] * 6 + ["test"] * 6
        self.metadata_df = pd.DataFrame({
            "sample_id": sample_ids,
            "split": splits,
            "stimulus_id": [f"stim_{i // 3:02d}.jpg" for i in range(self.n_samples)],
        })

        # Synthetic embeddings dictionary
        self.embed_dict = {
            sid: self.rng.normal(0, 1, size=(self.dim,)).astype(np.float32)
            for sid in sample_ids
        }

    # -------------------------------------------------------------------------
    # 5. Fit Clustering Strictly on Training Split
    # -------------------------------------------------------------------------
    def test_clustering_fits_on_train_split_only(self):
        """Clustering must be fitted strictly on train split samples."""
        clustering = EmbeddingClustering(n_clusters=2, random_state=42, expected_dim=self.dim)
        clustering.fit(self.embed_dict, self.metadata_df)

        self.assertIsNotNone(clustering.model)
        self.assertEqual(len(clustering.fitted_train_sample_ids_), 18)
        self.assertEqual(clustering.diagnostics_["fit_split"], "train")

        # Assign clusters to all splits
        assigned_df = clustering.assign_clusters(self.embed_dict, self.metadata_df)
        self.assertEqual(len(assigned_df), self.n_samples)
        self.assertIn("cluster_id", assigned_df.columns)
        self.assertIn("group_id", assigned_df.columns)

        # Verify all splits have assignments
        for sp in ("train", "val", "test"):
            sp_df = assigned_df[assigned_df["split"] == sp]
            self.assertFalse(sp_df["cluster_id"].isna().any())

    # -------------------------------------------------------------------------
    # 6. Stable-ID Joins & Input Validation
    # -------------------------------------------------------------------------
    def test_stable_id_joins_and_error_handling(self):
        """Verify deterministic ID joins, duplicate ID detection, and non-finite rejection."""
        clustering = EmbeddingClustering(n_clusters=2, expected_dim=self.dim)

        # Duplicate ID rejection
        dup_ids = ["id_01", "id_02", "id_01"]
        with self.assertRaises(ValueError):
            clustering.validate_embeddings(np.zeros((3, self.dim)), dup_ids, self.dim)

        # Non-finite values rejection
        bad_matrix = np.zeros((2, self.dim), dtype=np.float32)
        bad_matrix[0, 5] = np.nan
        with self.assertRaises(ValueError):
            clustering.validate_embeddings(bad_matrix, ["s1", "s2"], self.dim)

        # Dimension mismatch
        wrong_dim_matrix = np.zeros((2, 64), dtype=np.float32)
        with self.assertRaises(ValueError):
            clustering.validate_embeddings(wrong_dim_matrix, ["s1", "s2"], expected_dim=128)

        # Stable join preserves correct sample correspondence regardless of row ordering
        clustering.fit(self.embed_dict, self.metadata_df)
        shuffled_meta = self.metadata_df.sample(frac=1.0, random_state=99).reset_index(drop=True)
        assigned_shuffled = clustering.assign_clusters(self.embed_dict, shuffled_meta)

        # Look up by sample_id and compare with normal assignment
        normal_assigned = clustering.assign_clusters(self.embed_dict, self.metadata_df)
        for sid in self.metadata_df["sample_id"].head(5):
            norm_cid = normal_assigned[normal_assigned["sample_id"] == sid]["cluster_id"].iloc[0]
            shuf_cid = assigned_shuffled[assigned_shuffled["sample_id"] == sid]["cluster_id"].iloc[0]
            self.assertEqual(norm_cid, shuf_cid)


class TestEvaluationMustPass(unittest.TestCase):
    """Test suite validating GroupEvaluator metric directions and worst-group aggregation."""

    def setUp(self):
        self.evaluator = GroupEvaluator(min_group_size=1)

    # -------------------------------------------------------------------------
    # 7. Metric Direction & Worst-Group Calculation
    # -------------------------------------------------------------------------
    def test_metric_direction_and_gap_calculation(self):
        """Lower-is-better (loss) vs Higher-is-better (accuracy) worst-group logic."""
        # 3 groups: Group 0 (loss 1.0), Group 1 (loss 4.0), Group 2 (loss 2.0)
        losses = [1.0, 1.0, 4.0, 4.0, 2.0, 2.0]
        group_ids = [0, 0, 1, 1, 2, 2]

        loss_res = self.evaluator.evaluate(
            values=losses,
            group_ids=group_ids,
            metric_name="crossmodal_loss",
            higher_is_better=False,  # LOSS: higher is worse!
        )

        self.assertAlmostEqual(loss_res.overall_value, 2.333333, places=5)
        # Worst group for loss must be group 1 (loss = 4.0)
        self.assertEqual(loss_res.worst_group_id, 1)
        self.assertAlmostEqual(loss_res.worst_group_value, 4.0, places=5)
        # Gap = worst_group - overall = 4.0 - 2.3333 = 1.66667 (>= 0)
        self.assertAlmostEqual(loss_res.performance_gap, 4.0 - (7.0 / 3.0), places=5)

        # Accuracy / Higher-is-better: Group 0 (acc 0.9), Group 1 (acc 0.4), Group 2 (acc 0.7)
        accuracies = [0.9, 0.9, 0.4, 0.4, 0.7, 0.7]
        acc_res = self.evaluator.evaluate(
            values=accuracies,
            group_ids=group_ids,
            metric_name="top1_accuracy",
            higher_is_better=True,  # ACCURACY: lower is worse!
        )

        self.assertAlmostEqual(acc_res.overall_value, (0.9 + 0.4 + 0.7) / 3.0, places=5)
        # Worst group for accuracy must be group 1 (acc = 0.4)
        self.assertEqual(acc_res.worst_group_id, 1)
        self.assertAlmostEqual(acc_res.worst_group_value, 0.4, places=5)
        # Gap = overall - worst_group = 0.6667 - 0.4 = 0.2667 (>= 0)
        self.assertAlmostEqual(acc_res.performance_gap, (2.0 / 3.0) - 0.4, places=5)

    def test_missing_group_excluded_from_worst_group(self):
        """Missing groups with 0 samples must NOT become worst group (e.g. 0.0 accuracy)."""
        accuracies = [0.8, 0.9, 0.7]
        group_ids = [0, 0, 1]  # Group 2 is unobserved!

        res = self.evaluator.evaluate(
            values=accuracies,
            group_ids=group_ids,
            metric_name="accuracy",
            higher_is_better=True,
            all_group_ids=[0, 1, 2],
        )

        self.assertIn(2, res.inactive_groups)
        self.assertEqual(res.group_counts[2], 0)
        self.assertTrue(np.isnan(res.per_group_values[2]))
        # Worst group must be Group 1 (0.7), NOT unobserved Group 2!
        self.assertEqual(res.worst_group_id, 1)
        self.assertAlmostEqual(res.worst_group_value, 0.7, places=5)


class TestPipelineArtifactsMustPass(unittest.TestCase):
    """Test suite validating end-to-end pipeline execution and output artifacts."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pipeline_generates_all_expected_artifacts(self):
        """Pipeline must run and write metrics.csv, group_losses.csv, group_weights.csv, etc."""
        # Use existing baseline embeddings and metadata
        config = {
            "paths": {
                "embeddings_pt": "embeddings/embeddings.pt",
                "metadata_csv": "embeddings/metadata.csv",
                "output_dir": self.test_dir,
            },
            "clustering": {
                "n_clusters": 2,
                "random_state": 42,
                "expected_dim": 128,
            },
            "group_dro": {
                "learning_rate": 0.05,
                "smoothing": 0.01,
                "num_steps": 5,
            },
            "evaluation": {
                "temperature": 0.07,
            },
        }

        summary = run_member3_pipeline(config_or_path=config, output_dir=self.test_dir)
        self.assertEqual(summary["status"], "SUCCESS")

        out = Path(self.test_dir)
        self.assertTrue((out / "metrics.csv").exists(), "metrics.csv missing")
        self.assertTrue((out / "group_losses.csv").exists(), "group_losses.csv missing")
        self.assertTrue((out / "group_weights.csv").exists(), "group_weights.csv missing")
        self.assertTrue((out / "group_assignments.csv").exists(), "group_assignments.csv missing")
        self.assertTrue((out / "clustering_diagnostics.json").exists(), "clustering_diagnostics.json missing")
        self.assertTrue((out / "experiment_config.yaml").exists(), "experiment_config.yaml missing")

        # Verify metrics.csv content
        metrics_df = pd.read_csv(out / "metrics.csv")
        self.assertTrue(len(metrics_df) > 0)
        expected_cols = {
            "split", "metric_name", "higher_is_better", "overall_value",
            "worst_group_id", "worst_group_value", "performance_gap"
        }
        self.assertTrue(expected_cols.issubset(set(metrics_df.columns)))
        self.assertTrue((out / "group_metrics.csv").exists(), "group_metrics.csv missing")


class TestLargeDatasetReadiness(unittest.TestCase):
    """Test suite validating pipeline behavior for large-scale and variable datasets."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.rng = np.random.default_rng(123)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_dynamic_embedding_dimensions(self):
        """Pipeline and clustering must support variable embedding dimensions (e.g. 64-D, 256-D)."""
        dim = 256
        n_samples = 45
        sample_ids = [f"sample_dim_{i:03d}" for i in range(n_samples)]
        splits = ["train"] * 27 + ["val"] * 9 + ["test"] * 9

        meta_df = pd.DataFrame({
            "sample_id": sample_ids,
            "split": splits,
        })
        embed_dict = {
            sid: self.rng.normal(0, 1, size=(dim,)).astype(np.float32)
            for sid in sample_ids
        }

        # Auto-detect expected_dim=None
        clustering = EmbeddingClustering(n_clusters=3, expected_dim=None)
        clustering.fit(embed_dict, meta_df)
        self.assertEqual(clustering.expected_dim, dim)
        assigned = clustering.assign_clusters(embed_dict, meta_df)
        self.assertEqual(len(assigned), n_samples)
        self.assertIn("cluster_id", assigned.columns)

    def test_group_dro_from_group_labels(self):
        """GroupDRO.from_group_labels must dynamically create groups from arbitrary strings."""
        labels = ["animal", "tool", "food", "animal", "tool", "vehicle"]
        dro, label_to_idx = GroupDRO.from_group_labels(labels, learning_rate=0.05)

        self.assertEqual(dro.num_groups, 4)  # 4 unique categories
        self.assertEqual(set(dro.group_names), {"animal", "food", "tool", "vehicle"})
        self.assertEqual(len(dro.get_weights()), 4)

        # Update weights
        losses = torch.tensor([1.0, 2.0, 3.0, 0.5])
        updated = dro.update_weights(losses)
        self.assertEqual(len(updated), 4)
        torch.testing.assert_close(updated.sum(), torch.tensor(1.0))

    def test_variable_cluster_count_and_dynamic_metrics(self):
        """Pipeline must support K=3 clusters and dynamically export all group columns."""
        config = {
            "paths": {
                "embeddings_pt": "embeddings/embeddings.pt",
                "metadata_csv": "embeddings/metadata.csv",
                "output_dir": self.test_dir,
            },
            "clustering": {
                "n_clusters": 3,  # K=3
                "random_state": 42,
                "expected_dim": None,
            },
            "group_dro": {
                "learning_rate": 0.05,
                "smoothing": 0.01,
                "num_steps": 3,
            },
            "evaluation": {
                "temperature": 0.07,
            },
        }

        summary = run_member3_pipeline(config_or_path=config, output_dir=self.test_dir)
        self.assertEqual(summary["status"], "SUCCESS")
        self.assertEqual(summary["num_groups"], 3)

        metrics_df = pd.read_csv(Path(self.test_dir) / "metrics.csv")
        # Check that group 0, group 1, group 2 columns are all present
        for g in range(3):
            self.assertIn(f"group_{g}_value", metrics_df.columns)
            self.assertIn(f"group_{g}_count", metrics_df.columns)

        # Check granular group metrics long-table
        group_df = pd.read_csv(Path(self.test_dir) / "group_metrics.csv")
        self.assertTrue((group_df["group_id"].isin([0, 1, 2])).all())

    def test_modality_and_metadata_grouping_strategies(self):
        """Pipeline must support explicit modality and metadata-based grouping strategies."""
        # Modality grouping test
        mod_config = {
            "paths": {
                "embeddings_pt": "embeddings/embeddings.pt",
                "metadata_csv": "embeddings/metadata.csv",
                "output_dir": self.test_dir,
            },
            "grouping": {
                "strategy": "modality",
            },
            "group_dro": {
                "learning_rate": 0.05,
                "num_steps": 2,
            },
        }
        mod_summary = run_member3_pipeline(config_or_path=mod_config, output_dir=self.test_dir)
        self.assertIn(mod_summary["num_groups"], (2, 3))
        self.assertTrue(set(mod_summary["group_names"]).issubset({"eeg", "meg", "fmri"}))
        self.assertIn("eeg", mod_summary["group_names"])
        self.assertIn("meg", mod_summary["group_names"])

        # Metadata category grouping test
        meta_config = {
            "paths": {
                "embeddings_pt": "embeddings/embeddings.pt",
                "metadata_csv": "embeddings/metadata.csv",
                "output_dir": self.test_dir,
            },
            "grouping": {
                "strategy": "metadata",
                "metadata_column": "category_name",
            },
            "group_dro": {
                "learning_rate": 0.05,
                "num_steps": 2,
            },
        }
        meta_summary = run_member3_pipeline(config_or_path=meta_config, output_dir=self.test_dir)
        self.assertEqual(meta_summary["num_groups"], 20)  # 20 unique categories in pilot
        self.assertIn("bear", meta_summary["group_names"])
        self.assertIn("banner", meta_summary["group_names"])


if __name__ == "__main__":
    unittest.main()
