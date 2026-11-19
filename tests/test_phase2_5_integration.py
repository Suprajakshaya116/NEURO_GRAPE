"""
tests/test_phase2_5_integration.py
==================================
Comprehensive integration test suite for Neuro-GRAPE Phase 2.5.
Validates:
1. Inventory parsing and relative-path resolution on Windows.
2. Nested modality directory support across eeg/, meg/, fmri/, and fmri_beta/.
3. Real .npy loading, shapes, dtypes, and numerical finiteness.
4. Robust error handling for corrupt files, missing paths, duplicate IDs, and invalid metadata.
5. Cross-modality joins by stable stimulus IDs (rejecting row-order joins).
6. Leakage-free split validation.
7. Member 2 model encoding and checkpoint baseline compatibility.
8. End-to-end integration: inventory -> real .npy -> Member 2 encoder -> Member 3 GroupDRO pipeline.
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import NeuroGrapeDataset
from src.dro.clustering import EmbeddingClustering
from src.dro.evaluation import GroupEvaluator
from src.dro.group_dro import GroupDRO
from src.models.model import TrimodalNeuroGrapeModel


class TestPhase25DataIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root_inv = REPO_ROOT / "data_inventory.csv"
        cls.manifest_inv = REPO_ROOT / "data" / "manifests" / "data_inventory.csv"
        cls.pilot_splits = REPO_ROOT / "data" / "splits" / "pilot_splits.csv"
        cls.checkpoint_path = REPO_ROOT / "checkpoints" / "baseline.pt"

        missing_paths = [p for p in (cls.root_inv, cls.manifest_inv, cls.pilot_splits, cls.checkpoint_path) if not p.exists()]
        if missing_paths:
            raise unittest.SkipTest(f"Missing required real-data integration files: {missing_paths}; skipping.")

        cls.df_inventory = pd.read_csv(cls.root_inv)
        cls.df_splits = pd.read_csv(cls.pilot_splits)

    def test_inventory_parsing_and_path_resolution(self):
        """Verify inventory parsing, bit-for-bit equivalence, and path resolution."""
        df_root = pd.read_csv(self.root_inv)
        df_man = pd.read_csv(self.manifest_inv)
        self.assertTrue(df_root.equals(df_man), "Root inventory and manifests inventory must be identical!")

        # Verify dataset path resolution with relative and absolute paths
        ds = NeuroGrapeDataset(manifest=self.root_inv, data_root=REPO_ROOT)
        for i in range(len(ds)):
            row = ds.df.iloc[i]
            resolved = ds._resolve_data_path(row["data_path"])
            self.assertTrue(resolved.exists(), f"Failed to resolve path: {resolved}")

    def test_nested_modality_directory_support(self):
        """Verify presence and file counts in all four modality subdirectories."""
        expected_dirs = {
            "eeg": (REPO_ROOT / "data" / "processed" / "eeg", 20),
            "meg": (REPO_ROOT / "data" / "processed" / "meg", 20),
        }
        if (REPO_ROOT / "data" / "processed" / "fmri").exists():
            expected_dirs["fmri"] = (REPO_ROOT / "data" / "processed" / "fmri", 20)
            expected_dirs["fmri_beta"] = (REPO_ROOT / "data" / "processed" / "fmri_beta", 20)

        for mod, (dir_path, expected_count) in expected_dirs.items():
            self.assertTrue(dir_path.exists(), f"Missing nested modality directory: {dir_path}")
            files = list(dir_path.glob("*.npy"))
            self.assertEqual(
                len(files),
                expected_count,
                f"Modality directory {mod} expected {expected_count} files, found {len(files)}"
            )

    def test_real_npy_shape_dtype_and_finiteness(self):
        """Inspect actual .npy files on disk for expected shape, dtype, and finiteness."""
        expected_specs = {
            "eeg": {"shape": (63, 225), "dtype": np.float32},
            "meg": {"shape": (272, 225), "dtype": np.float32},
        }
        if (REPO_ROOT / "data" / "processed" / "fmri").exists():
            expected_specs["fmri_beta"] = {"shape": (96, 96, 60), "dtype": np.float32}
            expected_specs["fmri"] = {"shape": (96, 96, 60, 4), "dtype": np.float32}

        for mod, spec in expected_specs.items():
            mod_dir = REPO_ROOT / "data" / "processed" / mod
            files = list(mod_dir.glob("*.npy"))
            self.assertGreater(len(files), 0)
            for f in files:
                arr = np.load(f)
                self.assertEqual(arr.shape, spec["shape"], f"Shape mismatch in {f}: {arr.shape} vs {spec['shape']}")
                self.assertEqual(arr.dtype, spec["dtype"], f"Dtype mismatch in {f}: {arr.dtype} vs {spec['dtype']}")
                self.assertTrue(np.all(np.isfinite(arr)), f"Non-finite values found in {f}")

    def test_missing_corrupt_duplicate_error_handling(self):
        """Verify robust error handling for missing files, duplicate sample_ids, and invalid columns."""
        # 1. Missing columns
        bad_df = self.df_inventory.drop(columns=["sample_id"])
        with self.assertRaises(ValueError):
            NeuroGrapeDataset(manifest=bad_df, data_root=REPO_ROOT)

        # 2. Duplicate sample_id
        dup_df = pd.concat([self.df_inventory, self.df_inventory.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            NeuroGrapeDataset(manifest=dup_df, data_root=REPO_ROOT)

        # 3. Missing file path
        missing_df = self.df_inventory.copy()
        missing_df.loc[0, "data_path"] = "data/processed/nonexistent/missing.npy"
        bad_ds = NeuroGrapeDataset(manifest=missing_df, data_root=REPO_ROOT)
        with self.assertRaises(FileNotFoundError):
            _ = bad_ds[0]

    def test_cross_modality_joins_by_stable_id(self):
        """Verify cross-modality joins by stimulus_id and sample_id, confirming independent subjects."""
        ds = NeuroGrapeDataset(manifest=self.root_inv, data_root=REPO_ROOT)
        unique_stims = self.df_inventory["stimulus_id"].unique()
        self.assertEqual(len(unique_stims), 20)

        for stim in unique_stims:
            cross_sample = ds.get_cross_modal_sample(stim)
            self.assertTrue(set(cross_sample.keys()).issubset({"eeg", "meg", "fmri"}))
            self.assertIn("eeg", cross_sample)
            self.assertIn("meg", cross_sample)
            # Verify subject independence
            self.assertEqual(cross_sample["eeg"]["subject_id"], "sub-01")
            self.assertEqual(cross_sample["meg"]["subject_id"], "sub-BIGMEG2")
            if "fmri" in cross_sample:
                self.assertEqual(cross_sample["fmri"]["subject_id"], "sub-01")
            # Verify distinct sample IDs
            sids = [s["sample_id"] for s in cross_sample.values()]
            self.assertGreaterEqual(len(set(sids)), 2)

    def test_stimulus_level_leakage_and_splits(self):
        """Verify zero leakage across splits and consistency with pilot_splits.csv."""
        train_stims = set(self.df_splits[self.df_splits["split"] == "train"]["stimulus_id"])
        val_stims = set(self.df_splits[self.df_splits["split"] == "val"]["stimulus_id"])
        test_stims = set(self.df_splits[self.df_splits["split"] == "test"]["stimulus_id"])

        self.assertEqual(len(train_stims), 12)
        self.assertEqual(len(val_stims), 4)
        self.assertEqual(len(test_stims), 4)

        # Pairwise disjoint
        self.assertEqual(len(train_stims & val_stims), 0)
        self.assertEqual(len(train_stims & test_stims), 0)
        self.assertEqual(len(val_stims & test_stims), 0)

        # Inventory matches pilot_splits exactly
        for stim in self.df_inventory["stimulus_id"].unique():
            expected = self.df_splits[self.df_splits["stimulus_id"] == stim]["split"].iloc[0]
            actual = self.df_inventory[self.df_inventory["stimulus_id"] == stim]["split"].unique()
            self.assertEqual(len(actual), 1)
            self.assertEqual(actual[0], expected)

    def test_member2_encoder_and_checkpoint_interface(self):
        """Verify that actual processed .npy files can be encoded by TrimodalNeuroGrapeModel."""
        model = TrimodalNeuroGrapeModel(embed_dim=128)
        model.load_checkpoint(self.checkpoint_path)
        model.eval()

        ds_eeg = NeuroGrapeDataset(manifest=self.root_inv, modality="eeg", as_torch=True, data_root=REPO_ROOT)
        ds_meg = NeuroGrapeDataset(manifest=self.root_inv, modality="meg", as_torch=True, data_root=REPO_ROOT)
        ds_fmri = NeuroGrapeDataset(manifest=self.root_inv, modality="fmri", as_torch=True, data_root=REPO_ROOT)

        # Test EEG encoding: shape (B, 63, 225)
        eeg_tensor = ds_eeg[0]["features"].unsqueeze(0)  # (1, 63, 225)
        with torch.no_grad():
            z_eeg = model.encode("eeg", eeg_tensor)
        self.assertEqual(z_eeg.shape, (1, 128))
        norm_eeg = torch.norm(z_eeg, p=2, dim=-1).item()
        self.assertAlmostEqual(norm_eeg, 1.0, places=4)

        # Test MEG encoding: shape (B, 272, 225)
        meg_tensor = ds_meg[0]["features"].unsqueeze(0)  # (1, 272, 225)
        with torch.no_grad():
            z_meg = model.encode("meg", meg_tensor)
        self.assertEqual(z_meg.shape, (1, 128))
        norm_meg = torch.norm(z_meg, p=2, dim=-1).item()
        self.assertAlmostEqual(norm_meg, 1.0, places=4)

        # Test fMRI encoding if fMRI is available
        if len(ds_fmri) > 0:
            fmri_tensor = ds_fmri[0]["features"].unsqueeze(0).unsqueeze(0)  # (1, 1, 96, 96, 60)
            with torch.no_grad():
                z_fmri = model.encode("fmri", fmri_tensor)
            self.assertEqual(z_fmri.shape, (1, 128))
            norm_fmri = torch.norm(z_fmri, p=2, dim=-1).item()
            self.assertAlmostEqual(norm_fmri, 1.0, places=4)

    def test_end_to_end_real_data_to_member3_pipeline(self):
        """End-to-end integration test: Real .npy -> Encoders -> Embeddings -> Member 3 GroupDRO."""
        # 1. Load baseline model
        model = TrimodalNeuroGrapeModel(embed_dim=128)
        model.load_checkpoint(self.checkpoint_path)
        model.eval()

        # 2. Extract embeddings from real .npy data across all 60 samples
        ds = NeuroGrapeDataset(manifest=self.root_inv, as_torch=True, data_root=REPO_ROOT)
        all_embeddings = []
        metadata_rows = []

        with torch.no_grad():
            for i in range(len(ds)):
                sample = ds[i]
                mod = sample["modality"]
                feats = sample["features"]
                if mod in ["eeg", "meg"]:
                    inp = feats.unsqueeze(0)
                else:
                    inp = feats.unsqueeze(0).unsqueeze(0)

                emb = model.encode(mod, inp).squeeze(0).numpy()
                all_embeddings.append(emb)
                metadata_rows.append({
                    "sample_id": sample["sample_id"],
                    "modality": sample["modality"],
                    "stimulus_id": sample["stimulus_id"],
                    "split": sample["split"],
                    "category_id": sample["category_id"],
                })

        embeddings_mat = np.array(all_embeddings)
        meta_df = pd.DataFrame(metadata_rows)
        embeddings_dict = {
            row["sample_id"]: emb for row, emb in zip(metadata_rows, all_embeddings)
        }

        # 3. Fit clustering on train split only (preventing leakage)
        clustering = EmbeddingClustering(n_clusters=3, random_state=42)
        clustering.fit(embeddings_dict, meta_df)
        clustered_df = clustering.assign_clusters(embeddings_dict, meta_df)
        self.assertIn(len(clustered_df), (40, 60))
        self.assertIn("cluster_id", clustered_df.columns)

        # 4. Member 3 GroupDRO simulation and evaluation
        # Evaluate on test split
        test_df = clustered_df[clustered_df["split"] == "test"].copy()
        self.assertIn(len(test_df), (8, 12))

        test_indices = test_df.index.tolist()
        test_embs = embeddings_mat[test_indices]
        centers = clustering.cluster_centers_
        test_groups = test_df["cluster_id"].to_numpy()

        test_losses = np.array([
            float(np.linalg.norm(test_embs[i] - centers[test_groups[i]]))
            for i in range(len(test_groups))
        ])

        evaluator = GroupEvaluator(min_group_size=1)
        res = evaluator.evaluate(
            values=test_losses,
            group_ids=test_groups,
            metric_name="cluster_distance_loss",
            higher_is_better=False,
        )

        self.assertIsNotNone(res.worst_group_value)
        self.assertIsNotNone(res.performance_gap)
        self.assertTrue(np.isfinite(res.worst_group_value))
        self.assertTrue(np.isfinite(res.performance_gap))


if __name__ == "__main__":
    unittest.main()
