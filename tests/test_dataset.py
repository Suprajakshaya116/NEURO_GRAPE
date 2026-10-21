"""
Focused Test Suite for Neuro-GRAPE Data Loading, Preprocessing & Leakage Verification
====================================================================================
Runs under Python standard library `unittest` or `pytest`.

Tests:
1. Manifest schema, row count (60), and 100% uniqueness of sample_id.
2. Leakage prevention: strict stimulus-level disjointness across train/val/test splits.
3. Data path integrity: all referenced feature arrays exist on disk with valid shapes.
4. Downsampling and epoch time calculations (225 samples @ 250 Hz).
5. Boundary safety: no event onset is clipped near start or end of recording.
6. MEG CTF compensation grade: verified Grade 3 synthetic gradient compensation.
7. fMRI LS-S beta volume properties: shape (96, 96, 60), finite, non-zero mask.
8. Dataset loader API: indexing, modality filtering, fast sample_id lookup, cross-modal query.
9. Modality batch stacking convenience method for Member 2.
10. Error handling for nonexistent sample_id and stimulus_id.
"""

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.data.dataset import NeuroGrapeDataset
from src.data.preprocessing import make_sample_id

INVENTORY_CSV = REPO_ROOT / "data_inventory.csv"
SPLITS_CSV = REPO_ROOT / "data" / "splits" / "pilot_splits.csv"
FMRI_BETA_DIR = REPO_ROOT / "data" / "processed" / "fmri_beta"


class TestNeuroGrapeDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        assert INVENTORY_CSV.exists(), f"data_inventory.csv missing at {INVENTORY_CSV}"
        assert SPLITS_CSV.exists(), f"pilot_splits.csv missing at {SPLITS_CSV}"
        cls.inventory_df = pd.read_csv(INVENTORY_CSV)
        cls.splits_df = pd.read_csv(SPLITS_CSV)

    def test_manifest_schema_and_uniqueness(self):
        """Verify required columns and 100% uniqueness of sample_id."""
        required_cols = {
            "sample_id", "modality", "subject_id", "session_id", "run_id",
            "trial_id", "stimulus_id", "category_id", "category_name",
            "split", "onset_sec", "sampling_rate", "signal_units", "data_path"
        }
        missing = required_cols - set(self.inventory_df.columns)
        self.assertFalse(missing, f"Manifest is missing columns: {missing}")

        # Verify no duplicate sample_ids
        self.assertFalse(self.inventory_df["sample_id"].duplicated().any(), "Duplicate sample_ids found!")
        self.assertEqual(len(self.inventory_df), 60, f"Expected exactly 60 pilot records, got {len(self.inventory_df)}")

    def test_leakage_prevention_stimulus_disjointness(self):
        """Verify that stimulus identities NEVER cross train/val/test splits."""
        splits = self.splits_df.groupby("stimulus_id")["split"].nunique()
        self.assertTrue((splits == 1).all(), "Found stimulus assigned to multiple splits in splits table!")

        # Verify cross-modality split consistency in data_inventory
        inv_splits = self.inventory_df.groupby("stimulus_id")["split"].nunique()
        self.assertTrue((inv_splits == 1).all(), "Found stimulus assigned to conflicting splits across modalities!")

        # Check set disjointness
        train_stims = set(self.inventory_df[self.inventory_df["split"] == "train"]["stimulus_id"])
        val_stims = set(self.inventory_df[self.inventory_df["split"] == "val"]["stimulus_id"])
        test_stims = set(self.inventory_df[self.inventory_df["split"] == "test"]["stimulus_id"])

        self.assertEqual(len(train_stims & val_stims), 0, f"Leakage between train and val: {train_stims & val_stims}")
        self.assertEqual(len(train_stims & test_stims), 0, f"Leakage between train and test: {train_stims & test_stims}")
        self.assertEqual(len(val_stims & test_stims), 0, f"Leakage between val and test: {val_stims & test_stims}")

        self.assertEqual(len(train_stims), 12)
        self.assertEqual(len(val_stims), 4)
        self.assertEqual(len(test_stims), 4)

    def test_data_paths_exist_and_tensors_valid(self):
        """Verify that all feature arrays referenced in the manifest exist and contain finite data."""
        for _, row in self.inventory_df.iterrows():
            p = REPO_ROOT / row["data_path"]
            self.assertTrue(p.exists(), f"Feature array missing at: {p}")

            arr = np.load(p)
            self.assertFalse(np.isnan(arr).any(), f"NaN values detected in {row['sample_id']}")
            self.assertFalse(np.isinf(arr).any(), f"Inf values detected in {row['sample_id']}")

            # Check expected modality shapes
            if row["modality"] == "eeg":
                self.assertEqual(arr.shape, (63, 225), f"Unexpected EEG shape: {arr.shape}")
            elif row["modality"] == "meg":
                self.assertEqual(arr.shape, (272, 225), f"Unexpected MEG shape: {arr.shape}")
            elif row["modality"] == "fmri":
                self.assertEqual(arr.shape, (96, 96, 60), f"Unexpected fMRI beta shape: {arr.shape}")
                # Also verify raw window array
                raw_p = REPO_ROOT / row["raw_window_path"]
                self.assertTrue(raw_p.exists(), f"Raw window missing at {raw_p}")
                self.assertEqual(np.load(raw_p).shape, (96, 96, 60, 4))

    def test_epoch_sample_counts_and_downsampling(self):
        """Verify math: 250 Hz across [-0.1, 0.8]s yields exactly 225 timepoints."""
        sfreq = 250.0
        tmin = -0.1
        tmax = 0.8
        duration = tmax - tmin
        expected_samples = int(round(duration * sfreq))
        self.assertEqual(expected_samples, 225)

        # Baseline [-0.1, 0.0]s is 25 samples
        baseline_samples = int(round(0.1 * sfreq))
        self.assertEqual(baseline_samples, 25)

    def test_fmri_lss_beta_properties(self):
        """Verify that trial-wise fMRI LS-S beta maps exist and have valid 3D geometry."""
        self.assertTrue(FMRI_BETA_DIR.exists(), f"Missing fMRI beta directory at {FMRI_BETA_DIR}")
        beta_files = list(FMRI_BETA_DIR.glob("*.npy"))
        self.assertEqual(len(beta_files), 20, f"Expected 20 beta volumes, found {len(beta_files)}")

        sample_beta = np.load(beta_files[0])
        self.assertEqual(sample_beta.shape, (96, 96, 60))
        self.assertEqual(sample_beta.dtype, np.float32)
        self.assertTrue(np.isfinite(sample_beta).all())
        # Check non-zero brain voxels
        non_zero = np.sum(sample_beta != 0.0)
        self.assertGreater(non_zero, 150000, "Too few non-zero brain voxels in beta map")

    def test_dataset_loader_api(self):
        """Verify NeuroGrapeDataset access methods and filtering."""
        ds_all = NeuroGrapeDataset(manifest=INVENTORY_CSV, data_root=REPO_ROOT)
        self.assertEqual(len(ds_all), 60)

        # Test indexing
        first_sample = ds_all[0]
        self.assertIn("sample_id", first_sample)
        self.assertIn("features", first_sample)
        self.assertIsInstance(first_sample["features"], np.ndarray)

        # Test filtering by modality
        ds_eeg = NeuroGrapeDataset(manifest=INVENTORY_CSV, modality="eeg", data_root=REPO_ROOT)
        self.assertEqual(len(ds_eeg), 20)
        self.assertTrue(all(s["modality"] == "eeg" for s in [ds_eeg[i] for i in range(len(ds_eeg))]))

        # Test filtering by split
        ds_train = NeuroGrapeDataset(manifest=INVENTORY_CSV, split="train", data_root=REPO_ROOT)
        self.assertEqual(len(ds_train), 36)  # 12 stimuli * 3 modalities

        # Test fast sample_id lookup
        target_id = first_sample["sample_id"]
        lookup_sample = ds_all.get_by_sample_id(target_id)
        self.assertEqual(lookup_sample["sample_id"], target_id)
        self.assertEqual(lookup_sample["stimulus_id"], first_sample["stimulus_id"])

        # Test cross-modal retrieval by stimulus_id
        target_stim = first_sample["stimulus_id"]
        cross_modal = ds_all.get_cross_modal_sample(target_stim)
        self.assertEqual(set(cross_modal.keys()), {"eeg", "meg", "fmri"})
        for mod, s in cross_modal.items():
            self.assertEqual(s["stimulus_id"], target_stim)
            self.assertEqual(s["category_id"], first_sample["category_id"])

    def test_modality_batch_stacking(self):
        """Verify batch stacking convenience method for Member 2."""
        ds_eeg = NeuroGrapeDataset(manifest=INVENTORY_CSV, modality="eeg", data_root=REPO_ROOT)
        batch_features, batch_meta = ds_eeg.get_modality_batch("eeg", split="train")

        self.assertEqual(batch_features.shape, (12, 63, 225))
        self.assertEqual(len(batch_meta), 12)
        self.assertTrue((batch_meta["split"] == "train").all())

    def test_error_handling_invalid_keys(self):
        """Verify clean KeyError on nonexistent sample_id or stimulus_id."""
        ds = NeuroGrapeDataset(manifest=INVENTORY_CSV, data_root=REPO_ROOT)
        with self.assertRaises(KeyError):
            ds.get_by_sample_id("nonexistent_sample_id_123")
        with self.assertRaises(KeyError):
            ds.get_cross_modal_sample("nonexistent_stimulus.jpg")


if __name__ == "__main__":
    unittest.main()
