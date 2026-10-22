"""
Tests for Neuro-GRAPE Trimodal Normalization Pipeline
======================================================
Verifies leakage-safe input normalization across EEG, MEG, and fMRI.
"""

from __future__ import annotations

import unittest
import torch

from src.models.normalization import TrimodalNormalizer


def _build_synthetic_batch(
    b: int = 4,
    device: torch.device = torch.device("cpu"),
    eeg_std_scale: float = 1e-5,
    meg_std_scale: float = 1e-13,
) -> dict[str, torch.Tensor]:
    """Generate synthetic multimodal batch with physical unit scales."""
    torch.manual_seed(123)
    eeg = torch.randn(b, 63, 225, device=device) * eeg_std_scale
    meg = torch.randn(b, 272, 225, device=device) * meg_std_scale

    # fMRI: simulate brain mask where ~50% of voxels are background zeros
    fmri = torch.zeros(b, 1, 16, 16, 10, device=device)
    # central sphere of brain voxels
    mask = torch.zeros(16, 16, 10, dtype=torch.bool, device=device)
    mask[4:12, 4:12, 2:8] = True
    fmri[:, 0, mask] = torch.randn(b, int(mask.sum().item()), device=device) * 500.0 - 100.0

    return {"eeg": eeg, "meg": meg, "fmri": fmri}


class TestTrimodalNormalizer(unittest.TestCase):

    def test_fit_uses_training_data_only(self):
        """Verify normalizer fits statistics solely from the training batch."""
        normalizer = TrimodalNormalizer()
        self.assertFalse(normalizer.is_fitted)

        train_batch = _build_synthetic_batch(b=6)
        normalizer.fit(train_batch)

        self.assertTrue(normalizer.is_fitted)
        self.assertIsNotNone(normalizer.eeg_mean)
        self.assertIsNotNone(normalizer.eeg_std_safe)
        self.assertIsNotNone(normalizer.meg_mean)
        self.assertIsNotNone(normalizer.meg_std_safe)
        self.assertIsNotNone(normalizer.fmri_mask)
        self.assertIsNotNone(normalizer.fmri_in_mask_mean)
        self.assertIsNotNone(normalizer.fmri_in_mask_std_safe)

        # Check fitted shapes
        self.assertEqual(normalizer.eeg_mean.shape, (1, 63, 1))
        self.assertEqual(normalizer.eeg_std_safe.shape, (1, 63, 1))
        self.assertEqual(normalizer.meg_mean.shape, (1, 272, 1))
        self.assertEqual(normalizer.meg_std_safe.shape, (1, 272, 1))
        self.assertEqual(normalizer.fmri_mask.shape, (16, 16, 10))

    def test_val_and_test_reuse_training_statistics_without_leakage(self):
        """Verify that transforming validation/test batches does not alter fitted statistics."""
        normalizer = TrimodalNormalizer()
        train_batch = _build_synthetic_batch(b=6, eeg_std_scale=1e-5)
        normalizer.fit(train_batch)

        # Snapshot training stats
        saved_eeg_mean = normalizer.eeg_mean.clone()
        saved_eeg_std = normalizer.eeg_std_safe.clone()
        saved_meg_mean = normalizer.meg_mean.clone()
        saved_fmri_mean = normalizer.fmri_in_mask_mean.clone()

        # Transform validation batch with radically different scale
        val_batch = _build_synthetic_batch(b=2, eeg_std_scale=1e2)
        norm_val = normalizer.transform(val_batch)

        # Verify normalizer statistics were NOT modified by validation data
        self.assertTrue(torch.equal(normalizer.eeg_mean, saved_eeg_mean))
        self.assertTrue(torch.equal(normalizer.eeg_std_safe, saved_eeg_std))
        self.assertTrue(torch.equal(normalizer.meg_mean, saved_meg_mean))
        self.assertTrue(torch.equal(normalizer.fmri_in_mask_mean, saved_fmri_mean))

        # Verify validation data was transformed using training stats
        expected_val_eeg = (val_batch["eeg"] - saved_eeg_mean) / saved_eeg_std
        self.assertTrue(torch.allclose(norm_val["eeg"], expected_val_eeg, atol=1e-6))

    def test_constant_channels_and_voxels_safe(self):
        """Verify constant-zero channels and dead voxels do not produce NaN or Inf."""
        normalizer = TrimodalNormalizer(tau_eeg=1e-12, tau_meg=1e-18, tau_fmri=1e-8)
        batch = _build_synthetic_batch(b=4)

        # Inject dead (constant 0.0) channels
        batch["eeg"][:, 5, :] = 0.0
        batch["meg"][:, 10, :] = 0.0
        # Inject constant uniform channel
        batch["eeg"][:, 12, :] = 5.0e-5

        norm_batch = normalizer.fit_transform(batch)

        for mod in ("eeg", "meg", "fmri"):
            self.assertTrue(
                torch.isfinite(norm_batch[mod]).all().item(),
                f"Non-finite value produced in {mod} with constant channels!",
            )

        # Dead channels should be centered and safe (finite, no NaN)
        self.assertTrue(torch.isfinite(norm_batch["eeg"][:, 5, :]).all().item())
        self.assertTrue(torch.isfinite(norm_batch["meg"][:, 10, :]).all().item())

    def test_fmri_brain_mask_preservation(self):
        """Verify fMRI background zeros stay exactly zero and in-mask voxels are standardized."""
        normalizer = TrimodalNormalizer()
        train_batch = _build_synthetic_batch(b=8)

        # Confirm that before normalization, background is 0.0
        mask = normalizer.fit(train_batch).fmri_mask
        self.assertIsNotNone(mask)

        norm_train = normalizer.transform(train_batch)
        norm_fmri = norm_train["fmri"]

        # 1. Background voxels outside mask must be exactly 0.0
        bg_voxels = norm_fmri[:, 0, ~mask]
        self.assertTrue(
            (bg_voxels == 0.0).all().item(),
            "fMRI background voxels outside mask were modified to non-zero values!",
        )

        # 2. In-mask voxels must have zero mean and unit variance on training set
        in_mask_voxels = norm_fmri[:, 0, mask]
        self.assertAlmostEqual(in_mask_voxels.mean().item(), 0.0, places=4)
        self.assertAlmostEqual(in_mask_voxels.std(unbiased=False).item(), 1.0, places=4)

    def test_tensor_shapes_and_finiteness_preserved(self):
        """Verify output tensor shapes, dtypes, and devices match inputs."""
        normalizer = TrimodalNormalizer()
        batch = _build_synthetic_batch(b=3)

        norm_batch = normalizer.fit_transform(batch)

        self.assertEqual(norm_batch["eeg"].shape, batch["eeg"].shape)
        self.assertEqual(norm_batch["meg"].shape, batch["meg"].shape)
        self.assertEqual(norm_batch["fmri"].shape, batch["fmri"].shape)

        self.assertEqual(norm_batch["eeg"].dtype, batch["eeg"].dtype)
        self.assertEqual(norm_batch["meg"].dtype, batch["meg"].dtype)
        self.assertEqual(norm_batch["fmri"].dtype, batch["fmri"].dtype)

    def test_transform_before_fit_raises_runtime_error(self):
        """Verify calling transform before fit raises RuntimeError."""
        normalizer = TrimodalNormalizer()
        batch = _build_synthetic_batch(b=2)
        with self.assertRaises(RuntimeError):
            normalizer.transform(batch)

    def test_dimension_mismatch_raises_value_error(self):
        """Verify channel mismatch raises ValueError."""
        normalizer = TrimodalNormalizer()
        train_batch = _build_synthetic_batch(b=2)
        normalizer.fit(train_batch)

        bad_batch = {
            "eeg": torch.randn(2, 50, 225),  # 50 channels instead of 63
            "meg": train_batch["meg"],
            "fmri": train_batch["fmri"],
        }
        with self.assertRaises(ValueError):
            normalizer.transform(bad_batch)


if __name__ == "__main__":
    unittest.main()
