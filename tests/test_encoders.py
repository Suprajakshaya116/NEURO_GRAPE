"""
Focused Test Suite for Neuro-GRAPE Modality-Specific Encoders
=============================================================
Member 2 — Phase 1: Encoder architecture tests only.
Does NOT test contrastive loss, training loop, or data loading.

Tests
-----
1.  EEGEncoder output shape (B, 128).
2.  MEGEncoder output shape (B, 128).
3.  FMRIEncoder output shape (B, 128).
4.  All encoders produce finite (non-NaN, non-Inf) embeddings.
5.  Embeddings are L2-normalised (norm ~= 1 for non-zero inputs).
6.  Wrong channel count raises ValueError with informative message.
7.  Wrong tensor rank raises ValueError with informative message.
8.  Backpropagation: scalar loss produces gradients on trainable params.
9.  CPU execution passes.
10. MPS execution skips gracefully when MPS is unavailable.
11. Wrong n_times raises ValueError for EEG and MEG encoders.
12. Wrong volume_shape raises ValueError for fMRI encoder.
13. Batch-size-1 EEG: shape, finite, and unit norm.
14. Batch-size-1 MEG: shape, finite, and unit norm.
15. Batch-size-1 fMRI: shape, finite, and unit norm.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

# Resolve repo root for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.encoders import EEGEncoder, FMRIEncoder, MEGEncoder

# ---------------------------------------------------------------------------
# Small batch dimensions — avoid unnecessary memory use in CI
# ---------------------------------------------------------------------------
BATCH = 4
EEG_CHANNELS = 63
MEG_CHANNELS = 272
EEG_TIMES = 225
MEG_TIMES = 225
FMRI_X, FMRI_Y, FMRI_Z = 96, 96, 60
EMBED_DIM = 128


def _rand_eeg(batch: int = BATCH, device: str = "cpu") -> torch.Tensor:
    return torch.randn(batch, EEG_CHANNELS, EEG_TIMES, device=device)


def _rand_meg(batch: int = BATCH, device: str = "cpu") -> torch.Tensor:
    return torch.randn(batch, MEG_CHANNELS, MEG_TIMES, device=device)


def _rand_fmri(batch: int = BATCH, device: str = "cpu") -> torch.Tensor:
    return torch.randn(batch, 1, FMRI_X, FMRI_Y, FMRI_Z, device=device)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestEncoderOutputShapes(unittest.TestCase):

    def test_eeg_output_shape(self):
        """EEGEncoder maps (B, 63, 225) -> (B, 128)."""
        model = EEGEncoder()
        model.eval()
        with torch.no_grad():
            out = model(_rand_eeg())
        self.assertEqual(out.shape, (BATCH, EMBED_DIM),
                         f"Expected ({BATCH}, {EMBED_DIM}), got {tuple(out.shape)}")

    def test_meg_output_shape(self):
        """MEGEncoder maps (B, 272, 225) -> (B, 128)."""
        model = MEGEncoder()
        model.eval()
        with torch.no_grad():
            out = model(_rand_meg())
        self.assertEqual(out.shape, (BATCH, EMBED_DIM),
                         f"Expected ({BATCH}, {EMBED_DIM}), got {tuple(out.shape)}")

    def test_fmri_output_shape(self):
        """FMRIEncoder maps (B, 1, 96, 96, 60) -> (B, 128)."""
        model = FMRIEncoder()
        model.eval()
        with torch.no_grad():
            out = model(_rand_fmri())
        self.assertEqual(out.shape, (BATCH, EMBED_DIM),
                         f"Expected ({BATCH}, {EMBED_DIM}), got {tuple(out.shape)}")

    def test_custom_embed_dim(self):
        """All encoders honour a custom embed_dim argument."""
        for cls, x in [
            (EEGEncoder, _rand_eeg()),
            (MEGEncoder, _rand_meg()),
            (FMRIEncoder, _rand_fmri()),
        ]:
            model = cls(embed_dim=64)
            model.eval()
            with torch.no_grad():
                out = model(x)
            self.assertEqual(out.shape[-1], 64,
                             f"{cls.__name__} custom embed_dim mismatch")


# ---------------------------------------------------------------------------
# Finite value tests
# ---------------------------------------------------------------------------

class TestEncoderFiniteValues(unittest.TestCase):

    def _check_finite(self, out: torch.Tensor, name: str):
        self.assertFalse(torch.isnan(out).any().item(),
                         f"{name} embedding contains NaN values.")
        self.assertFalse(torch.isinf(out).any().item(),
                         f"{name} embedding contains Inf values.")

    def test_eeg_finite(self):
        model = EEGEncoder().eval()
        with torch.no_grad():
            out = model(_rand_eeg())
        self._check_finite(out, "EEGEncoder")

    def test_meg_finite(self):
        model = MEGEncoder().eval()
        with torch.no_grad():
            out = model(_rand_meg())
        self._check_finite(out, "MEGEncoder")

    def test_fmri_finite(self):
        model = FMRIEncoder().eval()
        with torch.no_grad():
            out = model(_rand_fmri())
        self._check_finite(out, "FMRIEncoder")


# ---------------------------------------------------------------------------
# L2 normalisation tests
# ---------------------------------------------------------------------------

class TestEmbeddingNormalisation(unittest.TestCase):

    def _check_unit_norm(self, out: torch.Tensor, name: str):
        norms = out.norm(p=2, dim=1)
        max_deviation = (norms - 1.0).abs().max().item()
        self.assertAlmostEqual(
            max_deviation, 0.0, places=5,
            msg=f"{name}: L2 norms deviate from 1 by {max_deviation:.2e}"
        )

    def test_eeg_unit_norm(self):
        model = EEGEncoder().eval()
        with torch.no_grad():
            out = model(_rand_eeg())
        self._check_unit_norm(out, "EEGEncoder")

    def test_meg_unit_norm(self):
        model = MEGEncoder().eval()
        with torch.no_grad():
            out = model(_rand_meg())
        self._check_unit_norm(out, "MEGEncoder")

    def test_fmri_unit_norm(self):
        model = FMRIEncoder().eval()
        with torch.no_grad():
            out = model(_rand_fmri())
        self._check_unit_norm(out, "FMRIEncoder")


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation(unittest.TestCase):

    # --- Wrong channel count ---

    def test_eeg_wrong_channels(self):
        """EEGEncoder raises ValueError for wrong number of channels."""
        model = EEGEncoder(n_channels=63)
        x = torch.randn(BATCH, 32, EEG_TIMES)  # 32 channels instead of 63
        with self.assertRaises(ValueError) as ctx:
            model(x)
        self.assertIn("63", str(ctx.exception))

    def test_meg_wrong_channels(self):
        """MEGEncoder raises ValueError for wrong number of channels."""
        model = MEGEncoder(n_channels=272)
        x = torch.randn(BATCH, 64, MEG_TIMES)
        with self.assertRaises(ValueError) as ctx:
            model(x)
        self.assertIn("272", str(ctx.exception))

    def test_fmri_wrong_channels(self):
        """FMRIEncoder raises ValueError for wrong number of input channels."""
        model = FMRIEncoder(in_channels=1)
        x = torch.randn(BATCH, 2, FMRI_X, FMRI_Y, FMRI_Z)
        with self.assertRaises(ValueError) as ctx:
            model(x)
        self.assertIn("1", str(ctx.exception))

    # --- Wrong tensor rank ---

    def test_eeg_wrong_rank(self):
        """EEGEncoder raises ValueError for non-3-D input."""
        model = EEGEncoder()
        x = torch.randn(BATCH, EEG_CHANNELS)  # 2-D, missing time axis
        with self.assertRaises(ValueError) as ctx:
            model(x)
        self.assertIn("3-D", str(ctx.exception))

    def test_meg_wrong_rank(self):
        """MEGEncoder raises ValueError for non-3-D input."""
        model = MEGEncoder()
        x = torch.randn(MEG_CHANNELS, MEG_TIMES)  # missing batch dim
        with self.assertRaises(ValueError) as ctx:
            model(x)
        self.assertIn("3-D", str(ctx.exception))

    def test_fmri_wrong_rank(self):
        """FMRIEncoder raises ValueError for non-5-D input."""
        model = FMRIEncoder()
        x = torch.randn(BATCH, 1, FMRI_X, FMRI_Y)  # 4-D, missing Z
        with self.assertRaises(ValueError) as ctx:
            model(x)
        self.assertIn("5-D", str(ctx.exception))


# ---------------------------------------------------------------------------
# Input dimension validation tests (n_times / volume_shape)
# ---------------------------------------------------------------------------

class TestInputDimensionValidation(unittest.TestCase):

    def test_eeg_wrong_n_times(self):
        """EEGEncoder raises ValueError when time dimension != n_times."""
        model = EEGEncoder(n_times=225)
        x = torch.randn(BATCH, EEG_CHANNELS, 200)  # 200 != 225
        with self.assertRaises(ValueError) as ctx:
            model(x)
        msg = str(ctx.exception)
        self.assertIn("225", msg)
        self.assertIn("200", msg)

    def test_meg_wrong_n_times(self):
        """MEGEncoder raises ValueError when time dimension != n_times."""
        model = MEGEncoder(n_times=225)
        x = torch.randn(BATCH, MEG_CHANNELS, 300)  # 300 != 225
        with self.assertRaises(ValueError) as ctx:
            model(x)
        msg = str(ctx.exception)
        self.assertIn("225", msg)
        self.assertIn("300", msg)

    def test_fmri_wrong_volume_shape(self):
        """FMRIEncoder raises ValueError when spatial dims != volume_shape."""
        model = FMRIEncoder(volume_shape=(96, 96, 60))
        x = torch.randn(BATCH, 1, 64, 64, 40)  # wrong X, Y, Z
        with self.assertRaises(ValueError) as ctx:
            model(x)
        msg = str(ctx.exception)
        # Both expected and actual spatial shapes should appear in the message
        self.assertIn("96", msg)
        self.assertIn("64", msg)

    def test_eeg_correct_n_times_accepted(self):
        """EEGEncoder accepts tensors whose time dimension equals n_times."""
        model = EEGEncoder(n_times=225).eval()
        x = torch.randn(BATCH, EEG_CHANNELS, 225)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (BATCH, EMBED_DIM))

    def test_fmri_correct_volume_shape_accepted(self):
        """FMRIEncoder accepts tensors whose spatial shape equals volume_shape."""
        model = FMRIEncoder(volume_shape=(96, 96, 60)).eval()
        x = torch.randn(BATCH, 1, 96, 96, 60)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (BATCH, EMBED_DIM))


# ---------------------------------------------------------------------------
# Backpropagation tests
# ---------------------------------------------------------------------------

class TestBackpropagation(unittest.TestCase):

    def _check_gradients(self, model: torch.nn.Module, x: torch.Tensor, name: str):
        model.train()
        out = model(x)
        # Simple MSE loss against zero target as a proxy for any real objective
        target = torch.zeros_like(out)
        loss = F.mse_loss(out, target)
        loss.backward()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_params), 0,
                           f"{name} has no trainable parameters.")

        grad_magnitudes = [
            p.grad.abs().max().item()
            for p in trainable_params
            if p.grad is not None
        ]
        params_without_grad = sum(
            1 for p in trainable_params if p.grad is None
        )
        self.assertEqual(params_without_grad, 0,
                         f"{name} has {params_without_grad} param(s) without gradients.")
        self.assertTrue(
            all(g > 0.0 for g in grad_magnitudes),
            f"{name} has zero-magnitude gradients: {grad_magnitudes}"
        )

    def test_eeg_backprop(self):
        model = EEGEncoder()
        x = _rand_eeg()
        self._check_gradients(model, x, "EEGEncoder")

    def test_meg_backprop(self):
        model = MEGEncoder()
        x = _rand_meg()
        self._check_gradients(model, x, "MEGEncoder")

    def test_fmri_backprop(self):
        model = FMRIEncoder()
        x = _rand_fmri()
        self._check_gradients(model, x, "FMRIEncoder")


# ---------------------------------------------------------------------------
# Batch-size-1 tests (eval mode)
# ---------------------------------------------------------------------------

class TestBatchSizeOne(unittest.TestCase):
    """Verify encoders behave correctly for single-sample inference."""

    def _check_output(self, out: torch.Tensor, name: str):
        self.assertEqual(out.shape, (1, EMBED_DIM),
                         f"{name} batch-1 shape wrong: {tuple(out.shape)}")
        self.assertFalse(torch.isnan(out).any().item(),
                         f"{name} batch-1 embedding has NaN.")
        self.assertFalse(torch.isinf(out).any().item(),
                         f"{name} batch-1 embedding has Inf.")
        norm = out.norm(p=2, dim=1).item()
        self.assertAlmostEqual(norm, 1.0, places=5,
                               msg=f"{name} batch-1 L2 norm={norm:.6f}, expected 1.0")

    def test_eeg_batch_size_1(self):
        """EEGEncoder: batch=1, output shape (1, 128), finite, L2-normed."""
        model = EEGEncoder().eval()
        x = torch.randn(1, EEG_CHANNELS, EEG_TIMES)
        with torch.no_grad():
            out = model(x)
        self._check_output(out, "EEGEncoder")

    def test_meg_batch_size_1(self):
        """MEGEncoder: batch=1, output shape (1, 128), finite, L2-normed."""
        model = MEGEncoder().eval()
        x = torch.randn(1, MEG_CHANNELS, MEG_TIMES)
        with torch.no_grad():
            out = model(x)
        self._check_output(out, "MEGEncoder")

    def test_fmri_batch_size_1(self):
        """FMRIEncoder: batch=1, output shape (1, 128), finite, L2-normed."""
        model = FMRIEncoder().eval()
        x = torch.randn(1, 1, FMRI_X, FMRI_Y, FMRI_Z)
        with torch.no_grad():
            out = model(x)
        self._check_output(out, "FMRIEncoder")


# ---------------------------------------------------------------------------
# CPU / MPS device tests
# ---------------------------------------------------------------------------

class TestDeviceCompatibility(unittest.TestCase):

    def test_eeg_cpu(self):
        model = EEGEncoder().to("cpu")
        model.eval()
        with torch.no_grad():
            out = model(_rand_eeg(device="cpu"))
        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.shape, (BATCH, EMBED_DIM))

    def test_meg_cpu(self):
        model = MEGEncoder().to("cpu")
        model.eval()
        with torch.no_grad():
            out = model(_rand_meg(device="cpu"))
        self.assertEqual(out.device.type, "cpu")

    def test_fmri_cpu(self):
        model = FMRIEncoder().to("cpu")
        model.eval()
        with torch.no_grad():
            out = model(_rand_fmri(device="cpu"))
        self.assertEqual(out.device.type, "cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available on this machine")
    def test_eeg_mps(self):
        model = EEGEncoder().to("mps")
        model.eval()
        with torch.no_grad():
            out = model(_rand_eeg(device="mps"))
        self.assertEqual(out.device.type, "mps")
        self.assertEqual(out.shape, (BATCH, EMBED_DIM))
        self.assertFalse(torch.isnan(out).any().item())

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available on this machine")
    def test_meg_mps(self):
        model = MEGEncoder().to("mps")
        model.eval()
        with torch.no_grad():
            out = model(_rand_meg(device="mps"))
        self.assertEqual(out.device.type, "mps")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available on this machine")
    def test_fmri_mps(self):
        model = FMRIEncoder().to("mps")
        model.eval()
        with torch.no_grad():
            out = model(_rand_fmri(device="mps"))
        self.assertEqual(out.device.type, "mps")


# ---------------------------------------------------------------------------
# Dropout tests
# ---------------------------------------------------------------------------

class TestEncoderDropout(unittest.TestCase):

    def test_dropout_active_during_training(self):
        """In train mode with dropout=0.2, successive forward passes produce different features."""
        torch.manual_seed(42)
        x_eeg = _rand_eeg()
        x_meg = _rand_meg()
        x_fmri = _rand_fmri()

        eeg_enc = EEGEncoder(dropout=0.2).train()
        meg_enc = MEGEncoder(dropout=0.2).train()
        fmri_enc = FMRIEncoder(dropout=0.2).train()

        # Successive passes in train mode
        out1_eeg = eeg_enc(x_eeg)
        out2_eeg = eeg_enc(x_eeg)
        self.assertFalse(torch.equal(out1_eeg, out2_eeg), "EEG dropout was inactive during training!")

        out1_meg = meg_enc(x_meg)
        out2_meg = meg_enc(x_meg)
        self.assertFalse(torch.equal(out1_meg, out2_meg), "MEG dropout was inactive during training!")

        out1_fmri = fmri_enc(x_fmri)
        out2_fmri = fmri_enc(x_fmri)
        self.assertFalse(torch.equal(out1_fmri, out2_fmri), "fMRI dropout was inactive during training!")

    def test_dropout_disabled_during_eval(self):
        """In eval mode with dropout=0.2, forward passes are deterministic and identical."""
        x_eeg = _rand_eeg()
        x_meg = _rand_meg()
        x_fmri = _rand_fmri()

        eeg_enc = EEGEncoder(dropout=0.2).eval()
        meg_enc = MEGEncoder(dropout=0.2).eval()
        fmri_enc = FMRIEncoder(dropout=0.2).eval()

        with torch.no_grad():
            out1_eeg = eeg_enc(x_eeg)
            out2_eeg = eeg_enc(x_eeg)
            self.assertTrue(torch.equal(out1_eeg, out2_eeg), "EEG eval mode was non-deterministic!")

            out1_meg = meg_enc(x_meg)
            out2_meg = meg_enc(x_meg)
            self.assertTrue(torch.equal(out1_meg, out2_meg), "MEG eval mode was non-deterministic!")

            out1_fmri = fmri_enc(x_fmri)
            out2_fmri = fmri_enc(x_fmri)
            self.assertTrue(torch.equal(out1_fmri, out2_fmri), "fMRI eval mode was non-deterministic!")

    def test_dropout_preserves_shapes_and_finiteness(self):
        """Dropout maintains (B, 128) shape and unit norm."""
        encs = {
            "eeg": EEGEncoder(dropout=0.2),
            "meg": MEGEncoder(dropout=0.2),
            "fmri": FMRIEncoder(dropout=0.2),
        }
        inputs = {
            "eeg": _rand_eeg(batch=3),
            "meg": _rand_meg(batch=3),
            "fmri": _rand_fmri(batch=3),
        }
        for mod, enc in encs.items():
            enc.eval()
            with torch.no_grad():
                out = enc(inputs[mod])
                self.assertEqual(out.shape, (3, 128))
                self.assertTrue(torch.isfinite(out).all().item())
                norms = torch.norm(out, p=2, dim=1)
                self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))



# ---------------------------------------------------------------------------
# LayerNorm projection-head tests (added for Phase 4 collapse fix)
# ---------------------------------------------------------------------------

class TestLayerNormProjection(unittest.TestCase):
    """Verify the nn.LayerNorm inserted between self.proj and F.normalize.

    These tests check four properties without running training:
      1. Structural: pre_norm attribute exists and has the correct type/shape.
      2. Diversity: mean off-diagonal cosine at fresh init is below +0.80,
         confirming the collapse is broken (diagnostic baseline was +0.98).
      3. Gradient flow: pre_norm weight and bias receive gradients on backward.
      4. Contract: output shape, finiteness, and unit norm are preserved.
    """

    # --- 1. Structural ---

    def test_pre_norm_attribute_exists_eeg(self):
        enc = EEGEncoder()
        self.assertTrue(hasattr(enc, "pre_norm"),
                        "EEGEncoder missing 'pre_norm' attribute.")
        self.assertIsInstance(enc.pre_norm, torch.nn.LayerNorm)
        self.assertEqual(enc.pre_norm.normalized_shape, (EMBED_DIM,),
                         f"EEGEncoder pre_norm normalized_shape wrong: {enc.pre_norm.normalized_shape}")

    def test_pre_norm_attribute_exists_meg(self):
        enc = MEGEncoder()
        self.assertTrue(hasattr(enc, "pre_norm"),
                        "MEGEncoder missing 'pre_norm' attribute.")
        self.assertIsInstance(enc.pre_norm, torch.nn.LayerNorm)
        self.assertEqual(enc.pre_norm.normalized_shape, (EMBED_DIM,))

    def test_pre_norm_attribute_exists_fmri(self):
        enc = FMRIEncoder()
        self.assertTrue(hasattr(enc, "pre_norm"),
                        "FMRIEncoder missing 'pre_norm' attribute.")
        self.assertIsInstance(enc.pre_norm, torch.nn.LayerNorm)
        self.assertEqual(enc.pre_norm.normalized_shape, (EMBED_DIM,))

    # --- 2. Embedding diversity at fresh init ---

    def _mean_off_diag_cosine(self, z: torch.Tensor) -> float:
        """Mean cosine similarity of all off-diagonal pairs in a (N, D) embedding."""
        cos = torch.mm(z, z.t())
        n = cos.shape[0]
        off = [cos[i, j].item() for i in range(n) for j in range(n) if i != j]
        return sum(off) / len(off)

    def test_eeg_init_cosine_below_threshold(self):
        """EEGEncoder at fresh init: mean off-diag cosine < +0.80.

        Diagnostic without LayerNorm showed +0.982; with LayerNorm this
        should be substantially lower due to per-sample centering.
        """
        torch.manual_seed(0)
        enc = EEGEncoder().eval()
        with torch.no_grad():
            z = enc(_rand_eeg(batch=8))
        mean_cos = self._mean_off_diag_cosine(z)
        self.assertLess(mean_cos, 0.80,
            f"EEGEncoder post-LayerNorm mean off-diag cosine = {mean_cos:.4f} "
            f">= 0.80; cone collapse may not be fixed.")

    def test_meg_init_cosine_below_threshold(self):
        """MEGEncoder at fresh init: mean off-diag cosine < +0.80."""
        torch.manual_seed(0)
        enc = MEGEncoder().eval()
        with torch.no_grad():
            z = enc(_rand_meg(batch=8))
        mean_cos = self._mean_off_diag_cosine(z)
        self.assertLess(mean_cos, 0.80,
            f"MEGEncoder post-LayerNorm mean off-diag cosine = {mean_cos:.4f} >= 0.80.")

    def test_fmri_init_cosine_below_threshold(self):
        """FMRIEncoder at fresh init: mean off-diag cosine < +0.80.

        Diagnostic without LayerNorm showed +0.999; this is the most
        critical test.
        """
        torch.manual_seed(0)
        enc = FMRIEncoder().eval()
        with torch.no_grad():
            z = enc(_rand_fmri(batch=4))
        mean_cos = self._mean_off_diag_cosine(z)
        self.assertLess(mean_cos, 0.80,
            f"FMRIEncoder post-LayerNorm mean off-diag cosine = {mean_cos:.4f} >= 0.80.")

    # --- 3. Gradient flow through pre_norm ---

    def _check_pre_norm_gradients(self, enc, x, name):
        enc.train()
        out = enc(x)
        loss = out.sum()
        loss.backward()
        weight_grad = enc.pre_norm.weight.grad
        bias_grad   = enc.pre_norm.bias.grad
        self.assertIsNotNone(weight_grad,
            f"{name} pre_norm.weight has no gradient.")
        self.assertIsNotNone(bias_grad,
            f"{name} pre_norm.bias has no gradient.")
        self.assertGreater(weight_grad.abs().max().item(), 0.0,
            f"{name} pre_norm.weight gradient is all-zero.")
        self.assertGreater(bias_grad.abs().max().item(), 0.0,
            f"{name} pre_norm.bias gradient is all-zero.")

    def test_eeg_pre_norm_gradient_flow(self):
        self._check_pre_norm_gradients(EEGEncoder(), _rand_eeg(), "EEGEncoder")

    def test_meg_pre_norm_gradient_flow(self):
        self._check_pre_norm_gradients(MEGEncoder(), _rand_meg(), "MEGEncoder")

    def test_fmri_pre_norm_gradient_flow(self):
        self._check_pre_norm_gradients(FMRIEncoder(), _rand_fmri(), "FMRIEncoder")

    # --- 4. Output contract preserved ---

    def test_shape_finite_unit_norm_with_layer_norm(self):
        """Shape (B,128), finite values, and L2 norm ≈ 1 hold with LayerNorm."""
        encs = {
            "eeg":  (EEGEncoder(),  _rand_eeg(batch=6)),
            "meg":  (MEGEncoder(),  _rand_meg(batch=6)),
            "fmri": (FMRIEncoder(), _rand_fmri(batch=6)),
        }
        for name, (enc, x) in encs.items():
            enc.eval()
            with torch.no_grad():
                out = enc(x)
            self.assertEqual(out.shape, (6, EMBED_DIM),
                             f"{name} shape wrong: {tuple(out.shape)}")
            self.assertTrue(torch.isfinite(out).all().item(),
                            f"{name} output contains non-finite values.")
            norms = out.norm(p=2, dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
                            f"{name} L2 norms deviate from 1: {norms.tolist()}")

    def test_train_mode_shape_and_finiteness_with_layer_norm(self):
        """LayerNorm does not break shape/finiteness in train mode."""
        encs = {
            "eeg":  (EEGEncoder(),  _rand_eeg(batch=4)),
            "meg":  (MEGEncoder(),  _rand_meg(batch=4)),
            "fmri": (FMRIEncoder(), _rand_fmri(batch=4)),
        }
        for name, (enc, x) in encs.items():
            enc.train()
            out = enc(x)
            self.assertEqual(out.shape, (4, EMBED_DIM),
                             f"{name} train-mode shape wrong: {tuple(out.shape)}")
            self.assertTrue(torch.isfinite(out).all().item(),
                            f"{name} train-mode output has non-finite values.")


if __name__ == "__main__":
    unittest.main()

