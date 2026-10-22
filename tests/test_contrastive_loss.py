"""
Focused Test Suite for Neuro-GRAPE Contrastive Alignment Loss
=============================================================
Member 2 — Phase 3: Loss module tests only.
Does NOT test encoder forward passes, training loops, or data loading.

Tests
-----
1.  Identical embeddings with matching IDs yield a finite loss.
2.  Permuting one modality's row order (with consistent ID permutation)
    does not change the loss value.
3.  Gradients propagate to both embedding tensors.
4.  Unequal row counts (N != M) with valid matches are accepted.
5.  Multiple positive matches per row are handled correctly.
6.  Trimodal helper combines all three pairs and returns a finite scalar.
7a. Invalid rank on z_a raises ValueError.
7b. Invalid rank on z_b raises ValueError.
7c. Embedding dimension mismatch raises ValueError.
7d. ids_a length mismatch raises ValueError.
7e. ids_b length mismatch raises ValueError.
7f. Non-positive temperature raises ValueError.
7g. No positive matches raises ValueError.
7h. Missing modality key in trimodal raises ValueError.
7i. Missing stimulus_ids key in trimodal raises ValueError.
8.  Numerically stable finite result for typical randn embeddings.
9.  All tests use CPU; no MPS requirement.
"""

import sys
import math
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.contrastive_loss import (
    multimodal_contrastive_loss,
    trimodal_contrastive_loss,
)

# ---------------------------------------------------------------------------
# Tiny helper factories
# ---------------------------------------------------------------------------
D = 128  # embedding dimension used throughout


def _unit(t: torch.Tensor) -> torch.Tensor:
    """L2-normalise rows of a 2-D tensor."""
    import torch.nn.functional as F
    return F.normalize(t, p=2, dim=1)


def _rand_emb(n: int, d: int = D) -> torch.Tensor:
    return _unit(torch.randn(n, d))


def _ids(n: int, prefix: str = "s") -> list[str]:
    """Return n distinct stimulus IDs, e.g. ['s0', 's1', ...]."""
    return [f"{prefix}{i}" for i in range(n)]


def _matched_ids(stims: list[str]) -> tuple[list[str], list[str]]:
    """Return two equal ID lists (trivially matched) for the same stimuli."""
    return list(stims), list(stims)


# ---------------------------------------------------------------------------
# 1. Basic finite loss
# ---------------------------------------------------------------------------

class TestBasicFiniteLoss(unittest.TestCase):

    def test_identical_embeddings_matching_ids(self):
        """Loss is finite when embeddings and IDs are identical across modalities."""
        n = 4
        z = _rand_emb(n)
        ids = _ids(n)
        loss = multimodal_contrastive_loss(z, z.clone(), ids, list(ids))
        self.assertFalse(torch.isnan(loss).item(), "Loss is NaN.")
        self.assertFalse(torch.isinf(loss).item(), "Loss is Inf.")
        self.assertTrue(loss.item() >= 0.0, f"Loss unexpectedly negative: {loss.item()}")

    def test_loss_is_scalar(self):
        """Loss tensor must be 0-D (scalar)."""
        n = 3
        z = _rand_emb(n)
        ids = _ids(n)
        loss = multimodal_contrastive_loss(z, z.clone(), ids, list(ids))
        self.assertEqual(loss.shape, torch.Size([]))


# ---------------------------------------------------------------------------
# 2. Row-order invariance
# ---------------------------------------------------------------------------

class TestRowOrderInvariance(unittest.TestCase):

    def test_permuted_b_same_loss(self):
        """Permuting z_b rows with a matching ID permutation must not change the loss."""
        n = 5
        z_a = _rand_emb(n)
        z_b = _rand_emb(n)
        ids_a = _ids(n, prefix="stim")
        ids_b = list(ids_a)  # same order

        perm = [3, 0, 4, 1, 2]
        z_b_perm = z_b[perm]
        ids_b_perm = [ids_b[i] for i in perm]

        loss_orig = multimodal_contrastive_loss(z_a, z_b, ids_a, ids_b)
        loss_perm = multimodal_contrastive_loss(z_a, z_b_perm, ids_a, ids_b_perm)

        self.assertAlmostEqual(
            loss_orig.item(), loss_perm.item(), places=5,
            msg=f"Loss changed after permutation: {loss_orig.item():.6f} vs {loss_perm.item():.6f}"
        )

    def test_permuted_a_same_loss(self):
        """Permuting z_a rows (with IDs) must not change the loss."""
        n = 4
        z_a = _rand_emb(n)
        z_b = _rand_emb(n)
        ids_a = _ids(n, prefix="img")
        ids_b = list(ids_a)

        perm = [2, 0, 3, 1]
        z_a_perm = z_a[perm]
        ids_a_perm = [ids_a[i] for i in perm]

        loss_orig = multimodal_contrastive_loss(z_a, z_b, ids_a, ids_b)
        loss_perm = multimodal_contrastive_loss(z_a_perm, z_b, ids_a_perm, ids_b)

        self.assertAlmostEqual(
            loss_orig.item(), loss_perm.item(), places=5,
            msg=f"Loss changed after A permutation: {loss_orig.item():.6f} vs {loss_perm.item():.6f}"
        )


# ---------------------------------------------------------------------------
# 3. Gradient propagation
# ---------------------------------------------------------------------------

class TestGradientPropagation(unittest.TestCase):

    def test_gradients_reach_both_tensors(self):
        """Loss.backward() must populate .grad on both z_a and z_b."""
        n = 4
        # Normalise first (detached), then mark as leaf requiring grad.
        z_a = _unit(torch.randn(n, D)).detach().requires_grad_(True)
        z_b = _unit(torch.randn(n, D)).detach().requires_grad_(True)
        ids = _ids(n)

        loss = multimodal_contrastive_loss(z_a, z_b, ids, list(ids))
        loss.backward()

        self.assertIsNotNone(z_a.grad, "z_a.grad is None after backward().")
        self.assertIsNotNone(z_b.grad, "z_b.grad is None after backward().")
        self.assertFalse(torch.isnan(z_a.grad).any().item(), "z_a.grad has NaN.")
        self.assertFalse(torch.isnan(z_b.grad).any().item(), "z_b.grad has NaN.")
        self.assertFalse((z_a.grad == 0).all().item(), "z_a.grad is all zeros.")
        self.assertFalse((z_b.grad == 0).all().item(), "z_b.grad is all zeros.")

    def test_gradient_is_nonzero_for_imperfect_embeddings(self):
        """Gradient should be non-zero when embeddings are not perfectly aligned."""
        n = 6
        z_a = _unit(torch.randn(n, D)).detach().requires_grad_(True)
        z_b = _unit(torch.randn(n, D)).detach().requires_grad_(True)  # independent
        ids = _ids(n)

        loss = multimodal_contrastive_loss(z_a, z_b, ids, list(ids))
        loss.backward()
        grad_norm_a = z_a.grad.norm().item()
        grad_norm_b = z_b.grad.norm().item()
        self.assertGreater(grad_norm_a, 0.0, "z_a gradient norm is zero.")
        self.assertGreater(grad_norm_b, 0.0, "z_b gradient norm is zero.")


# ---------------------------------------------------------------------------
# 4. Unequal row counts
# ---------------------------------------------------------------------------

class TestUnequalRowCounts(unittest.TestCase):

    def test_n_greater_than_m(self):
        """N > M: extra rows in A without any B match are simply excluded."""
        # 5 rows in A, 3 in B; first 3 of A match B
        ids_a = ["s0", "s1", "s2", "s3", "s4"]
        ids_b = ["s0", "s1", "s2"]
        z_a = _rand_emb(5)
        z_b = _rand_emb(3)
        loss = multimodal_contrastive_loss(z_a, z_b, ids_a, ids_b)
        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())

    def test_m_greater_than_n(self):
        """M > N: extra rows in B without any A match are excluded."""
        ids_a = ["s0", "s1"]
        ids_b = ["s0", "s1", "s2", "s3", "s4"]
        z_a = _rand_emb(2)
        z_b = _rand_emb(5)
        loss = multimodal_contrastive_loss(z_a, z_b, ids_a, ids_b)
        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())


# ---------------------------------------------------------------------------
# 5. Multiple positive matches
# ---------------------------------------------------------------------------

class TestMultiplePositives(unittest.TestCase):

    def test_two_positives_per_anchor(self):
        """Each row in A matches exactly two rows in B (duplicated stimulus IDs)."""
        # A has 2 unique stims; B has each stim twice => 2 positives per A row
        ids_a = ["stim_X", "stim_Y"]
        ids_b = ["stim_X", "stim_Y", "stim_X", "stim_Y"]  # duplicated
        z_a = _rand_emb(2)
        z_b = _rand_emb(4)
        loss = multimodal_contrastive_loss(z_a, z_b, ids_a, ids_b)
        self.assertFalse(torch.isnan(loss).item(), "Loss NaN with multiple positives.")
        self.assertFalse(torch.isinf(loss).item(), "Loss Inf with multiple positives.")
        self.assertTrue(loss.item() >= 0.0)

    def test_multi_positive_differs_from_single_positive(self):
        """Loss value should change when we go from 1 to 2 positives per anchor."""
        torch.manual_seed(0)
        ids_a = ["s0", "s1", "s2"]
        z_a = _rand_emb(3)

        # Single positive per row
        ids_b_single = ["s0", "s1", "s2"]
        z_b_single = _rand_emb(3)

        # Two positives per row: duplicate each stim in B
        ids_b_double = ["s0", "s1", "s2", "s0", "s1", "s2"]
        z_b_double = torch.cat([z_b_single, _rand_emb(3)], dim=0)

        loss_single = multimodal_contrastive_loss(z_a, z_b_single, ids_a, ids_b_single)
        loss_double = multimodal_contrastive_loss(z_a, z_b_double, ids_a, ids_b_double)

        # Values should be different (not guaranteed direction, but not equal)
        self.assertNotAlmostEqual(
            loss_single.item(), loss_double.item(), places=4,
            msg="Single vs double positive losses are unexpectedly identical."
        )


# ---------------------------------------------------------------------------
# 6. Trimodal helper
# ---------------------------------------------------------------------------

class TestTrimodalHelper(unittest.TestCase):

    def _make_trimodal_inputs(self, n: int = 4):
        stims = _ids(n, prefix="vis")
        embeddings = {
            "eeg":  _rand_emb(n),
            "meg":  _rand_emb(n),
            "fmri": _rand_emb(n),
        }
        stimulus_ids = {
            "eeg":  list(stims),
            "meg":  list(stims),
            "fmri": list(stims),
        }
        return embeddings, stimulus_ids

    def test_trimodal_returns_finite_scalar(self):
        emb, ids = self._make_trimodal_inputs()
        loss = trimodal_contrastive_loss(emb, ids)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertFalse(torch.isnan(loss).item(), "Trimodal loss is NaN.")
        self.assertFalse(torch.isinf(loss).item(), "Trimodal loss is Inf.")
        self.assertTrue(loss.item() >= 0.0)

    def test_trimodal_is_mean_of_three_pairs(self):
        """Trimodal result must equal the manual average of three pairwise losses."""
        n = 4
        stims = _ids(n, prefix="obj")
        emb = {
            "eeg":  _rand_emb(n),
            "meg":  _rand_emb(n),
            "fmri": _rand_emb(n),
        }
        ids = {k: list(stims) for k in ("eeg", "meg", "fmri")}

        loss_tri = trimodal_contrastive_loss(emb, ids)

        l_em  = multimodal_contrastive_loss(emb["eeg"],  emb["meg"],  ids["eeg"],  ids["meg"])
        l_ef  = multimodal_contrastive_loss(emb["eeg"],  emb["fmri"], ids["eeg"],  ids["fmri"])
        l_mf  = multimodal_contrastive_loss(emb["meg"],  emb["fmri"], ids["meg"],  ids["fmri"])
        expected = (l_em + l_ef + l_mf) / 3.0

        self.assertAlmostEqual(
            loss_tri.item(), expected.item(), places=5,
            msg=f"Trimodal mismatch: {loss_tri.item():.6f} vs {expected.item():.6f}"
        )

    def test_trimodal_gradients_propagate(self):
        """Gradients must reach all three embedding tensors via trimodal loss."""
        n = 3
        stims = _ids(n)
        emb = {
            "eeg":  _unit(torch.randn(n, D)).detach().requires_grad_(True),
            "meg":  _unit(torch.randn(n, D)).detach().requires_grad_(True),
            "fmri": _unit(torch.randn(n, D)).detach().requires_grad_(True),
        }
        ids = {k: list(stims) for k in ("eeg", "meg", "fmri")}
        loss = trimodal_contrastive_loss(emb, ids)
        loss.backward()
        for mod, z in emb.items():
            self.assertIsNotNone(z.grad, f"{mod} grad is None after backward().")
            self.assertFalse(torch.isnan(z.grad).any().item(), f"{mod} grad has NaN.")


# ---------------------------------------------------------------------------
# 7. Input validation errors
# ---------------------------------------------------------------------------

class TestInputValidationErrors(unittest.TestCase):

    def _default_inputs(self, n: int = 4):
        z = _rand_emb(n)
        ids = _ids(n)
        return z, z.clone(), ids, list(ids)

    # 7a — wrong rank on z_a
    def test_z_a_wrong_rank(self):
        z_b = _rand_emb(4)
        ids = _ids(4)
        z_a_bad = torch.randn(4, D, 1)  # 3-D
        with self.assertRaises(ValueError) as ctx:
            multimodal_contrastive_loss(z_a_bad, z_b, ids, ids)
        self.assertIn("z_a", str(ctx.exception))

    # 7b — wrong rank on z_b
    def test_z_b_wrong_rank(self):
        z_a = _rand_emb(4)
        ids = _ids(4)
        z_b_bad = torch.randn(4)  # 1-D
        with self.assertRaises(ValueError) as ctx:
            multimodal_contrastive_loss(z_a, z_b_bad, ids, ids)
        self.assertIn("z_b", str(ctx.exception))

    # 7c — embedding dimension mismatch
    def test_embedding_dim_mismatch(self):
        z_a = _rand_emb(4, d=128)
        z_b = _rand_emb(4, d=64)
        ids = _ids(4)
        with self.assertRaises(ValueError) as ctx:
            multimodal_contrastive_loss(z_a, z_b, ids, ids)
        self.assertIn("dimension", str(ctx.exception).lower())

    # 7d — ids_a length mismatch
    def test_ids_a_length_mismatch(self):
        z_a, z_b, _, ids_b = self._default_inputs(4)
        ids_a_bad = _ids(3)  # wrong length
        with self.assertRaises(ValueError) as ctx:
            multimodal_contrastive_loss(z_a, z_b, ids_a_bad, ids_b)
        self.assertIn("ids_a", str(ctx.exception))

    # 7e — ids_b length mismatch
    def test_ids_b_length_mismatch(self):
        z_a, z_b, ids_a, _ = self._default_inputs(4)
        ids_b_bad = _ids(5)  # wrong length
        with self.assertRaises(ValueError) as ctx:
            multimodal_contrastive_loss(z_a, z_b, ids_a, ids_b_bad)
        self.assertIn("ids_b", str(ctx.exception))

    # 7f — non-positive temperature
    def test_nonpositive_temperature_zero(self):
        z_a, z_b, ids_a, ids_b = self._default_inputs(4)
        with self.assertRaises(ValueError) as ctx:
            multimodal_contrastive_loss(z_a, z_b, ids_a, ids_b, temperature=0.0)
        self.assertIn("temperature", str(ctx.exception).lower())

    def test_nonpositive_temperature_negative(self):
        z_a, z_b, ids_a, ids_b = self._default_inputs(4)
        with self.assertRaises(ValueError) as ctx:
            multimodal_contrastive_loss(z_a, z_b, ids_a, ids_b, temperature=-1.0)
        self.assertIn("temperature", str(ctx.exception).lower())

    # 7g — no positive matches
    def test_no_positive_matches(self):
        z_a = _rand_emb(3)
        z_b = _rand_emb(3)
        ids_a = ["a0", "a1", "a2"]
        ids_b = ["b0", "b1", "b2"]  # completely disjoint
        with self.assertRaises(ValueError) as ctx:
            multimodal_contrastive_loss(z_a, z_b, ids_a, ids_b)
        msg = str(ctx.exception).lower()
        self.assertTrue(
            "positive" in msg or "match" in msg,
            f"Expected 'positive' or 'match' in error: {ctx.exception}"
        )

    # 7h — missing modality key in embeddings
    def test_trimodal_missing_embedding_key(self):
        emb = {"eeg": _rand_emb(3), "meg": _rand_emb(3)}  # fmri missing
        ids = {"eeg": _ids(3), "meg": _ids(3), "fmri": _ids(3)}
        with self.assertRaises(ValueError) as ctx:
            trimodal_contrastive_loss(emb, ids)
        self.assertIn("fmri", str(ctx.exception))

    # 7i — missing modality key in stimulus_ids
    def test_trimodal_missing_ids_key(self):
        n = 3
        emb = {k: _rand_emb(n) for k in ("eeg", "meg", "fmri")}
        ids = {"eeg": _ids(n), "fmri": _ids(n)}  # meg missing
        with self.assertRaises(ValueError) as ctx:
            trimodal_contrastive_loss(emb, ids)
        self.assertIn("meg", str(ctx.exception))

    # trimodal invalid temperature
    def test_trimodal_invalid_temperature(self):
        n = 3
        emb = {k: _rand_emb(n) for k in ("eeg", "meg", "fmri")}
        ids = {k: _ids(n) for k in ("eeg", "meg", "fmri")}
        with self.assertRaises(ValueError) as ctx:
            trimodal_contrastive_loss(emb, ids, temperature=0.0)
        self.assertIn("temperature", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# 8. Numerical stability
# ---------------------------------------------------------------------------

class TestNumericalStability(unittest.TestCase):

    def test_finite_for_random_unit_embeddings(self):
        """Loss should be finite for typical randomly-initialised unit embeddings."""
        torch.manual_seed(42)
        n = 12  # pilot train set size
        z_a = _rand_emb(n)
        z_b = _rand_emb(n)
        ids = _ids(n, prefix="stim")
        loss = multimodal_contrastive_loss(z_a, z_b, ids, list(ids))
        self.assertFalse(torch.isnan(loss).item(), "NaN loss for n=12 random embeddings.")
        self.assertFalse(torch.isinf(loss).item(), "Inf loss for n=12 random embeddings.")

    def test_loss_bounded_below_by_zero(self):
        """Cross-entropy is non-negative by construction."""
        n = 8
        z_a = _rand_emb(n)
        z_b = _rand_emb(n)
        ids = _ids(n)
        loss = multimodal_contrastive_loss(z_a, z_b, ids, list(ids))
        self.assertGreaterEqual(loss.item(), 0.0,
                                f"Loss is negative: {loss.item():.6f}")

    def test_perfect_alignment_low_loss(self):
        """When z_a == z_b exactly, the loss should be low (approaching log(1/N))."""
        # With identical embeddings, positive dot product = 1/tau >> negatives
        # for well-separated random unit vectors. The loss should be < log(N).
        n = 6
        torch.manual_seed(7)
        z = _rand_emb(n)  # random but fixed unit vectors
        ids = _ids(n)
        loss = multimodal_contrastive_loss(z, z.clone(), ids, list(ids), temperature=0.07)
        upper_bound = math.log(n)  # maximum theoretical value (uniform distribution)
        self.assertLess(
            loss.item(), upper_bound,
            f"Loss {loss.item():.4f} not below log(N)={upper_bound:.4f} for identical embeddings."
        )

    def test_different_temperatures(self):
        """Loss decreases or stays finite across a range of valid temperatures."""
        n = 5
        torch.manual_seed(1)
        z_a = _rand_emb(n)
        z_b = _rand_emb(n)
        ids = _ids(n)
        for tau in [0.01, 0.07, 0.1, 0.5, 1.0]:
            loss = multimodal_contrastive_loss(z_a, z_b, ids, list(ids), temperature=tau)
            self.assertFalse(
                torch.isnan(loss).item() or torch.isinf(loss).item(),
                f"Loss is not finite at temperature={tau}."
            )


if __name__ == "__main__":
    unittest.main()
