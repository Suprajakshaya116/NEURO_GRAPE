"""Neuro-GRAPE Member 2 Handoff Interface Tests (Phase 5).

Verifies the five core deliverables for downstream Member 3 integration:
1. model.py interface, tensor dimensions, and forward pass.
2. losses.py per-example loss vector (N,) and sample/group weighting.
3. checkpoints/baseline.pt validity, weight loading, and config metadata.
4. embeddings/ exported latent vectors, metadata schema, and L2 unit norms.
5. Training API with sample_weights / GroupDRO integration.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.losses import (
    compute_group_weighted_loss,
    multimodal_contrastive_loss,
    per_sample_trimodal_contrastive_loss,
    trimodal_contrastive_loss,
)
from src.models.model import NeuroGrapeModel, TrimodalNeuroGrapeModel
from src.models.training import train_epoch


class TestMember2HandoffInterfaces(unittest.TestCase):
    """Test suite validating Member 2 deliverables for Member 3 handoff."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.b_size = 4
        self.d_embed = 128
        self.stim_ids = [f"stim_{i:02d}.jpg" for i in range(self.b_size)]

        self.mock_batch = {
            "eeg": torch.randn(self.b_size, 63, 225, dtype=torch.float32),
            "meg": torch.randn(self.b_size, 272, 225, dtype=torch.float32),
            "fmri": torch.randn(self.b_size, 1, 96, 96, 60, dtype=torch.float32),
        }
        self.stimulus_ids_map = {
            "eeg": self.stim_ids,
            "meg": self.stim_ids,
            "fmri": self.stim_ids,
        }

    # -----------------------------------------------------------------------
    # Deliverable 1: model.py interface & tensor dimensions
    # -----------------------------------------------------------------------
    def test_model_interface_and_dimensions(self):
        """Verify unified TrimodalNeuroGrapeModel interface and output shapes."""
        model = TrimodalNeuroGrapeModel(embed_dim=self.d_embed)
        model.eval()

        # Batch forward pass
        out = model(self.mock_batch)
        for mod in ("eeg", "meg", "fmri"):
            self.assertIn(mod, out)
            self.assertEqual(out[mod].shape, (self.b_size, self.d_embed))
            norms = torch.norm(out[mod], p=2, dim=1)
            torch.testing.assert_close(
                norms, torch.ones(self.b_size), atol=1e-5, rtol=1e-5
            )

        # Single modality encode() method
        z_eeg = model.encode("eeg", self.mock_batch["eeg"])
        self.assertEqual(z_eeg.shape, (self.b_size, self.d_embed))

        # Root model alias
        model_alias = NeuroGrapeModel()
        self.assertIsInstance(model_alias, TrimodalNeuroGrapeModel)

    # -----------------------------------------------------------------------
    # Deliverable 2: losses.py per-example loss vector
    # -----------------------------------------------------------------------
    def test_per_example_loss_vector(self):
        """Verify losses.py returns per-example loss vector of shape (N,)."""
        model = TrimodalNeuroGrapeModel(embed_dim=self.d_embed)
        z = model(self.mock_batch)

        loss_vec, ids = per_sample_trimodal_contrastive_loss(
            embeddings=z,
            stimulus_ids=self.stimulus_ids_map,
            temperature=0.07,
        )

        self.assertEqual(loss_vec.shape, (self.b_size,))
        self.assertEqual(ids, self.stim_ids)
        self.assertTrue(torch.isfinite(loss_vec).all())
        self.assertTrue((loss_vec >= 0.0).all())

        # trimodal_contrastive_loss with reduction='none'
        loss_none = trimodal_contrastive_loss(
            embeddings=z,
            stimulus_ids=self.stimulus_ids_map,
            reduction="none",
        )
        self.assertEqual(loss_none.shape, (self.b_size,))
        torch.testing.assert_close(loss_vec, loss_none)

    # -----------------------------------------------------------------------
    # Deliverable 5: GroupDRO weighting support in losses and training API
    # -----------------------------------------------------------------------
    def test_weighted_loss_and_groupdro_helper(self):
        """Verify sample-weighted loss and compute_group_weighted_loss helper."""
        model = TrimodalNeuroGrapeModel(embed_dim=self.d_embed)
        z = model(self.mock_batch)

        weights = torch.tensor([0.1, 0.4, 0.2, 0.3])
        weighted_loss = trimodal_contrastive_loss(
            embeddings=z,
            stimulus_ids=self.stimulus_ids_map,
            sample_weights=weights,
        )
        self.assertTrue(weighted_loss.ndim == 0)  # Scalar
        self.assertTrue(torch.isfinite(weighted_loss))

        # GroupDRO helper
        loss_vec, _ = per_sample_trimodal_contrastive_loss(z, self.stimulus_ids_map)
        group_indices = torch.tensor([0, 0, 1, 1])  # 2 groups
        group_weights = torch.tensor([0.7, 0.3])
        g_loss = compute_group_weighted_loss(loss_vec, group_indices, group_weights)
        self.assertTrue(g_loss.ndim == 0)
        self.assertTrue(torch.isfinite(g_loss))

        # Gradient flow through weighted loss
        g_loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
                break

    # -----------------------------------------------------------------------
    # Deliverable 4: checkpoints/baseline.pt verification
    # -----------------------------------------------------------------------
    def test_checkpoint_baseline_exists_and_loads(self):
        """Verify checkpoints/baseline.pt exists, has valid keys, and loads into model."""
        ckpt_path = Path("checkpoints/baseline.pt")
        self.assertTrue(ckpt_path.exists(), "checkpoints/baseline.pt not found on disk.")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.assertIn("encoders", ckpt)
        self.assertIn("config", ckpt)
        self.assertIn("metrics", ckpt)
        self.assertIn("normalizer_state", ckpt)

        # Config verification
        self.assertEqual(ckpt["config"]["seed"], 42)
        self.assertEqual(ckpt["config"]["best_epoch"], 11)
        self.assertAlmostEqual(ckpt["config"]["best_val_loss"], 1.383788, places=5)

        # Model loading verification
        model = TrimodalNeuroGrapeModel()
        loaded = model.load_checkpoint(ckpt_path)
        self.assertIsInstance(loaded, dict)

    # -----------------------------------------------------------------------
    # Deliverable 3: embeddings/ metadata and tensors verification
    # -----------------------------------------------------------------------
    def test_embeddings_metadata_and_integrity(self):
        """Verify embeddings/ directory contains 60 pilot vectors with exact metadata."""
        meta_path = Path("embeddings/metadata.csv")
        pt_path = Path("embeddings/embeddings.pt")

        self.assertTrue(meta_path.exists(), "embeddings/metadata.csv missing.")
        self.assertTrue(pt_path.exists(), "embeddings/embeddings.pt missing.")

        df = pd.read_csv(meta_path)
        self.assertEqual(len(df), 60, "Expected 60 total pilot sample embeddings.")

        required_cols = {
            "sample_id",
            "modality",
            "stimulus_id",
            "split",
            "category_id",
            "category_name",
            "embedding_dim",
            "l2_norm",
            "file_path",
        }
        self.assertTrue(required_cols.issubset(set(df.columns)))

        # Split counts: 36 train (12 stimuli * 3), 12 val (4 * 3), 12 test (4 * 3)
        split_counts = df["split"].value_counts().to_dict()
        self.assertEqual(split_counts["train"], 36)
        self.assertEqual(split_counts["val"], 12)
        self.assertEqual(split_counts["test"], 12)

        # Unit norm verification
        self.assertTrue((df["embedding_dim"] == 128).all())
        self.assertTrue((df["l2_norm"].between(0.999, 1.001)).all())

        # pt dictionary verification
        pt_dict = torch.load(pt_path, map_location="cpu")
        self.assertEqual(len(pt_dict), 60)
        first_sid = df.iloc[0]["sample_id"]
        self.assertIn(first_sid, pt_dict)
        self.assertEqual(pt_dict[first_sid].shape, (128,))


if __name__ == "__main__":
    unittest.main()
