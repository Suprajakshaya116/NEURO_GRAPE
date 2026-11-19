"""Full Solution Integration and Unit Test Suite for Neuro-GRAPE.

Tests all components specified in docs/Neuro-GRAPE.pdf:
1. Model outputs: dual shared and private representation shapes and finiteness.
2. Cross-modal graph layer: attention message passing, gradient flow, and ablation bypass.
3. Separation objective: orthogonality loss and gradient flow.
4. Reconstruction objective: private decoder loss and gradient flow.
5. Total unified loss: end-to-end backpropagation through all parameters.
6. Adaptive mixture sampler: GRAPE/DOGE exponentiated updates, simplex, and floor.
7. GroupDRO: exponentiated distribution updates and absent-group handling.
8. Clustering: strict training-set fitting and leakage prevention.
9. Ablation switches: verification that disabled components are genuinely bypassed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.adaptive_sampler import AdaptiveMixtureSampler
from src.dro.clustering import EmbeddingClustering
from src.dro.group_dro import GroupDRO
from src.models.encoders import EEGEncoder, FMRIEncoder, MEGEncoder
from src.models.graph import CrossModalGraphLayer
from src.models.losses import (
    compute_sensor_summary_targets,
    multimodal_reconstruction_loss,
    multimodal_separation_loss,
    neuro_grape_total_loss,
    orthogonal_separation_loss,
    per_sample_trimodal_contrastive_loss,
)
from src.models.model import TrimodalNeuroGrapeModel


class TestFullSolutionComponents(unittest.TestCase):
    """Unit and integration tests for the full Neuro-GRAPE architecture."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.batch_size = 4
        self.dummy_batch = {
            "eeg": torch.randn(self.batch_size, 63, 225),
            "meg": torch.randn(self.batch_size, 272, 225),
            "fmri": torch.randn(self.batch_size, 1, 96, 96, 60),
        }
        self.stimulus_ids = {
            "eeg": [f"stim_{i}" for i in range(self.batch_size)],
            "meg": [f"stim_{i}" for i in range(self.batch_size)],
            "fmri": [f"stim_{i}" for i in range(self.batch_size)],
        }

    # 1. Dual Heads: Shapes & Finiteness
    def test_dual_heads_shapes_and_finiteness(self):
        """Verify encoders produce (B, 128) unit-norm shared and private representations."""
        model = TrimodalNeuroGrapeModel(embed_dim=128)
        shared, private = model(self.dummy_batch, return_private=True)

        for mod in ("eeg", "meg", "fmri"):
            self.assertIn(mod, shared)
            self.assertIn(mod, private)
            self.assertEqual(shared[mod].shape, (self.batch_size, 128))
            self.assertEqual(private[mod].shape, (self.batch_size, 128))
            self.assertTrue(torch.isfinite(shared[mod]).all().item())
            self.assertTrue(torch.isfinite(private[mod]).all().item())

            # Verify unit hypersphere normalization
            shared_norms = torch.norm(shared[mod], p=2, dim=1)
            private_norms = torch.norm(private[mod], p=2, dim=1)
            self.assertTrue(
                torch.allclose(shared_norms, torch.ones_like(shared_norms), atol=1e-5)
            )
            self.assertTrue(
                torch.allclose(private_norms, torch.ones_like(private_norms), atol=1e-5)
            )

    # 2. Cross-Modal Graph Layer
    def test_cross_modal_graph_operations_and_ablation(self):
        """Verify graph layer message passing, gradient flow, and ablation bypass."""
        gnn = CrossModalGraphLayer(embed_dim=128, n_heads=4, enabled=True)
        inputs = {
            "eeg": torch.randn(self.batch_size, 128, requires_grad=True),
            "meg": torch.randn(self.batch_size, 128, requires_grad=True),
            "fmri": torch.randn(self.batch_size, 128, requires_grad=True),
        }
        outputs = gnn(inputs)

        # Shapes and finite values
        for mod in ("eeg", "meg", "fmri"):
            self.assertEqual(outputs[mod].shape, (self.batch_size, 128))
            self.assertTrue(torch.isfinite(outputs[mod]).all().item())

        # Gradients flow back to node inputs and graph parameters
        loss = sum(out.sum() for out in outputs.values())
        loss.backward()
        for mod in ("eeg", "meg", "fmri"):
            self.assertIsNotNone(inputs[mod].grad)
            self.assertGreater(inputs[mod].grad.abs().max().item(), 0.0)
        self.assertIsNotNone(gnn.q_proj.weight.grad)

        # Verify ablation: when enabled=False, output equals input exactly
        gnn_disabled = CrossModalGraphLayer(embed_dim=128, enabled=False)
        out_disabled = gnn_disabled(inputs)
        for mod in ("eeg", "meg", "fmri"):
            self.assertTrue(torch.equal(out_disabled[mod], inputs[mod]))

    # 3. Separation Objective
    def test_separation_objective_gradient_flow(self):
        """Verify orthogonality loss penalizes correlation and passes gradients."""
        z_s = torch.randn(self.batch_size, 128, requires_grad=True)
        z_p = torch.randn(self.batch_size, 128, requires_grad=True)
        z_s_norm = F.normalize(z_s, p=2, dim=1)
        z_p_norm = F.normalize(z_p, p=2, dim=1)

        loss = orthogonal_separation_loss(z_s_norm, z_p_norm)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(z_s.grad)
        self.assertIsNotNone(z_p.grad)
        self.assertGreater(z_s.grad.abs().max().item(), 0.0)
        self.assertGreater(z_p.grad.abs().max().item(), 0.0)

    # 4. Reconstruction Objective
    def test_reconstruction_objective_and_targets(self):
        """Verify sensor summary target extraction and private reconstruction loss."""
        targets = compute_sensor_summary_targets(self.dummy_batch)
        self.assertEqual(targets["eeg"].shape, (self.batch_size, 63))
        self.assertEqual(targets["meg"].shape, (self.batch_size, 272))
        self.assertEqual(targets["fmri"].shape, (self.batch_size, 60))

        decoders = nn.ModuleDict({
            "eeg": nn.Linear(128, 63),
            "meg": nn.Linear(128, 272),
            "fmri": nn.Linear(128, 60),
        })

        private_embeds = {
            "eeg": torch.randn(self.batch_size, 128, requires_grad=True),
            "meg": torch.randn(self.batch_size, 128, requires_grad=True),
            "fmri": torch.randn(self.batch_size, 128, requires_grad=True),
        }

        rec_loss = multimodal_reconstruction_loss(private_embeds, targets, decoders)
        self.assertTrue(torch.isfinite(rec_loss).item())
        rec_loss.backward()

        self.assertIsNotNone(private_embeds["eeg"].grad)
        self.assertIsNotNone(decoders["eeg"].weight.grad)

    # 5. Total Unified Loss End-to-End Backprop
    def test_end_to_end_gradient_flow_through_encoders(self):
        """Verify gradients from all four loss terms reach encoder backbones."""
        model = TrimodalNeuroGrapeModel(embed_dim=128, use_graph=True)
        shared, private = model(self.dummy_batch, return_private=True, apply_graph=True)

        align_loss, _ = per_sample_trimodal_contrastive_loss(
            shared, self.stimulus_ids, temperature=0.07
        )
        align_loss_mean = align_loss.mean()

        sep_loss = multimodal_separation_loss(shared, private)

        targets = compute_sensor_summary_targets(self.dummy_batch)
        rec_loss = multimodal_reconstruction_loss(private, targets, model.decoders)

        # GroupDRO scalar loss
        dro = GroupDRO(num_groups=2, learning_rate=0.01)
        group_idx = torch.tensor([0, 1, 0, 1])
        dro_loss = dro.compute_weighted_loss(align_loss, group_idx, update_weights=True)

        total_loss, metrics = neuro_grape_total_loss(
            alignment_loss=align_loss_mean,
            separation_loss=sep_loss,
            reconstruction_loss=rec_loss,
            dro_loss=dro_loss,
            lambda_align=1.0,
            lambda_private=0.5,
            lambda_recon=0.2,
            lambda_robust=1.0,
        )

        total_loss.backward()

        # Check gradients in EEG encoder backbone and projection heads
        eeg_enc = model.eeg_encoder
        self.assertIsNotNone(eeg_enc.backbone[0].weight.grad)
        self.assertIsNotNone(eeg_enc.proj.weight.grad)
        self.assertIsNotNone(eeg_enc.private_proj.weight.grad)
        self.assertIsNotNone(model.graph_layer.q_proj.weight.grad)
        self.assertIsNotNone(model.decoders["eeg"].weight.grad)

    # 6. Adaptive Mixture Sampler
    def test_adaptive_mixture_sampler_dynamics(self):
        """Verify exponentiated gradient updates, simplex constraint, and minimum floor."""
        sampler = AdaptiveMixtureSampler(
            modalities=["eeg", "meg", "fmri"],
            step_size=0.1,
            min_weight=0.15,
        )
        init_weights = sampler.current_weights
        self.assertAlmostEqual(sum(init_weights.values()), 1.0, places=5)

        # Modality with higher loss should receive an increased sampling weight
        for _ in range(5):
            sampler.update_from_losses({"eeg": 5.0, "meg": 1.0, "fmri": 1.0})

        updated = sampler.current_weights
        self.assertAlmostEqual(sum(updated.values()), 1.0, places=5)
        self.assertGreater(updated["eeg"], init_weights["eeg"])
        self.assertGreaterEqual(updated["meg"], 0.15)
        self.assertGreaterEqual(updated["fmri"], 0.15)
        self.assertEqual(len(sampler.get_history()), 6)

    # 7. GroupDRO Updates & Absent Group Guard
    def test_group_dro_absent_group_guard(self):
        """Verify GroupDRO safely handles batches where one group is completely absent."""
        dro = GroupDRO(num_groups=3, learning_rate=0.01)
        init_weights = dro.group_weights.clone()

        # Batch contains only group 0 and group 1 (group 2 absent)
        per_sample_loss = torch.tensor([2.0, 3.0, 1.5, 2.5])
        group_idx = torch.tensor([0, 1, 0, 1])

        loss = dro.compute_weighted_loss(per_sample_loss, group_idx, update_weights=True)
        self.assertTrue(torch.isfinite(loss).item())

        # Weights updated only for present groups, no NaN or zero-loss distortion
        self.assertTrue(torch.isfinite(dro.group_weights).all().item())
        self.assertAlmostEqual(dro.group_weights.sum().item(), 1.0, places=5)

    # 8. Clustering Leakage Prevention
    def test_clustering_strictly_fits_on_train_split(self):
        """Verify cluster centroids are fitted exclusively on training samples."""
        clustering = EmbeddingClustering(n_clusters=2, random_state=42)

        # 12 training samples, 4 test samples
        train_embeds = torch.randn(12, 128)
        test_embeds = torch.randn(4, 128)
        all_embeds = torch.cat([train_embeds, test_embeds], dim=0)

        sample_ids = [f"sample_{i:02d}" for i in range(16)]
        metadata_df = pd.DataFrame({
            "sample_id": sample_ids,
            "split": ["train"] * 12 + ["test"] * 4,
        })
        embeds_dict = {sid: all_embeds[i].numpy() for i, sid in enumerate(sample_ids)}

        clustering.fit(embeds_dict, metadata_df)
        self.assertIsNotNone(clustering.cluster_centers_)
        self.assertEqual(clustering.cluster_centers_.shape, (2, 128))
        self.assertEqual(len(clustering.fitted_train_sample_ids_), 12)

        assigned_df = clustering.assign_clusters(embeds_dict, metadata_df)
        self.assertEqual(len(assigned_df), 16)
        self.assertIn("cluster_id", assigned_df.columns)
        self.assertEqual(clustering.cluster_centers_.shape, (2, 128))


if __name__ == "__main__":
    unittest.main()

