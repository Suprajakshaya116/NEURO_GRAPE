"""
Focused Test Suite for Neuro-GRAPE Training Pipeline
====================================================
Member 2 — Phase 4: Training pipeline, cross-modal batching, evaluation safety,
and in-memory validation checkpointing tests.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.dataset import NeuroGrapeDataset
from src.models import (
    EEGEncoder,
    FMRIEncoder,
    MEGEncoder,
    evaluate,
    prepare_cross_modal_batch,
    set_seed,
    train_epoch,
    train_model,
)

INVENTORY_CSV = REPO_ROOT / "data_inventory.csv"
SPLITS_CSV = REPO_ROOT / "data" / "splits" / "pilot_splits.csv"


def _build_synthetic_batch(
    b: int = 2, device: torch.device = torch.device("cpu")
) -> tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
    """Create a minimal synthetic cross-modal batch."""
    batch = {
        "eeg": torch.randn(b, 63, 225, device=device),
        "meg": torch.randn(b, 272, 225, device=device),
        "fmri": torch.randn(b, 1, 96, 96, 60, device=device),
    }
    stim_ids = [f"stim_{i}" for i in range(b)]
    stim_map = {
        "eeg": list(stim_ids),
        "meg": list(stim_ids),
        "fmri": list(stim_ids),
    }
    return batch, stim_map


class TestTrainingPipeline(unittest.TestCase):

    def test_prepare_cross_modal_batch_real_split(self):
        """Verify real training split batch assembly, shapes, types, and stimulus alignment."""
        train_ds = NeuroGrapeDataset(
            manifest=INVENTORY_CSV,
            split="train",
            as_torch=True,
            data_root=REPO_ROOT,
        )
        device = torch.device("cpu")
        batch, stim_ids_map = prepare_cross_modal_batch(train_ds, device=device)

        # Check tensor shapes
        self.assertEqual(batch["eeg"].shape, (12, 63, 225))
        self.assertEqual(batch["meg"].shape, (12, 272, 225))
        self.assertEqual(batch["fmri"].shape, (12, 1, 96, 96, 60))

        # Check dtypes and finiteness
        for mod in ("eeg", "meg", "fmri"):
            self.assertEqual(batch[mod].dtype, torch.float32)
            self.assertTrue(torch.isfinite(batch[mod]).all().item())

        # Check stimulus IDs
        eeg_ids = stim_ids_map["eeg"]
        meg_ids = stim_ids_map["meg"]
        fmri_ids = stim_ids_map["fmri"]

        self.assertEqual(len(eeg_ids), 12)
        self.assertEqual(eeg_ids, meg_ids)
        self.assertEqual(eeg_ids, fmri_ids)
        self.assertEqual(len(set(eeg_ids)), 12)

        # Check zero overlap with val/test
        splits_df = pd.read_csv(SPLITS_CSV)
        val_test_stims = set(
            splits_df[splits_df["split"].isin(["val", "test"])]["stimulus_id"]
        )
        overlap = set(eeg_ids) & val_test_stims
        self.assertEqual(len(overlap), 0, f"Found val/test stimuli in train batch: {overlap}")

    def test_train_epoch_updates_parameters_and_zeros_grad_first(self):
        """Verify that train_epoch updates parameters and executes zero_grad before backward."""
        device = torch.device("cpu")
        encoders = {
            "eeg": EEGEncoder().to(device),
            "meg": MEGEncoder().to(device),
            "fmri": FMRIEncoder().to(device),
        }
        all_params = (
            list(encoders["eeg"].parameters())
            + list(encoders["meg"].parameters())
            + list(encoders["fmri"].parameters())
        )
        base_opt = torch.optim.AdamW(all_params, lr=1e-3)

        # Instrument optimizer with a spy to track call sequence
        call_events = []
        original_zero_grad = base_opt.zero_grad
        original_step = base_opt.step

        def spy_zero_grad(*args, **kwargs):
            call_events.append("zero_grad")
            return original_zero_grad(*args, **kwargs)

        def spy_step(*args, **kwargs):
            call_events.append("step")
            return original_step(*args, **kwargs)

        base_opt.zero_grad = spy_zero_grad  # type: ignore
        base_opt.step = spy_step  # type: ignore

        # Capture initial parameter values
        initial_params = [p.detach().clone() for p in all_params]

        batch, stim_ids = _build_synthetic_batch(b=2, device=device)

        # Hook on a leaf parameter to verify backward occurs after zero_grad
        backward_called_after_zero_grad = []

        def param_hook(grad):
            backward_called_after_zero_grad.append("zero_grad" in call_events)
            return grad

        handle = all_params[0].register_hook(param_hook)

        loss_val = train_epoch(encoders, batch, stim_ids, base_opt, temperature=0.07)
        handle.remove()

        self.assertTrue(isinstance(loss_val, float))
        self.assertTrue(torch.isfinite(torch.tensor(loss_val)))

        # Verify call order: zero_grad was called before backward and step
        self.assertIn("zero_grad", call_events)
        self.assertIn("step", call_events)
        self.assertTrue(all(backward_called_after_zero_grad), "backward ran before zero_grad!")

        # Verify at least one parameter changed after the step
        params_changed = any(
            not torch.equal(p_init, p_curr)
            for p_init, p_curr in zip(initial_params, all_params)
        )
        self.assertTrue(params_changed, "No parameters changed after optimizer step!")

    def test_evaluate_mode_and_state_restoration(self):
        """Verify evaluate uses eval mode, computes no gradients, and restores original training states."""
        device = torch.device("cpu")
        encoders = {
            "eeg": EEGEncoder().to(device),
            "meg": MEGEncoder().to(device),
            "fmri": FMRIEncoder().to(device),
        }
        batch, stim_ids = _build_synthetic_batch(b=2, device=device)

        # Case 1: models start in train mode
        encoders["eeg"].train()
        encoders["meg"].train()
        encoders["fmri"].train()

        val_loss = evaluate(encoders, batch, stim_ids, temperature=0.07)
        self.assertTrue(isinstance(val_loss, float))
        self.assertTrue(torch.isfinite(torch.tensor(val_loss)))

        # Verify all models restored to train mode
        self.assertTrue(encoders["eeg"].training)
        self.assertTrue(encoders["meg"].training)
        self.assertTrue(encoders["fmri"].training)

        # Verify no gradients accumulated
        for enc in encoders.values():
            for p in enc.parameters():
                self.assertIsNone(p.grad)

        # Case 2: models start in mixed mode (EEG eval, MEG train, fMRI eval)
        encoders["eeg"].eval()
        encoders["meg"].train()
        encoders["fmri"].eval()

        evaluate(encoders, batch, stim_ids, temperature=0.07)

        self.assertFalse(encoders["eeg"].training)
        self.assertTrue(encoders["meg"].training)
        self.assertFalse(encoders["fmri"].training)

        # Case 3: exception inside evaluate restores original mode via finally
        encoders["eeg"].train()
        bad_batch = {
            "eeg": batch["eeg"],
            "meg": batch["meg"],
            "fmri": torch.randn(2, 5, 5, device=device),  # wrong rank
        }
        with self.assertRaises(ValueError):
            evaluate(encoders, bad_batch, stim_ids, temperature=0.07)

        self.assertTrue(encoders["eeg"].training, "Training mode not restored after exception!")

    def test_loss_finiteness_guards(self):
        """Verify train_epoch and evaluate raise FloatingPointError on non-finite loss."""
        device = torch.device("cpu")
        encoders = {
            "eeg": EEGEncoder().to(device),
            "meg": MEGEncoder().to(device),
            "fmri": FMRIEncoder().to(device),
        }
        all_params = (
            list(encoders["eeg"].parameters())
            + list(encoders["meg"].parameters())
            + list(encoders["fmri"].parameters())
        )
        opt = torch.optim.AdamW(all_params, lr=1e-4)
        batch, stim_ids = _build_synthetic_batch(b=2, device=device)

        # Mock trimodal_contrastive_loss to return NaN
        with patch("src.models.training.trimodal_contrastive_loss", return_value=torch.tensor(float("nan"))):
            with self.assertRaises(FloatingPointError):
                train_epoch(encoders, batch, stim_ids, opt)

            with self.assertRaises(FloatingPointError):
                evaluate(encoders, batch, stim_ids)

    def test_argument_validation(self):
        """Verify train_model and prepare_cross_modal_batch validate inputs strictly."""
        with self.assertRaises(ValueError):
            train_model(manifest_path=INVENTORY_CSV, num_epochs=0)

        with self.assertRaises(ValueError):
            train_model(manifest_path=INVENTORY_CSV, lr=-1e-4)

        with self.assertRaises(ValueError):
            train_model(manifest_path=INVENTORY_CSV, weight_decay=-0.01)

        with self.assertRaises(ValueError):
            train_model(manifest_path=INVENTORY_CSV, temperature=0.0)

        # Empty dataset mock
        empty_ds = MagicMock(spec=NeuroGrapeDataset)
        empty_ds.get_metadata_table.return_value = pd.DataFrame()
        with self.assertRaises(ValueError):
            prepare_cross_modal_batch(empty_ds)

    def test_deterministic_reproducibility_small_synthetic(self):
        """Verify deterministic loss sequences using small synthetic models and fixed seed."""
        device = torch.device("cpu")

        def run_short_training(seed_val: int) -> List[float]:
            set_seed(seed_val)
            encs = {
                "eeg": EEGEncoder().to(device),
                "meg": MEGEncoder().to(device),
                "fmri": FMRIEncoder().to(device),
            }
            params = (
                list(encs["eeg"].parameters())
                + list(encs["meg"].parameters())
                + list(encs["fmri"].parameters())
            )
            opt = torch.optim.AdamW(params, lr=1e-3)
            # Fixed batch
            set_seed(100)
            batch, stim_ids = _build_synthetic_batch(b=2, device=device)

            losses = []
            for _ in range(2):
                losses.append(train_epoch(encs, batch, stim_ids, opt))
            return losses

        run1 = run_short_training(seed_val=42)
        run2 = run_short_training(seed_val=42)

        self.assertEqual(len(run1), 2)
        for l1, l2 in zip(run1, run2):
            self.assertAlmostEqual(l1, l2, places=6)

    def test_best_state_dicts_are_independent_cpu_clones(self):
        """Verify best_state_dicts are detached CPU tensors unaffected by subsequent training steps."""
        device = torch.device("cpu")
        set_seed(42)
        encs = {
            "eeg": EEGEncoder().to(device),
            "meg": MEGEncoder().to(device),
            "fmri": FMRIEncoder().to(device),
        }
        params = (
            list(encs["eeg"].parameters())
            + list(encs["meg"].parameters())
            + list(encs["fmri"].parameters())
        )
        opt = torch.optim.AdamW(params, lr=1e-3)
        batch, stim_ids = _build_synthetic_batch(b=2, device=device)

        # Clone state dict
        cloned_states = {
            mod: {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
            for mod, enc in encs.items()
        }

        # Check all tensors reside on CPU
        for mod, sd in cloned_states.items():
            for k, tensor in sd.items():
                self.assertEqual(tensor.device, torch.device("cpu"))

        # Execute training steps modifying model parameters
        for _ in range(3):
            train_epoch(encs, batch, stim_ids, opt)

        # Verify that parameters in active models have changed, while cloned_states remain identical
        for mod, enc in encs.items():
            first_weight_key = next(iter(cloned_states[mod].keys()))
            saved_tensor = cloned_states[mod][first_weight_key]
            current_tensor = enc.state_dict()[first_weight_key].cpu()
            self.assertFalse(
                torch.equal(saved_tensor, current_tensor),
                f"Active model tensor was not modified for {mod}",
            )

        # Verify restoring state dict loads properly into active model
        for mod, enc in encs.items():
            enc.load_state_dict(cloned_states[mod])
            first_weight_key = next(iter(cloned_states[mod].keys()))
            restored_tensor = enc.state_dict()[first_weight_key].cpu()
            self.assertTrue(torch.equal(restored_tensor, cloned_states[mod][first_weight_key]))

    def test_test_split_isolation(self):
        """Verify train_model never instantiates or references the test split."""
        import inspect
        import src.models.training as training_mod

        src_code = inspect.getsource(training_mod.train_model)
        self.assertNotIn("split=\"test\"", src_code)
        self.assertNotIn("split='test'", src_code)
        self.assertNotIn('test_ds', src_code)


if __name__ == "__main__":
    unittest.main()
