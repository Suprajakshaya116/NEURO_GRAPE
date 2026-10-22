"""Neuro-GRAPE Trimodal Input Normalization (Member 2).

Provides leakage-safe, modality-specific input normalization for EEG, MEG,
and fMRI beta volumes prior to contrastive encoding.

Design & Guarantees
-------------------
1. Split Isolation: Normalization statistics (means, standard deviations, and
   fMRI spatial brain masks) are fitted strictly on training data.
2. Leakage Prevention: Validation and test splits strictly reuse the frozen
   training statistics; statistics are never updated on non-training splits.
3. Modality Conventions:
   - EEG: Per-channel standardization over training stimuli and time (dim 0 and 2),
     reducing (B, 63, 225) -> channel mean and std (1, 63, 1). Safe epsilon 1e-12 V.
   - MEG: Per-channel standardization over training stimuli and time (dim 0 and 2),
     reducing (B, 272, 225) -> channel mean and std (1, 272, 1). Safe epsilon 1e-18 T.
   - fMRI: Spatial brain-mask preservation. Identifies non-zero brain voxels across
     training stimuli. In-mask voxels are standardized using in-mask mean and std.
     Background voxels (outer air / non-brain zeros) remain strictly 0.0 to preserve
     spatial anatomical boundaries and avoid background elevation.
4. Numerical Safety: Handles near-zero or constant-zero channels/voxels safely with
   clamping/threshold guards, guaranteeing finite tensors without NaN or Inf.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union
import torch
import torch.nn as nn


class TrimodalNormalizer:
    """Modality-specific standardizer fitted strictly on training data.

    Parameters
    ----------
    tau_eeg : float, default=1e-12
        Lower threshold for EEG channel standard deviation in Volts. Channels with
        std <= tau_eeg use std=1.0 to prevent division by zero.
    tau_meg : float, default=1e-18
        Lower threshold for MEG channel standard deviation in Tesla. Channels with
        std <= tau_meg use std=1.0 to prevent division by zero.
    tau_fmri : float, default=1e-8
        Lower threshold for in-mask fMRI standard deviation.
    """

    def __init__(
        self,
        tau_eeg: float = 1e-12,
        tau_meg: float = 1e-18,
        tau_fmri: float = 1e-8,
    ) -> None:
        self.tau_eeg = float(tau_eeg)
        self.tau_meg = float(tau_meg)
        self.tau_fmri = float(tau_fmri)

        self.is_fitted: bool = False

        # Fitted statistics
        self.eeg_mean: Optional[torch.Tensor] = None
        self.eeg_std_safe: Optional[torch.Tensor] = None

        self.meg_mean: Optional[torch.Tensor] = None
        self.meg_std_safe: Optional[torch.Tensor] = None

        self.fmri_mask: Optional[torch.Tensor] = None
        self.fmri_in_mask_mean: Optional[torch.Tensor] = None
        self.fmri_in_mask_std_safe: Optional[torch.Tensor] = None

    def fit(self, batch: Mapping[str, torch.Tensor]) -> TrimodalNormalizer:
        """Fit normalization parameters strictly on training batch.

        Args:
            batch: Mapping containing 'eeg', 'meg', and 'fmri' tensors:
                - 'eeg':  (B, 63, 225) float tensor
                - 'meg':  (B, 272, 225) float tensor
                - 'fmri': (B, 1, 96, 96, 60) float tensor

        Returns:
            self: The fitted TrimodalNormalizer instance.

        Raises:
            KeyError: If any modality key is missing.
            ValueError: If input tensors have invalid ranks or non-finite values.
        """
        for mod in ("eeg", "meg", "fmri"):
            if mod not in batch:
                raise KeyError(f"Training batch missing required modality '{mod}'.")
            if not torch.isfinite(batch[mod]).all():
                raise ValueError(f"Training batch for '{mod}' contains non-finite values.")

        x_eeg = batch["eeg"]
        x_meg = batch["meg"]
        x_fmri = batch["fmri"]

        if x_eeg.ndim != 3:
            raise ValueError(f"EEG expects 3-D tensor (B, C, T); got {x_eeg.ndim}.")
        if x_meg.ndim != 3:
            raise ValueError(f"MEG expects 3-D tensor (B, C, T); got {x_meg.ndim}.")
        if x_fmri.ndim != 5:
            raise ValueError(f"fMRI expects 5-D tensor (B, 1, X, Y, Z); got {x_fmri.ndim}.")

        # 1. Fit EEG: reduce over batch (0) and time (2) -> shape (1, C, 1)
        eeg_mean = x_eeg.mean(dim=(0, 2), keepdim=True)
        eeg_std = x_eeg.std(dim=(0, 2), keepdim=True, unbiased=False)
        eeg_low = eeg_std < self.tau_eeg
        self.eeg_mean = eeg_mean.detach().clone()
        self.eeg_std_safe = torch.where(eeg_low, torch.ones_like(eeg_std), eeg_std).detach().clone()

        # 2. Fit MEG: reduce over batch (0) and time (2) -> shape (1, C, 1)
        meg_mean = x_meg.mean(dim=(0, 2), keepdim=True)
        meg_std = x_meg.std(dim=(0, 2), keepdim=True, unbiased=False)
        meg_low = meg_std < self.tau_meg
        self.meg_mean = meg_mean.detach().clone()
        self.meg_std_safe = torch.where(meg_low, torch.ones_like(meg_std), meg_std).detach().clone()

        # 3. Fit fMRI: brain-mask preservation
        # Identify active brain voxels where signal is non-zero in at least one training volume
        # fmri is (B, 1, X, Y, Z). Mask is spatial boolean across (X, Y, Z)
        spatial_nonzero = (x_fmri != 0.0).any(dim=0).squeeze(0)  # (X, Y, Z)
        num_brain_voxels = int(spatial_nonzero.sum().item())

        if num_brain_voxels == 0:
            # Fallback if entire training split has zero signal
            self.fmri_mask = spatial_nonzero.detach().clone()
            self.fmri_in_mask_mean = torch.tensor(0.0, device=x_fmri.device, dtype=x_fmri.dtype)
            self.fmri_in_mask_std_safe = torch.tensor(1.0, device=x_fmri.device, dtype=x_fmri.dtype)
        else:
            in_mask_vals = x_fmri[:, 0, spatial_nonzero]  # (B, num_brain_voxels)
            fmri_mean = in_mask_vals.mean()
            fmri_std = in_mask_vals.std(unbiased=False)
            fmri_std_safe = fmri_std if fmri_std > self.tau_fmri else torch.tensor(1.0, device=x_fmri.device, dtype=x_fmri.dtype)

            self.fmri_mask = spatial_nonzero.detach().clone()
            self.fmri_in_mask_mean = fmri_mean.detach().clone()
            self.fmri_in_mask_std_safe = fmri_std_safe.detach().clone()

        self.is_fitted = True
        return self

    def transform(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply fitted normalization parameters to an input batch.

        Args:
            batch: Mapping containing 'eeg', 'meg', and 'fmri' tensors.

        Returns:
            normalized_batch: Dict mapping 'eeg', 'meg', 'fmri' to standardized tensors.

        Raises:
            RuntimeError: If called before fit().
            ValueError: If input shapes mismatch fitted dimensions or contain non-finite values.
        """
        if not self.is_fitted:
            raise RuntimeError("TrimodalNormalizer must be fitted on training data before calling transform().")

        assert self.eeg_mean is not None and self.eeg_std_safe is not None
        assert self.meg_mean is not None and self.meg_std_safe is not None
        assert self.fmri_mask is not None and self.fmri_in_mask_mean is not None and self.fmri_in_mask_std_safe is not None

        out: Dict[str, torch.Tensor] = {}

        # 1. Transform EEG
        if "eeg" in batch:
            x_eeg = batch["eeg"]
            if not torch.isfinite(x_eeg).all():
                raise ValueError("EEG input tensor contains non-finite values.")
            if x_eeg.shape[1] != self.eeg_mean.shape[1]:
                raise ValueError(
                    f"EEG channel count mismatch: expected {self.eeg_mean.shape[1]}, got {x_eeg.shape[1]}."
                )
            mean = self.eeg_mean.to(device=x_eeg.device, dtype=x_eeg.dtype)
            std = self.eeg_std_safe.to(device=x_eeg.device, dtype=x_eeg.dtype)
            out["eeg"] = (x_eeg - mean) / std

        # 2. Transform MEG
        if "meg" in batch:
            x_meg = batch["meg"]
            if not torch.isfinite(x_meg).all():
                raise ValueError("MEG input tensor contains non-finite values.")
            if x_meg.shape[1] != self.meg_mean.shape[1]:
                raise ValueError(
                    f"MEG channel count mismatch: expected {self.meg_mean.shape[1]}, got {x_meg.shape[1]}."
                )
            mean = self.meg_mean.to(device=x_meg.device, dtype=x_meg.dtype)
            std = self.meg_std_safe.to(device=x_meg.device, dtype=x_meg.dtype)
            out["meg"] = (x_meg - mean) / std

        # 3. Transform fMRI (Masked Standardization)
        if "fmri" in batch:
            x_fmri = batch["fmri"]
            if not torch.isfinite(x_fmri).all():
                raise ValueError("fMRI input tensor contains non-finite values.")
            spatial_shape = tuple(x_fmri.shape[2:])
            mask_spatial = tuple(self.fmri_mask.shape)
            if spatial_shape != mask_spatial:
                raise ValueError(
                    f"fMRI spatial shape mismatch: expected {mask_spatial}, got {spatial_shape}."
                )

            mask = self.fmri_mask.to(device=x_fmri.device)
            mean = self.fmri_in_mask_mean.to(device=x_fmri.device, dtype=x_fmri.dtype)
            std = self.fmri_in_mask_std_safe.to(device=x_fmri.device, dtype=x_fmri.dtype)

            # Preserve background zeros: unmasked voxels remain 0.0
            norm_fmri = torch.zeros_like(x_fmri)
            if mask.any():
                norm_fmri[:, 0, mask] = (x_fmri[:, 0, mask] - mean) / std

            out["fmri"] = norm_fmri

        return out

    def fit_transform(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Fit on batch and return transformed batch."""
        return self.fit(batch).transform(batch)

    def to(self, device: torch.device) -> TrimodalNormalizer:
        """Move fitted statistics to specified device."""
        if self.eeg_mean is not None:
            self.eeg_mean = self.eeg_mean.to(device)
            self.eeg_std_safe = self.eeg_std_safe.to(device)
        if self.meg_mean is not None:
            self.meg_mean = self.meg_mean.to(device)
            self.meg_std_safe = self.meg_std_safe.to(device)
        if self.fmri_mask is not None:
            self.fmri_mask = self.fmri_mask.to(device)
            self.fmri_in_mask_mean = self.fmri_in_mask_mean.to(device)
            self.fmri_in_mask_std_safe = self.fmri_in_mask_std_safe.to(device)
        return self
