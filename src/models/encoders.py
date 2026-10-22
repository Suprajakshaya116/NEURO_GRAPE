"""Neuro-GRAPE Modality-Specific Encoders (Member 2 — Phase 1).

Three independent PyTorch modules that project heterogeneous brain recordings
into a common 128-dimensional L2-normalised latent space.

Supported modalities
--------------------
EEGEncoder   : (B, 63,  225)        → (B, embed_dim)
MEGEncoder   : (B, 272, 225)        → (B, embed_dim)
FMRIEncoder  : (B,  1,  96, 96, 60) → (B, embed_dim)

Design constraints (from integration_contract.md)
--------------------------------------------------
* Architectures are compact; the pilot has only 12 training stimuli.
* No cross-modal weight sharing — each encoder trains independently.
* Output embeddings are L2-normalised to unit hypersphere.
* Input validation raises ValueError on wrong rank or wrong channel count.
* A per-sample LayerNorm is applied between the linear projection head and
  F.normalize to break the shared-dominant-direction collapse confirmed by
  activation diagnostics (post-proj cosine ≈ +0.98–0.999 at init).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: 1-D CNN "backbone" factory
# ---------------------------------------------------------------------------

def _build_1d_backbone(in_channels: int, dropout: float = 0.0) -> nn.Sequential:
    """Return a shared-topology 1-D CNN backbone with *in_channels* input channels.

    Channel progression:  in_ch -> 32 -> 64 -> 128
    Temporal reduction:   stride-2 on first two blocks, then adaptive pool to 1.

    Args:
        in_channels: Number of sensor channels (63 for EEG, 272 for MEG).
        dropout:     Dropout probability (0.0 means disabled). Default 0.0.

    Returns:
        nn.Sequential comprising the backbone layers.
    """
    layers: list[nn.Module] = [
        # Block 1 -- wide kernel to capture temporal context
        nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm1d(32),
        nn.GELU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout1d(p=dropout))

    # Block 2
    layers.extend([
        nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
        nn.BatchNorm1d(64),
        nn.GELU(),
    ])
    if dropout > 0.0:
        layers.append(nn.Dropout1d(p=dropout))

    # Block 3 -- no stride; preserves temporal resolution for pooling
    layers.extend([
        nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.GELU(),
    ])
    if dropout > 0.0:
        layers.append(nn.Dropout1d(p=dropout))

    layers.append(nn.AdaptiveAvgPool1d(1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# EEG Encoder
# ---------------------------------------------------------------------------

class EEGEncoder(nn.Module):
    """Compact 1-D CNN encoder for EEG signals.

    Args:
        n_channels: Number of EEG channels. Default 63 (as in the pilot).
        n_times:    Number of time points.   Default 225 (250 Hz, -100 to 800 ms).
        embed_dim:  Dimensionality of the output embedding. Default 128.
    """

    def __init__(
        self,
        n_channels: int = 63,
        n_times: int = 225,
        embed_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.backbone = _build_1d_backbone(n_channels, dropout=dropout)

        # Projection head: 128 -> embed_dim (identity when embed_dim == 128)
        self.proj = nn.Linear(128, embed_dim)
        # LayerNorm centres the projected vector per-sample before L2 normalisation,
        # preventing the shared-dominant-direction collapse measured at init.
        self.pre_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: EEG tensor of shape (B, n_channels, n_times).

        Returns:
            L2-normalised embedding of shape (B, embed_dim).

        Raises:
            ValueError: If tensor rank is not 3, channel count is wrong,
                or time-point count does not match n_times.
        """
        if x.ndim != 3:
            raise ValueError(
                f"EEGEncoder expects a 3-D tensor (B, C, T); got ndim={x.ndim}."
            )
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"EEGEncoder expects {self.n_channels} input channels; "
                f"got {x.shape[1]}."
            )
        if x.shape[2] != self.n_times:
            raise ValueError(
                f"EEGEncoder expects {self.n_times} time points (dim 2); "
                f"got {x.shape[2]}."
            )

        # (B, 128, T) -> (B, 128, 1) -> (B, 128) -> LayerNorm -> L2-unit sphere
        h = self.backbone(x).squeeze(-1)
        z = self.pre_norm(self.proj(h))
        return F.normalize(z, p=2, dim=1)


# ---------------------------------------------------------------------------
# MEG Encoder
# ---------------------------------------------------------------------------

class MEGEncoder(nn.Module):
    """Compact 1-D CNN encoder for MEG signals.

    Identical topology to EEGEncoder but with independent parameters and
    a different default channel count (272 axial gradiometers).

    Args:
        n_channels: Number of MEG channels. Default 272 (CTF pilot).
        n_times:    Number of time points.   Default 225 (250 Hz, -100 to 800 ms).
        embed_dim:  Dimensionality of the output embedding. Default 128.
        dropout:    Dropout probability (0.0 means disabled). Default 0.0.
    """

    def __init__(
        self,
        n_channels: int = 272,
        n_times: int = 225,
        embed_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.backbone = _build_1d_backbone(n_channels, dropout=dropout)

        self.proj = nn.Linear(128, embed_dim)
        self.pre_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: MEG tensor of shape (B, n_channels, n_times).

        Returns:
            L2-normalised embedding of shape (B, embed_dim).

        Raises:
            ValueError: If tensor rank is not 3, channel count is wrong,
                or time-point count does not match n_times.
        """
        if x.ndim != 3:
            raise ValueError(
                f"MEGEncoder expects a 3-D tensor (B, C, T); got ndim={x.ndim}."
            )
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"MEGEncoder expects {self.n_channels} input channels; "
                f"got {x.shape[1]}."
            )
        if x.shape[2] != self.n_times:
            raise ValueError(
                f"MEGEncoder expects {self.n_times} time points (dim 2); "
                f"got {x.shape[2]}."
            )

        h = self.backbone(x).squeeze(-1)
        z = self.pre_norm(self.proj(h))
        return F.normalize(z, p=2, dim=1)


# ---------------------------------------------------------------------------
# fMRI Beta Encoder
# ---------------------------------------------------------------------------

class FMRIEncoder(nn.Module):
    """Compact 3-D CNN encoder for trial-wise fMRI beta volumes.

    The input is a single-channel 3-D beta map of shape (96, 96, 60).
    Channel progression: 1 -> 16 -> 32 -> 64, all with stride-2 Conv3d to
    aggressively shrink spatial dimensions before the projection head.

    Args:
        in_channels:  Number of input image channels. Default 1 (beta volume).
        volume_shape: Spatial shape (X, Y, Z). Default (96, 96, 60).
        embed_dim:    Dimensionality of the output embedding. Default 128.
        dropout:      Dropout probability (0.0 means disabled). Default 0.0.
    """

    def __init__(
        self,
        in_channels: int = 1,
        volume_shape: tuple = (96, 96, 60),
        embed_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.volume_shape = volume_shape
        self.embed_dim = embed_dim
        self.dropout = dropout

        layers: list[nn.Module] = [
            # Block 1: 1 -> 16
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.GELU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout3d(p=dropout))

        # Block 2: 16 -> 32
        layers.extend([
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
        ])
        if dropout > 0.0:
            layers.append(nn.Dropout3d(p=dropout))

        # Block 3: 32 -> 64
        layers.extend([
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
        ])
        if dropout > 0.0:
            layers.append(nn.Dropout3d(p=dropout))

        layers.append(nn.AdaptiveAvgPool3d(1))
        self.backbone = nn.Sequential(*layers)

        self.proj = nn.Linear(64, embed_dim)
        self.pre_norm = nn.LayerNorm(embed_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: fMRI beta tensor of shape (B, 1, X, Y, Z).

        Returns:
            L2-normalised embedding of shape (B, embed_dim).

        Raises:
            ValueError: If tensor rank is not 5, channel count is wrong,
                or spatial dimensions do not match volume_shape.
        """
        if x.ndim != 5:
            raise ValueError(
                f"FMRIEncoder expects a 5-D tensor (B, C, X, Y, Z); "
                f"got ndim={x.ndim}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"FMRIEncoder expects {self.in_channels} input channel(s); "
                f"got {x.shape[1]}."
            )
        actual_spatial = tuple(x.shape[2:])
        if actual_spatial != tuple(self.volume_shape):
            raise ValueError(
                f"FMRIEncoder expects spatial volume {tuple(self.volume_shape)} "
                f"(X, Y, Z); got {actual_spatial}."
            )

        # (B, 64, 1, 1, 1) -> (B, 64) -> proj -> LayerNorm -> L2-unit sphere
        h = self.backbone(x).view(x.shape[0], -1)
        z = self.pre_norm(self.proj(h))
        return F.normalize(z, p=2, dim=1)
