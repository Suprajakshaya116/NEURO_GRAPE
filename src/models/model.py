"""Neuro-GRAPE Unified Trimodal Model Interface (Member 2 — Phase 5 Handoff).

Unified PyTorch module wrapping modality-specific encoders for EEG, MEG, and fMRI.
Provides a consolidated interface and formal tensor dimension contract for
Member 3's downstream evaluation, probing, and GroupDRO alignment.

Tensor Dimension Contract
-------------------------
Modality | Input Shape               | Output Embedding Shape | Normalization
-------- | ------------------------- | ---------------------- | -------------
EEG      | (B, 63, 225) float32      | (B, 128) float32       | Unit L2 Sphere
MEG      | (B, 272, 225) float32     | (B, 128) float32       | Unit L2 Sphere
fMRI     | (B, 1, 96, 96, 60) float32| (B, 128) float32       | Unit L2 Sphere

Design Notes
------------
* Independent Encoders: EEGEncoder, MEGEncoder, and FMRIEncoder do not share
  weights; each projects independently into the shared 128-dimensional latent space.
* LayerNorm: Per-sample LayerNorm is applied after linear projection and before
  L2 hypersphere normalization (F.normalize).
* Missing modalities in batch dict are supported: forward() only encodes modalities
  present in the input mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

from src.models.encoders import EEGEncoder, FMRIEncoder, MEGEncoder
from src.models.graph import CrossModalGraphLayer


class TrimodalNeuroGrapeModel(nn.Module):
    """Unified trimodal model interface for Neuro-GRAPE representation learning.

    Args:
        embed_dim: Latent representation dimensionality. Default 128.
        dropout: Encoder backbone dropout probability (0.0 to 1.0). Default 0.0.
        eeg_channels: Number of EEG sensor channels. Default 63.
        eeg_times: Number of EEG timepoints (250 Hz, -100 to 800 ms). Default 225.
        meg_channels: Number of MEG sensor channels. Default 272.
        meg_times: Number of MEG timepoints (250 Hz, -100 to 800 ms). Default 225.
        fmri_channels: Input channels for fMRI beta volume. Default 1.
        fmri_volume_shape: Spatial dimensions (X, Y, Z) of fMRI volume. Default (96, 96, 60).
        use_graph: Whether cross-modal graph layer is enabled. Default True.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        dropout: float = 0.0,
        eeg_channels: int = 63,
        eeg_times: int = 225,
        meg_channels: int = 272,
        meg_times: int = 225,
        fmri_channels: int = 1,
        fmri_volume_shape: tuple[int, int, int] = (96, 96, 60),
        use_graph: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.use_graph = use_graph

        self.encoders = nn.ModuleDict({
            "eeg": EEGEncoder(
                n_channels=eeg_channels,
                n_times=eeg_times,
                embed_dim=embed_dim,
                dropout=dropout,
            ),
            "meg": MEGEncoder(
                n_channels=meg_channels,
                n_times=meg_times,
                embed_dim=embed_dim,
                dropout=dropout,
            ),
            "fmri": FMRIEncoder(
                in_channels=fmri_channels,
                volume_shape=fmri_volume_shape,
                embed_dim=embed_dim,
                dropout=dropout,
            ),
        })

        # Cross-modal graph attention layer
        self.graph_layer = CrossModalGraphLayer(
            embed_dim=embed_dim, n_heads=4, dropout=dropout, enabled=use_graph
        )

        # Private reconstruction decoders (predicting sensor statistics)
        self.decoders = nn.ModuleDict({
            "eeg": nn.Linear(embed_dim, eeg_channels),
            "meg": nn.Linear(embed_dim, meg_channels),
            "fmri": nn.Linear(embed_dim, fmri_volume_shape[2]),  # Z-slices = 60
        })

    @property
    def eeg_encoder(self) -> EEGEncoder:
        """Return the EEG encoder module."""
        return self.encoders["eeg"]  # type: ignore[return-value]

    @property
    def meg_encoder(self) -> MEGEncoder:
        """Return the MEG encoder module."""
        return self.encoders["meg"]  # type: ignore[return-value]

    @property
    def fmri_encoder(self) -> FMRIEncoder:
        """Return the fMRI encoder module."""
        return self.encoders["fmri"]  # type: ignore[return-value]

    def encode(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        """Encode a single modality tensor into a unit-norm embedding.

        Args:
            modality: One of 'eeg', 'meg', or 'fmri'.
            x: Input feature tensor matching the specified modality dimensions:
               - 'eeg':  (B, 63, 225)
               - 'meg':  (B, 272, 225)
               - 'fmri': (B, 1, 96, 96, 60)

        Returns:
            (B, embed_dim) L2-normalized embedding tensor.

        Raises:
            KeyError: If modality is not one of ('eeg', 'meg', 'fmri').
        """
        mod_key = modality.lower()
        if mod_key not in self.encoders:
            raise KeyError(
                f"Unknown modality '{modality}'. Expected one of {list(self.encoders.keys())}."
            )
        return self.encoders[mod_key](x)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        return_private: bool = False,
        apply_graph: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """Encode all available modalities in an input batch mapping.

        Args:
            batch: Mapping containing one or more modality tensors ('eeg', 'meg', 'fmri').
            return_private: If True, also returns private embeddings as a second dict.
            apply_graph: If True and graph_layer is enabled, refines shared representations.

        Returns:
            Dict mapping present modality keys to (B, embed_dim) embeddings, or a tuple
            of (shared_dict, private_dict) if return_private is True.
        """
        shared_embeddings: Dict[str, torch.Tensor] = {}
        private_embeddings: Dict[str, torch.Tensor] = {}

        for mod, enc in self.encoders.items():
            if mod in batch:
                if return_private:
                    z_s, z_p = enc(batch[mod], return_both=True)
                    shared_embeddings[mod] = z_s
                    private_embeddings[mod] = z_p
                else:
                    shared_embeddings[mod] = enc(batch[mod])

        if apply_graph and len(shared_embeddings) > 1:
            shared_embeddings = self.graph_layer(shared_embeddings)

        if return_private:
            return shared_embeddings, private_embeddings
        return shared_embeddings

    def get_encoder(self, modality: str) -> nn.Module:
        """Retrieve encoder module for the specified modality."""
        mod_key = modality.lower()
        if mod_key not in self.encoders:
            raise KeyError(f"Unknown modality '{modality}'.")
        return self.encoders[mod_key]

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load encoder weights from a saved checkpoint dict.

        Args:
            checkpoint_path: Path to PyTorch .pt checkpoint file.
            device: Target torch.device. Auto-detected if None.

        Returns:
            The loaded checkpoint dictionary.
        """
        ckpt = torch.load(checkpoint_path, map_location=device or "cpu")
        if "encoders" in ckpt:
            for mod in ("eeg", "meg", "fmri"):
                if mod in ckpt["encoders"]:
                    self.encoders[mod].load_state_dict(
                        ckpt["encoders"][mod], strict=False
                    )
        elif "model_state_dict" in ckpt:
            self.load_state_dict(ckpt["model_state_dict"], strict=False)
        return ckpt


# Alias for intuitive importing
NeuroGrapeModel = TrimodalNeuroGrapeModel
