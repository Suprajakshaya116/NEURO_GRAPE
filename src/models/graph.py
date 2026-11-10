"""Neuro-GRAPE Cross-Modal Graph Module.

Explicitly models cross-modal concept and representational relationships using
differentiable graph attention / message-passing across neural modalities
(EEG, MEG, fMRI).

Graph Semantics
---------------
* Nodes: For each stimulus concept $i \\in \\{1, \\dots, B\\}$, there are modality
  nodes $v_{i, m}$ for each active modality $m \\in \\{\\text{EEG}, \\text{MEG}, \\text{fMRI}\\}$.
  Initial node feature vector $u_{i, m}^{(0)} = z_{i, m, \\text{shared}} \\in \\mathbb{R}^{D}$.
* Edges: Cross-modal correspondence edges connecting heterogeneous neural views
  evoked by the same visual concept stimulus.
* Message Passing: Differentiable multi-head or scaled dot-product attention
  aggregates complementary neural dynamics across modalities.
* Residual Connection & LayerNorm:
  $u_{i, m}^{(1)} = \\text{LayerNorm}(u_{i, m}^{(0)} + \\sum_{m'} A_{i, (m, m')} W_V u_{i, m'}^{(0)})$
* Refined Embedding: Output is projected back to unit hypersphere $\\mathcal{S}^{D-1}$.
* Ablation Support: When enabled=False, returns input representations unchanged.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalGraphLayer(nn.Module):
    """Differentiable cross-modal graph attention and message-passing layer.

    Refines modality-specific shared representations through graph attention
    over concept-aligned multimodal nodes.

    Args:
        embed_dim: Node feature dimension. Default 128.
        n_heads: Number of attention heads for message passing. Default 4.
        dropout: Attention dropout probability. Default 0.0.
        enabled: If False, layer operates as identity bypass for ablation. Default True.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.0,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.enabled = enabled

        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})."
            )
        self.head_dim = embed_dim // n_heads

        # Graph attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        modality_embeddings: Mapping[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Execute graph message passing across available modality nodes.

        Args:
            modality_embeddings: Dict mapping modality names ('eeg', 'meg', 'fmri')
                to (B, embed_dim) normalized shared embeddings.
            mask: Optional attention mask of shape (B, num_modalities, num_modalities).

        Returns:
            Dict mapping each modality to its refined (B, embed_dim) unit-norm embedding.
        """
        if not self.enabled or len(modality_embeddings) <= 1:
            # Identity bypass when disabled or only a single modality is present
            return dict(modality_embeddings)

        modalities = sorted(modality_embeddings.keys())
        # Stack node features: (B, M, D)
        node_features = torch.stack(
            [modality_embeddings[m] for m in modalities], dim=1
        )  # (B, M, D)
        b, m, d = node_features.shape

        # Multi-head attention projections: (B, M, D) -> (B, n_heads, M, head_dim)
        q = (
            self.q_proj(node_features)
            .view(b, m, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(node_features)
            .view(b, m, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(node_features)
            .view(b, m, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention: (B, n_heads, M, M)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            # Broadcast mask across heads
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Message aggregation: (B, n_heads, M, head_dim)
        out = torch.matmul(attn_weights, v)
        # Reshape to (B, M, D)
        out = out.transpose(1, 2).contiguous().view(b, m, d)
        out = self.out_proj(out)

        # Residual connection + LayerNorm + L2 normalisation
        refined_nodes = self.norm(node_features + out)
        refined_nodes = F.normalize(refined_nodes, p=2, dim=-1)

        # Unpack back to modality dictionary
        refined_dict: Dict[str, torch.Tensor] = {}
        for idx, mod in enumerate(modalities):
            refined_dict[mod] = refined_nodes[:, idx, :]

        return refined_dict
