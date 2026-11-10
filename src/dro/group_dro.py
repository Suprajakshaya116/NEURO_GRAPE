"""Group Distributionally Robust Optimization (GroupDRO) Module.

Member 3 Implementation for Neuro-GRAPE.

Provides a robust, numerically stable GroupDRO optimizer module that:
1. Calculates per-group mean losses from per-example loss vectors.
2. Updates group distribution weights q_k using exponentiated gradient updates
   computed in log-space via softmax to prevent numeric overflow/underflow.
3. Gracefully handles empty, missing, or undersized groups without division-by-zero,
   preserving weights and normalizing across active groups.
4. Supports configurable group learning rate, update frequency, weight smoothing,
   and weight clipping.
5. Integrates cleanly with Member 2's model, loss functions, and training loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union,Sequence,Any
import torch
import torch.nn.functional as F


class GroupDRO:
    """Group Distributionally Robust Optimization (GroupDRO) Manager.

    Optimizes worst-group risk over K predefined or discovered subgroups.
    Group weights are updated via exponentiated gradient ascent on group losses:
        q_k^{(t+1)} \\propto q_k^{(t)} \\exp(\\eta \\cdot L_k)

    Implemented stably in log-space:
        s_k^{(t+1)} = s_k^{(t)} + \\eta \\cdot L_k
        q^{(t+1)} = \\text{softmax}(s)

    Parameters
    ----------
    num_groups : int
        Total number of groups K (must be >= 1).
    learning_rate : float, default=0.01
        Group weight step size \\eta (> 0).
    update_frequency : int, default=1
        Perform weight update every `update_frequency` steps.
    min_group_size : int, default=1
        Minimum number of samples required for a group to be considered active.
    smoothing : float, default=0.0
        Uniform label smoothing factor \\alpha \\in [0.0, 1.0) applied to weights:
        q \\leftarrow (1 - \\alpha) q + \\alpha / K.
    min_weight : float, default=0.0
        Minimum weight floor (clipping) per group before re-normalizing.
    device : torch.device, optional
        Device to store tensors on. Defaults to CPU.
    """

    def __init__(
        self,
        num_groups: int,
        learning_rate: float = 0.01,
        update_frequency: int = 1,
        min_group_size: int = 1,
        smoothing: float = 0.0,
        min_weight: float = 0.0,
        group_names: Optional[Sequence[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if update_frequency < 1:
            raise ValueError(f"update_frequency must be >= 1, got {update_frequency}")
        if min_group_size < 1:
            raise ValueError(f"min_group_size must be >= 1, got {min_group_size}")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0.0, 1.0), got {smoothing}")
        if not (0.0 <= min_weight < (1.0 / num_groups)):
            raise ValueError(
                f"min_weight must be in [0.0, {1.0 / num_groups:.4f}), got {min_weight}"
            )

        self.num_groups = num_groups
        self.learning_rate = float(learning_rate)
        self.update_frequency = int(update_frequency)
        self.min_group_size = int(min_group_size)
        self.smoothing = float(smoothing)
        self.min_weight = float(min_weight)
        self.device = device if device is not None else torch.device("cpu")

        if group_names is not None:
            if len(group_names) != num_groups:
                raise ValueError(
                    f"group_names length ({len(group_names)}) must equal num_groups ({num_groups})"
                )
            self.group_names = [str(n) for n in group_names]
        else:
            self.group_names = [f"group_{i}" for i in range(num_groups)]

        # Internal state
        self._step_count = 0
        self.log_weights = torch.zeros(self.num_groups, dtype=torch.float32, device=self.device)
        self.group_weights = torch.full(
            (self.num_groups,), 1.0 / self.num_groups, dtype=torch.float32, device=self.device
        )
        self.last_group_losses = torch.zeros(self.num_groups, dtype=torch.float32, device=self.device)
        self.last_group_counts = torch.zeros(self.num_groups, dtype=torch.long, device=self.device)

    @classmethod
    def from_group_labels(
        cls,
        labels: Sequence[Any],
        learning_rate: float = 0.01,
        update_frequency: int = 1,
        min_group_size: int = 1,
        smoothing: float = 0.0,
        min_weight: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> Tuple["GroupDRO", Dict[Any, int]]:
        """Instantiate GroupDRO dynamically from arbitrary group labels.

        Parameters
        ----------
        labels : Sequence[Any]
            Collection of group labels (e.g., category names or modality strings).

        Returns
        -------
        dro : GroupDRO
            Initialized GroupDRO instance.
        label_to_idx : Dict[Any, int]
            Mapping from label to integer group index [0, K-1].
        """
        unique_labels = sorted(list(set(labels)))
        if not unique_labels:
            raise ValueError("labels collection cannot be empty.")
        label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        dro = cls(
            num_groups=len(unique_labels),
            learning_rate=learning_rate,
            update_frequency=update_frequency,
            min_group_size=min_group_size,
            smoothing=smoothing,
            min_weight=min_weight,
            group_names=[str(lbl) for lbl in unique_labels],
            device=device,
        )
        return dro, label_to_idx

    def reset(self) -> None:
        """Reset group weights to uniform distribution and step counter to 0."""
        self._step_count = 0
        self.log_weights.zero_()
        self.group_weights.fill_(1.0 / self.num_groups)
        self.last_group_losses.zero_()
        self.last_group_counts.zero_()

    def compute_group_losses(
        self,
        per_example_losses: torch.Tensor,
        group_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean loss and sample count for each group.

        Parameters
        ----------
        per_example_losses : torch.Tensor
            1-D tensor of shape (N,) containing losses per example.
        group_indices : torch.Tensor
            1-D tensor of shape (N,) containing integer group IDs in [0, num_groups - 1].

        Returns
        -------
        group_losses : torch.Tensor
            1-D tensor of shape (num_groups,) with mean loss per group.
            For inactive/empty groups, returns 0.0.
        group_counts : torch.Tensor
            1-D tensor of shape (num_groups,) with sample counts per group.
        """
        if per_example_losses.ndim != 1:
            raise ValueError(
                f"per_example_losses must be 1D, got shape {per_example_losses.shape}"
            )
        if group_indices.ndim != 1:
            raise ValueError(f"group_indices must be 1D, got shape {group_indices.shape}")
        if len(per_example_losses) != len(group_indices):
            raise ValueError(
                f"per_example_losses length ({len(per_example_losses)}) "
                f"must match group_indices length ({len(group_indices)})"
            )

        losses = per_example_losses.to(self.device)
        groups = group_indices.to(self.device)

        if len(groups) > 0 and ((groups < 0) | (groups >= self.num_groups)).any():
            invalid = groups[(groups < 0) | (groups >= self.num_groups)]
            raise ValueError(
                f"Group indices out of bounds [0, {self.num_groups - 1}]: {invalid.tolist()[:5]}"
            )

        group_losses = torch.zeros(self.num_groups, dtype=torch.float32, device=self.device)
        group_counts = torch.zeros(self.num_groups, dtype=torch.long, device=self.device)

        for g in range(self.num_groups):
            mask = groups == g
            count = mask.sum().item()
            group_counts[g] = count
            if count >= self.min_group_size:
                group_losses[g] = losses[mask].mean()
            else:
                group_losses[g] = 0.0

        self.last_group_losses = group_losses.detach().clone()
        self.last_group_counts = group_counts.detach().clone()
        return group_losses, group_counts

    def update_weights(
        self,
        group_losses: torch.Tensor,
        group_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Update group weights using exponentiated gradient ascent in log-space.

        Parameters
        ----------
        group_losses : torch.Tensor
            (num_groups,) tensor of mean losses per group.
        group_counts : torch.Tensor, optional
            (num_groups,) tensor of sample counts. If provided, groups with
            count < min_group_size are considered inactive and not updated.

        Returns
        -------
        group_weights : torch.Tensor
            Updated and normalized (num_groups,) probability vector summing to 1.0.
        """
        g_losses = group_losses.to(self.device)

        if group_counts is not None:
            active_mask = (group_counts.to(self.device) >= self.min_group_size)
        else:
            # If counts not provided, treat groups with non-zero or positive loss as active
            active_mask = torch.ones(self.num_groups, dtype=torch.bool, device=self.device)

        # Update log-weights only for active groups
        if active_mask.any():
            # Exponentiated update in log-space: s_k <- s_k + eta * L_k
            update_delta = self.learning_rate * g_losses.detach()
            self.log_weights[active_mask] = self.log_weights[active_mask] + update_delta[active_mask]

        # Normalize via softmax to ensure numerical stability
        weights = F.softmax(self.log_weights, dim=0)

        # Optional smoothing: (1 - alpha) * q + alpha / K
        if self.smoothing > 0.0:
            weights = (1.0 - self.smoothing) * weights + (self.smoothing / self.num_groups)

        # Optional clipping / min-weight floor
        if self.min_weight > 0.0:
            weights = torch.clamp(weights, min=self.min_weight)
            weights = weights / weights.sum()

        # Keep log_weights in sync with post-processed weights
        self.log_weights = torch.log(weights.clamp(min=1e-12))
        self.group_weights = weights.detach().clone()
        return self.group_weights

    def compute_weighted_loss(
        self,
        per_example_losses: torch.Tensor,
        group_indices: torch.Tensor,
        update_weights: bool = False,
    ) -> torch.Tensor:
        """Compute the scalar group-weighted loss objective.

        Loss is weighted across active groups:
            \\mathcal{L} = \\frac{\\sum_{k \\in \\text{active}} q_k L_k}{\\sum_{k \\in \\text{active}} q_k}

        If `update_weights=True` and the update counter matches `update_frequency`,
        group weights are updated via exponentiated gradient ascent.

        Parameters
        ----------
        per_example_losses : torch.Tensor
            (N,) tensor of loss values per example.
        group_indices : torch.Tensor
            (N,) tensor of group indices in [0, num_groups - 1].
        update_weights : bool, default=False
            Whether to update group weights with this batch.

        Returns
        -------
        weighted_loss : torch.Tensor
            Scalar differentiable tensor representing the group-weighted objective.
        """
        group_losses, group_counts = self.compute_group_losses(
            per_example_losses, group_indices
        )

        active_mask = group_counts >= self.min_group_size
        if not active_mask.any():
            # Fallback if no active groups meet min_group_size
            return per_example_losses.mean()

        # Re-normalize weights across active groups in this batch
        active_weights = self.group_weights[active_mask]
        norm_factor = active_weights.sum().clamp(min=1e-8)
        normalized_active_weights = active_weights / norm_factor

        # Differentiable weighted sum
        weighted_loss = (normalized_active_weights * group_losses[active_mask]).sum()

        if update_weights:
            self._step_count += 1
            if (self._step_count % self.update_frequency) == 0:
                self.update_weights(group_losses, group_counts)

        return weighted_loss

    def step(
        self,
        per_example_losses: torch.Tensor,
        group_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method combining loss computation and weight update.

        Returns
        -------
        weighted_loss : torch.Tensor
            Differentiable scalar loss.
        group_weights : torch.Tensor
            Updated group weights.
        """
        loss = self.compute_weighted_loss(
            per_example_losses, group_indices, update_weights=True
        )
        return loss, self.group_weights

    def get_weights(self) -> torch.Tensor:
        """Return current group weights tensor of shape (num_groups,)."""
        return self.group_weights.clone()

    def set_weights(self, weights: Union[torch.Tensor, List[float]]) -> None:
        """Manually set group weights (must sum to 1.0)."""
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            weights = weights.to(dtype=torch.float32, device=self.device)

        if weights.shape != (self.num_groups,):
            raise ValueError(
                f"Weights shape {weights.shape} does not match (num_groups={self.num_groups},)"
            )
        if (weights < 0).any():
            raise ValueError("Group weights cannot be negative.")

        w_sum = weights.sum().item()
        if abs(w_sum - 1.0) > 1e-4:
            weights = weights / weights.sum()

        self.group_weights = weights.detach().clone()
        self.log_weights = torch.log(self.group_weights.clamp(min=1e-12))
