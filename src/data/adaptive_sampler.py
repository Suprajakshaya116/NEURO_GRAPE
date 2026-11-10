"""Neuro-GRAPE Adaptive Data Mixture & Dynamic Sampler (GRAPE / DOGE).

Implements dynamic feedback-driven sampling weight updates across modalities and
domains, aligning with the principles of GRAPE (Group-Robust Adaptive Pretraining)
and DOGE (Domain Optimization with Group-robust Exponentiation).

Design & Mechanics
------------------
1. Feedback Signal:
   Tracks running exponential moving average (EMA) of training loss per modality
   or regime:
       \\bar{L}_m^{(t)} = \\beta \\bar{L}_m^{(t-1)} + (1 - \\beta) L_m^{(t)}
   where \\beta \\in [0, 1) is a configurable smoothing coefficient.
2. Lagging Rate / Relative Difficulty:
   Computes normalized difficulty:
       r_m^{(t)} = \\bar{L}_m^{(t)} / \\sum_{m'} \\bar{L}_{m'}^{(t)}
3. Dynamic Exponentiated Update:
       \\tilde{\\alpha}_m^{(t+1)} = \\alpha_m^{(t)} \\exp(\\gamma (r_m^{(t)} - 1/M))
   where \\gamma is the update step size.
4. Simplex Projection with Minimum Allocation Floor:
   Enforces \\sum_m \\alpha_m = 1 and \\alpha_m \\ge \\alpha_{\\min} > 0 to prevent modality
   or group starvation:
       \\alpha_m = \\alpha_{\\min} + (1 - M \\alpha_{\\min}) \\cdot \\frac{\\tilde{\\alpha}_m}{\\sum_{m'} \\tilde{\\alpha}_{m'}}
5. Batch-Level Selection:
   Directly changes which modalities or modality batches are passed to encoders
   and optimized at each training step, logging empirical selection counts.
6. Leakage Protection:
   Rejects any update not explicitly flagged as training-split data.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch


class AdaptiveMixtureSampler:
    """Adaptive data-mixture sampler that updates modality/domain sampling weights.

    Args:
        modalities: Sequence of modality names (e.g. ['eeg', 'meg', 'fmri']).
        step_size: Learning rate gamma for exponentiated weight update. Default 0.05.
        ema_beta: Decay factor for loss exponential moving average. Default 0.9.
        min_weight: Minimum allocation floor alpha_min per modality. Default 0.15.
        seed: Random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        modalities: Sequence[str] = ("eeg", "meg", "fmri"),
        step_size: float = 0.05,
        ema_beta: float = 0.9,
        min_weight: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.modalities = sorted(list(modalities))
        self.num_modalities = len(self.modalities)
        if self.num_modalities == 0:
            raise ValueError("modalities sequence must not be empty.")

        if min_weight * self.num_modalities >= 1.0:
            raise ValueError(
                f"min_weight ({min_weight}) * num_modalities ({self.num_modalities}) "
                f"must be strictly less than 1.0."
            )

        self.step_size = step_size
        self.ema_beta = ema_beta
        self.min_weight = min_weight
        self.seed = seed

        # Deterministic generator
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

        # Initialize log-space weights and uniform probabilities
        self.logits: Dict[str, float] = {m: 0.0 for m in self.modalities}
        init_w = 1.0 / self.num_modalities
        self.weights: Dict[str, float] = {m: init_w for m in self.modalities}

        # Initialize loss EMA tracker
        self.loss_ema: Dict[str, float] = {m: 1.0 for m in self.modalities}
        self.step_count: int = 0

        # Empirical selection counters
        self.selection_counts: Dict[str, int] = {m: 0 for m in self.modalities}
        self.total_samples_drawn: int = 0

        # Trajectory history for diagnostics
        self.history: List[Dict[str, float]] = [dict(self.weights)]

    @property
    def current_weights(self) -> Dict[str, float]:
        """Return a copy of the current sampling probability distribution."""
        return dict(self.weights)

    def update_from_losses(
        self,
        modality_losses: Mapping[str, float],
        is_train: bool = True,
    ) -> Dict[str, float]:
        """Update mixture weights using latest observed modality training losses.

        Args:
            modality_losses: Dict mapping modality name to observed scalar loss.
            is_train: Guard flag ensuring only training-split losses update weights.

        Returns:
            Updated weights dictionary.

        Raises:
            ValueError: If is_train is False (preventing validation/test leakage).
        """
        if not is_train:
            raise ValueError(
                "Adaptive sampler updates are strictly prohibited from validation or test losses."
            )

        self.step_count += 1

        # 1. Update EMA of losses safely (handling missing, NaN, Inf)
        valid_losses = {}
        for m, loss in modality_losses.items():
            if m in self.loss_ema:
                val = float(loss)
                if math.isfinite(val) and val >= 0.0:
                    valid_losses[m] = val
                    self.loss_ema[m] = (
                        self.ema_beta * self.loss_ema[m] + (1.0 - self.ema_beta) * val
                    )

        if not valid_losses:
            # Degenerate case: no valid losses provided; retain current weights
            return dict(self.weights)

        # 2. Compute relative difficulty / lagging ratio
        total_loss = sum(self.loss_ema[m] for m in self.modalities)
        if total_loss <= 1e-8 or not math.isfinite(total_loss):
            return dict(self.weights)

        uniform_share = 1.0 / self.num_modalities

        # 3. Exponentiated gradient update in stable log-space
        for m in self.modalities:
            rel_diff = (self.loss_ema[m] / total_loss) - uniform_share
            self.logits[m] += self.step_size * rel_diff
            # Clip logits to prevent unbounded drift
            self.logits[m] = max(-20.0, min(20.0, self.logits[m]))

        # 4. Softmax with max subtraction for numerical stability
        max_logit = max(self.logits.values())
        exps = {m: math.exp(self.logits[m] - max_logit) for m in self.modalities}
        unnorm_sum = sum(exps.values())
        if unnorm_sum <= 1e-8 or not math.isfinite(unnorm_sum):
            return dict(self.weights)

        normalized = {m: exps[m] / unnorm_sum for m in self.modalities}

        # 5. Apply minimum allocation floor
        remaining_budget = 1.0 - (self.min_weight * self.num_modalities)
        self.weights = {
            m: (normalized[m] * remaining_budget) + self.min_weight
            for m in self.modalities
        }

        # Ensure exact sum to 1.0
        final_sum = sum(self.weights.values())
        self.weights = {m: max(0.0, w / final_sum) for m, w in self.weights.items()}
        re_sum = sum(self.weights.values())
        self.weights = {m: w / re_sum for m, w in self.weights.items()}

        self.history.append(dict(self.weights))
        return dict(self.weights)

    def sample_active_modalities(
        self,
        k: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> List[str]:
        """Sample k active modalities according to the current mixture distribution.

        Args:
            k: Number of modalities to sample without replacement. If None, returns all.
            generator: PyTorch random generator for reproducibility. If None, uses internal RNG.

        Returns:
            List of sampled modality names.
        """
        gen = generator if generator is not None else self._rng

        if k is None or k >= self.num_modalities:
            sampled = list(self.modalities)
        else:
            weights_tensor = torch.tensor(
                [self.weights[m] for m in self.modalities], dtype=torch.float32
            )
            indices = torch.multinomial(
                weights_tensor, num_samples=k, replacement=False, generator=gen
            )
            sampled = [self.modalities[i.item()] for i in indices]

        # Update empirical counters
        for m in sampled:
            self.selection_counts[m] += 1
        self.total_samples_drawn += len(sampled)

        return sampled

    def sample_batch_modalities(
        self,
        batch: Mapping[str, torch.Tensor],
        k: Optional[int] = 2,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Filter an input batch to include only the sampled active modalities.

        Directly links the sampler to training batch composition: only the sampled
        modalities will be passed to encoders, graph layers, and loss functions.

        Args:
            batch: Full multimodal batch mapping (e.g. {'eeg': ..., 'meg': ..., 'fmri': ...}).
            k: Number of modalities to retain (e.g. 2 for cross-modal pair). If None, all.
            generator: Optional RNG generator.

        Returns:
            Sub-batch containing only the sampled modalities.
        """
        available = [m for m in self.modalities if m in batch]
        if not available:
            return {}

        sample_k = min(k, len(available)) if k is not None else len(available)
        # Sample from available modalities using current relative weights
        sub_weights = [self.weights[m] for m in available]
        sub_sum = sum(sub_weights)
        if sub_sum <= 1e-8:
            sub_probs = torch.ones(len(available), dtype=torch.float32) / len(available)
        else:
            sub_probs = torch.tensor([w / sub_sum for w in sub_weights], dtype=torch.float32)

        gen = generator if generator is not None else self._rng
        if sample_k >= len(available):
            selected = available
        else:
            indices = torch.multinomial(
                sub_probs, num_samples=sample_k, replacement=False, generator=gen
            )
            selected = [available[i.item()] for i in indices]

        # Update counters
        for m in selected:
            self.selection_counts[m] += 1
        self.total_samples_drawn += len(selected)

        return {m: batch[m] for m in selected}

    def get_history(self) -> List[Dict[str, float]]:
        """Return history of mixture weights across training updates."""
        return list(self.history)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return comprehensive diagnostic status of the sampler."""
        return {
            "weights": dict(self.weights),
            "loss_ema": dict(self.loss_ema),
            "selection_counts": dict(self.selection_counts),
            "total_samples_drawn": self.total_samples_drawn,
            "step_count": self.step_count,
            "min_weight": self.min_weight,
            "ema_beta": self.ema_beta,
            "step_size": self.step_size,
        }
