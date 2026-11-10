"""Evaluation Utilities for GroupDRO and Subgroup Generalization.

Member 3 Implementation for Neuro-GRAPE.

Provides evaluation routines that compute:
1. Overall mean loss or task metric.
2. Per-group mean loss or task performance.
3. Group sample counts.
4. Worst-group performance, respecting the correct optimization direction:
   - For loss (higher_is_better=False): worst group is argmax (maximum loss).
   - For accuracy/retrieval (higher_is_better=True): worst group is argmin (minimum metric).
5. The performance gap between overall and worst-group metrics:
   - If higher_is_better=False: gap = worst_group_value - overall_value (>= 0).
   - If higher_is_better=True:  gap = overall_value - worst_group_value (>= 0).
6. Safe Handling: Inactive/missing groups (count < min_group_size) are excluded from
   worst-group selection so that unobserved groups with 0 counts do not corrupt metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class EvaluationResult:
    """Structured evaluation report containing overall, per-group, and worst-group statistics.

    Attributes
    ----------
    metric_name : str
        Name of the evaluated metric (e.g. 'contrastive_loss', 'cosine_similarity').
    higher_is_better : bool
        True if higher metric indicates better performance (e.g. accuracy),
        False if lower is better (e.g. loss).
    overall_value : float
        Population-wide mean metric across all evaluated examples.
    per_group_values : Dict[int, float]
        Dictionary mapping group ID -> mean metric value.
    group_counts : Dict[int, int]
        Dictionary mapping group ID -> number of evaluated samples in that group.
    worst_group_id : Optional[int]
        Group ID exhibiting the worst performance among active groups (None if no active groups).
    worst_group_value : Optional[float]
        Metric value achieved by the worst group.
    performance_gap : Optional[float]
        Disparity between overall performance and worst-group performance.
        Non-negative by definition (0.0 means perfect group parity).
    active_groups : List[int]
        List of group IDs that met the minimum sample size threshold.
    inactive_groups : List[int]
        List of group IDs with sample count below minimum threshold.
    """

    metric_name: str
    higher_is_better: bool
    overall_value: float
    per_group_values: Dict[int, float]
    group_counts: Dict[int, int]
    worst_group_id: Optional[int]
    worst_group_value: Optional[float]
    performance_gap: Optional[float]
    active_groups: List[int]
    inactive_groups: List[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)

    def to_summary_series(self) -> pd.Series:
        """Convert scalar summary metrics to a pandas Series."""
        return pd.Series({
            "metric_name": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "overall_value": self.overall_value,
            "worst_group_id": self.worst_group_id,
            "worst_group_value": self.worst_group_value,
            "performance_gap": self.performance_gap,
            "num_active_groups": len(self.active_groups),
            "num_inactive_groups": len(self.inactive_groups),
        })

    def to_group_dataframe(self) -> pd.DataFrame:
        """Convert per-group metrics into a tabular DataFrame."""
        rows = []
        for g in sorted(self.group_counts.keys()):
            rows.append({
                "group_id": g,
                "metric_name": self.metric_name,
                "sample_count": self.group_counts[g],
                "mean_value": self.per_group_values.get(g, np.nan),
                "is_active": g in self.active_groups,
                "is_worst_group": g == self.worst_group_id,
            })
        return pd.DataFrame(rows)


class GroupEvaluator:
    """Evaluates per-group and worst-group performance metrics across splits and modalities.

    Parameters
    ----------
    min_group_size : int, default=1
        Minimum number of samples required to evaluate a group and consider it active.
    """

    def __init__(self, min_group_size: int = 1) -> None:
        if min_group_size < 1:
            raise ValueError(f"min_group_size must be >= 1, got {min_group_size}")
        self.min_group_size = int(min_group_size)

    def evaluate(
        self,
        values: Union[np.ndarray, Sequence[float], "torch.Tensor"],
        group_ids: Union[np.ndarray, Sequence[int], "torch.Tensor"],
        metric_name: str = "loss",
        higher_is_better: bool = False,
        all_group_ids: Optional[Sequence[int]] = None,
    ) -> EvaluationResult:
        """Compute comprehensive group evaluation metrics.

        Parameters
        ----------
        values : array-like of shape (N,)
            Per-sample metric or loss values.
        group_ids : array-like of shape (N,)
            Integer or string group identifier per sample.
        metric_name : str, default='loss'
            Descriptive name of the metric.
        higher_is_better : bool, default=False
            Whether higher values are better (True for accuracy, False for loss).
        all_group_ids : sequence of int, optional
            List of all expected group IDs in the universe, to detect unobserved groups.

        Returns
        -------
        result : EvaluationResult
            Complete evaluation statistics.
        """
        # Convert values to 1D float numpy array
        if HAS_TORCH and isinstance(values, torch.Tensor):
            vals = values.detach().cpu().numpy().astype(np.float64)
        else:
            vals = np.asarray(values, dtype=np.float64)

        if vals.ndim != 1:
            raise ValueError(f"values must be 1D, got shape {vals.shape}")

        # Convert group_ids to 1D numpy array
        if HAS_TORCH and isinstance(group_ids, torch.Tensor):
            g_ids = group_ids.detach().cpu().numpy()
        else:
            g_ids = np.asarray(group_ids)

        if g_ids.ndim != 1:
            raise ValueError(f"group_ids must be 1D, got shape {g_ids.shape}")

        if len(vals) != len(g_ids):
            raise ValueError(
                f"Length mismatch between values ({len(vals)}) and group_ids ({len(g_ids)})"
            )

        # Check finite values
        if not np.isfinite(vals).all():
            raise ValueError("Evaluated metric values contain non-finite numbers (NaN or Inf).")

        # Determine all expected groups
        observed_groups = sorted(list(set(g_ids.tolist())))
        if all_group_ids is not None:
            expected_groups = sorted(list(set(all_group_ids)))
        else:
            expected_groups = observed_groups

        if len(vals) == 0:
            return EvaluationResult(
                metric_name=metric_name,
                higher_is_better=higher_is_better,
                overall_value=np.nan,
                per_group_values={g: np.nan for g in expected_groups},
                group_counts={g: 0 for g in expected_groups},
                worst_group_id=None,
                worst_group_value=None,
                performance_gap=None,
                active_groups=[],
                inactive_groups=expected_groups,
            )

        overall_value = float(np.mean(vals))
        per_group_values: Dict[int, float] = {}
        group_counts: Dict[int, int] = {}
        active_groups: List[int] = []
        inactive_groups: List[int] = []

        for g in expected_groups:
            mask = g_ids == g
            count = int(np.sum(mask))
            group_counts[g] = count

            if count >= self.min_group_size:
                mean_val = float(np.mean(vals[mask]))
                per_group_values[g] = mean_val
                active_groups.append(g)
            else:
                per_group_values[g] = np.nan
                inactive_groups.append(g)

        # Identify worst group among ACTIVE groups only
        if not active_groups:
            worst_group_id = None
            worst_group_value = None
            performance_gap = None
        else:
            if higher_is_better:
                # Higher is better: worst group has minimum metric value
                worst_group_id = min(active_groups, key=lambda g: per_group_values[g])
                worst_group_value = per_group_values[worst_group_id]
                performance_gap = float(overall_value - worst_group_value)
            else:
                # Lower is better (Loss): worst group has maximum loss value
                worst_group_id = max(active_groups, key=lambda g: per_group_values[g])
                worst_group_value = per_group_values[worst_group_id]
                performance_gap = float(worst_group_value - overall_value)

        return EvaluationResult(
            metric_name=metric_name,
            higher_is_better=higher_is_better,
            overall_value=overall_value,
            per_group_values=per_group_values,
            group_counts=group_counts,
            worst_group_id=worst_group_id,
            worst_group_value=worst_group_value,
            performance_gap=performance_gap,
            active_groups=active_groups,
            inactive_groups=inactive_groups,
        )


def evaluate_per_group(
    values: Union[np.ndarray, Sequence[float], "torch.Tensor"],
    group_ids: Union[np.ndarray, Sequence[int], "torch.Tensor"],
    metric_name: str = "loss",
    higher_is_better: bool = False,
    min_group_size: int = 1,
    all_group_ids: Optional[Sequence[int]] = None,
) -> EvaluationResult:
    """Functional convenience wrapper for GroupEvaluator."""
    evaluator = GroupEvaluator(min_group_size=min_group_size)
    return evaluator.evaluate(
        values=values,
        group_ids=group_ids,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        all_group_ids=all_group_ids,
    )
