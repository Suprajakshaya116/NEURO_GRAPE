"""
Neuro-GRAPE Standard Dataset & Loader Interface
===============================================
Provides a unified, stable, and PyTorch-compatible data interface for Members 2 and 3.

Key Guarantees:
- Stable Identifiers: Every sample is uniquely identified by `sample_id`.
  Row-order matching is strictly prohibited.
- Modality & Cohort Independence: Explicitly represents EEG, MEG, and fMRI as
  independent experiments with stimulus-identity alignment only.
- Leakage Prevention: Transparent cross-validation split assignment per stimulus.
- Fast, Memory-Conscious Loading: Arrays are loaded from disk on-demand with caching options.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Safe optional torch import
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    TorchDataset = object  # type: ignore


class NeuroGrapeDataset(TorchDataset):
    """Unified dataset class for Neuro-GRAPE multimodal neural representations.

    Parameters
    ----------
    manifest : str, Path, or pd.DataFrame
        Path to data_inventory.csv or an existing DataFrame.
    modality : str, optional
        Filter by modality ('eeg', 'meg', 'fmri'). If None, includes all modalities.
    split : str, optional
        Filter by split ('train', 'val', 'test'). If None, includes all splits.
    stimuli : list of str, optional
        Filter by specific stimulus IDs (e.g. ['puck_02s.jpg', ...]).
    as_torch : bool, default=False
        If True and PyTorch is installed, returns features as torch.Tensor.
    cache_data : bool, default=False
        If True, caches loaded numpy arrays in memory for fast epoch access.
    data_root : str or Path, optional
        Base directory to resolve relative data paths in manifest.
    """

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame],
        modality: Optional[str] = None,
        split: Optional[str] = None,
        stimuli: Optional[List[str]] = None,
        as_torch: bool = False,
        cache_data: bool = False,
        data_root: Optional[Union[str, Path]] = None,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")
            self.df = pd.read_csv(manifest_path)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy()
        else:
            raise TypeError("Manifest must be a path or a pandas DataFrame.")

        self.data_root = Path(data_root) if data_root is not None else Path.cwd()
        self.as_torch = as_torch and HAS_TORCH
        self.cache_data = cache_data
        self._cache: Dict[str, np.ndarray] = {}

        # Validate required columns
        required_cols = {
            "sample_id", "modality", "subject_id", "session_id",
            "stimulus_id", "split", "data_path"
        }
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

        # Check uniqueness of sample_id
        if self.df["sample_id"].duplicated().any():
            dups = self.df["sample_id"][self.df["sample_id"].duplicated()].tolist()
            raise ValueError(f"Found duplicate sample_ids in manifest: {dups[:5]}")

        # Apply filters
        if modality is not None:
            modality_clean = modality.lower()
            self.df = self.df[self.df["modality"].str.lower() == modality_clean].copy()

        if split is not None:
            split_clean = split.lower()
            self.df = self.df[self.df["split"].str.lower() == split_clean].copy()

        if stimuli is not None:
            stim_set = set(stimuli)
            self.df = self.df[self.df["stimulus_id"].isin(stim_set)].copy()

        # Build index lookup
        self.df = self.df.reset_index(drop=True)
        self._id_to_idx = {sid: idx for idx, sid in enumerate(self.df["sample_id"])}

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_data_path(self, rel_path: str) -> Path:
        p = Path(rel_path)
        if p.is_absolute() and p.exists():
            return p
        # Check relative to data_root
        candidate = self.data_root / rel_path
        if candidate.exists():
            return candidate
        # Check relative to repo root
        return p

    def _load_array(self, sample_id: str, rel_path: str) -> np.ndarray:
        if self.cache_data and sample_id in self._cache:
            return self._cache[sample_id]

        full_path = self._resolve_data_path(rel_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Feature array not found at: {full_path}")

        arr = np.load(full_path)
        if self.cache_data:
            self._cache[sample_id] = arr
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve a single sample record by integer index.

        Returns
        -------
        sample : dict
            Standardized record for downstream consumption by Member 2/3.
        """
        row = self.df.iloc[idx]
        sample_id = row["sample_id"]
        rel_path = row["data_path"]

        features = self._load_array(sample_id, rel_path)

        if self.as_torch:
            features = torch.from_numpy(features)

        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "modality": row["modality"],
            "subject_id": row["subject_id"],
            "session_id": row["session_id"],
            "run_id": row.get("run_id", "run-01"),
            "stimulus_id": row["stimulus_id"],
            "category_id": int(row["category_id"]) if pd.notna(row.get("category_id")) else -1,
            "category_name": str(row.get("category_name", "")),
            "split": row["split"],
            "features": features,
            "signal_units": row.get("signal_units", ""),
            "sampling_rate": float(row.get("sampling_rate", 0.0)),
            "duration_sec": float(row.get("duration_sec", 0.0)),
        }
        return record

    def get_by_sample_id(self, sample_id: str) -> Dict[str, Any]:
        """Fast lookup of a sample record by its deterministic sample_id."""
        if sample_id not in self._id_to_idx:
            raise KeyError(f"sample_id '{sample_id}' not found in active dataset.")
        return self.__getitem__(self._id_to_idx[sample_id])

    def get_cross_modal_sample(self, stimulus_id: str) -> Dict[str, Dict[str, Any]]:
        """Retrieve all modality samples available for a given stimulus filename.

        Returns
        -------
        cross_modal : dict
            Mapping from modality name to sample dictionary.
        """
        sub_df = self.df[self.df["stimulus_id"] == stimulus_id]
        if sub_df.empty:
            raise KeyError(f"stimulus_id '{stimulus_id}' not found in dataset.")

        results = {}
        for _, row in sub_df.iterrows():
            sid = row["sample_id"]
            results[row["modality"]] = self.get_by_sample_id(sid)
        return results

    def get_modality_batch(
        self, modality: str, split: Optional[str] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Convenience method for Member 2 to obtain a full stacked feature tensor.

        Returns
        -------
        features : np.ndarray
            Stacked array of shape (N, C, T) or (N, ...).
        metadata_df : pd.DataFrame
            Corresponding metadata aligned row-for-row with features.
        """
        sub_ds = NeuroGrapeDataset(
            manifest=self.df,
            modality=modality,
            split=split,
            as_torch=False,
            data_root=self.data_root,
        )
        if len(sub_ds) == 0:
            raise ValueError(f"No samples found for modality '{modality}' and split '{split}'.")

        arrays = [sub_ds[i]["features"] for i in range(len(sub_ds))]
        stacked = np.stack(arrays, axis=0)
        return stacked, sub_ds.df.copy()

    def get_metadata_table(self) -> pd.DataFrame:
        """Convenience method for Member 3 to inspect groups, splits, and metadata."""
        return self.df.copy()
