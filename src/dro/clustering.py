"""Embedding-Based Neural Regime Clustering Module.

Member 3 Implementation for Neuro-GRAPE.

Provides unsupervised clustering over Member 2's multimodal shared latent embeddings.
Guarantees and Scientific Guardrails:
1. Validates embedding dimensionality, finite values, and deterministic sample IDs.
2. Leakage Prevention: Fits clustering models strictly on the training split ('train').
   Validation and test splits are assigned via the fitted model and NEVER used to fit clusters.
3. Stable-ID Joins: Assigns clusters via deterministic `sample_id` joins, NEVER relying
   on CSV row order or tensor indices.
4. Exploratory Disclaimer: Clusters are treated as exploratory data groupings for
   distributionally robust optimization (GroupDRO), NOT proven biological neural regimes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class EmbeddingClustering:
    """Clustering manager for discovering exploratory regime groups from shared embeddings.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters (groups) to discover (K >= 2).
    random_state : int, default=42
        Random seed for KMeans initialization and reproducibility.
    expected_dim : int, default=128
        Expected embedding vector dimension.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        random_state: int = 42,
        expected_dim: Optional[int] = None,
    ) -> None:
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        if expected_dim is not None and expected_dim < 1:
            raise ValueError(f"expected_dim must be >= 1, got {expected_dim}")
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.expected_dim = int(expected_dim) if expected_dim is not None else None

        self.model: Optional[KMeans] = None
        self.diagnostics_: Dict[str, Any] = {}
        self.cluster_centers_: Optional[np.ndarray] = None
        self.fitted_train_sample_ids_: List[str] = []

    @staticmethod
    def validate_embeddings(
        embeddings: Union[Dict[str, Any], np.ndarray],
        sample_ids: Sequence[str],
        expected_dim: Optional[int] = 128,
    ) -> np.ndarray:
        """Validate embedding vectors for dimension, finite values, and unique IDs.

        Parameters
        ----------
        embeddings : dict or np.ndarray
            Dictionary mapping sample_id -> 1D vector, or 2D array of shape (N, D).
        sample_ids : sequence of str
            Unique sample IDs corresponding to the embeddings.
        expected_dim : int, optional, default=128
            Expected feature dimension. If None, dimension is inferred from input.

        Returns
        -------
        embed_matrix : np.ndarray
            Validated 2D float32 array of shape (N, D).

        Raises
        ------
        ValueError: If duplicate IDs, NaN/Inf values, or shape mismatches occur.
        """
        # 1. Validate unique IDs
        id_list = list(sample_ids)
        if len(id_list) != len(set(id_list)):
            duplicates = [sid for sid in id_list if id_list.count(sid) > 1]
            raise ValueError(f"Found duplicate sample_ids: {sorted(set(duplicates))[:5]}")

        # 2. Extract matrix
        if isinstance(embeddings, dict):
            missing_ids = [sid for sid in id_list if sid not in embeddings]
            if missing_ids:
                raise KeyError(
                    f"Embeddings dict missing {len(missing_ids)} requested sample IDs: {missing_ids[:5]}"
                )
            vectors = []
            for sid in id_list:
                vec = embeddings[sid]
                if HAS_TORCH and isinstance(vec, torch.Tensor):
                    vec = vec.detach().cpu().numpy()
                else:
                    vec = np.asarray(vec, dtype=np.float32)
                vectors.append(vec)
            matrix = np.stack(vectors, axis=0)
        elif isinstance(embeddings, np.ndarray):
            matrix = np.asarray(embeddings, dtype=np.float32)
            if matrix.shape[0] != len(id_list):
                raise ValueError(
                    f"Embedding matrix rows ({matrix.shape[0]}) does not match sample_ids count ({len(id_list)})"
                )
        else:
            raise TypeError(f"Unsupported embeddings type: {type(embeddings)}")

        # 3. Validate dimension
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got shape {matrix.shape}")
        if expected_dim is not None and matrix.shape[1] != expected_dim:
            raise ValueError(
                f"Expected 2D embedding matrix of shape (N, {expected_dim}), got {matrix.shape}"
            )

        # 4. Validate finite values (0 NaNs, 0 Infs)
        if not np.isfinite(matrix).all():
            nan_count = int(np.isnan(matrix).sum())
            inf_count = int(np.isinf(matrix).sum())
            raise ValueError(
                f"Embeddings contain non-finite values (NaNs: {nan_count}, Infs: {inf_count})."
            )

        return matrix

    def fit(
        self,
        embeddings: Union[Dict[str, Any], np.ndarray],
        metadata_df: pd.DataFrame,
    ) -> "EmbeddingClustering":
        """Fit K-Means clustering model strictly on the training split ('train').

        Parameters
        ----------
        embeddings : dict or np.ndarray
            Exported shared embeddings from Member 2.
        metadata_df : pd.DataFrame
            Metadata containing at least 'sample_id' and 'split' columns.

        Returns
        -------
        self : EmbeddingClustering
        """
        for req_col in ("sample_id", "split"):
            if req_col not in metadata_df.columns:
                raise ValueError(f"metadata_df missing required column '{req_col}'")

        # 1. Filter strictly to training split
        train_df = metadata_df[metadata_df["split"].str.lower() == "train"].copy()
        if len(train_df) == 0:
            raise ValueError("Training split is empty; cannot fit clustering model.")

        if len(train_df) < self.n_clusters:
            raise ValueError(
                f"Training sample count ({len(train_df)}) is less than n_clusters ({self.n_clusters})."
            )

        train_sample_ids = train_df["sample_id"].tolist()
        self.fitted_train_sample_ids_ = list(train_sample_ids)

        # 2. Extract and validate training embeddings
        train_matrix = self.validate_embeddings(
            embeddings, train_sample_ids, expected_dim=self.expected_dim
        )
        if self.expected_dim is None:
            self.expected_dim = int(train_matrix.shape[1])


        # 3. Fit KMeans on training split ONLY
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        train_labels = self.model.fit_predict(train_matrix)
        self.cluster_centers_ = self.model.cluster_centers_

        # 4. Compute training diagnostics
        cluster_counts = pd.Series(train_labels).value_counts().to_dict()
        inertia = float(self.model.inertia_)
        sil_score = None
        if self.n_clusters > 1 and len(train_matrix) > self.n_clusters:
            try:
                sil_score = float(silhouette_score(train_matrix, train_labels))
            except Exception:
                sil_score = None

        self.diagnostics_ = {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "expected_dim": self.expected_dim,
            "fit_split": "train",
            "train_sample_count": len(train_df),
            "train_cluster_counts": cluster_counts,
            "inertia": inertia,
            "silhouette_score_train": sil_score,
            "disclaimer": (
                "Clusters represent exploratory mathematical groupings across shared "
                "neural embeddings for GroupDRO evaluation, NOT proven biological neural regimes."
            ),
        }

        return self

    def assign_clusters(
        self,
        embeddings: Union[Dict[str, Any], np.ndarray],
        metadata_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assign clusters to samples across any split using stable sample_id joins.

        Parameters
        ----------
        embeddings : dict or np.ndarray
            Exported shared embeddings.
        metadata_df : pd.DataFrame
            Metadata table containing 'sample_id'.

        Returns
        -------
        assigned_df : pd.DataFrame
            DataFrame with original columns plus 'cluster_id' and 'group_id'
            joined deterministically on 'sample_id'.
        """
        if self.model is None:
            raise RuntimeError("Clustering model must be fitted on training data before assigning clusters.")

        if "sample_id" not in metadata_df.columns:
            raise ValueError("metadata_df missing 'sample_id' column.")

        # Extract and validate all samples in metadata
        sample_ids = metadata_df["sample_id"].tolist()
        matrix = self.validate_embeddings(embeddings, sample_ids, expected_dim=self.expected_dim)

        # Predict cluster assignments using fitted training centroids
        predicted_clusters = self.model.predict(matrix)

        # Build assignment table keyed by sample_id
        cluster_mapping_df = pd.DataFrame({
            "sample_id": sample_ids,
            "cluster_id": predicted_clusters,
            "group_id": predicted_clusters,  # Alias for GroupDRO consumption
        })

        # Deterministic join on sample_id — never rely on row order
        result_df = pd.merge(metadata_df.copy(), cluster_mapping_df, on="sample_id", how="left")
        return result_df

    def save_diagnostics(self, output_path: Union[str, Path]) -> None:
        """Save cluster diagnostics to a JSON file."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(self.diagnostics_, f, indent=2)


def fit_regime_clustering(
    embeddings_source: Union[str, Path, Dict[str, Any]],
    metadata_source: Union[str, Path, pd.DataFrame] = "embeddings/metadata.csv",
    n_clusters: int = 2,
    random_state: int = 42,
    expected_dim: Optional[int] = None,
) -> Tuple[EmbeddingClustering, pd.DataFrame]:
    """Helper function to load embeddings, fit on train split, and assign clusters.

    Parameters
    ----------
    embeddings_source : Path, str, or dict
        Path to .pt dictionary, directory of .npy files, or in-memory dict.
    metadata_source : Path, str, or DataFrame
        Metadata table containing 'sample_id' and 'split'.
    n_clusters : int, default=2
        Number of clusters.
    random_state : int, default=42
        Seed.
    expected_dim : int, default=128
        Expected embedding dimension.

    Returns
    -------
    clustering : EmbeddingClustering
        Fitted clustering object.
    assignments_df : pd.DataFrame
        Complete DataFrame with joined cluster_id.
    """
    # 1. Load metadata
    if isinstance(metadata_source, (str, Path)):
        meta_path = Path(metadata_source)
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        meta_df = pd.read_csv(meta_path)
    elif isinstance(metadata_source, pd.DataFrame):
        meta_df = metadata_source.copy()
    else:
        raise TypeError(f"Invalid metadata_source type: {type(metadata_source)}")

    # 2. Load embeddings
    if isinstance(embeddings_source, (str, Path)):
        embed_path = Path(embeddings_source)
        if not embed_path.exists():
            raise FileNotFoundError(f"Embeddings path not found: {embed_path}")

        if embed_path.is_file() and embed_path.suffix == ".pt":
            embed_dict = torch.load(embed_path, map_location="cpu")
        elif embed_path.is_dir():
            embed_dict = {}
            for sid in meta_df["sample_id"]:
                npy_file = embed_path / f"{sid}.npy"
                if npy_file.exists():
                    embed_dict[sid] = np.load(npy_file)
        else:
            raise ValueError(f"Unsupported embeddings path: {embed_path}")
    elif isinstance(embeddings_source, dict):
        embed_dict = embeddings_source
    else:
        raise TypeError(f"Invalid embeddings_source type: {type(embeddings_source)}")

    # 3. Fit and assign
    clustering = EmbeddingClustering(
        n_clusters=n_clusters,
        random_state=random_state,
        expected_dim=expected_dim,
    )
    clustering.fit(embed_dict, meta_df)
    assignments_df = clustering.assign_clusters(embed_dict, meta_df)
    return clustering, assignments_df
