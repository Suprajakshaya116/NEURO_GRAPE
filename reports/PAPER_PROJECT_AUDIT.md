# Neuro-GRAPE: Factual Implementation Inventory & Pre-Paper Project Audit

**Document:** Pre-Manuscript Architectural & Algorithmic Inventory  
**Auditor:** Senior Research Scientist & Machine Learning Auditor  
**Date:** November 2026  
**Scope:** Verification of implemented components vs. proposed/planned features for the scientific paper draft  

---

## 1. Inventory of Implemented Components

Each component claimed in the research paper must correspond to concrete, verified implementation in the repository:

### 1.1 Data Preprocessing & Alignment Pipeline
* **What it does**: Ingests, filters, standardizes, downsamples, and epochs raw EEG, MEG, and fMRI data from THINGS-data into standardized arrays.
* **Code Location**: [`src/data/preprocessing.py`](file:///d:/projects/NEURO_GRAPE/src/data/preprocessing.py), [`src/data/dataset.py`](file:///d:/projects/NEURO_GRAPE/src/data/dataset.py), [`src/models/normalization.py`](file:///d:/projects/NEURO_GRAPE/src/models/normalization.py).
* **Modality Shapes**:
  * EEG: `(63, 225)` at 250 Hz, $[-100, 800]\text{ ms}$, channel Z-scored.
  * MEG: `(272, 225)` at 250 Hz, $[-100, 800]\text{ ms}$, channel Z-scored.
  * fMRI: `(96, 96, 60)` single-trial LS-S GLM beta volumes, normalized inside an anatomical brain mask (228,068 voxels).
* **Alignment Mechanism**: Strictly associative over `stimulus_id`. Zero raw temporal synchronization, zero subject overlap.
* **Implementation Status**: **VERIFIED & ACTIVE**.

### 1.2 Modality-Specific Encoders
* **What it does**: Projects heterogeneous input geometries into a shared $D=128$ dimensional latent representation normalized onto a unit hypersphere.
* **Code Location**: [`src/models/encoders.py`](file:///d:/projects/NEURO_GRAPE/src/models/encoders.py), [`src/models/model.py`](file:///d:/projects/NEURO_GRAPE/src/models/model.py).
* **Architectures**:
  * `EEGEncoder`: 3-block 1D CNN (kernel sizes 7, 5, 3 with stride 2), BatchNorm1d, GELU, AdaptiveAvgPool1d, Linear projection head (128 $\rightarrow$ 128), LayerNorm, and $L_2$ sphere projection.
  * `MEGEncoder`: 3-block 1D CNN (kernel sizes 7, 5, 3 with stride 2), BatchNorm1d, GELU, AdaptiveAvgPool1d, Linear projection head (128 $\rightarrow$ 128), LayerNorm, and $L_2$ sphere projection.
  * `FMRIEncoder`: 3-block 3D CNN (kernel size 3 with stride 2), BatchNorm3d, GELU, AdaptiveAvgPool3d, Linear projection head (64 $\rightarrow$ 128), LayerNorm, and $L_2$ sphere projection.
* **Implementation Status**: **VERIFIED & ACTIVE**.

### 1.3 Shared and Private Representations (Dual Heads)
* **What it does**: Encoders feature dual projection heads: a shared projection head $\mathbf{z}_s^{(m)}$ for cross-modal alignment and a private projection head $\mathbf{z}_p^{(m)}$ for modality-specific sensor dynamics.
* **Code Location**: [`src/models/encoders.py`](file:///d:/projects/NEURO_GRAPE/src/models/encoders.py) (lines 112, 178, 303), [`src/models/losses.py`](file:///d:/projects/NEURO_GRAPE/src/models/losses.py).
* **Losses**:
  * Orthogonal Separation Penalty:
    $$\mathcal{L}_{\text{sep}} = \frac{1}{M} \sum_{m} \left( \frac{\mathbf{z}_s^{(m)} \cdot \mathbf{z}_p^{(m)}}{\|\mathbf{z}_s^{(m)}\|_2 \|\mathbf{z}_p^{(m)}\|_2} \right)^2$$
  * Sensor Reconstruction Decoder Loss:
    $$\mathcal{L}_{\text{recon}} = \frac{1}{M} \sum_{m} \|\text{Decoder}_m(\mathbf{z}_p^{(m)}) - \mathbf{y}_{\text{target}}^{(m)}\|_2^2$$
* **Implementation Status**: **VERIFIED & ACTIVE IN TRAINING**. (Note: Test retrieval evaluates shared heads).

### 1.4 Cross-Modal Graph Attention Layer
* **What it does**: Differentiable multi-head attention ($h=4$, $d=128$) across modality tokens for a given visual stimulus concept.
* **Code Location**: [`src/models/graph.py`](file:///d:/projects/NEURO_GRAPE/src/models/graph.py).
* **Formulation**:
  $$\mathbf{H}_{\text{nodes}} = [\mathbf{z}_s^{\text{EEG}}, \mathbf{z}_s^{\text{MEG}}, \mathbf{z}_s^{\text{fMRI}}] \in \mathbb{R}^{B \times 3 \times 128}$$
  $$\mathbf{U} = \text{LayerNorm}(\mathbf{H}_{\text{nodes}} + \text{MHA}(\mathbf{H}_{\text{nodes}}, \mathbf{H}_{\text{nodes}}, \mathbf{H}_{\text{nodes}}))$$
* **Implementation Status**: **VERIFIED IN TRAINING**. (Evaluation must avoid joint test-batch target mixing).

### 1.5 Group Distributionally Robust Optimization (GroupDRO)
* **What it does**: Tracks per-group contrastive losses and dynamically shifts training loss weights toward the worst-performing subgroup via exponentiated gradient updates with EMA smoothing.
* **Code Location**: [`src/dro/group_dro.py`](file:///d:/projects/NEURO_GRAPE/src/dro/group_dro.py).
* **Formulation**:
  $$\mathcal{L}_{\text{DRO}} = \sum_{g=1}^G q_g \bar{\ell}_g, \quad q_g \leftarrow \frac{q_g \exp(\eta_q \bar{\ell}_g)}{\sum_j q_j \exp(\eta_q \bar{\ell}_j)}$$
* **Implementation Status**: **VERIFIED & ACTIVE**.

### 1.6 Adaptive Data Mixture Sampler
* **What it does**: Modality-level sampling reweighting based on relative training loss trajectory, enforcing an allocation floor ($\alpha_m \ge 0.15$).
* **Code Location**: [`src/data/adaptive_sampler.py`](file:///d:/projects/NEURO_GRAPE/src/data/adaptive_sampler.py).
* **Implementation Status**: **VERIFIED & ACTIVE**.

### 1.7 Neural Regime Discovery
* **What it does**: Partitions training embeddings into $K=2$ unsupervised clusters using KMeans fitted strictly on training latent representations, or uses semantic metadata (Animate vs Inanimate).
* **Code Location**: [`src/dro/clustering.py`](file:///d:/projects/NEURO_GRAPE/src/dro/clustering.py).
* **Implementation Status**: **VERIFIED & ACTIVE**.

---

## 2. Inventory of Proposed vs. Implemented Distinctions

To ensure academic integrity, the manuscript must strictly maintain these distinctions:

| Architectural Concept | Implemented Status | Paper Presentation Guidance |
| :--- | :---: | :--- |
| **Masked Autoencoder Tokenization (LaBraM style)** | **Not Implemented** | Present as conceptual literature inspiration in Related Work; do NOT claim our encoders tokenize neural spectrum patches with VQ-VAE. |
| **Unified Sensor-Agnostic Tokenization (BrainOmni style)** | **Not Implemented** | Present as related foundation model work; Neuro-GRAPE uses dedicated heterogeneous encoders (1D CNN for EEG/MEG, 3D CNN for fMRI). |
| **Bi-level Proxy Model Optimization (DoReMi / DoGE style)** | **Not Implemented** | Present as algorithmic inspiration; Neuro-GRAPE adapts mixture weights online via running loss statistics, not an external proxy model. |
| **Minimax Task Re-weighting (GRAPE style)** | **Adapted for Neural Modalities** | Clarify that GRAPE addressed multi-task NLP pretraining; Neuro-GRAPE adapts these principles to multimodal neural representation learning and group robustness. |
| **Foundation Model Scale** | **Pilot Prototype (12 train stimuli)** | Describe the current system as a *prototype multimodal representation learning framework* rather than claiming a fully pretrained planetary foundation model. |
