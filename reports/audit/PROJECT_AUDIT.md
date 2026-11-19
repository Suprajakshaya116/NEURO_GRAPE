# Neuro-GRAPE: Comprehensive Project & System Audit

**Date:** November 2026  
**Auditor:** Senior Multimodal Machine Learning Researcher & Neuro-AI Systems Auditor  
**Project:** Neuro-GRAPE (Cross-Modality Foundation Model with Group-Robust Adaptive Pretraining for Neural Data)  
**Status:** Audit Completed — Major Structural & Optimization Bottlenecks Diagnosed  

---

## 1. Executive Summary

A comprehensive, ground-truth audit of the Neuro-GRAPE codebase was conducted. The audit inspected data loading, feature pipelines, encoder architectures, loss functions, GroupDRO mechanics, the cross-modal attention graph, training/validation loops, ablation flags, and evaluation routines.

### Key Finding:
The preliminary results (overall accuracy hovering at ~22–27%, near the 25% chance level of a 4-class retrieval task) **are not caused by bad data or broken code**, but rather by a combination of:
1. **Severe Under-Optimization & Premature Early Stopping**: The training loop runs only 25–50 epochs with a single batch ($B=12$) at $\text{lr}=10^{-4}$. More critically, the validation checkpoint selection restored **Epoch 1** (loss $\approx \ln(4) \approx 1.386$), causing the evaluated model to be functionally an untrained random-initialization network.
2. **Positive-Cone Representation Collapse**: The 1D and 3D backbones terminate in positive activations (`GELU`) followed by `AdaptiveAvgPool`, producing pooled vectors that lie in a narrow positive cone (mean off-diagonal cosine similarity $> +0.997$).
3. **Artifact of Constant Predicition**: The model collapsed to predicting a single dominant target index across all queries, yielding an apparent 25% accuracy ($1/4$) and causing extreme disparity gaps (0% vs 50%) under semantic subgroup evaluation.
4. **Inactive Ablation Pathways**: Dual-head separation and sensor reconstruction losses contribute to auxiliary objectives during training, but during evaluation only shared representations are tested, explaining why ablating them produced near-identical test metrics.

---

## 2. Project Architecture & End-to-End Data Flow

```text
+---------------------------------------------------------------------------------------+
|                                    INPUT DATA                                         |
|  EEG: sub-01, RSVP (63 channels, 225 timepoints @ 250 Hz, [-100, 800] ms)             |
|  MEG: sub-BIGMEG2, Ses-07 (272 axial gradiometers, 225 timepoints @ 250 Hz)           |
|  fMRI: sub-01, Ses-things08 (Single-trial LS-S GLM beta volume, 96x96x60 voxels)      |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                            LEAKAGE-FREE NORMALIZATION                                 |
|  TrimodalNormalizer: Channel-wise Z-score (EEG/MEG), Intracranial in-mask mean/std    |
|  (Statistics strictly computed on 12 training stimuli; val & test transformed)        |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                             MODALITY-SPECIFIC ENCODERS                                |
|  EEGEncoder: 1D CNN (stride-2 Conv1d x 2, GELU, BatchNorm1d, AdaptiveAvgPool1d)       |
|  MEGEncoder: 1D CNN (stride-2 Conv1d x 2, GELU, BatchNorm1d, AdaptiveAvgPool1d)       |
|  FMRIEncoder: 3D CNN (stride-2 Conv3d x 3, GELU, BatchNorm3d, AdaptiveAvgPool3d)      |
|  Shared Projection: Linear(C -> 128) + LayerNorm(128) + F.normalize(p=2)              |
|  Private Projection: Linear(C -> 128) + LayerNorm(128) + F.normalize(p=2)            |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                          CROSS-MODAL GRAPH ATTENTION LAYER                            |
|  Multi-head attention across active modality tokens (n_heads=4, embed_dim=128)        |
+---------------------------------------------------------------------------------------+
                                           │
                                           ▼
+---------------------------------------------------------------------------------------+
|                                 UNIFIED LOSS OBJECTIVE                                |
|  L_total = (1 - lambda_robust)*L_align + lambda_robust*L_DRO                          |
|          + lambda_private*L_sep + lambda_recon*L_recon                                |
+---------------------------------------------------------------------------------------+
```

---

## 3. Dataset Sizes, Classes, and Groups

- **Total Pilot Records**: 60 samples across 3 modalities (20 EEG, 20 MEG, 20 fMRI).
- **Unique Stimuli**: 20 distinct visual concepts from the THINGS dataset.
- **Split Allocation** (Stimulus-disjoint):
  - **Train**: 12 stimuli (36 records)
  - **Validation**: 4 stimuli (12 records: `pinwheel_02s`, `puck_02s`, `screwdriver_02s`, `sprinkler_02n`)
  - **Test**: 4 stimuli (12 records: `cucumber_02s`, `lampshade_02s`, `pea_02s`, `tarp_02s`)
- **Groups**:
  - **Semantic**: Animate (living things: `bear`, `boa`, `tadpole`, `woman`) vs Inanimate (all tools, food, objects).
  - **Learned Cluster Regimes**: Unsupervised $K=2$ KMeans clustering fitted strictly on training embedding representations.

---

## 4. Input Tensor Shapes & Parameter Counts

| Modality | Input Tensor Shape | Input Type | Model Parameters | Output Dimension | Normalization |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EEG** | `(B, 63, 225)` | float32 (Volts) | 82,656 | `(B, 128)` | Unit $L_2$ Sphere |
| **MEG** | `(B, 272, 225)` | float32 (Tesla) | 129,472 | `(B, 128)` | Unit $L_2$ Sphere |
| **fMRI** | `(B, 1, 96, 96, 60)` | float32 (Beta weight) | 86,800 | `(B, 128)` | Unit $L_2$ Sphere |
| **Graph Layer** | Tokens: `[B, 3, 128]` | float32 | 66,048 | `(B, 128)` each | Residual Add + Norm |
| **Decoders** | Latent `(B, 128)` | float32 | 50,699 | Sensor Targets | MSE Loss |
| **Total Model** | — | — | **415,675** | — | — |

---

## 5. Loss Terms & Formulations

The complete optimization objective in `src/models/losses.py` is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{align-effective}} + \lambda_{\text{private}} \mathcal{L}_{\text{sep}} + \lambda_{\text{recon}} \mathcal{L}_{\text{recon}}$$

Where:
1. **Trimodal Symmetric Contrastive Loss** ($\mathcal{L}_{\text{align}}$):
   $$\mathcal{L}_{\text{align}} = \frac{1}{3} \left( \mathcal{L}_{\text{EEG}\leftrightarrow\text{MEG}} + \mathcal{L}_{\text{EEG}\leftrightarrow\text{fMRI}} + \mathcal{L}_{\text{MEG}\leftrightarrow\text{fMRI}} \right)$$
   Evaluated with temperature $\tau = 0.07$.
2. **GroupDRO Loss** ($\mathcal{L}_{\text{DRO}}$):
   $$\mathcal{L}_{\text{DRO}} = \sum_{g=1}^G q_g \cdot \bar{\ell}_g, \quad q_g \leftarrow q_g \exp(\eta \bar{\ell}_g)$$
   Interpolated with empirical alignment loss:
   $$\mathcal{L}_{\text{align-effective}} = (1 - \lambda_{\text{robust}}) \lambda_{\text{align}} \mathcal{L}_{\text{align}} + \lambda_{\text{robust}} \mathcal{L}_{\text{DRO}}$$
3. **Orthogonal Separation Loss** ($\mathcal{L}_{\text{sep}}$):
   $$\mathcal{L}_{\text{sep}} = \frac{1}{M} \sum_{m} \left( \frac{\mathbf{z}_{s}^{(m)} \cdot \mathbf{z}_{p}^{(m)}}{\|\mathbf{z}_{s}^{(m)}\|_2 \|\mathbf{z}_{p}^{(m)}\|_2} \right)^2$$
4. **Sensor Reconstruction Loss** ($\mathcal{L}_{\text{recon}}$):
   $$\mathcal{L}_{\text{recon}} = \frac{1}{M} \sum_{m} \text{MSE}\left( \text{Decoder}_m(\mathbf{z}_p^{(m)}), \mathbf{y}_{\text{target}}^{(m)} \right)$$

---

## 6. Audit of Flaws, Bugs, and Vulnerabilities

### Bug 1: Early-Stopping Metric Trapped at Random Guessing (Critical)
- **Location**: `scripts/run_full_solution_benchmark.py`, line 408; `src/models/training.py`, line 379.
- **Cause**: Validation loss is measured on 4 stimuli. At Epoch 1, an untrained network produces near-uniform logits, yielding $\text{Loss} \approx \ln(4) = 1.386294$. As the network starts fitting the training set, validation loss immediately ticks upward to $1.387–1.400$.
- **Impact**: The early stopping criterion immediately flags Epoch 1 as the "best" model and restores Epoch 1 weights. Consequently, **the evaluated model was never trained**!

### Bug 2: Representation Collapse into a Positive Cone
- **Location**: `src/models/encoders.py`, `_build_1d_backbone` and `FMRIEncoder`.
- **Cause**: The backbone ends with `GELU()` followed by `AdaptiveAvgPool(1)`. All pooled values are non-negative.
- **Impact**: Pairwise cosine similarity among different stimuli is $> +0.997$. When projected and normalized, every sample points in virtually the same direction.

### Bug 3: Constant Argmax Prediction Artifact
- **Impact**: In cross-modal retrieval, because all query vectors and all gallery vectors point in the same direction, every query selects the same gallery sample (the one closest to the center of the cone). Top-1 accuracy is exactly 1 out of 4 (25.0%).

### Bug 4: Inactive Ablations in Retrieval Evaluation
- **Impact**: Ablations G1 (`No Dual Heads`), G2 (`No Separation Loss`), and G7 (`No Reconstruction`) only change private heads and decoders. But test retrieval only evaluates the shared representation $\mathbf{z}_s$. Hence, their test metrics were identical.

---

## 7. Prioritized Action Plan

1. **Fix Optimization & Early Stopping**: Track validation retrieval accuracy or use validation loss with appropriate regularization and patience, avoiding restoring Epoch 1.
2. **Increase Epochs & Learning Rate**: Train for 30–50 epochs at $\text{lr}=5\times 10^{-4}$ with cosine decay.
3. **Decouple Representations**: Add batch-level centering or contrastive spread to avoid positive cone collapse.
4. **Verify Scalability**: Ensure the ingestion pipeline automatically validates new data when the full dataset arrives.
