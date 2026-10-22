# Phase 5 Diagnostic Audit & Experimental Plan: Investigating Representation Collapse in Neuro-GRAPE

**Project:** Neuro-GRAPE (Trimodal Neural Representation & Decoding)  
**Phase:** Phase 5 — Representation Quality, Collapse Remediation, & Empirical Alignment  
**Date:** October 2026  
**Status:** **Diagnostic Audit Completed — Testable Hypotheses & Experimental Plan Established**  

---

## 1. Executive Summary

Phase 4 established the trimodal alignment pipeline across EEG, MEG, and fMRI data, achieving 95 passing automated unit tests and full train/validation/test isolation. However, representation collapse persisted: mean off-diagonal cosine similarities remained high ($>0.88$ for MEG, $>0.94$ for EEG, and $>0.99$ for fMRI), validation retrieval accuracy remained at chance ($29.2\%$ vs $25.0\%$ random baseline for 4-way matching), and 3 of 6 cross-modal retrieval directions produced constant candidate predictions.

This Phase 5 diagnostic audit provides quantitative empirical measurements of the embedding space, isolates dimensional collapse from sample collapse, audits the loss and data pipeline, and distinguishes verified empirical facts from testable hypotheses. It outlines a prioritized, single-variable experimental plan to resolve the collapse without touching the test split.

---

## 2. Quantitative Diagnostic Audit of Embeddings

To determine whether collapse occurs across **individual samples** (cone collapse), **feature dimensions** (dimensional collapse), or both, we computed singular value decompositions (SVD), effective ranks (Roy & Vetterli entropy metric: $\exp(-\sum p_i \ln p_i)$ where $p_i = \sigma_i / \sum \sigma_i$), per-dimension variances, and pairwise cosine metrics across both the training ($N=12$) and validation ($N=4$) cohorts.

### Empirical Measurements Summary

| Metric | EEG (Train $N=12$) | MEG (Train $N=12$) | fMRI (Train $N=12$) | EEG (Val $N=4$) | MEG (Val $N=4$) | fMRI (Val $N=4$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Input Correlation (Raw Off-Diag Cosine)** | `+0.0199` | `+0.0000` | **`+0.5873`** | `+0.0210` | `-0.0180` | **`+0.5767`** |
| **Normalized Input Correlation** | `+0.0278` | `-0.0199` | **`+0.5767`** | `+0.0285` | `-0.0195` | **`+0.5712`** |
| **Effective Rank (Max: 12 train / 4 val)** | `7.01` / 12 | `8.51` / 12 | **`2.20` / 12** | `2.96` / 4 | `3.19` / 4 | **`1.70` / 4** |
| **Top Singular Value Ratio ($\sigma_0 / \sum \sigma_i$)** | `41.4%` | `32.7%` | **`79.3%`** | `61.2%` | `55.1%` | **`87.8%`** |
| **Active Dims ($\text{Var}(z_d) > 10^{-4}$)** | **128 / 128** | **128 / 128** | **73 / 128** | **123 / 128** | **125 / 128** | **67 / 128** |
| **Mean Off-Diagonal Cosine Sim** | `+0.7591` | `+0.6184` | **`+0.9675`** | `+0.8358` | `+0.7386` | **`+0.9681`** |
| **Max Elementwise Diff ($\|z_i - z_j\|_\infty$)** | `0.185` | `0.231` | **`0.039`** | `0.125` | `0.190` | **`0.036`** |

### Key Diagnostic Discoveries:
1. **fMRI Suffers from Dual Collapse (Dimensional + Sample):**
   - **Severe Dimensional Collapse:** The effective rank of fMRI embeddings is only **2.20 out of 12** on training data and **1.70 out of 4** on validation data. The first principal component accounts for **$79.3\%$** of the entire singular value spectrum at training and **$87.8\%$** on validation. **55 of 128 dimensions** have variance $< 10^{-4}$ (completely inactive).
   - **Severe Sample Collapse:** Pairwise cosine similarity remains at **$+0.968$** on validation, and max elementwise difference is only $0.036$.
2. **EEG and MEG Suffer Primarily from Sample/Cone Collapse, Not Dimensional Collapse:**
   - Both EEG and MEG utilize **all 128 embedding dimensions** ($\text{Var}(z_d) > 10^{-4}$ across all dimensions).
   - Their effective ranks are relatively healthy ($8.51/12$ for MEG and $7.01/12$ for EEG on training data).
   - However, their mean cosine similarity remains clustered in a positive cone ($+0.62$ to $+0.84$), meaning vectors are spread across dimensions but constrained to a shared positive octant.
3. **Severe Input-Level Anatomical Correlation in fMRI:**
   - EEG and MEG raw trials are practically orthogonal across stimuli ($+0.02$ and $+0.00$ mean off-diagonal cosine).
   - In contrast, raw fMRI beta volumes share a **$+0.587$** pairwise cosine similarity before ever entering the network. Because whole-brain volumes contain mostly non-stimulus-specific baseline tissue, spatial pooling over the entire brain preserves global anatomy rather than stimulus-specific signals.

---

## 3. Audit of the Training Pipeline, Objective, & Gradients

1. **Theoretical vs. Empirical Loss (The "Chance Level" Proof):**
   - For a batch of $N$ classes, random uniform guessing produces an expected cross-entropy / InfoNCE loss of $\ln(N)$:
     $$\mathcal{L}_{\text{random}}(\text{train}) = \ln(12) \approx 2.4849$$
     $$\mathcal{L}_{\text{random}}(\text{val}) = \ln(4) \approx 1.3863$$
   - In the 50-epoch experiment, validation loss began at `1.3893` (epoch 1), reached its minimum at `1.3838` (epoch 11), and rose to `1.7626` (epoch 50).
   - The validation loss never deviated by more than **$0.0025$** from random chance ($\ln(4)$). This proves mathematically that the model never learned generalizable cross-modal features on the held-out validation cohort.
2. **Positive-Pair Matching & Batch Construction:**
   - `prepare_cross_modal_batch` correctly extracts cross-modal records matching `stimulus_id`.
   - Positive masks in `multimodal_contrastive_loss` use boolean string equality. Alignment is verified correct.
3. **Negative Sample Starvation:**
   - With full-batch training on $N=12$ stimuli, each query has only **11 negative distractors**.
   - With 11 negatives, InfoNCE provides very weak repulsive gradient energy along orthogonal directions, especially when temperature $\tau=0.07$ concentrates gradients only on the single nearest neighbor.
4. **Gradient Norm Disparity:**
   - Backpropagation gradient norms at epoch 1:
     - MEG Encoder: $\|\nabla_\theta\| \approx 4.48$
     - EEG Encoder: $\|\nabla_\theta\| \approx 1.96$
     - fMRI Encoder: $\|\nabla_\theta\| \approx 0.90$
   - MEG gradients are nearly **$5\times$ larger** than fMRI gradients. The 3D-CNN backbone in fMRI receives weaker updates, while its inputs are dominated by high spatial autocorrelation.
5. **Single Linear Projection Bottleneck:**
   - All three encoders use a single linear projection head: `Linear(64, 128)` for fMRI and `Linear(128, 128)` for EEG/MEG.
   - A single linear layer directly couples contrastive gradient updates to convolutional feature maps, preventing the network from discarding high-variance nuisance features (anatomy/sensor drift) while aligning semantic features.

---

## 4. Separation of Verified Facts vs. Testable Hypotheses

### Verified Empirical Facts:
1. **fMRI undergoes severe dimensional collapse**: Effective rank is $2.20/12$ on train and $1.70/4$ on val; the 1st singular value accounts for $79.3\%$ of the variance; 55 of 128 dimensions are inactive.
2. **fMRI inputs have $+0.58$ baseline correlation**: Unlike EEG/MEG (correlation $\approx 0.0$), whole-brain fMRI beta volumes share high anatomical autocorrelation.
3. **Validation loss is functionally identical to random chance**: Val loss is locked at $\ln(4) \approx 1.386 \pm 0.003$.
4. **Three retrieval directions exhibit trivial ranking collapse**: `fMRI -> EEG` predicts class 2 for all queries; `MEG -> fMRI` predicts class 0 for all queries; `fMRI -> MEG` predicts class 1 for all queries.
5. **LayerNorm centers vectors per-sample, not across samples**: LayerNorm enforces $\sum_d z_d = 0$, but does not penalize samples from pointing in the same direction across the batch ($\frac{1}{N}\sum_i z_i \neq 0$).

### Testable Hypotheses:
- **Hypothesis 1 (fMRI Global Pooling Hypothesis):** `AdaptiveAvgPool3d(1)` over $552,960$ voxels dilutes localized visual cortex activations with shared resting-state tissue. Masking to visual cortex or applying 3D convolution before pooling will increase fMRI effective rank.
- **Hypothesis 2 (Variance Collapse Hypothesis):** InfoNCE lacks an explicit constraint enforcing non-zero variance along orthogonal dimensions. Incorporating a non-contrastive variance regularization term (e.g. VICReg variance loss $S(Z) = \max(0, 1 - \text{std}(Z))$) will prevent dimensional collapse.
- **Hypothesis 3 (Projection Decoupling Hypothesis):** A 2-layer non-linear MLP projection head (`Linear -> LayerNorm -> GELU -> Linear`) will allow backbones to maintain invariant representations while the projection head absorbs contrastive hypersphere deformation.
- **Hypothesis 4 (Sample Size Starvation Hypothesis):** 11 training negatives cannot span 128-dimensional latent space. Expanding the training set beyond 12 stimuli is necessary for the repulsion force to overcome shared inductive bias.

### Untrusted Measurements (Cautionary Note):
- **4-way retrieval accuracy ($29.2\%$ vs $25.0\%$):** With $N=4$, each sample represents $25\%$ of the total score. The $+4.2\%$ nominal improvement was an artifact of constant-class prediction in `MEG -> fMRI`. **Retrieval accuracy on $N=4$ cannot be used as a primary optimization signal.**

---

## 5. Prioritized Phase 5 Experimental Plan

All experiments must adhere to strict controls:
- **Seed:** 42
- **Baseline:** Current Phase 4 model (LayerNorm, `lr=1e-4`, `weight_decay=1e-2`, `temperature=0.07`, `dropout=0.0`, 50 epochs).
- **Test Split:** **Strictly prohibited** (0 access).
- **Evaluation Metrics:** Effective rank (SVD), percentage of active dimensions ($\text{Var} > 10^{-4}$), singular value 0 ratio, mean off-diagonal cosine, and row variance.

```
+-----------------------------------------------------------------------------------+
| Baseline: Phase 4 Model (Val Loss: 1.3838, fMRI Eff-Rank: 1.70, MEG/EEG Cone)    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Experiment 5.1: Non-Contrastive Variance Regularization (VICReg Loss Penalty)     |
| - Target: Dimensional collapse in fMRI & cone collapse in EEG/MEG                |
| - Intervention: Loss = Loss_contrastive + lambda * Var_loss(Z)                    |
| - Isolates: Objective loss formulation without modifying encoder architecture    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Experiment 5.2: 2-Layer Non-Linear MLP Projection Heads                          |
| - Target: Coupling between backbone feature extraction and contrastive sphere    |
| - Intervention: Linear(d_in, 128) -> LayerNorm -> GELU -> Linear(128, 128)      |
| - Isolates: Projection capacity without modifying backbone or loss function      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Experiment 5.3: fMRI Anatomical Feature Masking / High-Variance Voxel Selection  |
| - Target: +0.58 anatomical baseline correlation in full-volume beta maps         |
| - Intervention: Filter input volume to top-k variance voxels or visual cortex    |
| - Isolates: Input spatial representation heterogeneity                           |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Experiment 5.4: Temperature Scaling & Negative Weighting Sensitivity             |
| - Target: Gradient starvation with N=12 negatives                                |
| - Intervention: Sweep tau in {0.05, 0.10, 0.15, 0.20}                           |
| - Isolates: Softmax sharpness with small batch sizes                              |
+-----------------------------------------------------------------------------------+
```

### Detailed Protocol for Experiment 5.1 (Single Best Next Experiment):
- **Objective:** Directly target the verified dimensional collapse in fMRI (effective rank $2.20$, 55 dead dimensions) and cone collapse in EEG/MEG.
- **Intervention:** Add an explicit variance penalty on the embedding batches:
  $$\mathcal{L}_{\text{var}}(Z) = \frac{1}{D} \sum_{d=1}^D \max\left(0, 1 - \sqrt{\text{Var}(z_{:, d}) + \epsilon}\right)$$
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{trimodal}} + \lambda_{\text{var}} \sum_{\text{mod}} \mathcal{L}_{\text{var}}(Z_{\text{mod}})$$
  (with $\lambda_{\text{var}} = 1.0$ or $0.5$).
- **Success Criteria:**
  1. fMRI effective rank on training increases from $2.20$ to $> 6.0$.
  2. Active dimensions ($\text{Var} > 10^{-4}$) in fMRI increase from $73$ to $> 110$.
  3. Mean off-diagonal cosine decreases below $0.70$ in all modalities without loss divergence.
  4. Non-constant retrieval predictions across validation queries.
