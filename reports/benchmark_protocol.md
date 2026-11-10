# Neuro-GRAPE: Comprehensive Benchmark Protocol & Evaluation Standards

**Author:** Member 3 (Downstream Evaluation, GroupDRO, & Architecture Audit)  
**Date:** October 2026  
**Reference Document:** `docs/Neuro-GRAPE.pdf`  
**Deliverable:** Benchmark Protocol Specification

---

## 1. Executive Protocol Overview

This benchmark protocol establishes the rigorous, reproducible evaluation matrix for the **Neuro-GRAPE** foundation learning framework. It defines the formal data partitions, metric definitions, experimental conditions, ablation controls, and statistical standards.

### 1.1 Dataset and Split Integrity
- **Dataset Cohort:** 20 verified visual stimulus concepts from the THINGS database across 3 neural recording modalities:
  - EEG: 63 sensor channels, 225 timepoints ($[-100, 800]$ ms, 250 Hz)
  - MEG: 272 axial gradiometers, 225 timepoints ($[-100, 800]$ ms, 250 Hz)
  - fMRI: Single trial-wise beta volume of shape $(96, 96, 60)$ voxels
- **Splits:**
  - **Train:** 12 stimuli (6 animate, 6 inanimate) = 36 modality records
  - **Validation:** 4 stimuli (2 animate, 2 inanimate) = 12 modality records (used strictly for model selection and checkpoint tuning)
  - **Test (Held-out):** 4 stimuli (2 animate, 2 inanimate) = 12 modality records (locked until final evaluation)
- **Data Leakage Safeguards:**
  - Normalization parameters (mean, std) fitted exclusively on training data.
  - Cluster centroids for regime discovery fitted exclusively on training data; validation and test representations are assigned using frozen centroids.

---

## 2. Quantitative Evaluation Metrics

All experimental conditions are assessed using the following standardized metrics:

1. **Top-1 Cross-Modal Retrieval Accuracy:**
   Percentage of test queries where the true stimulus counterpart has the highest cosine similarity in the target modality:
   $$\text{Top-1} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \arg\max_j \frac{z_{i, p}^\top z_{j, q}}{\|z_{i, p}\| \|z_{j, q}\|} = i \right)$$
   Evaluated across all 3 pairwise directions: EEG $\to$ MEG, EEG $\to$ fMRI, MEG $\to$ fMRI (and their inverses).

2. **Top-k Cross-Modal Retrieval Accuracy ($k=2$):**
   Percentage of queries where the correct stimulus appears within the top-2 nearest neighbors.

3. **Average Loss ($\bar{L}_{\text{test}}$):**
   Unweighted symmetric InfoNCE alignment loss on the test cohort.

4. **Worst-Group Loss ($\max_g L_g$) & Worst-Group Accuracy:**
   The loss and Top-1 accuracy evaluated on the hardest subgroup (either hardest semantic category or hardest learned regime).

5. **Group Disparity Gap ($\Delta_{\text{group}}$):**
   Difference between the best-performing and worst-performing subgroup:
   $$\Delta_{\text{group}} = \max_g \text{Acc}_g - \min_g \text{Acc}_g$$
   Lower values indicate superior group fairness and representational balance.

6. **Subspace Orthogonality Diagnostic ($S_{\text{ortho}}$):**
   Mean squared cosine similarity between shared and private heads:
   $$S_{\text{ortho}} = \frac{1}{M \cdot N} \sum_{m \in \mathcal{M}} \sum_{i=1}^N (z_{i, m, \text{shared}}^\top z_{i, m, \text{private}})^2$$
   Ideal target: $S_{\text{ortho}} \approx 0.0$.

7. **Expected Calibration Error (ECE):**
   Measures confidence calibration of retrieval similarity softmax probabilities against empirical Top-1 correctness across $M=5$ confidence bins:
   $$\text{ECE} = \sum_{b=1}^M \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$

8. **Compute & Resource Profile:**
   Parameter count, training wall-clock time (seconds), and peak memory consumption.

---

## 3. Benchmark Matrix: Named Experimental Conditions

| Condition Code | Experiment Name | Description | Key Configuration Flags |
| :--- | :--- | :--- | :--- |
| **A1** | **Single-Modality Baseline: EEG** | 1D-CNN autoencoder pretraining on EEG alone | `modality='eeg'`, unimodal |
| **A2** | **Single-Modality Baseline: MEG** | 1D-CNN autoencoder pretraining on MEG alone | `modality='meg'`, unimodal |
| **A3** | **Single-Modality Baseline: fMRI** | 3D-CNN autoencoder pretraining on fMRI alone | `modality='fmri'`, unimodal |
| **B** | **Fixed-Mixture Trimodal ERM** | Standard multimodal contrastive learning with equal weights, no DRO, no graph, single head | `dro=False`, `adaptive=False`, `graph=False`, `dual_head=False` |
| **C** | **Modality-Balanced Training** | Trimodal ERM with explicit modality balancing loss weighting | `dro=False`, `balanced=True` |
| **D** | **Standard GroupDRO (Semantic)** | GroupDRO optimizing worst-case semantic category (animate vs inanimate) | `group_type='semantic'`, `dro=True`, $\eta_q=0.01$ |
| **E** | **Learned-Regime GroupDRO** | GroupDRO optimizing worst-case learned neural cluster regime ($K=2$) | `group_type='cluster'`, `dro=True`, $\eta_q=0.01$ |
| **F** | **Full Neuro-GRAPE** | Complete framework: Dual heads + Orthogonality + Sensor Reconstruction + Cross-modal Graph + Adaptive Sampler + Learned-Regime GroupDRO + 3-Phase Curriculum | `dual_head=True`, `graph=True`, `adaptive=True`, `dro=True`, `curriculum=True` |
| **G1** | **Ablation: No Shared/Private Split** | Full framework with private head disabled (single shared head only) | `dual_head=False`, $\lambda_{\text{private}}=0, \lambda_{\text{recon}}=0$ |
| **G2** | **Ablation: No Separation Loss** | Dual heads active but orthogonality loss set to zero ($\lambda_{\text{private}} = 0$) | `dual_head=True`, $\lambda_{\text{private}}=0$ |
| **G3** | **Ablation: No Regime Discovery** | Full framework using random uniform groups instead of discovered clusters | `group_type='random'` |
| **G4** | **Ablation: No GroupDRO** | Full framework with GroupDRO disabled ($\lambda_{\text{robust}} = 0$, ERM optimization) | `dro=False`, $\lambda_{\text{robust}}=0$ |
| **G5** | **Ablation: No Adaptive Mixture** | Full framework with fixed static sampling mixture ($\alpha = [1/3, 1/3, 1/3]$) | `adaptive=False` |
| **G6** | **Ablation: No Cross-Modal Graph** | Full framework with graph attention bypassed (`enabled=False`) | `graph=False` |
| **G7** | **Ablation: No Reconstruction Loss** | Full framework with private sensor reconstruction disabled ($\lambda_{\text{recon}} = 0$) | `lambda_recon=0` |
| **H1** | **Transfer: Held-Out Concepts** | Zero-shot cross-modal retrieval on the 4 unseen test stimuli | Standard test partition |
| **H2** | **Transfer: Held-Out Subjects** | Generalization across different human subjects | **BLOCKED:** Pilot contains only 1 subject per modality (`sub-01`, `sub-BIGMEG2`) |
| **H3** | **Transfer: Pairwise Cross-Modal** | Retrieval evaluated across independent sensor pairings (EEG $\leftrightarrow$ MEG, etc.) | Evaluated for all runs |
| **H4** | **Transfer: Regime-Shift Disparity** | Evaluation of worst-regime performance on held-out test data | Evaluated for all runs |
| **H5** | **Calibration Benchmark** | Expected Calibration Error (ECE) evaluated across confidence bins | Evaluated for all runs |

---

## 4. Multi-Seed Replication Protocol

To ensure rigorous statistical validation and reproducibility:
- Every condition is executed across three fixed random seeds: `seed=42`, `seed=123`, `seed=999`.
- Mean and sample standard deviation ($\mu \pm \sigma$) are reported for all metrics.
- All experiment configurations, random seeds, training checkpoints, and per-epoch metrics are written to machine-readable JSON and CSV files in `reports/artifacts/full_benchmark/`.
