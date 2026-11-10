# Neuro-GRAPE Member 3: Phase 1 Implementation & Evaluation Report

**Project:** Neuro-GRAPE (Multimodal Neural Representation & Decoding)  
**Author:** Member 3 (Evaluation, Regime Clustering, GroupDRO & Generalization Benchmarking)  
**Date:** October 2026  
**Status:** **Completed, Tested, and Verified Against Real Pilot Embeddings**

---

## 1. Executive Summary

In Phase 1, Member 3 was responsible for implementing the downstream distributionally robust optimization (GroupDRO), unsupervised regime clustering, multi-split evaluation framework, and reproducible pipeline integration.

All components were built as modular Python units, verified through automated tests, and evaluated on the 20-stimulus multimodal pilot dataset using Member 2's pre-computed 128-D latent embeddings across 60 samples (EEG, MEG, and fMRI).

---

## 2. Deliverables & Implementation Overview

| Module | File Location | Core Functionality |
| :--- | :--- | :--- |
| **GroupDRO** | [`src/dro/group_dro.py`](file:///d:/hciproj/NEURO_GRAPE/src/dro/group_dro.py) | Numerically stable log-space exponentiated weight updates, group loss computation, safe normalization, empty/undersized group guards. |
| **Regime Clustering** | [`src/dro/clustering.py`](file:///d:/hciproj/NEURO_GRAPE/src/dro/clustering.py) | Unsupervised K-Means clustering fitted strictly on training data; deterministic `sample_id` joins; exploratory data regime disclaimers. |
| **Evaluation Framework** | [`src/dro/evaluation.py`](file:///d:/hciproj/NEURO_GRAPE/src/dro/evaluation.py) | Group-aware evaluator computing overall metrics, per-group performance, worst-group identification, and disparity gaps for loss and performance metrics. |
| **Pipeline Runner** | [`src/dro/pipeline.py`](file:///d:/hciproj/NEURO_GRAPE/src/dro/pipeline.py) | Orchestrator tying together embedding loading, clustering, GroupDRO simulation, multi-split evaluation, and artifact generation. |
| **Declarative Config** | [`configs/groupdro_config.yaml`](file:///d:/hciproj/NEURO_GRAPE/configs/groupdro_config.yaml) | Centralized YAML configuration decoupling hyperparameters, paths, and clustering settings from code. |
| **CLI Runner** | [`scripts/run_member3_pipeline.py`](file:///d:/hciproj/NEURO_GRAPE/scripts/run_member3_pipeline.py) | Command-line interface for reproducible runs. |
| **Automated Test Suite**| [`tests/test_group_dro.py`](file:///d:/hciproj/NEURO_GRAPE/tests/test_group_dro.py) | 9 focused automated unit/integration tests verifying all mathematical and integration invariants. |

---

## 3. Mathematical & Algorithmic Design

### 3.1 GroupDRO Formulations
Group weights $q \in \Delta_K$ are updated via exponentiated gradient ascent over group losses $L_k$:
$$s_k^{(t+1)} = s_k^{(t)} + \eta \cdot L_k$$
$$q^{(t+1)} = \text{softmax}(s)$$

- **Numerical Stability**: Updates are computed in log-space ($s_k = \log q_k$) via PyTorch's `log_softmax` / `softmax`, eliminating floating-point overflow under arbitrary loss magnitudes.
- **Empty / Inactive Group Safety**: Groups with $N_k < \text{min\_group\_size}$ preserve their unnormalized weights and are excluded from batch updates without causing `NaN` or zero-division errors.
- **Smoothing & Clipping**: Optional label smoothing $q \leftarrow (1-\alpha)q + \frac{\alpha}{K}$ and weight floors are supported.

### 3.2 Unsupervised Regime Clustering
- **Zero-Leakage Invariant**: K-Means ($K=2$, seed 42) is fitted **strictly on the 36 training samples**. Validation (12) and test (12) samples are assigned via the fitted model; their representations are never observed during centroid optimization.
- **Stable-ID Joins**: Cluster assignments are joined against metadata strictly using globally unique `sample_id` keys, never CSV row order.
- **Diagnostics**: Achieved training inertia of `14.207` and training silhouette score of `0.478`. Discovered that MEG and fMRI embeddings cluster together (Cluster 0), while EEG embeddings form Cluster 1.

### 3.3 Metric Direction & Disparity Gaps
The evaluator strictly enforces metric directionality:
- **Loss (`higher_is_better=False`)**:
  $$\text{Worst Group} = \arg\max_k L_k, \quad \text{Gap} = \max_k L_k - L_{\text{overall}} \ge 0$$
- **Accuracy / Similarity (`higher_is_better=True`)**:
  $$\text{Worst Group} = \arg\min_k S_k, \quad \text{Gap} = S_{\text{overall}} - \min_k S_k \ge 0$$
- Missing/unobserved groups are excluded from worst-group selection to prevent fabricated 0.0 scores from distorting conclusions.

---

## 4. Empirical Results on the 20-Stimulus Pilot Dataset

The pipeline was executed end-to-end on Member 2's exported 128-D embeddings across the 60 multimodal pilot samples:

### 4.1 Multi-Split Evaluation Summary (`metrics.csv`)

| Split | Metric | Direction | Overall | Group 0 (MEG/fMRI) | Group 1 (EEG) | Worst Group | Disparity Gap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** ($N=36$) | Contrastive Loss | Lower is better | **2.2884** | 2.3248 ($n=24$) | 2.2157 ($n=12$) | **Group 0** | **0.03637** |
| **Train** ($N=36$) | Cosine Similarity | Higher is better | **+0.00691**| +0.01540 ($n=24$)| -0.01009 ($n=12$)| **Group 1** | **0.01699** |
| **Val** ($N=12$) | Contrastive Loss | Lower is better | **1.38379** | 1.38297 ($n=8$) | 1.38542 ($n=4$) | **Group 1** | **0.00163** |
| **Val** ($N=12$) | Cosine Similarity | Higher is better | **-0.01280**| -0.00026 ($n=8$)| -0.03789 ($n=4$)| **Group 1** | **0.02509** |
| **Test** ($N=12$) | Contrastive Loss | Lower is better | **1.44230** | 1.44705 ($n=8$) | 1.43279 ($n=4$) | **Group 0** | **0.00475** |
| **Test** ($N=12$) | Cosine Similarity | Higher is better | **-0.00828**| +0.00677 ($n=8$)| -0.03837 ($n=4$)| **Group 1** | **0.03010** |

### 4.2 GroupDRO Weight Dynamics (`group_weights.csv`)
- **Initial Weights**: $q_0 = 0.5000$, $q_1 = 0.5000$.
- **Step 1 Update**: Group 0 exhibited higher mean loss ($2.3248$ vs $2.2157$), adjusting weights to $q_0 = 0.5014$, $q_1 = 0.4986$.
- **Final Weights (Step 10)**: $q_0$ smoothly rose to **$0.5129$**, while $q_1$ relaxed to **$0.4871$**, confirming numerically stable distributional adaptation.

---

## 5. Artifacts Generated

All outputs were exported to [`reports/artifacts/`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts):
1. [`metrics.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/metrics.csv): Complete multi-split metrics table.
2. [`group_losses.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_losses.csv): Step-wise tracking of group losses.
3. [`group_weights.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_weights.csv): Step-wise tracking of GroupDRO distribution weights.
4. [`group_assignments.parquet`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_assignments.parquet): Parquet assignment table mapping all 60 `sample_id` records to `cluster_id`.
5. [`group_assignments.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_assignments.csv): CSV companion copy of cluster assignments.
6. [`clustering_diagnostics.json`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/clustering_diagnostics.json): Inertia, silhouette score, and metadata.
7. [`experiment_config.yaml`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/experiment_config.yaml): Exact run hyperparameter snapshot.
8. [`pipeline_summary.json`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/pipeline_summary.json): Overall pipeline execution metadata.

---

## 6. Test Verification Summary

- **Member 3 Test Suite** (`tests/test_group_dro.py`): **9/9 tests passed** (100%).
- **Member 2 Handoff Interface Suite** (`tests/test_handoff.py`): **5/5 tests passed** (100%).
- **Combined Non-Diagnostic Regression Suite**: **49/49 tests passed** (100%).

---

## 7. Known Blockers & Limitations

1. **Member 1 Raw Preprocessed Arrays (`data/processed/*.npy`)**:
   - The `.npy` epoch arrays are gitignored and omitted from this repository clone. While `tests/test_dataset.py` fails on loading raw `.npy` files from disk, Member 2's pre-computed latent embeddings (`embeddings/`) are 100% complete and verified, allowing Member 3's pipeline to run end-to-end.
2. **Pilot Cohort Size**:
   - The 20 shared stimuli serve to verify interfaces, zero-leakage mechanics, and tensor flow. They do not constitute a statistically powered machine learning benchmark for claiming out-of-distribution robustness.
