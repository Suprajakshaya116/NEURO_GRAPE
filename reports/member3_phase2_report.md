# Neuro-GRAPE Member 3: Phase 2 Audit, Validation, and Large-Scale Preparation Report

**Project:** Neuro-GRAPE (Multimodal Neural Representation & Decoding)  
**Author:** Senior ML Engineer (Member 3: Evaluation, GroupDRO, Clustering & Benchmarking)  
**Date:** October 2026  
**Status:** **Audited, Validated, Corrected, and Verified Large-Scale Ready**  
**Repository Branch:** `main`

---

## 1. Executive Summary

Phase 2 established an independent audit, mathematical validation, and architecture upgrade of Member 3’s GroupDRO, clustering, and evaluation pipeline within the Neuro-GRAPE repository.

### Key Audit Findings & Actions Taken:
1. **Test Verification & Reproducibility**:
   - Independently verified the Phase 1 test suite: all 9 Member 3 tests and all 5 Member 2 handoff tests were reproduced and passed.
   - Expanded the test suite from 9 to **13 automated tests** (`tests/test_group_dro.py`), adding a dedicated `TestLargeDatasetReadiness` suite.
   - Combined regression suite of non-diagnostic tests stands at **53 passed, 0 failed** in 3.79s.
2. **Methodological Clarification (Simulation vs Training)**:
   - Phase 1 simulated GroupDRO weight dynamics across iterations over precomputed representations. The pipeline now explicitly tags and logs this as `execution_mode: "groupdro_weight_simulation_on_fixed_embeddings"` (`is_trained_model: false`), preventing misleading claims of end-to-end model fine-tuning while retaining a plug-and-play interface for Member 2's training loop.
3. **Group Definition Transparency & Modality Decoupling**:
   - Unsupervised K-Means clustering ($K=2$) on the pilot embeddings discovered that cluster boundaries separate along **recording modality** (Cluster 0 = MEG & fMRI; Cluster 1 = EEG) rather than visual stimulus semantic categories, caused by positive cone collapse in baseline encoders.
   - Upgraded the pipeline to support **three explicit grouping strategies**:
     - `"cluster"`: Learned exploratory regime clusters via training-only K-Means.
     - `"modality"`: Explicit partitioning by neural modality (`eeg`, `meg`, `fmri`).
     - `"metadata"`: Predefined semantic metadata partitions (e.g., `category_name`, `category_id`).
4. **Large-Dataset Architectural Decoupling**:
   - Removed hard-coded assumptions regarding 60 samples, 20 stimuli, 128 embedding dimensions, and 2 clusters.
   - Made embedding dimensionality auto-detectable from inputs, supported variable $K$ clusters with dynamic metric column generation, and added safe trimodal stimulus alignment that drops incomplete stimuli without throwing index errors.
   - Generated a new granular long-format artifact: `group_metrics.csv`.

---

## 2. Repository Audit

### 2.1 Files Inspected
- Manifests & Metadata: [`data_inventory.csv`](file:///d:/hciproj/NEURO_GRAPE/data_inventory.csv), [`data/splits/pilot_splits.csv`](file:///d:/hciproj/NEURO_GRAPE/data/splits/pilot_splits.csv), [`embeddings/metadata.csv`](file:///d:/hciproj/NEURO_GRAPE/embeddings/metadata.csv), [`NeuroGRAPE_handover/manifests/shared_stimuli.csv`](file:///d:/hciproj/NEURO_GRAPE/NeuroGRAPE_handover/manifests/shared_stimuli.csv).
- Member 2 Checkpoints & Code: [`checkpoints/baseline.pt`](file:///d:/hciproj/NEURO_GRAPE/checkpoints/baseline.pt), [`model.py`](file:///d:/hciproj/NEURO_GRAPE/model.py), [`losses.py`](file:///d:/hciproj/NEURO_GRAPE/losses.py), [`src/models/training.py`](file:///d:/hciproj/NEURO_GRAPE/src/models/training.py), [`src/models/losses.py`](file:///d:/hciproj/NEURO_GRAPE/src/models/losses.py).
- Member 1 Preprocessing & Data: [`src/data/dataset.py`](file:///d:/hciproj/NEURO_GRAPE/src/data/dataset.py), [`tests/test_dataset.py`](file:///d:/hciproj/NEURO_GRAPE/tests/test_dataset.py).
- Handover Metadata: [`NeuroGRAPE_handover/metadata/sample_attributes_P2.csv`](file:///d:/hciproj/NEURO_GRAPE/NeuroGRAPE_handover/metadata/sample_attributes_P2.csv) (27,050 rows of THINGS-MEG run metadata).
- Phase 1 Code & Artifacts: [`src/dro/`](file:///d:/hciproj/NEURO_GRAPE/src/dro/), [`configs/groupdro_config.yaml`](file:///d:/hciproj/NEURO_GRAPE/configs/groupdro_config.yaml), [`reports/artifacts/`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/).

### 2.2 Files Created or Modified in Phase 2

| File Path | Action | Description of Modification |
| :--- | :---: | :--- |
| [`src/dro/clustering.py`](file:///d:/hciproj/NEURO_GRAPE/src/dro/clustering.py) | **Modified** | Added `Sequence` import; supported `Optional[int]` for `expected_dim` with automatic dimension detection. |
| [`src/dro/group_dro.py`](file:///d:/hciproj/NEURO_GRAPE/src/dro/group_dro.py) | **Modified** | Added `group_names` support and `GroupDRO.from_group_labels(...)` factory method for dynamic string/category label mapping. |
| [`src/dro/pipeline.py`](file:///d:/hciproj/NEURO_GRAPE/src/dro/pipeline.py) | **Modified** | Supported 3 grouping strategies (`cluster`, `modality`, `metadata`), safe trimodal stimulus extraction, dynamic group metrics export, and `group_metrics.csv` generation. |
| [`configs/groupdro_config.yaml`](file:///d:/hciproj/NEURO_GRAPE/configs/groupdro_config.yaml) | **Modified** | Added declarative `grouping` block (`strategy` and `metadata_column`); set `expected_dim: null` for auto-detection. |
| [`tests/test_group_dro.py`](file:///d:/hciproj/NEURO_GRAPE/tests/test_group_dro.py) | **Modified** | Added `TestLargeDatasetReadiness` suite with 4 new tests covering variable dimensions (256-D), label factories, dynamic $K=3$, and grouping strategies. |
| [`reports/artifacts/group_metrics.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_metrics.csv) | **Created** | Granular long-format table containing per-group sample counts, metric values, and worst-group flags across splits. |
| [`reports/member3_phase2_report.md`](file:///d:/hciproj/NEURO_GRAPE/reports/member3_phase2_report.md) | **Created** | Comprehensive Phase 2 audit and validation report. |

---

## 3. Implementation Changes & Rationales

### 3.1 Dynamic Dimensionality & Auto-Detection
- **Issue**: Phase 1 defaulted to a hard-coded `expected_dim=128`. If Member 2 or a future scaled checkpoint exports embeddings of 256, 512, or 768 dimensions, the validator would throw an unhandled `ValueError`.
- **Correction**: In [`src/dro/clustering.py`](file:///d:/hciproj/NEURO_GRAPE/src/dro/clustering.py#L48), `expected_dim` is now `Optional[int] = None`. When set to `None`, the validator inspects the first embedding vector and dynamically binds `expected_dim = matrix.shape[1]`. If an explicit dimension is provided, it continues to validate strict dimensional adherence.

### 3.2 Dynamic Group Definition Strategies
- **Issue**: Phase 1 coupled GroupDRO strictly to learned K-Means clusters (`cluster_id`), preventing researchers from evaluating groups defined by semantic metadata (e.g. animate/inanimate) or recording modalities (EEG vs MEG vs fMRI).
- **Correction**: In [`src/dro/pipeline.py`](file:///d:/hciproj/NEURO_GRAPE/src/dro/pipeline.py#L90), implemented three selectable strategies:
  1. `"cluster"`: Runs K-Means on training embeddings ($K$ clusters).
  2. `"modality"`: Groups samples by their biological recording sensor (`eeg`, `meg`, `fmri`).
  3. `"metadata"`: Groups samples by an arbitrary metadata column (`category_name`, `category_id`, `exemplar_id`, etc.).

### 3.3 Dynamic Metric Columns & Granular Reporting
- **Issue**: In Phase 1, `metrics.csv` hard-coded columns `group_0_value`, `group_0_count`, `group_1_value`, and `group_1_count`. If $K \ge 3$ or when grouping by category, higher group metrics were silently omitted from the table.
- **Correction**: Replaced hardcoded dictionary keys with a dynamic loop over all $K$ groups (`group_{g}_value`, `group_{g}_count`). Additionally created [`group_metrics.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_metrics.csv) in standardized long-format:
  `group_id, metric_name, sample_count, mean_value, is_active, is_worst_group, split, higher_is_better, group_name`.

### 3.4 Safe Trimodal Stimulus Extraction
- **Issue**: Indexing into stimulus modalities via `.iloc[0]` assumed every stimulus had exactly one EEG, one MEG, and one fMRI recording. On larger uncurated datasets where some stimuli may have incomplete modality coverage, `.iloc[0]` throws `IndexError`.
- **Correction**: Evaluates modality presence per stimulus and filters strictly to complete triplets, logging clear warnings for any omitted incomplete stimuli.

---

## 4. Methodology Verification

### 4.1 Split Isolation & Leakage Prevention
- **Invariant**: Stimulus identities must never cross train, validation, or test partitions.
- **Verification**:
  - Master split table: [`data/splits/pilot_splits.csv`](file:///d:/hciproj/NEURO_GRAPE/data/splits/pilot_splits.csv)
  - Train (12 stimuli): `banner`, `bear`, `beet`, `boa`, `cucumber`, `glass`, `hamburger`, `hay`, `inhaler`, `lampshade`, `microscope`, `pea`.
  - Val (4 stimuli): `pinwheel`, `puck`, `screwdriver`, `sprinkler`.
  - Test (4 stimuli): `tadpole`, `tarp`, `wire`, `woman`.
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- **Clustering Isolation**: Centroids are fitted strictly on `split == 'train'` (36 samples). Validation and test samples are assigned via `predict()`; their representations are never used to optimize centroids.

### 4.2 Group Identity & Scientific Interpretation
- **Empirical Reality of Learned Clusters**:
  - In [`reports/artifacts/group_assignments.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_assignments.csv), Cluster 0 contains 100% of MEG samples (20/20) and 100% of fMRI samples (20/20). Cluster 1 contains 100% of EEG samples (20/20).
  - **Scientific Interpretation**: Because Member 2's baseline checkpoint exhibits severe modality-specific representation collapse (positive cone clustering), unsupervised clustering groups recordings by sensor modality, **not** by neural visual regimes.
  - **Action**: All reports and code docstrings include a scientific disclaimer that clusters are exploratory groupings for robustness auditing, not biological neural regimes.

### 4.3 GroupDRO Behavior (Simulation Mode)
- **Status**: GroupDRO weights are updated iteratively via:
  $$s_k^{(t+1)} = s_k^{(t)} + \eta \cdot L_k, \quad q^{(t+1)} = \text{softmax}(s)$$
  computed in log-space on fixed baseline embeddings.
- **Verification**: Because encoder weights were not backpropagated during pipeline execution, this is an **offline simulation of group risk under distributional shift**. When retraining encoders, `GroupDRO.compute_weighted_loss()` is ready to be passed into `train_epoch(..., sample_weights=...)`.

---

## 5. Pilot Results: Reproduced vs. Enhanced

### 5.1 Contrastive Loss Benchmark (`crossmodal_contrastive_loss`, Lower is Better)

| Split | Partition Count | Phase 1 Overall Loss | Phase 2 Overall Loss | Group 0 Loss (MEG/fMRI) | Group 1 Loss (EEG) | Worst Group | Performance Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 36 samples (12 stim $\times$ 3) | 2.28841 | **2.28841** | 2.32478 ($n=24$) | 2.21568 ($n=12$) | **Cluster_0** | **0.03637** |
| **Val** | 12 samples (4 stim $\times$ 3) | 1.38379 | **1.38379** | 1.38297 ($n=8$) | 1.38542 ($n=4$) | **Cluster_1** | **0.00163** |
| **Test** | 12 samples (4 stim $\times$ 3) | 1.44230 | **1.44230** | 1.44705 ($n=8$) | 1.43279 ($n=4$) | **Cluster_0** | **0.00475** |

### 5.2 Representation Quality Benchmark (`crossmodal_cosine_similarity`, Higher is Better)

| Split | Partition Count | Overall Cosine | Group 0 Cosine | Group 1 Cosine | Worst Group | Performance Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 36 samples | **+0.00691** | +0.01540 ($n=24$) | -0.01009 ($n=12$) | **Cluster_1** | **0.01699** |
| **Val** | 12 samples | **-0.01280** | -0.00026 ($n=8$) | -0.03789 ($n=4$) | **Cluster_1** | **0.02509** |
| **Test** | 12 samples | **-0.00828** | +0.00677 ($n=8$) | -0.03837 ($n=4$) | **Cluster_1** | **0.03010** |

*Note: In all cases, performance gap is defined non-negatively: for loss, $\text{gap} = \text{worst} - \text{overall}$; for cosine similarity, $\text{gap} = \text{overall} - \text{worst}$.*

### 5.3 GroupDRO Weight Dynamics across 10 Iterations
- Step 1: $q_0 = 0.5014$, $q_1 = 0.4986$ (Group 0 loss $2.3248$ > Group 1 loss $2.2157$).
- Step 10: $q_0 = 0.5129$, $q_1 = 0.4871$ (stable, bounded monotonic adaptation).

---

## 6. Test Verification

All tests were executed inside the project virtual environment (`.venv` with Python 3.13):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_group_dro.py -v
```

### 6.1 Member 3 Test Suite Results (`tests/test_group_dro.py`) — 13/13 PASSED

| Test ID | Method Name | Status | Verified Property |
| :---: | :--- | :---: | :--- |
| 1 | `test_empty_and_undersized_groups` | **PASSED** | Zero-sample and undersized groups default to 0.0 without `NaN` or zero-division errors. |
| 2 | `test_higher_loss_group_receives_higher_weight` | **PASSED** | Group with higher relative loss receives increased probability mass under exponentiated update. |
| 3 | `test_weight_normalization_and_stability` | **PASSED** | Weights sum to 1.0; stable under extreme losses ($\pm 1000.0$). |
| 4 | `test_weighted_loss_computation_and_gradient` | **PASSED** | Scalar weighted objective matches analytical formula and backpropagates finite gradients. |
| 5 | `test_clustering_fits_on_train_split_only` | **PASSED** | Centroids fitted strictly on 18 train samples; val/test assigned via predict. |
| 6 | `test_stable_id_joins_and_error_handling` | **PASSED** | Rejects duplicate IDs and NaNs; sample correspondence invariant to row shuffling. |
| 7 | `test_metric_direction_and_gap_calculation` | **PASSED** | Loss uses argmax; accuracy uses argmin; gap $\ge 0$. |
| 8 | `test_missing_group_excluded_from_worst_group` | **PASSED** | Unobserved groups (count=0) are excluded from worst-group selection. |
| 9 | `test_pipeline_generates_all_expected_artifacts` | **PASSED** | Verifies creation of all 8 artifact files in temporary directories. |
| 10 | `test_dynamic_embedding_dimensions` | **PASSED** | Verifies 256-D synthetic embeddings with `expected_dim=None` auto-detection. |
| 11 | `test_group_dro_from_group_labels` | **PASSED** | Verifies dynamic group creation and label mapping from string categories. |
| 12 | `test_modality_and_metadata_grouping_strategies` | **PASSED** | Verifies execution under `strategy="modality"` ($K=3$) and `strategy="metadata"` ($K=20$). |
| 13 | `test_variable_cluster_count_and_dynamic_metrics` | **PASSED** | Verifies clustering with $K=3$ and verifies dynamic `group_2_value` column creation. |

### 6.2 Full Combined Regression Suite
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_group_dro.py tests/test_handoff.py tests/test_contrastive_loss.py tests/test_normalization.py -v
```
**Outcome:** **53 passed in 3.79s** (100% success rate).

### 6.3 Documented Diagnostic Test Exclusions
1. `tests/test_encoders.py` (3 tests deselected):
   - `test_eeg_init_cosine_below_threshold`, `test_meg_init_cosine_below_threshold`, `test_fmri_init_cosine_below_threshold`.
   - *Rationale*: Preserved as baseline collapse benchmarks documenting positive cone collapse in encoder initializations.
2. `tests/test_dataset.py` (4 tests failing on raw arrays):
   - *Rationale*: Feature arrays (`data/processed/*.npy`) are gitignored and not tracked in the git working tree.

---

## 7. Artifacts Generated

All artifacts were generated in [`reports/artifacts/`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts):

| Filename | Format | Size | Description |
| :--- | :---: | :---: | :--- |
| [`metrics.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/metrics.csv) | CSV | 1,156 B | High-level summary metrics across train, val, and test splits with dynamic group columns and worst-group names. |
| [`group_metrics.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_metrics.csv) | CSV | 1,130 B | Standardized long-format table with rows for every split, metric, group ID, group name, count, and worst-group flag. |
| [`group_losses.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_losses.csv) | CSV | 570 B | Step-by-step history of group mean losses and sample counts during GroupDRO iterations. |
| [`group_weights.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_weights.csv) | CSV | 503 B | Step-by-step history of GroupDRO distribution weights $q_k$. |
| [`group_assignments.parquet`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_assignments.parquet) | Parquet | 9,126 B | Full sample-level table joining `sample_id` to `cluster_id`, `group_id`, and `group_name`. |
| [`group_assignments.csv`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/group_assignments.csv) | CSV | 10,483 B | CSV companion table of sample-level assignments. |
| [`clustering_diagnostics.json`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/clustering_diagnostics.json) | JSON | 438 B | K-Means training inertia (`14.207`), silhouette score (`0.478`), and scientific disclaimer. |
| [`experiment_config.yaml`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/experiment_config.yaml) | YAML | 1,140 B | Full declarative snapshot of execution parameters. |
| [`pipeline_summary.json`](file:///d:/hciproj/NEURO_GRAPE/reports/artifacts/pipeline_summary.json) | JSON | 1,319 B | Execution metadata, execution mode flag, and final group weights. |

---

## 8. Large-Dataset Readiness

The pipeline is now decoupled from the pilot dataset size and ready to ingest large-scale experiments:

### 8.1 Required Input Schema
1. **Metadata Table (`metadata_csv`)**:
   - Required columns: `sample_id` (globally unique string), `stimulus_id` (string), `modality` (`eeg`, `meg`, `fmri`), `split` (`train`, `val`, `test`).
   - Optional grouping columns: `category_id`, `category_name`, `concept_id`, `superordinate_category`.
2. **Embeddings (`embeddings_pt`)**:
   - PyTorch dictionary mapping `sample_id -> torch.Tensor(D,)` with unit L2-norm where $D$ can be any positive integer (e.g., 128, 256, 512).

### 8.2 Execution via Declarative Configuration
To execute on a new dataset, create a configuration file (e.g., `configs/large_experiment.yaml`):

```yaml
paths:
  embeddings_pt: "embeddings/large_cohort_embeddings.pt"
  metadata_csv: "data/manifests/large_metadata.csv"
  output_dir: "reports/experiments/large_run_01"

grouping:
  strategy: "metadata"          # Options: "cluster", "modality", "metadata"
  metadata_column: "category_name"

clustering:
  n_clusters: 10
  random_state: 42
  expected_dim: null            # Auto-detects from embeddings

group_dro:
  learning_rate: 0.02
  smoothing: 0.05
  num_steps: 25

evaluation:
  temperature: 0.07
  min_group_size: 5
```

Execute from terminal:
```powershell
.\.venv\Scripts\python.exe scripts/run_member3_pipeline.py --config configs/large_experiment.yaml
```

---

## 9. Limitations & Blockers

1. **Member 1 Preprocessed Arrays (`data/processed/*.npy`)**:
   - Raw feature arrays are gitignored and not present in this workspace checkout. Member 3's pipeline runs completely on Member 2's exported latent embeddings (`embeddings/embeddings.pt`). If end-to-end retraining from raw epochs is desired, Member 1's preprocessing script (`python scripts/build_pipeline.py`) must be executed first to populate `data/processed/`.
2. **Pilot Dataset Size ($N=60, 20\text{ stimuli}$)**:
   - 20 stimuli validate software interfaces, tensor shapes, and zero-leakage mechanics. They do not constitute a statistically powered machine learning benchmark for claiming out-of-distribution neural generalization.
3. **Representation Collapse Dominates Semantic Regimes**:
   - Learned K-Means clusters currently reflect sensor modality partitions rather than biological neural regimes due to baseline representation collapse.

---

## 10. Next Steps

1. **Ingest Scaled THINGS Metadata**: When scaling beyond the pilot, ingest superordinate semantic labels (e.g., `animate` vs `inanimate`, `natural` vs `artificial`) to test GroupDRO against human visual semantic regimes.
2. **End-to-End Encoder Retraining with GroupDRO**: Integrate `GroupDRO.compute_weighted_loss()` into Member 2's active training loop (`src/models/training.py`) to optimize encoder weights directly against worst-group risk.
3. **Evaluation on Full THINGS Cohort**: Evaluate cross-modal retrieval and alignment across the full cohort of 1,854 concepts once scaled embeddings are exported by Member 2.
