# Phase 2: Implementation of Minimal Reusable Data Pipeline

**Project:** Neuro-GRAPE  
**Author:** Member 1 (Dataset Engineering, Preprocessing, Data Quality & Leakage Prevention)  
**Date:** October 2026  
**Status:** Complete — All Unit Tests Passed  

---

## 1. Executive Summary

Phase 2 successfully implemented the reusable data pipeline and integration contracts for Neuro-GRAPE. Using a project-isolated virtual environment (`.venv`), domain-standard libraries (`mne` 1.13.2 and `nibabel` 5.4.2) were integrated without modifying the global system environment or raw data files.

### Key Milestones Achieved:
1. **Isolated Scientific Environment**: Provisioned `d:\NEURO_GRAPE\.venv` with `--system-site-packages` and installed `mne` and `nibabel`.
2. **Standardized Preprocessing Pipeline** (`src/data/preprocessing.py`):
   - **EEG**: Extracted $[-100, 800]\text{ ms}$ epochs (baseline corrected $[-100, 0]\text{ ms}$) using verified 1000 Hz sample indices; downsampled to 250 Hz (`(63, 225)`).
   - **MEG**: Extracted $[-100, 800]\text{ ms}$ epochs (baseline corrected $[-100, 0]\text{ ms}$) from 272 primary axial gradiometers; downsampled to 250 Hz (`(272, 225)`).
   - **fMRI**: Inspected 4D BOLD header via `nibabel.load` and extracted on-demand trial sub-volume windows ($[0, 6.0]\text{ s}$, 4 TRs, `(96, 96, 60, 4)`) without loading all 284 volumes into RAM.
3. **Unified Dataset Class** (`src/data/dataset.py`):
   - Fully supports indexing, modality/split filtering, fast `sample_id` lookup, and cross-modal associative retrieval.
   - Fully compatible with PyTorch `DataLoader` while operating natively on NumPy arrays.
4. **Master Inventory & Leakage-Proof Splits**:
   - `data_inventory.csv`: 60 verified records (20 shared stimuli × 3 modalities) with deterministic `sample_id` keys.
   - `data/splits/pilot_splits.csv`: Zero-shot stimulus generalization partition (12 Train, 4 Val, 4 Test) strictly preventing cross-split stimulus contamination.
5. **Automated Validation Test Suite** (`tests/test_dataset.py`):
   - 5/5 unit tests passed in 1.86s verifying manifest schema, ID uniqueness, path resolution, tensor validity, and zero split leakage.

---

## 2. Files Created & Modified

| File Path | Action | Description |
| :--- | :---: | :--- |
| `src/data/preprocessing.py` | Created | MNE & NiBabel epoching, on-demand slicing, deterministic sample ID generation. |
| `src/data/dataset.py` | Created | `NeuroGrapeDataset` loader class with batch stacking and cross-modal queries. |
| `data_inventory.csv` | Created | Master tabular manifest (60 rows) with stable identifiers and metadata. |
| `data/manifests/data_inventory.csv` | Created | Repository copy of master manifest. |
| `data/splits/pilot_splits.csv` | Created | Stimulus-disjoint split table (12 train, 4 val, 4 test). |
| `data/processed/eeg/*.npy` | Created | 20 preprocessed EEG epoch arrays `(63, 225)` @ 250 Hz. |
| `data/processed/meg/*.npy` | Created | 20 preprocessed MEG epoch arrays `(272, 225)` @ 250 Hz. |
| `data/processed/fmri/*.npy` | Created | 20 candidate fMRI trial sub-volume arrays `(96, 96, 60, 4)`. |
| `tests/test_dataset.py` | Created | Standard library `unittest` suite (5 test cases). |
| `configs/dataset_config.yaml` | Created | Pipeline parameters, epoch windows, channel counts, and paths. |
| `requirements.txt` | Created | Dependency specifications for Python 3.13. |
| `docs/integration_contract.md` | Created | Handoff specifications and code snippets for Members 2 and 3. |
| `docs/reproducible_setup_guide.md` | Created | Step-by-step reproduction instructions. |

---

## 3. Commands Executed & Test Results

```powershell
# 1. Environment creation and package installation
python -m venv --system-site-packages --clear d:\NEURO_GRAPE\.venv
.\.venv\Scripts\pip.exe install nibabel mne

# 2. Pipeline execution (preprocessing and inventory build)
.\.venv\Scripts\python.exe d:\NEURO_GRAPE\scratch_build_pipeline.py

# 3. Test suite execution
.\.venv\Scripts\python.exe tests/test_dataset.py -v
```

### Actual Test Output:
```
test_data_paths_exist_and_tensors_valid (__main__.TestNeuroGrapeDataset.test_data_paths_exist_and_tensors_valid)
Verify that all feature arrays referenced in the manifest exist and contain finite data. ... ok
test_dataset_loader_api (__main__.TestNeuroGrapeDataset.test_dataset_loader_api)
Verify NeuroGrapeDataset access methods and filtering. ... ok
test_leakage_prevention_stimulus_disjointness (__main__.TestNeuroGrapeDataset.test_leakage_prevention_stimulus_disjointness)
Verify that stimulus identities NEVER cross train/val/test splits. ... ok
test_manifest_schema_and_uniqueness (__main__.TestNeuroGrapeDataset.test_manifest_schema_and_uniqueness)
Verify required columns and 100% uniqueness of sample_id. ... ok
test_modality_batch_stacking (__main__.TestNeuroGrapeDataset.test_modality_batch_stacking)
Verify batch stacking convenience method for Member 2. ... ok

----------------------------------------------------------------------
Ran 5 tests in 1.865s

OK
```

---

## 4. Unresolved Scientific Decisions

1. **fMRI Trial-wise Feature Estimation**:
   - *Status*: Pending GLM single-trial beta estimation (LS-S).
   - *Rationale*: Raw BOLD slices without motion correction and high-pass drift regression are contaminated by physiological noise. Simple fixed-lag averaging ($4.5\text{–}6.0\text{ s}$) is scientifically unvalidated. A full GLM beta extraction script should be scheduled in Phase 3.
2. **MEG Sensor Transformations**:
   - *Status*: Primary axial gradiometers (272 channels) are extracted directly.
   - *Question for Team*: Should CTF 3rd-order synthetic gradient compensation or planar gradient conversion be applied prior to encoder modeling?
3. **Pilot Evaluation Scope**:
   - *Limitation*: The 20-stimulus pilot demonstrates pipeline functionality and leakage-proof partitioning, but is statistically insufficient for a final machine learning benchmark. Full model claims must wait for scaled data ingestion.

---

## 5. Next Steps for Phase 3

1. **Implement fMRI GLM / Beta Feature Extractor**: Build single-trial beta estimator using NiBabel and scipy / nilearn conventions.
2. **Data Quality and Leakage Report**: Author `reports/data_quality_and_leakage_report.md` detailing signal-to-noise ratios, trial timing sanity checks, and strict split isolation.
3. **Cross-Modality Compatibility Report**: Author `reports/compatibility_report.md` formalizing shared stimulus representations for Member 2 and Member 3.
