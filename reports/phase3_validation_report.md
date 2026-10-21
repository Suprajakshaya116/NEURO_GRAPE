# Phase 3: Pilot Validation, Scientific Quality, & Integration Report

**Project:** Neuro-GRAPE  
**Author:** Member 1 (Dataset Engineering, Preprocessing, Data Quality & Leakage Prevention)  
**Date:** October 2026  
**Status:** Complete — All Phase 3 Objectives Validated  

---

## 1. Executive Summary

Phase 3 validated the empirical integrity of the Neuro-GRAPE pilot data pipeline, resolved key scientific gaps in trial-wise feature modeling, and completed comprehensive quality and leakage audits.

### Key Outcomes:
1. **Targeted EEG & MEG Validation**:
   - Confirmed timing, sampling rates, anti-aliased polyphase downsampling to 250 Hz, and zero boundary clipping.
   - Verified that MEG CTF 3rd-order synthetic gradient compensation (`compensation_grade == 3`) was already applied during recording.
   - Verified that all 20 EEG epochs and 20 MEG epochs are 100% finite and baseline-centered.
2. **fMRI Feature Modeling Resolved**:
   - Implemented and validated the **Mumford et al. (2012) Least Squares-Separate (LS-S) single-trial GLM** using the SPM canonical double-gamma HRF and a 7-harmonic Discrete Cosine Transform (DCT) drift basis ($>128\text{ s}$ cutoff).
   - Produced 20 trial-wise activation beta volumes `(96, 96, 60)` saved under `data/processed/fmri_beta/` with an intensity-derived intracranial brain mask (228,068 voxels).
   - Documented the archive limitation that raw confound regressors (motion realignment parameters) were not provided in the pilot archive.
3. **Leakage & Manifest Verification**:
   - Manifest `data_inventory.csv` contains 60 verified records with 100% unique, deterministic sample IDs.
   - Zero split leakage confirmed across train (12 stimuli), val (4 stimuli), and test (4 stimuli).
4. **Extended Regression Test Suite**:
   - 8/8 unit tests passed cleanly in 0.52s ([tests/test_dataset.py](file:///d:/NEURO_GRAPE/tests/test_dataset.py)).
5. **Quality & Integration Documentation**:
   - Published `reports/data_quality_report.md`, `reports/data_compatibility_and_stimulus_overlap.md`, `reports/leakage_audit.md`, and this summary report.

---

## 2. Regression Test Execution Summary

Executed using the project-isolated virtual environment:
```powershell
.\.venv\Scripts\python.exe tests/test_dataset.py -v
```

```
test_data_paths_exist_and_tensors_valid (__main__.TestNeuroGrapeDataset.test_data_paths_exist_and_tensors_valid) ... ok
test_dataset_loader_api (__main__.TestNeuroGrapeDataset.test_dataset_loader_api) ... ok
test_epoch_sample_counts_and_downsampling (__main__.TestNeuroGrapeDataset.test_epoch_sample_counts_and_downsampling) ... ok
test_error_handling_invalid_keys (__main__.TestNeuroGrapeDataset.test_error_handling_invalid_keys) ... ok
test_fmri_lss_beta_properties (__main__.TestNeuroGrapeDataset.test_fmri_lss_beta_properties) ... ok
test_leakage_prevention_stimulus_disjointness (__main__.TestNeuroGrapeDataset.test_leakage_prevention_stimulus_disjointness) ... ok
test_manifest_schema_and_uniqueness (__main__.TestNeuroGrapeDataset.test_manifest_schema_and_uniqueness) ... ok
test_modality_batch_stacking (__main__.TestNeuroGrapeDataset.test_modality_batch_stacking) ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.522s

OK
```

---

## 3. Preprocessed Feature Specifications

| Modality | Pipeline Method | Output Array Shape | Signal Units | File Location |
| :--- | :--- | :--- | :--- | :--- |
| **EEG** | MNE Epoching + Polyphase Resampling | `(63, 225)` @ 250 Hz | Volts ($\text{V}$) | `data/processed/eeg/*.npy` |
| **MEG** | MNE Epoching + CTF Grade 3 + Resampling | `(272, 225)` @ 250 Hz | Tesla ($\text{T}$) | `data/processed/meg/*.npy` |
| **fMRI** (Raw Window) | NiBabel On-Demand Slicing | `(96, 96, 60, 4)` | Raw BOLD counts | `data/processed/fmri/*.npy` |
| **fMRI** (Trial Betas) | Mumford LS-S GLM + SPM HRF + DCT Drift | `(96, 96, 60)` | Estimated Beta Weight | `data/processed/fmri_beta/*.npy` |

---

## 4. Unresolved Decisions & Limitations

1. **fMRI Head Motion Regressors**:
   - The pilot archive provided only the raw continuous BOLD file without motion parameters (`trans_x`, `trans_y`, `trans_z`, etc.). While LS-S deconvolves overlapping trial BOLD signals, residual motion artifacts cannot be regressed out until full fMRIPrep derivatives are ingested.
2. **Evaluation Sample Size**:
   - 20 stimuli (12 train, 4 val, 4 test) provide complete verification of software interfaces, tensor dimensions, and leakage rules, but are not statistically sufficient for final model benchmarking.

---

## 5. Next Steps for Phase 4 (Team Integration)

1. **Member 2 Integration**: Supply batch loaders from `NeuroGrapeDataset` for encoder representation alignment across EEG `(63, 225)`, MEG `(272, 225)`, and fMRI beta `(96, 96, 60)`.
2. **Member 3 Integration**: Deliver stable metadata tables and group keys for evaluation and GroupDRO modeling.
