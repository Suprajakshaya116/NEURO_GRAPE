# Neuro-GRAPE Reproducible Setup Guide
**Author:** Member 1 (Dataset Engineering, Preprocessing, Data Loading, Leakage Prevention)  
**Target Environment:** Python 3.13.x (Windows AMD64)  
**Date:** October 2026  

---

## 1. Environment Requirements & Setup

The pipeline requires Python 3.13 (or compatible 3.10–3.12). All dependencies are managed in an isolated virtual environment (`.venv`):

```powershell
# 1. Navigate to workspace root
cd D:\NEURO_GRAPE

# 2. Create virtual environment with system site-packages inheritance
python -m venv --system-site-packages .venv

# 3. Install domain packages (MNE and NiBabel)
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Verified Installed Packages:
- `numpy==2.4.2`
- `scipy==1.18.0`
- `pandas==3.0.2`
- `scikit-learn==1.9.0`
- `matplotlib==3.10.8`
- `mne==1.13.2`
- `nibabel==5.4.2`

---

## 2. Using Existing Preprocessed Pilot Data (No Rerunning Required)

Teammates do **not** need to touch the raw 1.53 GB recordings or rerun preprocessing. All 60 preprocessed arrays are already stored on disk:
- `data/processed/eeg/` (`*.npy`, shape `(63, 225)` @ 250 Hz)
- `data/processed/meg/` (`*.npy`, shape `(272, 225)` @ 250 Hz)
- `data/processed/fmri/` (`*.npy`, raw 4-TR window `(96, 96, 60, 4)`)
- `data/processed/fmri_beta/` (`*.npy`, LS-S single-trial beta maps `(96, 96, 60)`)

### Example: Instant Loading
```python
from pathlib import Path
from src.data.dataset import NeuroGrapeDataset

ds = NeuroGrapeDataset(manifest="data_inventory.csv", modality="eeg", split="train")
sample = ds[0]
print(f"Loaded {sample['sample_id']} with shape {sample['features'].shape}")
```

---

## 3. Regenerating Preprocessed Outputs from Scratch (If Required)

If preprocessing parameters change (e.g. changing the epoch window from $[-100, 800]\text{ ms}$ or altering the resample frequency), the full pipeline can be reproduced using:

```powershell
.\.venv\Scripts\python.exe scripts/build_pipeline.py
```

This script will:
1. Load `NeuroGRAPE_handover/manifests/shared_stimuli.csv`.
2. Construct and save `data/splits/pilot_splits.csv`.
3. Preprocess and epoch EEG recordings via MNE into `data/processed/eeg/`.
4. Preprocess and epoch MEG recordings via MNE into `data/processed/meg/`.
5. Extract fMRI raw windows into `data/processed/fmri/` and estimate LS-S beta maps into `data/processed/fmri_beta/`.
6. Write the master manifest to `data_inventory.csv` and `data/manifests/data_inventory.csv`.

---

## 4. Running Automated Regression Tests

The test suite runs using standard library `unittest`:

```powershell
.\.venv\Scripts\python.exe tests/test_dataset.py -v
```

### Verified Test Cases:
1. `test_manifest_schema_and_uniqueness`: Verifies required columns and 100% uniqueness of 60 `sample_id` records.
2. `test_leakage_prevention_stimulus_disjointness`: Verifies mathematically that train, val, and test stimulus sets have zero intersection.
3. `test_data_paths_exist_and_tensors_valid`: Verifies all 60 files exist on disk, are finite (0 NaNs/Infs), and match expected shapes (`(63, 225)`, `(272, 225)`, `(96, 96, 60)`).
4. `test_epoch_sample_counts_and_downsampling`: Verifies sampling rate arithmetic (250 Hz × 0.9 s = 225 timepoints).
5. `test_fmri_lss_beta_properties`: Verifies that 3D beta volumes have $>150,000$ non-zero intracranial voxels.
6. `test_dataset_loader_api`: Verifies indexing, modality filtering, split filtering, and fast `sample_id` lookups.
7. `test_modality_batch_stacking`: Verifies stacked array generation `(12, 63, 225)` for Member 2.
8. `test_error_handling_invalid_keys`: Verifies clean `KeyError` handling.

---

## 5. Artifact & File Locations

- **Manifests**: `data_inventory.csv`, `data/manifests/data_inventory.csv`
- **Splits**: `data/splits/pilot_splits.csv`
- **Feature Arrays**: `data/processed/{eeg, meg, fmri, fmri_beta}/`
- **Configuration**: `configs/dataset_config.yaml`
- **Integration Contracts**: `docs/integration_contract.md`
- **Audit & Validation Reports**:
  - Initial Plan: `reports/phase1_audit_and_plan.md`
  - Implementation Report: `reports/phase2_implementation_report.md`
  - Validation Report: `reports/phase3_validation_report.md`
  - Data Quality Report: `reports/data_quality_report.md`
  - Compatibility Report: `reports/data_compatibility_and_stimulus_overlap.md`
  - Leakage Audit: `reports/leakage_audit.md`
  - Final Handoff Report: `reports/member1_final_handoff.md`

---

## 6. Known Limitations & Data Privacy

1. **Missing Confound Regressors in Pilot**: Raw head motion parameters and preprocessed BOLD derivatives are absent from the pilot archive. Future phases will ingest fMRIPrep derivatives.
2. **Pilot Sample Size**: The 20 shared stimuli are intended for software engineering, tensor verification, and pipeline testing. They do not constitute a statistically powered machine learning benchmark.
3. **Data Sharing & Licensing**: All source datasets are published under **CC0 (Public Domain Dedication)** on OpenNeuro. Raw data directories must remain read-only and outside Git version control. No credentials, tokens, or restricted files are committed.
