# Member 1 Final Handoff Report: Dataset Engineering & Integration

**Project:** Neuro-GRAPE (Multimodal Neural Representation & Decoding)  
**Author:** Member 1 (Dataset Engineering, Preprocessing, Reusable Loaders, Data Quality & Leakage Prevention)  
**Date:** October 2026  
**Status:** Completed & Validated — Ready for Member 2 & 3 Integration  

---

## 1. Executive Summary & Deliverables Inventory

Member 1's responsibilities for Phases 1–4 are complete. All required datasets, preprocessing pipelines, loaders, validation tests, documentation, and handoff archives have been verified on disk without modifying raw data or installing global packages.

### Deliverables Master Table

| Deliverable Name | File / Directory Path | Status | Verification Summary |
| :--- | :--- | :---: | :--- |
| **Phase 1 Audit & Plan** | `reports/phase1_audit_and_plan.md` | Verified | Verified 20-stimulus intersection, identified 1000 Hz sample onset gotcha in EEG. |
| **Phase 2 Implementation Report** | `reports/phase2_implementation_report.md` | Verified | Documented virtual environment setup, MNE/NiBabel pipeline creation. |
| **Phase 3 Validation Report** | `reports/phase3_validation_report.md` | Verified | Documented fMRI LS-S GLM feature resolution, boundary safety, and CTF grade 3. |
| **Data Quality Report** | `reports/data_quality_report.md` | Verified | SNR, baseline offsets ($<1\%$), 100% finite values (0 NaNs/Infs). |
| **Compatibility & Overlap Report**| `reports/data_compatibility_and_stimulus_overlap.md` | Verified | 5-dimensional distinction: filenames, concepts, subjects, timing, representations. |
| **Leakage Audit Report** | `reports/leakage_audit.md` | Verified | Mathematical verification: zero intersection across train (12), val (4), test (4). |
| **Final Handoff Report** | `reports/member1_final_handoff.md` | Verified | This document. |
| **Integration Contract** | `docs/integration_contract.md` | Verified | Handoff specifications and executable snippets for Members 2 and 3. |
| **Reproducible Setup Guide** | `docs/reproducible_setup_guide.md` | Verified | Detailed reproduction, environment, and execution instructions. |
| **Root Repository Guide** | `README.md` | Verified | Repository overview, directory map, quickstart, and licensing. |
| **Configuration Specification**| `configs/dataset_config.yaml` | Verified | Declarative parameters for EEG, MEG, and fMRI pipelines. |
| **Master Data Inventory** | `data_inventory.csv` & `data/manifests/data_inventory.csv` | Verified | 60 verified records (20 stimuli × 3 modalities) with unique `sample_id`. |
| **Pilot Splits** | `data/splits/pilot_splits.csv` | Verified | Stimulus-disjoint split table (12 train, 4 val, 4 test). |
| **Preprocessing Pipeline** | `src/data/preprocessing.py` | Verified | MNE epoching, NiBabel header/window slicing, and LS-S single-trial GLM. |
| **Reusable Dataset Loader** | `src/data/dataset.py` | Verified | PyTorch-compatible `NeuroGrapeDataset` with fast indexing and batch stacking. |
| **Automated Test Suite** | `tests/test_dataset.py` | Verified | 8/8 regression unit tests passing cleanly in 0.63s. |
| **Preprocessed EEG Epochs** | `data/processed/eeg/*.npy` | Verified | 20 arrays of shape `(63, 225)` @ 250 Hz in Volts. |
| **Preprocessed MEG Epochs** | `data/processed/meg/*.npy` | Verified | 20 arrays of shape `(272, 225)` @ 250 Hz in Tesla. |
| **Preprocessed fMRI Betas** | `data/processed/fmri_beta/*.npy`| Verified | 20 activation volumes of shape `(96, 96, 60)` in beta weights. |
| **Preprocessed fMRI Windows** | `data/processed/fmri/*.npy` | Verified | 20 raw 4-TR candidate windows `(96, 96, 60, 4)`. |
| **Handoff Archive** | `NeuroGRAPE_pilot_processed_handoff.tar.gz` | Verified | 103.68 MB archive (122 files) excluding 1.53 GB raw recordings. |

---

## 2. Actual Verification Commands & Test Results

All verification tests were executed against the actual stored artifacts in the project virtual environment:

```powershell
.\.venv\Scripts\python.exe tests/test_dataset.py -v
```

### Output:
```
test_data_paths_exist_and_tensors_valid (__main__.TestNeuroGrapeDataset.test_data_paths_exist_and_tensors_valid)
Verify that all feature arrays referenced in the manifest exist and contain finite data. ... ok
test_dataset_loader_api (__main__.TestNeuroGrapeDataset.test_dataset_loader_api)
Verify NeuroGrapeDataset access methods and filtering. ... ok
test_epoch_sample_counts_and_downsampling (__main__.TestNeuroGrapeDataset.test_epoch_sample_counts_and_downsampling)
Verify math: 250 Hz across [-0.1, 0.8]s yields exactly 225 timepoints. ... ok
test_error_handling_invalid_keys (__main__.TestNeuroGrapeDataset.test_error_handling_invalid_keys)
Verify clean KeyError on nonexistent sample_id or stimulus_id. ... ok
test_fmri_lss_beta_properties (__main__.TestNeuroGrapeDataset.test_fmri_lss_beta_properties)
Verify that trial-wise fMRI LS-S beta maps exist and have valid 3D geometry. ... ok
test_leakage_prevention_stimulus_disjointness (__main__.TestNeuroGrapeDataset.test_leakage_prevention_stimulus_disjointness)
Verify that stimulus identities NEVER cross train/val/test splits. ... ok
test_manifest_schema_and_uniqueness (__main__.TestNeuroGrapeDataset.test_manifest_schema_and_uniqueness)
Verify required columns and 100% uniqueness of sample_id. ... ok
test_modality_batch_stacking (__main__.TestNeuroGrapeDataset.test_modality_batch_stacking)
Verify batch stacking convenience method for Member 2. ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.629s

OK
```

---

## 3. Input / Output Interface for Member 2 (Model Architecture & Encoders)

### 3.1 Loading Features and Batches
Member 2 consumes standardized NumPy arrays (or PyTorch tensors) directly via `NeuroGrapeDataset`.

```python
from pathlib import Path
from src.data.dataset import NeuroGrapeDataset

# 1. Initialize dataset
dataset = NeuroGrapeDataset(manifest="data_inventory.csv", modality="eeg", split="train")

# 2. Access single sample
sample = dataset[0]
print(sample["sample_id"])      # 'eeg_sub-01_task-rsvp_run-01_trial-000000_banner-02s'
print(sample["stimulus_id"])    # 'banner_02s.jpg'
print(sample["features"].shape) # (63, 225)
print(sample["signal_units"])   # 'V'

# 3. Stack an entire split into a batch tensor (N, C, T)
batch_features, batch_meta = dataset.get_modality_batch("eeg", split="train")
print(batch_features.shape)     # (12, 63, 225)
```

### 3.2 Representational Geometry by Modality

| Modality | Output Array Shape | Time / Spatial Window | Signal Units | Preprocessing Applied |
| :--- | :--- | :--- | :--- | :--- |
| **EEG** | `(63, 225)` | $[-100, 800]\text{ ms}$ @ 250 Hz | Volts ($\text{V}$) | MNE Epoching, $[-100, 0]\text{ ms}$ baseline, polyphase downsampling to 250 Hz. |
| **MEG** | `(272, 225)` | $[-100, 800]\text{ ms}$ @ 250 Hz | Tesla ($\text{T}$) | CTF Grade 3 synthetic gradient, 272 axial gradiometers, polyphase downsampling. |
| **fMRI** | `(96, 96, 60)` | Single-trial beta activation | Beta weight | Mumford LS-S GLM, canonical SPM HRF, 7-harmonic DCT drift ($>128\text{ s}$), brain mask. |

> [!WARNING]
> **Representation Heterogeneity Warning**:
> EEG/MEG arrays have shape `(channels, time)` whereas fMRI has shape `(X, Y, Z)`. They **cannot be stacked into a single multi-modal tensor** without modality-specific encoders projecting them into a shared latent space.

---

## 4. Metadata & Split Interface for Member 3 (Evaluation, GroupDRO & Leakage)

### 4.1 Metadata Retrieval
```python
from src.data.dataset import NeuroGrapeDataset

ds = NeuroGrapeDataset(manifest="data_inventory.csv")
meta_df = ds.get_metadata_table()

# Grouping and cross-modal inspection
print(meta_df[["sample_id", "modality", "stimulus_id", "category_id", "category_name", "split"]].head())

# Cross-modal query for a given stimulus
cross_modal = ds.get_cross_modal_sample("puck_02s.jpg")
print("EEG shape:  ", cross_modal["eeg"]["features"].shape)   # (63, 225)
print("MEG shape:  ", cross_modal["meg"]["features"].shape)   # (272, 225)
print("fMRI shape: ", cross_modal["fmri"]["features"].shape)  # (96, 96, 60)
```

### 4.2 Split Partitions & Generalization Guarantee
- **Generalization Goal**: Zero-shot held-out stimulus decoding.
- **Leakage Invariant**:
  $$\text{Split}(\text{EEG}_s) \equiv \text{Split}(\text{MEG}_s) \equiv \text{Split}(\text{fMRI}_s) \quad \forall s \in \text{Stimuli}$$
- **Partitions**:
  - **Train (12 stimuli / 36 samples)**: `banner`, `bear`, `beet`, `boa`, `cucumber`, `glass`, `hamburger`, `hay`, `inhaler`, `lampshade`, `microscope`, `pea`
  - **Val (4 stimuli / 12 samples)**: `pinwheel`, `puck`, `screwdriver`, `sprinkler`
  - **Test (4 stimuli / 12 samples)**: `tadpole`, `tarp`, `wire`, `woman`
- **Scope Limitation**: The 20-stimulus pilot verifies pipeline and interface correctness. It is **not** sufficient for claiming statistically reliable worst-group generalization. GroupDRO is not implemented in the data layer.

---

## 5. Instructions for Using the Current Pilot (No Preprocessing Needed)

Teammates can immediately start developing without touching raw files or rerunning preprocessing:
1. Activate `.venv`:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Verify tests:
   ```powershell
   python tests/test_dataset.py -v
   ```
3. Load data directly via `NeuroGrapeDataset(manifest="data_inventory.csv")`.

---

## 6. Handoff Package Archive Details

A standalone handoff archive was created to enable portable sharing of code, manifests, and processed features without the 1.53 GB raw continuous recordings:
- **Archive Path**: `D:\NEURO_GRAPE\NeuroGRAPE_pilot_processed_handoff.tar.gz`
- **Archive Size**: **103.68 MB** (108,719,939 bytes)
- **Members**: 122 files (includes `configs/`, `data/manifests/`, `data/splits/`, `data/processed/`, `docs/`, `reports/`, `src/`, `tests/`, `README.md`, `requirements.txt`, and `data_inventory.csv`).
- **Raw Data Protection**: The original raw archive `NeuroGRAPE_handover.tar.gz` (986 MB) remains completely intact, untouched, and uncompressed in workspace root.

---

## 7. Status Checklist: Completed vs. Pending Responsibilities

### Completed by Member 1 (Data Layer):
- [x] Full pilot inventory and integrity audit.
- [x] Verified 20-stimulus intersection and category indexing formulas.
- [x] Handled 1000 Hz sample onset interpretation in EEG.
- [x] Verified CTF 3rd-order synthetic gradient compensation in MEG.
- [x] Implemented MNE EEG/MEG epoching and anti-aliased downsampling to 250 Hz.
- [x] Implemented Mumford LS-S single-trial GLM beta estimation with SPM canonical HRF and DCT drift filtering for fMRI.
- [x] Established deterministic, stable `sample_id` format and master inventory.
- [x] Implemented leakage-proof stimulus-disjoint train/val/test splits.
- [x] Implemented reusable PyTorch-compatible `NeuroGrapeDataset`.
- [x] Automated regression test suite (8/8 tests passing).
- [x] Authored integration contracts, setup guides, and quality reports.
- [x] Built and verified lightweight handoff package archive.

### Pending for Later Team Members:
- [ ] **Member 2**: Implement modality-specific neural encoders (1D-CNN / Transformer for EEG/MEG; 3D-CNN / Swin for fMRI) and latent alignment loss functions (e.g. InfoNCE / CLIP-style contrastive objectives).
- [ ] **Member 3**: Ingest full THINGS semantic category hierarchies (animate/inanimate, natural/artificial) to define subgroup labels for GroupDRO optimization and worst-group generalization benchmarking.
- [ ] **Upstream Data Scaling**: When moving beyond the pilot, ingest full fMRIPrep derivatives to include head motion regressors and anatomical normalization.
