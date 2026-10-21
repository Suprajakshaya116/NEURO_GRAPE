# Neuro-GRAPE: Multimodal Neural Representation & Decoding

**Phase 1–4 Completed**: Dataset Engineering, Preprocessing, Data Quality, Leakage Prevention & Integration Contracts.  
**Author**: Member 1 (Dataset Engineer)  

---

## 1. Project Overview

Neuro-GRAPE investigates multimodal neural representation alignment across human **EEG**, **MEG**, and **fMRI** using the OpenNeuro THINGS datasets:
- **EEG**: THINGS-EEG1 ([OpenNeuro ds003825](https://doi.org/10.18112/openneuro.ds003825.v1.1.0)), 63 scalp electrodes.
- **MEG**: THINGS-MEG ([OpenNeuro ds004212](https://doi.org/10.18112/openneuro.ds004212.v2.0.0)), CTF 272 axial gradiometers.
- **fMRI**: THINGS-fMRI ([OpenNeuro ds004192](https://doi.org/10.18112/openneuro.ds004192.v1.0.7)), Siemens Prisma 3T 2.0 mm BOLD.

### Critical Cohort & Alignment Notice:
The pilot recordings originate from **three independent human participants and three separate experiments**:
- EEG: `sub-01`, RSVP recording (Univ. of Sydney).
- MEG: `sub-BIGMEG2`, Session 07, Run 06 (NIH).
- fMRI: `sub-01`, Session `things08`, Run 10 (NIH).

**The three modalities share visual stimulus identity (`stimulus_id`) only**. There is **zero simultaneous recording, zero subject correspondence, and zero raw temporal synchronization**. Representation models must align features in a semantic/latent space, not via paired time-series loss functions.

---

## 2. Directory Structure

```
d:/NEURO_GRAPE/
├── configs/
│   └── dataset_config.yaml           # Declarative pipeline & model parameters
├── data/
│   ├── manifests/
│   │   └── data_inventory.csv        # Master tabular inventory (60 samples)
│   ├── processed/
│   │   ├── eeg/*.npy                 # 20 preprocessed EEG epochs (63, 225) @ 250 Hz
│   │   ├── meg/*.npy                 # 20 preprocessed MEG epochs (272, 225) @ 250 Hz
│   │   ├── fmri/*.npy                # 20 raw candidate BOLD windows (96, 96, 60, 4)
│   │   └── fmri_beta/*.npy           # 20 trial-wise LS-S GLM beta volumes (96, 96, 60)
│   └── splits/
│       └── pilot_splits.csv          # Leakage-resistant split table (12 train, 4 val, 4 test)
├── data_inventory.csv                # Primary root copy of master manifest
├── docs/
│   ├── integration_contract.md       # Consumer contract & executable examples for Members 2 & 3
│   └── reproducible_setup_guide.md   # Step-by-step reproduction instructions
├── NeuroGRAPE_handover/              # Read-only extracted raw pilot archive (1.53 GB)
│   ├── raw/ (EEG, MEG, fMRI)
│   ├── manifests/shared_stimuli.csv
│   └── metadata/
├── reports/
│   ├── phase1_audit_and_plan.md      # Initial baseline audit & architecture plan
│   ├── phase2_implementation_report.md# Minimal reusable pipeline implementation report
│   ├── phase3_validation_report.md   # Scientific validation & fMRI feature resolution report
│   ├── data_quality_report.md        # Detailed SNR, baseline, and noise audit
│   ├── data_compatibility_and_stimulus_overlap.md # 5-dimensional compatibility report
│   ├── leakage_audit.md              # Mathematical verification of split disjointness
│   └── member1_final_handoff.md      # Final handoff specification
├── requirements.txt                  # Pinned environment requirements
├── scripts/
│   └── build_pipeline.py             # Reproducible pipeline extraction script
├── src/
│   └── data/
│       ├── __init__.py
│       ├── dataset.py                # NeuroGrapeDataset loader class
│       └── preprocessing.py          # MNE & NiBabel preprocessing & LS-S beta estimation
└── tests/
    └── test_dataset.py               # Automated unit & regression test suite (8 test cases)
```

---

## 3. Quickstart: Loading Preprocessed Pilot Data

No need to rerun preprocessing! All preprocessed features are stored on disk.

```python
from pathlib import Path
from src.data.dataset import NeuroGrapeDataset

# 1. Load dataset
manifest_file = Path("data_inventory.csv")
dataset_eeg = NeuroGrapeDataset(manifest=manifest_file, modality="eeg", split="train")

# 2. Access a sample
sample = dataset_eeg[0]
print(sample["sample_id"])      # 'eeg_sub-01_task-rsvp_run-01_trial-000000_banner-02s'
print(sample["features"].shape) # (63 channels, 225 timepoints)
print(sample["signal_units"])   # 'V' (Volts)

# 3. Stack an entire split for Member 2
eeg_batch, meta_df = dataset_eeg.get_modality_batch("eeg", split="train")
print(eeg_batch.shape)          # (12, 63, 225)
```

---

## 4. Environment Setup & Reproduction

```powershell
# 1. Activate project environment
cd D:\NEURO_GRAPE
.\.venv\Scripts\Activate.ps1

# 2. Run the test suite (8 tests)
python tests/test_dataset.py -v

# 3. (Optional) Rebuild preprocessed arrays from raw recordings
python scripts/build_pipeline.py
```

---

## 5. Known Limitations & Scientific Decisions

1. **Raw fMRI Confounders**: The pilot archive contains raw functional BOLD without motion realignment regressors (`confounds_timeseries.tsv`) or an anatomical T1w brain mask. The implemented LS-S beta estimation deconvolves overlapping trial BOLD signals using the canonical SPM HRF and high-pass DCT drift basis, but residual motion variance remains un-regressed until fMRIPrep derivatives are ingested.
2. **Pilot Sample Size**: The 20 shared stimuli are intended for software engineering, tensor verification, and pipeline testing. They do not constitute a statistically powered machine learning benchmark.
3. **Data Licensing & Privacy**: The OpenNeuro source datasets are licensed under CC0 (Public Domain Dedication). Raw files must remain read-only and outside Git version control.
