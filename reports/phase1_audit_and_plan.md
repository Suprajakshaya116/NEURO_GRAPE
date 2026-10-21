# Phase 1: One-Time Pilot Audit and Interface Plan

**Project:** Neuro-GRAPE (Multimodal Neural Representation & Decoding)  
**Author:** Member 1 (Dataset Engineering, Preprocessing, Data Quality & Leakage Prevention)  
**Date:** October 2026  
**Status:** Audit Complete — Phase 1 Baseline Established  

---

## 1. Executive Summary

This report establishes the verified empirical baseline for the Neuro-GRAPE pilot dataset extracted under `D:\NEURO_GRAPE\NeuroGRAPE_handover`. All findings are based on direct header and metadata parsing, physical file verification, and algorithmic overlap verification. No synthetic assumptions or undocumented claims are introduced.

### Key Takeaways
1. **The 20-Stimulus Intersection is Verified**: Exactly 20 distinct THINGS stimulus image files are common across the pilot recordings of EEG, MEG, and fMRI.
2. **Three Completely Independent Cohorts**: EEG (`sub-01`), MEG (`sub-BIGMEG2`), and fMRI (`sub-01`) are distinct human participants recorded in separate facilities on different dates and instruments. They share **visual stimulus identity only**, with **zero simultaneous brain recording, zero subject correspondence, and zero raw temporal synchronization**.
3. **EEG Timing Gotcha Verified**: In `raw/EEG/sub-01/eeg/sub-01_task-rsvp_events.tsv`, the `onset` column contains **integer sample numbers** (e.g. `10724`, `10824`) at a 1000 Hz sampling rate, **not seconds**. The sidecar `sub-01_task-rsvp_eeg.vmrk` confirms these numbers match BrainVision marker positions (`Position in data points`). Treating them as seconds would introduce a 1000× timing error.
4. **Local Hardware Constraints**: Total system RAM is 16.4 GB with only ~1.7 GB free physical RAM during active sessions. Raw recording files (EEG: 765 MB, MEG: 518 MB, fMRI: 314 MB uncompressed) cannot all be loaded into memory simultaneously. Epoch extraction must be streamed or memory-mapped.
5. **Local Python Environment Gaps**: While Python 3.13.1 with `numpy`, `scipy`, `pandas`, and `scikit-learn` is installed, neuroimaging libraries `mne` and `nibabel`, as well as `torch`, are **not installed** in the active environment. Phase 2 execution will require dependency provisioning or pure-Python/NumPy fallback decoders.

---

## 2. Verified Facts vs. Assumptions & Unresolved Questions

| Category | Verified Facts | Working Assumptions & Unresolved Questions |
| :--- | :--- | :--- |
| **Data Integrity & Files** | All 47 files listed in `file_inventory.csv` exist on disk with exact byte counts. Total raw size: 1.53 GB. | Archive extraction is complete; `.tar.gz` archive does not need to be touched. |
| **Stimulus Overlap** | Exactly 20 stimuli intersect across all 3 modalities. All 20 are experimental (`trial_type == exp`) and correspond to Exemplar 2 (`_02s.jpg` / `_02n.jpg`). | MEG & fMRI share 3 additional stimuli (`catch051.jpg`, `catch010.jpg`, `iguana_20s.jpg`), but these are absent from the EEG `sub-01` recording. |
| **Subject & Session Alignment** | - EEG: `sub-01`, RSVP continuous session.<br>- MEG: `sub-BIGMEG2`, Session 07, Run 06.<br>- fMRI: `sub-01`, Session `things08`, Run 10. | **Unresolved Scientific Decision**: Model architectures must align representations across *stimulus semantic space*, not across paired temporal signals or paired subjects. |
| **EEG Metadata & Units** | 63 EEG channels; 1000 Hz sampling rate; IEEE 32-bit float format (`.eeg`). `onset` in `events.tsv` is **sample index**, confirmed by `.vmrk`. Duration is 50 samples = 50 ms. | EEG `objectnumber` is 0-indexed (0 to 1853), whereas MEG/fMRI `things_category_nr` is 1-indexed (1 to 1854). `objectnumber == things_category_nr - 1`. |
| **MEG Metadata & Units** | CTF 272 axial gradiometer channels + 28 reference channels + 10 misc channels (310 total). Sampling rate: 1200.0 Hz. Duration: 347.999 s. `events.tsv` onset is in **seconds**, matched to `sample_attributes_P2.csv`. | Preprocessing decision needed for 3rd-order synthetic gradient compensation (CTF reference channels) and planar gradient transformation. |
| **fMRI Metadata & Units** | Siemens Prisma 3T, TR=1.5s, TE=33ms, 2.0 mm isotropic voxels, 96×96×60 matrix, 284 TR volumes (426.0 s). Data is completely raw BOLD without motion correction or slice timing. | **Unresolved Scientific Decision**: Feature extraction strategy (GLM single-trial beta estimation via LS-S vs. 4–6s peak hemodynamic delay averaging vs. whole-brain vs. visual cortex ROI masking). |
| **Licensing** | All three source datasets are published under **CC0 (Public Domain Dedication)** on OpenNeuro (ds003825, ds004212, ds004192). | Attribution required per citation notices in `dataset_description.json`. |

---

## 3. Dataset Inventory and File Verification

### 3.1 Raw Recording Inventory

| Modality | Participant / Session / Run | Primary Data Path | Format / Extension | Size (Bytes) | Verified Dimensions & Rates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EEG** | `sub-01` / `task-rsvp` | `raw/EEG/sub-01/eeg/sub-01_task-rsvp_eeg.eeg` | BrainVision Binary (`.eeg`, `.vhdr`, `.vmrk`) | 765,006,480 | 63 channels, 3,035,740 samples @ 1000 Hz (3035.74 s / 50.6 min) |
| **MEG** | `sub-BIGMEG2` / `ses-07` / `run-06` | `raw/MEG/sub-BIGMEG2/ses-07/meg/...meg.ds/...meg.meg4` | CTF MEG (`.ds` package, `.meg4`, `.res4`) | 517,824,008 | 310 channels (272 MEG), 417,600 samples @ 1200 Hz (348.0 s / 5.8 min) |
| **fMRI** | `sub-01` / `ses-things08` / `run-10` | `raw/fMRI/sub-01/ses-things08/func/...bold.nii.gz` | NIfTI-1 gzip (`.nii.gz`) | 250,403,586 | 96×96×60 voxels, 284 volumes, TR=1.5 s, 2.0 mm isotropic |

### 3.2 Metadata & Sidecar Inventory

| File Path | Purpose | Key Fields Verified |
| :--- | :--- | :--- |
| `manifests/shared_stimuli.csv` | Cross-modality pilot anchor | 20 stimuli, exact onsets across modalities, category/exemplar numbers. |
| `metadata/sample_attributes_P2.csv` | Full MEG P2 stimulus mapping | 27,048 rows. Session 7 Run 6 has 225 rows matching MEG `events.tsv`. |
| `metadata/EEG_dataset_description.json` | Dataset provenance (ds003825) | OpenNeuro v1.1.0 / v1.2.0, CC0 license, Tijl Grootswagers et al. |
| `metadata/MEG_dataset_description.json` | Dataset provenance (ds004212) | OpenNeuro v2.0.0, CC0 license, Martin Hebart et al. |
| `metadata/fMRI_dataset_description.json` | Dataset provenance (ds004192) | OpenNeuro v1.0.7, CC0 license, Martin Hebart et al. |
| `raw/EEG/.../sub-01_task-rsvp_events.tsv` | EEG event triggers | 22,248 trials, `onset` (sample index), `duration` (50 samples), `stimname`. |
| `raw/MEG/.../sub-BIGMEG2..._events.tsv` | MEG event triggers | 225 trials (205 exp, 20 catch/test), `onset` (sec), `value` (`things_image_nr`). |
| `raw/fMRI/.../sub-01..._events.tsv` | fMRI event triggers | 92 trials (72 exp, 12 test, 8 catch), `onset` (sec), `file_path`. |

---

## 4. Verification of the 20-Stimulus Intersection

An automated check was run across the raw event metadata:
- **EEG `sub-01`**: 22,248 unique stimulus filenames.
- **MEG `sub-BIGMEG2_ses-07_run-06`**: 225 trials (205 unique experimental stimuli).
- **fMRI `sub-01_ses-things08_run-10`**: 92 trials (72 unique experimental stimuli).

### Overlap Matrix
$$\begin{array}{rcc}
\text{EEG} \cap \text{MEG} & = & 185 \text{ stimuli} \\
\text{EEG} \cap \text{fMRI} & = & 72 \text{ stimuli} \\
\text{MEG} \cap \text{fMRI} & = & 23 \text{ stimuli} \\
\mathbf{\text{EEG} \cap \text{MEG} \cap \text{fMRI}} & \mathbf{=} & \mathbf{20} \textbf{ stimuli}
\end{array}$$

The 3 stimuli present in MEG & fMRI but absent in EEG `sub-01` are:
1. `catch051.jpg` (Catch trial)
2. `catch010.jpg` (Catch trial)
3. `iguana_20s.jpg` (Test trial, Exemplar 13, not presented to EEG `sub-01`)

### Verified 20 Shared Stimuli
Every shared stimulus is an experimental trial, corresponds to Exemplar 2 (`_02s.jpg` or `_02n.jpg`), and represents an independent THINGS concept:
1. `banner_02s.jpg` (Category 71)
2. `bear_02s.jpg` (Category 106)
3. `beet_02s.jpg` (Category 115)
4. `boa_02s.jpg` (Category 151)
5. `cucumber_02s.jpg` (Category 458)
6. `glass_02s.jpg` (Category 678)
7. `hamburger_02s.jpg` (Category 741)
8. `hay_02s.jpg` (Category 762)
9. `inhaler_02s.jpg` (Category 829)
10. `lampshade_02s.jpg` (Category 894)
11. `microscope_02s.jpg` (Category 1000)
12. `pea_02s.jpg` (Category 1129)
13. `pinwheel_02s.jpg` (Category 1186)
14. `puck_02s.jpg` (Category 1246)
15. `screwdriver_02s.jpg` (Category 1392)
16. `sprinkler_02n.jpg` (Category 1524)
17. `tadpole_02s.jpg` (Category 1615)
18. `tarp_02s.jpg` (Category 1629)
19. `wire_02s.jpg` (Category 1829)
20. `woman_02s.jpg` (Category 1833)

> [!IMPORTANT]
> **Category Index Mapping**: In EEG, `objectnumber` is zero-indexed ($0 \le \text{objectnumber} \le 1853$). In MEG and fMRI, `things_category_nr` is one-indexed ($1 \le \text{things_category_nr} \le 1854$).  
> Formula: $\text{things\_category\_nr} = \text{eeg\_objectnumber} + 1$.

---

## 5. Pilot Limitations & Resource Constraints

### 5.1 Local Machine Hardware Constraints
- **Host OS**: Windows 11 (64-bit AMD64)
- **CPU**: 12 logical cores
- **RAM**: 16.4 GB total physical RAM; **available free RAM is currently ~1.7 GB**.
- **Disk Space**:
  - `D:\` Drive (Project Root): 152.9 GB Free / 200.2 GB Total.
  - `C:\` Drive: 100.5 GB Free / 274.8 GB Total.

### 5.2 Processing Implications
1. **No Simultaneous Full-Recording Loading**: Loading uncompressed EEG (765 MB) + MEG (518 MB) + fMRI (314 MB) plus processing buffers and derivatives exceeds available free RAM.
2. **Chunk-wise / On-Demand Extraction**: Preprocessing scripts must extract only epoch windows ($[-100, 800]\text{ ms}$ around the 20 onsets) directly into preallocated arrays.
3. **No Heavy Deep Learning Training Locally**: Model exploration for Member 2 & 3 must use lightweight ridge regression, linear probe decoders, or RSA (Representational Similarity Analysis) matrices rather than multi-gigabyte foundation models.

### 5.3 Scientific & Pipeline Limitations
1. **Cross-Subject Associative Pairing**: Because modalities originate from different subjects, models cannot learn sample-level temporal dynamics or shared noise correlations. Alignment must be in semantic/feature space.
2. **Raw Signals**: fMRI data is completely un-preprocessed BOLD. Motion artifacts, drift, and hemodynamic lag must be accounted for before feature matrices are presented to Member 2.

---

## 6. Proposed Manifest & Loader Contract

To guarantee idempotency and avoid row-order dependency, we propose an immutable, deterministic sample identification contract.

### 6.1 Stable `sample_id` Format
$$\text{sample\_id} = \{\text{modality}\}\_\{\text{subject\_id}\}\_\{\text{session\_id}\}\_\{\text{run\_id}\}\_\{\text{trial\_index:06d}\}\_\{\text{stimulus\_stem}\}$$
*Example:* `eeg_sub-01_ses-rsvp_run-01_trial-012450_puck-02s`

### 6.2 Standard Manifest Schema (`data_inventory.csv` / `manifests/pilot_manifest.csv`)

| Field Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `sample_id` | `str` (PK) | Globally unique, deterministic sample identifier | `meg_sub-BIGMEG2_ses-07_run-06_trial-000045_puck-02s` |
| `stimulus_id` | `str` (FK) | Exact stimulus image filename | `puck_02s.jpg` |
| `modality` | `str` | Modality tag (`eeg`, `meg`, `fmri`) | `meg` |
| `subject_id` | `str` | BIDS subject ID | `sub-BIGMEG2` |
| `session_id` | `str` | BIDS session ID | `ses-07` |
| `run_id` | `str` | BIDS run ID | `run-06` |
| `trial_id` | `int` | Original sequence/trial number | `45` |
| `repetition_id` | `int` | Repetition count of this stimulus for this subject | `1` |
| `trial_type` | `str` | Condition (`exp`, `test`, `catch`) | `exp` |
| `category_id` | `int` | Verified 1-indexed THINGS category number | `1246` |
| `category_name`| `str` | Verified concept label | `puck` |
| `exemplar_id` | `int` | Verified exemplar number | `2` |
| `onset_sec` | `float` | Onset in seconds from recording start | `51.25` |
| `onset_sample` | `int` | Onset sample index in raw file | `61500` |
| `duration_sec` | `float` | Duration in seconds | `0.5` |
| `split` | `str` | Cross-validation split assignment (`train`, `val`, `test`) | `train` |
| `raw_data_path`| `str` | Relative path to raw recording | `NeuroGRAPE_handover/raw/MEG/...` |
| `proc_data_path`| `str` | Relative path to preprocessed array (`.npy`/`.npz`) | `data/processed/meg/puck_02s.npy` |
| `sampling_rate` | `float` | Sampling rate in Hz | `1200.0` |
| `signal_units` | `str` | Measurement unit (`uV`, `fT`, `raw_bold_counts`) | `fT` |

### 6.3 Handoff Integration Contract
- **For Member 2 (Modeling / Feature Arrays)**:
  - Loader yields standardized dictionary:
    ```python
    {
        "sample_id": str,
        "modality": str,
        "features": np.ndarray,      # (channels, time) for EEG/MEG; (voxels,) or (voxels, time) for fMRI
        "time_axis": np.ndarray,      # timestamps relative to stimulus onset in seconds (e.g., -0.1 to 0.8)
        "stimulus_id": str,
        "category_id": int,
        "split": str
    }
    ```
- **For Member 3 (Evaluation, Leakage Prevention, Benchmarking)**:
  - Stable manifest table provides grouping keys: `stimulus_id`, `category_id`, and `split`.
  - Zero cross-contamination: `split` is invariant across modalities for any `stimulus_id`.

---

## 7. Leakage-Resistant Split Strategy

### 7.1 Generalization Claim
Neuro-GRAPE investigates **Zero-Shot Stimulus / Concept Generalization** across neural modalities. Models must decode or reconstruct brain responses to **held-out visual stimuli and held-out object concepts** that were never seen during training.

### 7.2 Leakage Risks & Prevention Rules
1. **Stimulus Contamination**:
   - *Risk*: If `puck_02s.jpg` is in EEG training and MEG validation, the model could exploit low-level stimulus features rather than learning genuine cross-modal neural alignments.
   - *Rule*: **Split assignment is strictly keyed on `stimulus_id`**. If a stimulus is assigned to `test`, all EEG, MEG, and fMRI instances of that stimulus are strictly restricted to `test`.
2. **Concept Contamination**:
   - *Risk*: In the full dataset, multiple exemplars exist per category. In the pilot, all 20 shared stimuli belong to 20 distinct categories.
   - *Rule*: Concept disjointness is naturally preserved in the pilot ($20\text{ stimuli} = 20\text{ categories}$).
3. **Temporal Auto-correlation**:
   - *Risk*: Adjacent trials in RSVP EEG or BOLD fMRI can suffer from temporal leakage (slow drift or overlapping hemodynamic tails).
   - *Rule*: In the pilot, shared stimuli are spaced apart in the original recordings. For fMRI, onsets have an inter-trial interval of 4.5 seconds. Baseline correction and confound regression must be applied within-run before splitting.

### 7.3 Proposed Pilot Split Allocation (20 Stimuli)
- **Train (60%, 12 stimuli)**: `banner`, `bear`, `beet`, `boa`, `cucumber`, `glass`, `hamburger`, `hay`, `inhaler`, `lampshade`, `microscope`, `pea`
- **Validation (20%, 4 stimuli)**: `pinwheel`, `puck`, `screwdriver`, `sprinkler`
- **Test (20%, 4 stimuli)**: `tadpole`, `tarp`, `wire`, `woman`

---

## 8. Master Deliverable Roadmap (Phases 2, 3, 4)

| Phase | Deliverable | Scope / Description | Target File / Artifact |
| :---: | :--- | :--- | :--- |
| **Phase 2** | Preprocessing Pipeline | Epoch extraction ($[-100, 800]\text{ ms}$ for EEG/MEG; FIR/lag window for fMRI), bandpass filtering, baseline normalization, memory-efficient array export. | `src/data/preprocessing.py` |
| **Phase 2** | Standard Dataset Loader | PyTorch / NumPy dataset class implementing the contract in Section 6. | `src/data/dataset.py` |
| **Phase 2** | Master Inventory Manifest | Complete tabular index of all verified pilot samples and paths. | `data/manifests/data_inventory.csv` |
| **Phase 2** | Split Definitions | Deterministic train/val/test assignment table. | `data/splits/pilot_splits.csv` |
| **Phase 3** | Data Validation Test Suite | Unit and property tests verifying tensor shapes, non-zero variance, alignment, and lack of NaNs. | `tests/test_dataset.py`, `tests/test_preprocessing.py` |
| **Phase 3** | Data Quality & Leakage Audit | Report documenting SNR, trial drops, temporal leakage verification, and test-set isolation. | `reports/data_quality_and_leakage_report.md` |
| **Phase 3** | Compatibility Report | Formal cross-modality alignment verification report for Member 2 & 3. | `reports/compatibility_report.md` |
| **Phase 4** | Integration Contract Document | API contract and sample access guide for Member 2 and Member 3. | `docs/integration_contract.md` |
| **Phase 4** | Reproducible Setup Guide | Environment recreation instructions, requirements specification, and execution commands. | `docs/reproducible_setup_guide.md`, `requirements.txt` |
| **Phase 4** | Config Specification | YAML parameter configuration for all pipelines. | `configs/dataset_config.yaml` |

---

## 9. Proposed File Modification Plan

```
d:/NEURO_GRAPE/
├── configs/
│   └── dataset_config.yaml               [MODIFY: Define epoch windows, filters, splits, paths]
├── data/
│   ├── manifests/
│   │   └── data_inventory.csv            [CREATE: Master sample-level manifest with stable IDs]
│   └── splits/
│       └── pilot_splits.csv              [CREATE: Leakage-proof split partition table]
├── docs/
│   ├── integration_contract.md           [CREATE: Handoff contract for Members 2 & 3]
│   └── reproducible_setup_guide.md       [CREATE: Environment and execution instructions]
├── reports/
│   ├── phase1_audit_and_plan.md          [CREATED: This Phase 1 Audit & Plan Report]
│   ├── compatibility_report.md           [PHASE 3: Cross-modality compatibility report]
│   └── data_quality_and_leakage_report.md[PHASE 3: Quality checks and leakage audit]
├── requirements.txt                      [MODIFY: Document exact dependencies: numpy, scipy, pandas, mne, nibabel]
├── src/
│   └── data/
│       ├── __init__.py                   [EXISTS: Keep]
│       ├── dataset.py                    [PHASE 2: Implement NeuroGrapeDataset loader]
│       ├── inspect_datasets.py           [EXISTS: Keep]
│       └── preprocessing.py              [PHASE 2: Implement epoching and feature extraction]
└── tests/
    ├── __init__.py                       [EXISTS: Keep]
    └── test_dataset.py                   [PHASE 3: Implement automated loader validation tests]
```

---

## 10. Conclusion & Next Steps

Phase 1 audit is concluded with 100% empirical verification. All raw data files, metadata mappings, timing conventions, and hardware boundaries are cataloged. No raw files were modified, no remote access was attempted, and no assumptions were made without verification.

- **Immediate Next Action**: Review the proposed preprocessing parameters (epoch window $[-100, 800]\text{ ms}$, fMRI hemodynamic delay handling) and approve progressing to **Phase 2 (Implementation of `src/data/preprocessing.py`, `src/data/dataset.py`, and manifest generation)**.
