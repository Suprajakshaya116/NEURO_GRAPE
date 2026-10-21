# Neuro-GRAPE Data Quality Report

**Project:** Neuro-GRAPE  
**Author:** Member 1 (Dataset Engineering, Preprocessing, Data Quality & Leakage Prevention)  
**Date:** October 2026  
**Status:** Audit Verified & Completed  

---

## 1. Executive Summary

This report establishes the data quality and signal integrity assessment for the Neuro-GRAPE multimodal pilot dataset across EEG (`sub-01`), MEG (`sub-BIGMEG2`), and fMRI (`sub-01`). All assessments are based on direct empirical inspections of recording headers, event files, and processed feature arrays.

---

## 2. Modality-Specific Quality Audits

### 2.1 EEG Signal Quality (`raw/EEG/sub-01/eeg/`)
- **Format & Instrumentation**: BrainVision continuous 63-channel recording, sampled at 1000 Hz with IEEE 32-bit floating point precision.
- **Hardware Filtering**: Analog anti-aliasing low-pass filter at 280.0 Hz; high-pass DC coupling (0.0 Hz).
- **Channel Integrity**: All 63 channels (`Fp1` through `FC3`) are confirmed present and active. The header specifies zero bad channels (`bads: []`).
- **Timing & Boundary Safety**:
  - Events are verified as integer sample indices at 1000 Hz.
  - The earliest stimulus onset is at sample 25,025 (25.025 s into the run).
  - The latest stimulus onset is at sample 2,920,691 (2920.691 s; 114 s before the run ends at sample 3,035,740).
  - Zero epoch clipping occurs at recording boundaries.
- **Signal Quality & Noise Profile**:
  - Epoched signals exhibit realistic neurophysiological amplitudes: mean $8.96 \times 10^{-6}\text{ V}$ ($\sim 9\ \mu\text{V}$), standard deviation $2.92 \times 10^{-5}\text{ V}$ ($\sim 29\ \mu\text{V}$).
  - Baseline correction over $[-100, 0]\text{ ms}$ reduces pre-stimulus mean offset to $-2.32 \times 10^{-7}\text{ V}$ ($< 1\%$ of signal variance).
  - Anti-aliasing filtering in MNE's polyphase resampler ensures no frequency foldover when downsampling to 250 Hz.

### 2.2 MEG Signal Quality (`raw/MEG/sub-BIGMEG2/ses-07/meg/`)
- **Format & Instrumentation**: CTF 272-channel axial gradiometer system, sampled at 1200.0 Hz.
- **Compensation Grade**: Verified directly from header: **CTF 3rd-order synthetic gradient compensation (`compensation_grade == 3`)** was applied during acquisition using the 28 reference magnetometers/gradiometers.
- **Channel Selection**: Exactly 272 primary axial gradiometers (`MLC11` to `MZR03`) are selected for feature arrays; reference channels and 10 misc/trigger channels are excluded from feature tensors.
- **Timing & Boundary Safety**:
  - Onsets in seconds match `sample_attributes_P2.csv`.
  - Earliest stimulus onset is at 16.15 s (sample 19,380; $>16\text{ s}$ after recording starts).
  - Latest stimulus onset is at 309.75 s (sample 371,700; $>37\text{ s}$ before recording ends at 348.0 s).
  - Zero boundary clipping.
- **Signal Quality & Noise Profile**:
  - Epoched signals exhibit standard biomagnetic field strengths: mean $9.16 \times 10^{-15}\text{ T}$ ($\sim 9\ \text{fT}$), standard deviation $2.21 \times 10^{-13}\text{ T}$ ($\sim 221\ \text{fT}$).
  - Baseline window mean is $-3.29 \times 10^{-16}\text{ T}$ ($\sim 0.15\%$ of signal standard deviation).
  - Downsampling from 1200 Hz to 250 Hz with polyphase low-pass filtering maintains temporal resolution while reducing tensor size by $4.8\times$.

### 2.3 fMRI Signal Quality & Feature Modeling (`raw/fMRI/sub-01/ses-things08/func/`)
- **Format & Acquisition**: Siemens Prisma 3T, $\text{TR} = 1.5\text{ s}$, $\text{TE} = 33\text{ ms}$, matrix $96 \times 96 \times 60$, 2.0 mm isotropic voxels, 284 volumes.
- **Pilot Limitations & Archive Contents**:
  - **Raw BOLD Only**: The pilot archive contains no slice-timing correction, no motion correction parameters, no confound regressors (`confounds_timeseries.tsv`), and no anatomical T1w brain mask.
  - **Spatial Masking**: An empirical intensity threshold ($>10\%$ of 99th percentile mean signal) was constructed, isolating 228,068 intracranial voxels (41.2% of the $96 \times 96 \times 60$ bounding box) from outer air background.
- **Feature Modeling (LS-S Beta Maps vs. Raw Windows)**:
  - *Option 1 (Raw Window)*: 4 TR post-onset window ($[0, 6.0]\text{ s}$, `(96, 96, 60, 4)`). Documented as uncorrected raw BOLD.
  - *Option 2 (LS-S GLM Beta Maps)*: Least Squares-Separate single-trial regression with SPM canonical double-gamma HRF and Discrete Cosine Transform (DCT) drift basis ($>128\text{ s}$ cutoff) yielding 3D activation volumes `(96, 96, 60)`.
  - *Quality Caveat*: While LS-S deconvolves overlapping BOLD tails from adjacent trials (ITI = 4.5 s), residual motion artifacts cannot be modeled without motion parameter logs.

---

## 3. Data Integrity & Verification Checklist

| Quality Dimension | EEG Status | MEG Status | fMRI Status |
| :--- | :---: | :---: | :---: |
| **NaN / Inf Absence** | 100% Finite (0 NaNs) | 100% Finite (0 NaNs) | 100% Finite (0 NaNs) |
| **Boundary Margin** | $>24\text{ s}$ pre, $>114\text{ s}$ post | $>16\text{ s}$ pre, $>37\text{ s}$ post | $>3\text{ s}$ pre, $>13\text{ s}$ post |
| **Baseline Centering** | Centered near $0\ \text{V}$ | Centered near $0\ \text{T}$ | Detrended via DCT basis |
| **Channel Selection** | 63 scalp electrodes | 272 axial gradiometers | 228,068 brain voxels |
| **Motion Regressors** | N/A | N/A | Missing in raw archive |
| **Downsampling** | Anti-alias filtered to 250 Hz | Anti-alias filtered to 250 Hz | N/A (Acquired at $0.667\text{ Hz}$) |
