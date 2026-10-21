"""
Neuro-GRAPE Pipeline Execution Script:
1. Preprocess EEG epochs (20 shared stimuli) using MNE.
2. Preprocess MEG epochs (20 shared stimuli) using MNE.
3. Inspect and extract fMRI trial windows using NiBabel.
4. Export data/splits/pilot_splits.csv.
5. Export data/manifests/data_inventory.csv and data_inventory.csv.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src is on path
REPO_ROOT = Path(r"D:\NEURO_GRAPE")
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import (
    preprocess_eeg_epochs,
    preprocess_meg_epochs,
    inspect_fmri_header,
    extract_fmri_trial_windows,
    make_sample_id,
)

HANDOVER = REPO_ROOT / "NeuroGRAPE_handover"
SHARED_CSV = HANDOVER / "manifests" / "shared_stimuli.csv"
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFESTS_DIR = DATA_DIR / "manifests"
SPLITS_DIR = DATA_DIR / "splits"

MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load shared stimuli
df_shared = pd.read_csv(SHARED_CSV)
print(f"Loaded {len(df_shared)} shared stimuli from {SHARED_CSV}")

# 2. Define Leakage-Proof Splits
# 20 stimuli -> 12 Train (60%), 4 Val (20%), 4 Test (20%)
# All occurrences of any stimulus are bound to the exact same split
train_stims = [
    "banner_02s.jpg", "bear_02s.jpg", "beet_02s.jpg", "boa_02s.jpg",
    "cucumber_02s.jpg", "glass_02s.jpg", "hamburger_02s.jpg", "hay_02s.jpg",
    "inhaler_02s.jpg", "lampshade_02s.jpg", "microscope_02s.jpg", "pea_02s.jpg"
]
val_stims = [
    "pinwheel_02s.jpg", "puck_02s.jpg", "screwdriver_02s.jpg", "sprinkler_02n.jpg"
]
test_stims = [
    "tadpole_02s.jpg", "tarp_02s.jpg", "wire_02s.jpg", "woman_02s.jpg"
]

split_map = {}
for s in train_stims:
    split_map[s] = "train"
for s in val_stims:
    split_map[s] = "val"
for s in test_stims:
    split_map[s] = "test"

# Build pilot_splits.csv
splits_records = []
for idx, row in df_shared.iterrows():
    stim = row["stimulus_filename"]
    cat_id = int(row["meg_category_nr"])
    cat_name = stim.split("_")[0]
    splits_records.append({
        "stimulus_id": stim,
        "category_id": cat_id,
        "category_name": cat_name,
        "split": split_map[stim],
        "eeg_subject": row["eeg_subject"],
        "meg_subject": row["meg_subject"],
        "fmri_subject": row["fmri_subject"],
    })

df_splits = pd.DataFrame(splits_records)
splits_csv_path = SPLITS_DIR / "pilot_splits.csv"
df_splits.to_csv(splits_csv_path, index=False)
print(f"Saved splits to {splits_csv_path} (Counts: {df_splits['split'].value_counts().to_dict()})")

# 3. Preprocess EEG
print("\n--- Processing EEG Epochs ---")
eeg_vhdr = HANDOVER / "raw" / "EEG" / "sub-01" / "eeg" / "sub-01_task-rsvp_eeg.vhdr"
eeg_events_tsv = HANDOVER / "raw" / "EEG" / "sub-01" / "eeg" / "sub-01_task-rsvp_events.tsv"
df_eeg_events = pd.read_csv(eeg_events_tsv, sep="\t")
eeg_out_dir = PROCESSED_DIR / "eeg"

eeg_epochs, eeg_times, eeg_meta = preprocess_eeg_epochs(
    vhdr_path=eeg_vhdr,
    events_df=df_eeg_events,
    stimulus_manifest=df_shared,
    tmin=-0.1,
    tmax=0.8,
    baseline=(-0.1, 0.0),
    resample_freq=250.0,  # Memory-conscious downsampling from 1000 Hz to 250 Hz (226 timepoints)
    output_dir=eeg_out_dir,
)
print(f"EEG Epochs extracted: {len(eeg_epochs)} samples saved to {eeg_out_dir}")
sample_eeg_arr = next(iter(eeg_epochs.values()))
print(f"EEG epoch shape: {sample_eeg_arr.shape}, dtype: {sample_eeg_arr.dtype}, sfreq: {eeg_meta['sfreq']} Hz")

# 4. Preprocess MEG
print("\n--- Processing MEG Epochs ---")
meg_ds = HANDOVER / "raw" / "MEG" / "sub-BIGMEG2" / "ses-07" / "meg" / "sub-BIGMEG2_ses-07_task-main_run-06_meg.ds"
sample_attr = HANDOVER / "metadata" / "sample_attributes_P2.csv"
meg_out_dir = PROCESSED_DIR / "meg"

meg_epochs, meg_times, meg_meta = preprocess_meg_epochs(
    meg_ds_path=meg_ds,
    sample_attributes_path=sample_attr,
    stimulus_manifest=df_shared,
    session_nr=7,
    run_nr=6,
    tmin=-0.1,
    tmax=0.8,
    baseline=(-0.1, 0.0),
    resample_freq=250.0,  # Memory-conscious downsampling from 1200 Hz to 250 Hz (226 timepoints)
    output_dir=meg_out_dir,
)
print(f"MEG Epochs extracted: {len(meg_epochs)} samples saved to {meg_out_dir}")
sample_meg_arr = next(iter(meg_epochs.values()))
print(f"MEG epoch shape: {sample_meg_arr.shape}, dtype: {sample_meg_arr.dtype}, sfreq: {meg_meta['sfreq']} Hz")

# 5. Process fMRI
print("\n--- Processing fMRI Header & Pilot Windows ---")
fmri_nii = HANDOVER / "raw" / "fMRI" / "sub-01" / "ses-things08" / "func" / "sub-01_ses-things08_task-things_run-10_bold.nii.gz"
fmri_events_tsv = HANDOVER / "raw" / "fMRI" / "sub-01" / "ses-things08" / "func" / "sub-01_ses-things08_task-things_run-10_events.tsv"
df_fmri_events = pd.read_csv(fmri_events_tsv, sep="\t")
fmri_hdr = inspect_fmri_header(fmri_nii)
print(f"fMRI header: shape={fmri_hdr['shape']}, TR={fmri_hdr['tr_sec']}s, voxel_size={fmri_hdr['voxel_size_mm']}")

# Option A: Raw sub-volume slices
fmri_out_dir = PROCESSED_DIR / "fmri"
fmri_windows, fmri_meta = extract_fmri_trial_windows(
    bold_nii_path=fmri_nii,
    events_df=df_fmri_events,
    stimulus_manifest=df_shared,
    window_tr=(0, 4),  # 4 TRs (0 to 6.0s post onset)
    output_dir=fmri_out_dir,
)
print(f"fMRI trial windows extracted: {len(fmri_windows)} samples saved to {fmri_out_dir}")

# Option B: LS-S Single-trial GLM Beta Volumes
fmri_beta_out_dir = PROCESSED_DIR / "fmri_beta"
from src.data.preprocessing import estimate_fmri_lss_betas
fmri_betas, fmri_beta_meta = estimate_fmri_lss_betas(
    bold_nii_path=fmri_nii,
    events_df=df_fmri_events,
    stimulus_manifest=df_shared,
    output_dir=fmri_beta_out_dir,
)
print(f"fMRI LS-S beta maps extracted: {len(fmri_betas)} samples saved to {fmri_beta_out_dir}")

# 6. Build Master Data Inventory
print("\n--- Building Master data_inventory.csv ---")
inventory_records = []

# EEG records
for idx, row in df_shared.iterrows():
    stim = row["stimulus_filename"]
    cat_id = int(row["meg_category_nr"])
    cat_name = stim.split("_")[0]
    sample_id = make_sample_id("eeg", "sub-01", "task-rsvp", "run-01", idx, stim)
    rel_path = f"data/processed/eeg/{sample_id}.npy"
    inventory_records.append({
        "sample_id": sample_id,
        "modality": "eeg",
        "subject_id": "sub-01",
        "session_id": "task-rsvp",
        "run_id": "run-01",
        "trial_id": idx,
        "stimulus_id": stim,
        "category_id": cat_id,
        "category_name": cat_name,
        "exemplar_id": 2,
        "trial_type": "exp",
        "split": split_map[stim],
        "onset_sec": float(row["eeg_onset"]) / 1000.0,
        "onset_sample": int(row["eeg_onset"]),
        "duration_sec": 0.05,
        "sampling_rate": float(eeg_meta["sfreq"]),
        "signal_units": eeg_meta["signal_units"],
        "data_path": rel_path,
        "raw_window_path": rel_path,
        "raw_source_path": "NeuroGRAPE_handover/raw/EEG/sub-01/eeg/sub-01_task-rsvp_eeg.eeg",
    })

# MEG records
for idx, row in df_shared.iterrows():
    stim = row["stimulus_filename"]
    cat_id = int(row["meg_category_nr"])
    cat_name = stim.split("_")[0]
    sample_id = make_sample_id("meg", "sub-BIGMEG2", "ses-07", "run-06", idx, stim)
    rel_path = f"data/processed/meg/{sample_id}.npy"
    onset_sec = float(row["meg_onset"])
    inventory_records.append({
        "sample_id": sample_id,
        "modality": "meg",
        "subject_id": "sub-BIGMEG2",
        "session_id": "ses-07",
        "run_id": "run-06",
        "trial_id": idx,
        "stimulus_id": stim,
        "category_id": cat_id,
        "category_name": cat_name,
        "exemplar_id": 2,
        "trial_type": "exp",
        "split": split_map[stim],
        "onset_sec": onset_sec,
        "onset_sample": int(round(onset_sec * 1200.0)),
        "duration_sec": 0.5,
        "sampling_rate": float(meg_meta["sfreq"]),
        "signal_units": meg_meta["signal_units"],
        "data_path": rel_path,
        "raw_window_path": rel_path,
        "raw_source_path": "NeuroGRAPE_handover/raw/MEG/sub-BIGMEG2/ses-07/meg/...meg.meg4",
    })

# fMRI records (Default features: 3D LS-S Beta maps)
for idx, row in df_shared.iterrows():
    stim = row["stimulus_filename"]
    cat_id = int(row["meg_category_nr"])
    cat_name = stim.split("_")[0]
    sample_id = make_sample_id("fmri", "sub-01", "ses-things08", "run-10", idx, stim)
    beta_path = f"data/processed/fmri_beta/{sample_id}.npy"
    window_path = f"data/processed/fmri/{sample_id}.npy"
    onset_sec = float(row["fmri_onset"])
    inventory_records.append({
        "sample_id": sample_id,
        "modality": "fmri",
        "subject_id": "sub-01",
        "session_id": "ses-things08",
        "run_id": "run-10",
        "trial_id": idx,
        "stimulus_id": stim,
        "category_id": cat_id,
        "category_name": cat_name,
        "exemplar_id": 2,
        "trial_type": "exp",
        "split": split_map[stim],
        "onset_sec": onset_sec,
        "onset_sample": int(round(onset_sec / 1.5)),
        "duration_sec": 0.5,
        "sampling_rate": 1.0 / 1.5,
        "signal_units": "beta_weight",
        "data_path": beta_path,
        "raw_window_path": window_path,
        "raw_source_path": "NeuroGRAPE_handover/raw/fMRI/sub-01/ses-things08/func/...bold.nii.gz",
    })

df_inventory = pd.DataFrame(inventory_records)
inv_path1 = MANIFESTS_DIR / "data_inventory.csv"
inv_path2 = REPO_ROOT / "data_inventory.csv"
df_inventory.to_csv(inv_path1, index=False)
df_inventory.to_csv(inv_path2, index=False)
print(f"Saved master inventory to {inv_path1} and {inv_path2}")
print(f"Total inventory rows: {len(df_inventory)} (Unique sample_ids: {df_inventory['sample_id'].nunique()})")
