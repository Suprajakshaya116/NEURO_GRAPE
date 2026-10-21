"""
Neuro-GRAPE Preprocessing Pipeline
==================================
Provides memory-conscious, reproducible preprocessing for EEG, MEG, and fMRI pilot recordings.

Key Design Decisions:
- EEG: Reads BrainVision continuous data using MNE. Onsets in events.tsv are verified
  as 1000 Hz sample indices matching .vmrk marker positions.
- MEG: Reads CTF continuous data using MNE. Event onsets are verified in seconds from
  sample_attributes_P2.csv and events.tsv, converted to 1200 Hz samples.
- fMRI: Reads NIfTI-1 header and volume metadata using NiBabel without loading the full
  volume series into memory. Full trial-wise HRF deconvolution is marked as pending.
- Candidate epoch window: [-100, 800] ms with pre-stimulus baseline [-100, 0] ms.
- Output arrays are exported with deterministic, stable sample IDs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Safe import of domain libraries
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def check_dependencies() -> Dict[str, bool]:
    """Return availability of domain-specific dependencies."""
    return {
        "mne": HAS_MNE,
        "nibabel": HAS_NIBABEL,
    }


def make_sample_id(
    modality: str,
    subject_id: str,
    session_id: str,
    run_id: str,
    trial_id: int,
    stimulus_id: str,
) -> str:
    """Generate a stable, unique, deterministic sample identifier.

    Format: {modality}_{subject}_{session}_{run}_trial-{trial:06d}_{stimulus_stem}
    """
    stim_stem = Path(stimulus_id).stem.replace("_", "-")
    sub_clean = subject_id.replace("_", "-")
    ses_clean = session_id.replace("_", "-")
    run_clean = run_id.replace("_", "-")
    return f"{modality.lower()}_{sub_clean}_{ses_clean}_{run_clean}_trial-{trial_id:06d}_{stim_stem}"


# =============================================================================
# EEG Preprocessing
# =============================================================================

def preprocess_eeg_epochs(
    vhdr_path: Union[str, Path],
    events_df: pd.DataFrame,
    stimulus_manifest: pd.DataFrame,
    tmin: float = -0.1,
    tmax: float = 0.8,
    baseline: Optional[Tuple[float, float]] = (-0.1, 0.0),
    resample_freq: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """Epoch EEG continuous recording for selected stimuli using MNE.

    Parameters
    ----------
    vhdr_path : str or Path
        Path to BrainVision .vhdr header file.
    events_df : pd.DataFrame
        Events dataframe from sub-01_task-rsvp_events.tsv.
    stimulus_manifest : pd.DataFrame
        Table containing target 'stimulus_filename' and verified 'eeg_onset' (sample indices).
    tmin : float
        Epoch start time in seconds relative to stimulus onset. Candidate: -0.1.
    tmax : float
        Epoch end time in seconds relative to stimulus onset. Candidate: 0.8.
    baseline : tuple or None
        Baseline interval in seconds (tmin, tmax) for baseline correction.
    resample_freq : float, optional
        Target sampling rate in Hz for downsampling (e.g. 250 Hz).
    output_dir : str or Path, optional
        Directory where .npy arrays will be saved.

    Returns
    -------
    epochs_dict : dict
        Mapping from sample_id to np.ndarray of shape (n_channels, n_times).
    times : np.ndarray
        Time points relative to onset in seconds.
    metadata : dict
        Provenance and recording metadata.
    """
    if not HAS_MNE:
        raise ImportError("MNE is required for EEG preprocessing. Please install mne.")

    raw_eeg = mne.io.read_raw_brainvision(str(vhdr_path), preload=False, verbose=False)
    sfreq = float(raw_eeg.info["sfreq"])
    assert sfreq == 1000.0, f"Expected 1000 Hz sampling rate, got {sfreq}"

    # Build chronologically sorted events array
    event_list = []
    stim_order = []
    for idx, row in stimulus_manifest.iterrows():
        sample = int(row["eeg_onset"])
        stim = row["stimulus_filename"]
        event_list.append((sample, stim, idx))

    # Sort strictly by sample index to ensure MNE Epochs integrity
    event_list.sort(key=lambda x: x[0])

    mne_events = []
    event_id = {}
    for code, (sample, stim, original_idx) in enumerate(event_list, start=1):
        mne_events.append([sample, 0, code])
        event_id[stim] = code
        stim_order.append((stim, original_idx))

    mne_events = np.array(mne_events, dtype=int)

    # Epoch directly from raw without full continuous data load into RAM
    epochs = mne.Epochs(
        raw_eeg,
        mne_events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
    )

    if resample_freq is not None and resample_freq < sfreq:
        epochs = epochs.resample(resample_freq, verbose=False)
        sfreq = resample_freq

    times = epochs.times
    ch_names = epochs.ch_names
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    epochs_dict = {}
    for epoch_idx, (stim, orig_idx) in enumerate(stim_order):
        sample_id = make_sample_id(
            modality="eeg",
            subject_id="sub-01",
            session_id="task-rsvp",
            run_id="run-01",
            trial_id=orig_idx,
            stimulus_id=stim,
        )
        epoch_array = data[epoch_idx].astype(np.float32)
        epochs_dict[sample_id] = epoch_array

        if output_dir is not None:
            np.save(os.path.join(output_dir, f"{sample_id}.npy"), epoch_array)

    meta = {
        "modality": "eeg",
        "sfreq": sfreq,
        "n_channels": len(ch_names),
        "ch_names": ch_names,
        "tmin": tmin,
        "tmax": tmax,
        "baseline": baseline,
        "signal_units": "V",
        "time_units": "sec",
    }
    return epochs_dict, times, meta


# =============================================================================
# MEG Preprocessing
# =============================================================================

def preprocess_meg_epochs(
    meg_ds_path: Union[str, Path],
    sample_attributes_path: Union[str, Path],
    stimulus_manifest: pd.DataFrame,
    session_nr: int = 7,
    run_nr: int = 6,
    tmin: float = -0.1,
    tmax: float = 0.8,
    baseline: Optional[Tuple[float, float]] = (-0.1, 0.0),
    resample_freq: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """Epoch CTF MEG continuous recording for selected stimuli using MNE.

    Parameters
    ----------
    meg_ds_path : str or Path
        Path to CTF .ds directory.
    sample_attributes_path : str or Path
        Path to sample_attributes_P2.csv for timing/exemplar mapping.
    stimulus_manifest : pd.DataFrame
        Table containing target 'stimulus_filename'.
    session_nr : int
        Session number (7 for sub-BIGMEG2 ses-07).
    run_nr : int
        Run number (6 for sub-BIGMEG2 run-06).
    tmin, tmax : float
        Candidate epoch bounds in seconds.
    baseline : tuple or None
        Baseline window.
    resample_freq : float, optional
        Target downsampling rate.
    output_dir : str or Path, optional
        Output directory for exported .npy files.

    Returns
    -------
    epochs_dict : dict
        Mapping from sample_id to np.ndarray of shape (272, n_times).
    times : np.ndarray
    metadata : dict
    """
    if not HAS_MNE:
        raise ImportError("MNE is required for MEG preprocessing. Please install mne.")

    raw_meg = mne.io.read_raw_ctf(str(meg_ds_path), preload=False, verbose=False)
    sfreq = float(raw_meg.info["sfreq"])
    assert sfreq == 1200.0, f"Expected 1200 Hz sampling rate, got {sfreq}"

    # Filter sample attributes for this session and run
    df_sa = pd.read_csv(sample_attributes_path)
    df_sa_run = df_sa[(df_sa["session_nr"] == session_nr) & (df_sa["run_nr"] == run_nr)].copy()
    df_sa_run["basename"] = df_sa_run["image_path"].apply(
        lambda x: os.path.basename(str(x)) if pd.notna(x) else None
    )

    event_list = []
    stim_order = []
    for idx, row in stimulus_manifest.iterrows():
        stim = row["stimulus_filename"]
        match = df_sa_run[df_sa_run["basename"] == stim]
        if match.empty:
            continue
        sa_row = match.iloc[0]
        onset_sec = float(sa_row["onset"])
        sample = int(round(onset_sec * sfreq))
        event_list.append((sample, stim, idx))

    # Sort chronologically by sample index
    event_list.sort(key=lambda x: x[0])

    mne_events = []
    event_id = {}
    for code, (sample, stim, orig_idx) in enumerate(event_list, start=1):
        mne_events.append([sample, 0, code])
        event_id[stim] = code
        stim_order.append((stim, orig_idx))

    mne_events = np.array(mne_events, dtype=int)

    # Pick 272 primary MEG sensor channels (excluding reference magnetometers & misc)
    picks_meg = mne.pick_types(raw_meg.info, meg=True, ref_meg=False, eeg=False, stim=False)

    epochs = mne.Epochs(
        raw_meg,
        mne_events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        picks=picks_meg,
        baseline=baseline,
        preload=True,
        verbose=False,
    )

    if resample_freq is not None and resample_freq < sfreq:
        epochs = epochs.resample(resample_freq, verbose=False)
        sfreq = resample_freq

    times = epochs.times
    ch_names = epochs.ch_names
    data = epochs.get_data()  # shape: (n_epochs, 272, n_times)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    epochs_dict = {}
    for epoch_idx, (stim, orig_idx) in enumerate(stim_order):
        sample_id = make_sample_id(
            modality="meg",
            subject_id="sub-BIGMEG2",
            session_id=f"ses-{session_nr:02d}",
            run_id=f"run-{run_nr:02d}",
            trial_id=orig_idx,
            stimulus_id=stim,
        )
        epoch_array = data[epoch_idx].astype(np.float32)
        epochs_dict[sample_id] = epoch_array

        if output_dir is not None:
            np.save(os.path.join(output_dir, f"{sample_id}.npy"), epoch_array)

    meta = {
        "modality": "meg",
        "sfreq": sfreq,
        "n_channels": len(ch_names),
        "ch_names": ch_names,
        "tmin": tmin,
        "tmax": tmax,
        "baseline": baseline,
        "signal_units": "T",
        "time_units": "sec",
    }
    return epochs_dict, times, meta


# =============================================================================
# fMRI Preprocessing & Inspection
# =============================================================================

def inspect_fmri_header(
    bold_nii_path: Union[str, Path]
) -> Dict[str, Any]:
    """Inspect NIfTI-1 BOLD header without loading the complete volume series.

    Returns metadata dictionary including voxel dimensions, TR, and data type.
    """
    if not HAS_NIBABEL:
        raise ImportError("NiBabel is required for fMRI header inspection.")

    img = nib.load(str(bold_nii_path))
    hdr = img.header
    zooms = hdr.get_zooms()

    return {
        "modality": "fmri",
        "shape": img.shape,
        "ndim": len(img.shape),
        "voxel_size_mm": tuple(float(z) for z in zooms[:3]),
        "tr_sec": float(zooms[3]) if len(zooms) > 3 else None,
        "datatype": str(img.get_data_dtype()),
        "affine": img.affine.tolist(),
    }


def extract_fmri_trial_windows(
    bold_nii_path: Union[str, Path],
    events_df: pd.DataFrame,
    stimulus_manifest: pd.DataFrame,
    window_tr: Tuple[int, int] = (0, 4),
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Extract candidate post-stimulus volume windows for target trials.

    Parameters
    ----------
    bold_nii_path : str or Path
        Path to raw functional BOLD NIfTI file.
    events_df : pd.DataFrame
        Events dataframe with onsets in seconds.
    stimulus_manifest : pd.DataFrame
        Table with target stimulus filenames.
    window_tr : tuple of (int, int)
        Range of TR offsets relative to stimulus onset volume (e.g. 0 to 4 TRs).
    output_dir : str or Path, optional
        Target directory for saved arrays.

    Note:
    Scientific feature extraction (e.g., GLM beta estimation via LS-S) requires
    prior motion correction, slice timing, and high-pass filtering. This function
    provides raw sub-volume slices for pipeline testing while formal GLM modeling
    remains pending.
    """
    if not HAS_NIBABEL:
        raise ImportError("NiBabel is required for fMRI access.")

    img = nib.load(str(bold_nii_path))
    hdr = img.header
    tr = float(hdr.get_zooms()[3])
    data_proxy = img.dataobj  # Memory-mapped on-demand slice access

    events_df = events_df.copy()
    events_df["basename"] = events_df["file_path"].apply(
        lambda x: os.path.basename(str(x)) if pd.notna(x) else None
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    trial_windows = {}
    for idx, row in stimulus_manifest.iterrows():
        stim = row["stimulus_filename"]
        match = events_df[events_df["basename"] == stim]
        if match.empty:
            continue
        ev_row = match.iloc[0]
        onset_sec = float(ev_row["onset"])
        onset_tr = int(round(onset_sec / tr))

        # Extract window of TRs
        start_tr = max(0, onset_tr + window_tr[0])
        end_tr = min(img.shape[3], onset_tr + window_tr[1])

        # Slice dataobj on demand without loading all 284 volumes
        window_data = np.asarray(data_proxy[:, :, :, start_tr:end_tr], dtype=np.float32)

        sample_id = make_sample_id(
            modality="fmri",
            subject_id="sub-01",
            session_id="ses-things08",
            run_id="run-10",
            trial_id=idx,
            stimulus_id=stim,
        )
        trial_windows[sample_id] = window_data

        if output_dir is not None:
            np.save(os.path.join(output_dir, f"{sample_id}.npy"), window_data)

    meta = {
        "modality": "fmri",
        "tr_sec": tr,
        "window_tr": window_tr,
        "signal_units": "raw_bold_counts",
        "hrf_status": "raw_window_fallback",
    }
    return trial_windows, meta


def compute_spm_hrf(tr: float = 1.5, oversampling: int = 10, duration: float = 32.0) -> Tuple[np.ndarray, float]:
    """Compute the canonical SPM double-gamma Hemodynamic Response Function (HRF)."""
    from scipy.special import gamma
    dt = tr / oversampling
    t = np.arange(0, duration, dt)
    a1, b1 = 6.0, 1.0
    a2, b2 = 16.0, 1.0
    c = 1.0 / 6.0

    hrf = (t ** (a1 - 1)) * np.exp(-t / b1) / (gamma(a1) * (b1 ** a1)) - \
          c * ((t ** (a2 - 1)) * np.exp(-t / b2) / (gamma(a2) * (b2 ** a2)))
    return hrf / np.sum(hrf), dt


def make_dct_drift_basis(n_vols: int, tr: float, cutoff_sec: float = 128.0) -> np.ndarray:
    """Create a discrete cosine transform (DCT) basis for high-pass drift filtering."""
    total_time = n_vols * tr
    n_harmonics = int(np.floor(2 * total_time / cutoff_sec))
    t = (np.arange(n_vols) + 0.5) / n_vols
    basis = [np.ones(n_vols)]
    for h in range(1, n_harmonics + 1):
        basis.append(np.sqrt(2) * np.cos(np.pi * h * t))
    return np.column_stack(basis)


def estimate_fmri_lss_betas(
    bold_nii_path: Union[str, Path],
    events_df: pd.DataFrame,
    stimulus_manifest: pd.DataFrame,
    cutoff_sec: float = 128.0,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Estimate trial-wise activation beta maps using Least Squares-Separate (LS-S).

    Parameters
    ----------
    bold_nii_path : str or Path
        Path to raw functional BOLD NIfTI file.
    events_df : pd.DataFrame
        Events dataframe with onsets and durations in seconds.
    stimulus_manifest : pd.DataFrame
        Table with target stimulus filenames.
    cutoff_sec : float, default=128.0
        High-pass filter cutoff for DCT drift regressors.
    output_dir : str or Path, optional
        Target directory for saved arrays.

    Returns
    -------
    betas_dict : dict
        Mapping from sample_id to 3D numpy array of shape (96, 96, 60).
    metadata : dict
        Detailed modeling specifications and documented pilot limitations.
    """
    if not HAS_NIBABEL:
        raise ImportError("NiBabel is required for fMRI GLM beta estimation.")

    img = nib.load(str(bold_nii_path))
    hdr = img.header
    tr = float(hdr.get_zooms()[3])
    bold_4d = np.asarray(img.dataobj, dtype=np.float32)
    spatial_shape = bold_4d.shape[:3]
    n_vols = bold_4d.shape[3]

    # Reshape BOLD to (n_vols, n_voxels)
    Y = bold_4d.reshape(-1, n_vols).T

    # Generate intensity brain mask (excludes outer air / zero regions)
    mean_signal = np.mean(bold_4d, axis=3)
    p99 = float(np.percentile(mean_signal, 99))
    brain_mask = mean_signal > (0.10 * p99)
    in_mask_voxels = np.sum(brain_mask)
    Y_brain = Y[:, brain_mask.ravel()]

    events_df = events_df.copy()
    events_df["basename"] = events_df["file_path"].apply(
        lambda x: os.path.basename(str(x)) if pd.notna(x) else None
    )

    oversampling = 10
    hrf, dt = compute_spm_hrf(tr=tr, oversampling=oversampling)
    time_grid = np.arange(0, n_vols * tr, dt)
    dct_basis = make_dct_drift_basis(n_vols, tr, cutoff_sec=cutoff_sec)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    betas_dict = {}
    for idx, row in stimulus_manifest.iterrows():
        stim = row["stimulus_filename"]
        match = events_df[events_df["basename"] == stim]
        if match.empty:
            continue
        target_idx = match.index[0]
        target_onset = float(match.loc[target_idx, "onset"])
        target_dur = float(match.loc[target_idx, "duration"])

        # Target trial regressor
        boxcar_target = np.zeros_like(time_grid)
        boxcar_target[(time_grid >= target_onset) & (time_grid < target_onset + target_dur)] = 1.0

        # All other trials regressor (LS-S nuisance)
        boxcar_other = np.zeros_like(time_grid)
        for ev_i, ev_r in events_df.iterrows():
            if ev_i == target_idx:
                continue
            boxcar_other[(time_grid >= ev_r["onset"]) & (time_grid < ev_r["onset"] + ev_r["duration"])] = 1.0

        conv_target = np.convolve(boxcar_target, hrf)[:len(time_grid)][::oversampling][:n_vols]
        conv_other = np.convolve(boxcar_other, hrf)[:len(time_grid)][::oversampling][:n_vols]

        X = np.column_stack([conv_target, conv_other, dct_basis])
        pinv_X = np.linalg.pinv(X)

        # Estimate target trial beta for in-mask voxels
        target_beta_brain = pinv_X[0] @ Y_brain

        full_beta_vol = np.zeros(spatial_shape, dtype=np.float32)
        full_beta_vol[brain_mask] = target_beta_brain

        sample_id = make_sample_id(
            modality="fmri",
            subject_id="sub-01",
            session_id="ses-things08",
            run_id="run-10",
            trial_id=idx,
            stimulus_id=stim,
        )
        betas_dict[sample_id] = full_beta_vol

        if output_dir is not None:
            np.save(os.path.join(output_dir, f"{sample_id}.npy"), full_beta_vol)

    meta = {
        "modality": "fmri",
        "method": "least_squares_separate_lss",
        "hrf_model": "spm_canonical_double_gamma",
        "drift_cutoff_sec": cutoff_sec,
        "n_brain_voxels": int(in_mask_voxels),
        "total_voxels": int(np.prod(spatial_shape)),
        "mask_fraction": float(in_mask_voxels / np.prod(spatial_shape)),
        "motion_correction_applied": False,
        "slice_timing_applied": False,
        "signal_units": "beta_weight",
    }
    return betas_dict, meta
