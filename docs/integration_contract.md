# Neuro-GRAPE Integration Contract
**Author:** Member 1 (Dataset Engineering, Preprocessing, Data Loading, Leakage Prevention)  
**Target Audience:** Member 2 (Model Architecture & Decoders) and Member 3 (Evaluation, GroupDRO, Benchmarking)  
**Date:** October 2026  
**Status:** Validated & Finalized (Phase 4)  

---

## 1. Overview & Core Architectural Guarantees

The Neuro-GRAPE data pipeline provides a unified, leak-free interface to multimodal brain recordings across EEG, MEG, and fMRI.

### Non-Negotiable Contract Guarantees:
1. **Deterministic Identifiers**: Every record is keyed by a globally unique, stable `sample_id`. **Never match or index samples by CSV row order**.
2. **Independent Cohort Semantics**: EEG (`sub-01`), MEG (`sub-BIGMEG2`), and fMRI (`sub-01`) originate from **independent participants and separate experiments**. The alignment is purely associative over visual stimulus identity (`stimulus_id`), not simultaneous biological co-activation.
3. **Representational Heterogeneity**:
   - EEG and MEG represent temporal epoched dynamics: `(channels, timepoints)`.
   - fMRI represents spatial activation maps: `(X, Y, Z)` voxel volumes.
   - **Do not attempt to stack EEG, MEG, and fMRI into a single tensor**. Cross-modal fusion requires modality-specific encoders to project heterogeneous representations into a shared latent space.
4. **Leakage-Proof Partitions**: Cross-validation splits are partitioned strictly by `stimulus_id`. All modal instances of a stimulus belong to the exact same split (`train`, `val`, or `test`).

---

## 2. Handoff for Member 2 (Model Architecture, Encoders & Feature Loading)

Member 2 consumes standardized arrays via `src.data.dataset.NeuroGrapeDataset`.

### 2.1 Executable Example: Loading Modality Samples and Batches

```python
from pathlib import Path
from src.data.dataset import NeuroGrapeDataset

# Path to master inventory (root of workspace)
manifest_path = Path("data_inventory.csv")

# -------------------------------------------------------------
# 1. EEG: Load training split (channels=63, time=225 @ 250 Hz)
# -------------------------------------------------------------
eeg_dataset = NeuroGrapeDataset(manifest=manifest_path, modality="eeg", split="train")
print(f"EEG Train Samples: {len(eeg_dataset)}")

sample_eeg = eeg_dataset[0]
print("Sample ID:     ", sample_eeg["sample_id"])       # 'eeg_sub-01_task-rsvp_run-01_trial-000000_banner-02s'
print("Stimulus ID:   ", sample_eeg["stimulus_id"])     # 'banner_02s.jpg'
print("EEG Tensor:    ", sample_eeg["features"].shape)  # (63, 225)
print("Units:         ", sample_eeg["signal_units"])    # 'V' (Volts)
print("Sampling Rate: ", sample_eeg["sampling_rate"])   # 250.0 Hz

# Batch convenience: stack all 12 training EEG samples into (N, C, T)
eeg_batch_features, eeg_batch_meta = eeg_dataset.get_modality_batch("eeg", split="train")
print("Stacked EEG batch: ", eeg_batch_features.shape)  # (12, 63, 225)

# -------------------------------------------------------------
# 2. MEG: Load training split (channels=272, time=225 @ 250 Hz)
# -------------------------------------------------------------
meg_dataset = NeuroGrapeDataset(manifest=manifest_path, modality="meg", split="train")
meg_batch_features, _ = meg_dataset.get_modality_batch("meg", split="train")
print("Stacked MEG batch: ", meg_batch_features.shape)  # (12, 272, 225)

# -------------------------------------------------------------
# 3. fMRI: Load trial-wise 3D activation beta volumes (96, 96, 60)
# -------------------------------------------------------------
fmri_dataset = NeuroGrapeDataset(manifest=manifest_path, modality="fmri", split="train")
sample_fmri = fmri_dataset[0]
print("fMRI Beta Map: ", sample_fmri["features"].shape) # (96, 96, 60)
print("Units:         ", sample_fmri["signal_units"])   # 'beta_weight'
```

### 2.2 Feature Specifications & Preprocessing Lineage

| Modality | Stored Tensor Shape | Sampling Rate | Physical Units | Preprocessing History |
| :--- | :--- | :--- | :--- | :--- |
| **EEG** | `(63, 225)` | 250 Hz | Volts ($\text{V}$) | BrainVision raw continuous $\rightarrow$ MNE Epoching $[-100, 800]\text{ ms}$ $\rightarrow$ Baseline $[-100, 0]\text{ ms}$ $\rightarrow$ Polyphase low-pass filtered and resampled from 1000 Hz to 250 Hz. |
| **MEG** | `(272, 225)` | 250 Hz | Tesla ($\text{T}$) | CTF raw continuous with 3rd-order synthetic gradient compensation (`grade=3`) $\rightarrow$ 272 axial gradiometers picked $\rightarrow$ MNE Epoching $[-100, 800]\text{ ms}$ $\rightarrow$ Baseline $[-100, 0]\text{ ms}$ $\rightarrow$ Polyphase resampled from 1200 Hz to 250 Hz. |
| **fMRI** | `(96, 96, 60)` | $0.667\text{ Hz}$ ($\text{TR}=1.5\text{ s}$) | Beta weight | Siemens Prisma 3T raw BOLD $\rightarrow$ Least Squares-Separate (LS-S) single-trial GLM (Mumford et al., 2012) $\rightarrow$ SPM canonical double-gamma HRF $\rightarrow$ 7-harmonic DCT drift basis ($>128\text{ s}$) $\rightarrow$ Intracranial intensity mask (228,068 voxels). |

---

## 3. Handoff for Member 3 (Evaluation, Metadata, Groups, & Splits)

Member 3 consumes metadata, group keys, and splits to evaluate generalization without leakage.

### 3.1 Metadata Access & Split Inspections

```python
import pandas as pd
from src.data.dataset import NeuroGrapeDataset

ds = NeuroGrapeDataset(manifest="data_inventory.csv")
meta_df = ds.get_metadata_table()

# Inspect stable identifiers and split allocations
print(meta_df[["sample_id", "modality", "stimulus_id", "category_id", "category_name", "split"]].head())

# Cross-modal pairing query for a given visual stimulus
puck_data = ds.get_cross_modal_sample("puck_02s.jpg")
print("EEG puck shape:  ", puck_data["eeg"]["features"].shape)   # (63, 225)
print("MEG puck shape:  ", puck_data["meg"]["features"].shape)   # (272, 225)
print("fMRI puck shape: ", puck_data["fmri"]["features"].shape)  # (96, 96, 60)
```

### 3.2 Split Rules & Pilot Generalization Scope
- **Partition Rule**: Partitions are keyed strictly on `stimulus_id`. All modal instances of a stimulus belong to the exact same split.
- **Split Allocation (20 stimuli)**:
  - **Train (12 stimuli / 36 samples)**: `banner`, `bear`, `beet`, `boa`, `cucumber`, `glass`, `hamburger`, `hay`, `inhaler`, `lampshade`, `microscope`, `pea`
  - **Val (4 stimuli / 12 samples)**: `pinwheel`, `puck`, `screwdriver`, `sprinkler`
  - **Test (4 stimuli / 12 samples)**: `tadpole`, `tarp`, `wire`, `woman`
- **Scope & Limitations for Member 3**:
  - The 20-stimulus pilot demonstrates software interface compatibility, data loading, and zero-leakage mechanics.
  - **It is NOT sufficient for claiming statistically reliable worst-group generalization or GroupDRO benchmarks**.
  - Group labels (e.g. animate vs. inanimate, natural vs. artificial) can be derived from THINGS semantic metadata in later phases, but GroupDRO is not yet trained or evaluated in Member 1's data layer.

---

## 4. PyTorch DataLoader Integration

`NeuroGrapeDataset` natively implements PyTorch's `Dataset` interface:

```python
try:
    import torch
    from torch.utils.data import DataLoader

    dataset = NeuroGrapeDataset(manifest="data_inventory.csv", modality="meg", split="train", as_torch=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in loader:
        x = batch["features"]          # torch.Tensor of shape (4, 272, 225)
        sids = batch["sample_id"]      # list of 4 stable sample IDs
        y_cat = batch["category_id"]   # torch.Tensor of 4 category IDs
        break
except ImportError:
    pass
```
