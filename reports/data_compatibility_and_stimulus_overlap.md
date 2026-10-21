# Neuro-GRAPE Data Compatibility & Stimulus Overlap Report

**Project:** Neuro-GRAPE  
**Author:** Member 1 (Dataset Engineering, Preprocessing, Data Quality & Leakage Prevention)  
**Date:** October 2026  
**Status:** Validated  

---

## 1. Overview & Conceptual Distinctions

A core responsibility of dataset engineering in Neuro-GRAPE is preventing category errors in representation alignment. Downstream modeling (Member 2) and benchmarking (Member 3) must operate under explicit conceptual distinctions across five dimensions:

```
[Shared Image Filenames]  --> 20 exact image files (e.g., puck_02s.jpg)
        |
[Object Concepts/Categories] --> 20 distinct THINGS categories (e.g., puck, category 1246)
        |
[Subject & Session]       --> Independent cohorts (EEG sub-01, MEG sub-BIGMEG2, fMRI sub-01)
        |
[Recording Synchronization]--> Zero temporal simultaneity; completely separate experiments
        |
[Raw vs. Processed]       --> Continuous raw files vs. standardized epochs/beta volumes
```

---

## 2. Dimension 1: Shared Image Filenames

The pilot intersection comprises **exactly 20 unique stimulus filenames**. Every shared stimulus represents an experimental presentation (`trial_type == exp`) of Exemplar 2:
1. `banner_02s.jpg`
2. `bear_02s.jpg`
3. `beet_02s.jpg`
4. `boa_02s.jpg`
5. `cucumber_02s.jpg`
6. `glass_02s.jpg`
7. `hamburger_02s.jpg`
8. `hay_02s.jpg`
9. `inhaler_02s.jpg`
10. `lampshade_02s.jpg`
11. `microscope_02s.jpg`
12. `pea_02s.jpg`
13. `pinwheel_02s.jpg`
14. `puck_02s.jpg`
15. `screwdriver_02s.jpg`
16. `sprinkler_02n.jpg`
17. `tadpole_02s.jpg`
18. `tarp_02s.jpg`
19. `wire_02s.jpg`
20. `woman_02s.jpg`

*Note on Excluded Overlaps*: MEG and fMRI share 3 additional images (`catch051.jpg`, `catch010.jpg`, `iguana_20s.jpg`). These are excluded from the tri-modal pilot because they were not presented to EEG `sub-01`.

---

## 3. Dimension 2: Shared Object Concepts & Indexing Conventions

All 20 stimuli belong to **20 distinct, non-overlapping THINGS categories**.

### Indexing Formula
- In **EEG** (`events.tsv`): `objectnumber` is 0-indexed ($0 \le \text{objectnumber} \le 1853$).
- In **MEG** (`sample_attributes_P2.csv`) & **fMRI** (`events.tsv`): `things_category_nr` is 1-indexed ($1 \le \text{things\_category\_nr} \le 1854$).

$$\mathbf{\text{things\_category\_nr} = \text{eeg\_objectnumber} + 1}$$

| Stimulus Filename | EEG `objectnumber` | MEG/fMRI `things_category_nr` | Concept Name |
| :--- | :---: | :---: | :--- |
| `banner_02s.jpg` | 70 | 71 | banner |
| `bear_02s.jpg` | 105 | 106 | bear |
| `beet_02s.jpg` | 114 | 115 | beet |
| `boa_02s.jpg` | 150 | 151 | boa |
| `cucumber_02s.jpg` | 457 | 458 | cucumber |
| `glass_02s.jpg` | 677 | 678 | glass |
| `hamburger_02s.jpg` | 740 | 741 | hamburger |
| `hay_02s.jpg` | 761 | 762 | hay |
| `inhaler_02s.jpg` | 828 | 829 | inhaler |
| `lampshade_02s.jpg` | 893 | 894 | lampshade |
| `microscope_02s.jpg` | 999 | 1000 | microscope |
| `pea_02s.jpg` | 1128 | 1129 | pea |
| `pinwheel_02s.jpg` | 1185 | 1186 | pinwheel |
| `puck_02s.jpg` | 1245 | 1246 | puck |
| `screwdriver_02s.jpg` | 1391 | 1392 | screwdriver |
| `sprinkler_02n.jpg` | 1523 | 1524 | sprinkler |
| `tadpole_02s.jpg` | 1614 | 1615 | tadpole |
| `tarp_02s.jpg` | 1628 | 1629 | tarp |
| `wire_02s.jpg` | 1828 | 1829 | wire |
| `woman_02s.jpg` | 1832 | 1833 | woman |

---

## 4. Dimension 3 & 4: Cohort Independence vs. Separate Recordings

| Modality | Participant ID | Dataset / DOI | Recording Facility | Acquisition Year |
| :--- | :--- | :--- | :--- | :--- |
| **EEG** | `sub-01` | OpenNeuro ds003825 | University of Sydney, Australia | 2019 |
| **MEG** | `sub-BIGMEG2` | OpenNeuro ds004212 | NIMH, Bethesda, USA | 2019 |
| **fMRI** | `sub-01` | OpenNeuro ds004192 | NIMH, Bethesda, USA | 2019 |

### Operational Rule for Modeling:
There is **zero joint distribution of biological noise across modalities**. Models must never attempt temporal point-to-point loss functions or paired trial correlation. Alignment must occur in a **semantic representation space** or via stimulus-conditioned encoders.

---

## 5. Dimension 5: Raw Recordings vs. Processed Feature Tensors

| Modality | Raw Continuous Format | Preprocessed Feature Representation | Tensor Shape | Sampling Rate | Physical Units |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EEG** | BrainVision binary (765 MB) | Band-limited epochs $[-100, 800]\text{ ms}$ | `(63, 225)` | 250 Hz | Volts ($\text{V}$) |
| **MEG** | CTF `.ds` package (518 MB) | Band-limited epochs $[-100, 800]\text{ ms}$ | `(272, 225)` | 250 Hz | Tesla ($\text{T}$) |
| **fMRI** | NIfTI 4D BOLD (250 MB) | Option A: Raw 4-TR window<br>Option B: LS-S Single-Trial Beta Map | `(96, 96, 60, 4)`<br>`(96, 96, 60)` | $0.667\text{ Hz}$ | Raw counts<br>Beta weight |
