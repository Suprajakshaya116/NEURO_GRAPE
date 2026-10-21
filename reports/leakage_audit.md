# Neuro-GRAPE Data Leakage & Split Audit Report

**Project:** Neuro-GRAPE  
**Author:** Member 1 (Dataset Engineering, Preprocessing, Data Quality & Leakage Prevention)  
**Date:** October 2026  
**Status:** Audit Verified — Zero Leakage Confirmed  

---

## 1. Objective & Generalization Target

In representation learning across neuroimaging modalities, data leakage can lead to inflated decoding performance that fails to generalize. 

The evaluation target for Neuro-GRAPE is **Held-Out Visual Stimulus Generalization**:
> *Can an encoder trained on brain responses to a set of visual objects generalize to neural responses evoked by completely novel, unseen visual objects?*

Under this objective, any overlap of stimulus identities between training and evaluation splits represents critical scientific leakage.

---

## 2. Partition Strategy & Verification

### 2.1 Split Allocation
Splits were constructed strictly over the **20 unique stimulus filenames** ($N=20$ concepts):
- **Train (60%, 12 stimuli)**: `banner_02s.jpg`, `bear_02s.jpg`, `beet_02s.jpg`, `boa_02s.jpg`, `cucumber_02s.jpg`, `glass_02s.jpg`, `hamburger_02s.jpg`, `hay_02s.jpg`, `inhaler_02s.jpg`, `lampshade_02s.jpg`, `microscope_02s.jpg`, `pea_02s.jpg`
- **Validation (20%, 4 stimuli)**: `pinwheel_02s.jpg`, `puck_02s.jpg`, `screwdriver_02s.jpg`, `sprinkler_02n.jpg`
- **Test (20%, 4 stimuli)**: `tadpole_02s.jpg`, `tarp_02s.jpg`, `wire_02s.jpg`, `woman_02s.jpg`

### 2.2 Mathematical Set Disjointness Verification
Let $\mathcal{S}_{\text{train}}$, $\mathcal{S}_{\text{val}}$, and $\mathcal{S}_{\text{test}}$ denote the sets of unique stimulus IDs assigned to each partition. Automated assertions in `tests/test_dataset.py` verify:

$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset$$
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
$$\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

### 2.3 Row-Count vs. Unique Identity Disjointness
In `data_inventory.csv`, each of the 20 stimuli has 3 rows (1 EEG, 1 MEG, 1 fMRI), totaling 60 rows:
- **Train rows**: $12 \times 3 = 36$
- **Val rows**: $4 \times 3 = 12$
- **Test rows**: $4 \times 3 = 12$

Every stimulus maps to **exactly one split across all modalities**:
$$\forall s \in \mathcal{S}, \quad \text{nunique}(\text{split}(s)) \equiv 1$$

---

## 3. Transformation Fitting & Preprocessing Boundaries

To prevent data-dependent information leakage:
1. **Baseline Correction**: Applied strictly within-trial on pre-stimulus intervals ($[-100, 0]\text{ ms}$ for EEG/MEG; within-run DCT detrending for fMRI). No global standardization parameters are learned across trials.
2. **Standardization / Normalization for Modeling (Member 2)**:
   - Any z-scoring, whitening, PCA, or spatial filtering that computes dataset statistics (mean $\mu$, covariance $\Sigma$) **must be fit exclusively on training split samples ($\mathcal{S}_{\text{train}}$)**.
   - Test and validation samples must be transformed using parameters fixed on the training set.

---

## 4. Exclusions & Limitations

1. **Excluded Stimuli**:
   - `catch051.jpg`, `catch010.jpg`, and `iguana_20s.jpg` are present in MEG and fMRI but absent in EEG `sub-01`. To prevent modality imbalance, they were excluded from the tri-modal manifest.
2. **Pilot Sample Size**:
   - With 20 shared stimuli, the test set contains 4 visual concepts. While perfectly suitable for integration tests, pipeline debugging, and tensor verification, 4 test samples are insufficient for statistical hypothesis testing.
   - Scaled benchmarks in later phases will ingest the full THINGS datasets.
