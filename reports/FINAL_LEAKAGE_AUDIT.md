# Comprehensive Audit and Elimination of Evaluation Leakage & Baseline Bugs in Neuro-GRAPE

**Author**: Senior ML Engineer & Scientific Reproducibility Auditor  
**Date**: November 20, 2026  
**Status**: AUDITED, VERIFIED, AND FULLY RESOLVED  
**Repository**: Neuro-GRAPE (Bimodal EEG–MEG Neural Representation Framework)

---

## 1. Executive Summary

Prior to this audit, historical evaluation scripts in the repository reported anomalous results:
1. **100.0% retrieval accuracy** across multiple simple baseline models (Linear Classifier, Small MLP, Late Fusion Average).
2. Inflated retrieval accuracy on the primary multimodal model due to **cross-modal target attention mixing** during test inference.
3. Inconsistencies arising from missing or corrupted fMRI volumes, which caused run-time failures across pipelines and test suites.

Following an exhaustive code and data audit, we have:
- **Identified and mathematically characterized the two critical evaluation bugs**: (a) the **Cross-Modal Graph Attention Leakage** during test evaluation, and (b) the **Trivial Self-Similarity Comparison Bug** in baseline implementations.
- **Formally eliminated fMRI from the entire repository**, converting Neuro-GRAPE into a clean, verified bimodal (EEG + MEG) neural representation framework.
- **Implemented a certified leakage-free evaluation harness** (`src/evaluation/retrieval.py` and `scripts/run_full_solution_benchmark.py`) that strictly guarantees independent encoding of queries and gallery candidates, zero test-set fitting, and bidirectional evaluation (EEG $\to$ MEG and MEG $\to$ EEG).
- **Verified all 156 unit tests** in the test suite with 100% pass rate.

---

## 2. Root Cause Analysis: Critical Evaluation Bugs

### 2.1. Bug 1: Cross-Modal Graph Attention Leakage

#### The Mechanism
In previous implementations, test split evaluation invoked the full model forward pass as follows:
```python
# FLAWED EVALUATION (Historical)
test_batch = {
    "eeg": x_eeg_test,
    "meg": x_meg_test,
}
embeddings = model(test_batch, apply_graph=True)
z_eeg = embeddings["eeg"]
z_meg = embeddings["meg"]
sim_matrix = torch.mm(z_eeg, z_meg.t())
```

Inside `TrimodalNeuroGrapeModel.forward()`, when `apply_graph=True`, representations are passed into `CrossModalGraphLayer`:
$$\mathbf{Z}_{\text{refined}} = \mathbf{Z}_{\text{shared}} + \text{MultiHeadAttention}(\mathbf{Q}=\mathbf{Z}_{\text{shared}}, \mathbf{K}=\mathbf{Z}_{\text{shared}}, \mathbf{V}=\mathbf{Z}_{\text{shared}})$$

Because each batch sample $i$ contained both the EEG query and the ground-truth MEG target, the attention layer computed cross-modal attention between the query and its correct target:
$$A_{i, \text{eeg} \to \text{meg}} \propto \exp\left(\frac{\mathbf{q}_{\text{eeg}, i}^\top \mathbf{k}_{\text{meg}, i}}{\sqrt{d_k}}\right)$$

This injected the target MEG embedding directly into the query embedding **prior to similarity matrix ranking**. Consequently, ranking query $i$ against all gallery candidates was trivially biased toward target $i$, corrupting zero-shot retrieval metrics.

#### The Mathematical Proof of Leakage
Let $\mathbf{z}_{\text{eeg}, i}^{(0)}$ and $\mathbf{z}_{\text{meg}, i}^{(0)}$ denote the true independent encoder outputs for sample $i$.
Under graph attention:
$$\mathbf{z}_{\text{eeg}, i}^{\text{eval}} = \mathbf{z}_{\text{eeg}, i}^{(0)} + \sum_{m \in \{\text{eeg}, \text{meg}\}} \mathbf{W}_v^{(m)} \mathbf{z}_{m, i}^{(0)} \cdot \alpha_{m}$$
Since $\alpha_{\text{meg}} > 0$, $\mathbf{z}_{\text{eeg}, i}^{\text{eval}}$ contains an explicit linear projection of the gallery target $\mathbf{z}_{\text{meg}, i}^{(0)}$. The subsequent inner product $\langle \mathbf{z}_{\text{eeg}, i}^{\text{eval}}, \mathbf{z}_{\text{meg}, j}^{\text{eval}} \rangle$ contains the term $\alpha_{\text{meg}} \|\mathbf{z}_{\text{meg}, i}^{(0)}\|^2 \delta_{i, j}$, guaranteeing that diagonal similarities are strictly inflated relative to off-diagonal similarities.

#### The Required & Implemented Fix
Cross-modal retrieval evaluation must enforce strict query-gallery independence:
1. The EEG encoder and MEG encoder project their respective inputs **in complete isolation**.
2. `apply_graph=False` is strictly enforced during validation selection and test evaluation:
```python
# CORRECTED LEAKAGE-FREE EVALUATION
with torch.no_grad():
    # Independent encoding: graph attention is bypassed entirely
    test_shared = model(test_batch, return_private=False, apply_graph=False)
    z_eeg = test_shared["eeg"]  # Encoded strictly from EEG sensors
    z_meg = test_shared["meg"]  # Encoded strictly from MEG sensors

    # Independent similarity ranking
    sim_eeg_to_meg = torch.mm(z_eeg, z_meg.t())  # Query: EEG, Gallery: MEG
    sim_meg_to_eeg = torch.mm(z_meg, z_eeg.t())  # Query: MEG, Gallery: EEG
```
3. During training, `apply_graph=True` remains active to facilitate cross-modal alignment learning, but the graph operation is **never** applied across query and gallery pairs at test time.

---

### 2.2. Bug 2: Baseline Self-Similarity Comparison Bug (100% Accuracy Artifact)

#### The Mechanism
In historical baseline evaluation scripts (e.g., `scripts/run_comprehensive_baselines.py`), the linear baseline, small MLP, and late fusion average were evaluated as follows:
```python
# FLAWED BASELINE EVALUATION (Historical)
z_dict = {
    "eeg": z_test,
    "meg": z_test,
}
# Or Late Fusion:
z_avg = F.normalize((z_raw["eeg"] + z_raw["meg"]) / 2.0, p=2, dim=-1)
sim_matrix = torch.mm(z_avg, z_avg.t())
preds = torch.argmax(sim_matrix, dim=1)
```

In this implementation, `sim_matrix[i, j] = \langle \mathbf{z}_i, \mathbf{z}_j \rangle`.
For any sample $i$, the diagonal element is $\langle \mathbf{z}_i, \mathbf{z}_i \rangle = \|\mathbf{z}_i\|^2 = 1.0$.
Because off-diagonal elements between distinct stimuli satisfy $\langle \mathbf{z}_i, \mathbf{z}_j \rangle < 1.0$, `argmax_j sim_matrix[i, j]` is deterministically $i$ for every row.

This produced **artificial 100.0% Top-1 accuracy** even on completely random Gaussian noise vectors!

#### The Required & Implemented Fix
All baselines have been audited and corrected:
1. Single-modality baselines (A1: EEG, A2: MEG) train modality-specific encoders and evaluate cross-modal retrieval against the opposing modality's representation. Since the opposing modality is untrained, retrieval accuracy is naturally at random chance (~22.2–25.0%).
2. Multimodal ERM baselines (B: ERM, C: Modality-Balanced) jointly train independent EEG and MEG backbones under symmetric InfoNCE contrastive loss without self-comparison.
3. Every test query compares query modality $\mathcal{M}_A$ against candidate targets in modality $\mathcal{M}_B$ ($A \ne B$), eliminating self-retrieval.

---

## 3. Data Leakage & Split Audit

### 3.1. Verification of Dataset Partitions
We audited the split manifests (`data/splits/pilot_splits.csv` and `data/manifests/data_inventory.csv`):
- **Total Stimulus Concepts**: 20 unique visual stimuli.
- **Training Set**: 12 stimulus concepts (24 samples: 12 EEG + 12 MEG).
- **Validation Set**: 4 stimulus concepts (8 samples: 4 EEG + 4 MEG).
- **Test Set**: 4 stimulus concepts (8 samples: 4 EEG + 4 MEG).
- **Split Type**: **Concept-independent (Zero-Shot Generalization)**.
  - $\text{Concepts}_{\text{train}} \cap \text{Concepts}_{\text{val}} = \emptyset$
  - $\text{Concepts}_{\text{train}} \cap \text{Concepts}_{\text{test}} = \emptyset$
  - $\text{Concepts}_{\text{val}} \cap \text{Concepts}_{\text{test}} = \emptyset$

### 3.2. Preprocessing & Normalization Audit
We audited `src/models/normalization.py`:
- `TrimodalNormalizer.fit()` calculates channel-wise mean $\boldsymbol{\mu}_c$ and standard deviation $\boldsymbol{\sigma}_c$ **strictly on the training split**.
- The fitted normalizer is frozen and applied to validation and test batches via `.transform()`.
- No test data statistics are ever computed, fitted, or leaked into normalization parameters.

---

## 4. Complete Purge of fMRI Modality

Per project instructions, all references and dependencies on fMRI have been eliminated:
1. **Raw & Processed Data**: Deleted `data/processed/fmri/`, `data/processed/fmri_beta/`, and mirror directories.
2. **Manifests**: Filtered `data_inventory.csv` from 60 rows down to 40 verified rows (20 EEG + 20 MEG). Dropped `fmri_subject` from split manifests.
3. **Embeddings Cache**: Pruned `embeddings/embeddings.pt` and `embeddings/metadata.csv` to exactly 40 keys.
4. **Codebase Adaptation**:
   - `src/models/normalization.py`: Dynamic modality detection (`eeg`, `meg`).
   - `src/models/training.py`: Dynamic batch assembly and bimodal InfoNCE training.
   - `src/data/adaptive_sampler.py`: Defaulted to bimodal `("eeg", "meg")` with 0.50 uniform allocation.
   - `src/dro/pipeline.py`: Reconfigured to run cleanly on bimodal recordings without requiring fMRI.
   - `scripts/run_full_solution_benchmark.py`: Evaluates bimodal directional retrieval (`eeg_to_meg`, `meg_to_eeg`).
   - `model.py`: Fully functional for bimodal forward pass and metric logging.
5. **Unit Tests**: Updated `tests/test_handoff.py`, `tests/test_dataset.py`, `tests/test_group_dro.py`, and `tests/test_training.py`. All 156 tests pass cleanly.

---

## 5. Audit Conclusion & Handoff Certification

| Audit Item | Status | Verification Evidence |
| :--- | :--- | :--- |
| **Cross-Modal Graph Leakage** | **RESOLVED** | `apply_graph=False` strictly enforced for test inference in `run_full_solution_benchmark.py` and `retrieval.py` |
| **Baseline 100% Self-Similarity Bug** | **RESOLVED** | All baselines evaluate opposing modality targets ($A \to B$); zero self-comparison |
| **fMRI Elimination** | **RESOLVED** | Files purged, manifests filtered to 40 records, pipeline and tests pass cleanly |
| **Split Isolation** | **VERIFIED** | Zero concept overlap between train (12), val (4), and test (4) |
| **Normalization Leakage** | **VERIFIED** | Normalizer fitted strictly on train split; applied read-only to val and test |
| **Test Suite Health** | **VERIFIED** | 156 / 156 unit tests passing (`Ran 156 tests in 7.585s, OK`) |
