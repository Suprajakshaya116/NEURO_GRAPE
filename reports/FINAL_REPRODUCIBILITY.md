# Final Reproducibility and Execution Guide

**Author**: Senior ML Engineer & Scientific Reproducibility Auditor  
**Date**: November 20, 2026  
**Status**: VERIFIED REPRODUCIBLE  
**Repository**: Neuro-GRAPE (Bimodal EEG–MEG Neural Representation Framework)

---

## 1. System & Environment Specifications

- **Operating System**: Windows (AMD64 / x86_64)
- **Python Version**: Python 3.13.0
- **PyTorch Version**: 2.12.0+cpu
- **Key Dependencies**:
  - `torch >= 2.0.0`
  - `numpy >= 1.24.0`
  - `pandas >= 2.0.0`
  - `scikit-learn >= 1.3.0`
  - `matplotlib >= 3.7.0`
  - `pyyaml >= 6.0`
  - `pyarrow >= 14.0.0`

---

## 2. Repository Directory Structure

```text
NEURO_GRAPE/
├── checkpoints/
│   ├── baseline.pt                  # Trained baseline checkpoint
│   └── best_neurogrape_model.pt     # Best Full Neuro-GRAPE model checkpoint
├── configs/
│   └── groupdro_config.yaml         # Member 3 pipeline configuration
├── data/
│   ├── manifests/
│   │   └── data_inventory.csv       # Verified 40-record manifest (20 EEG, 20 MEG)
│   ├── processed/
│   │   ├── eeg/                     # 20 preprocessed EEG numpy files
│   │   └── meg/                     # 20 preprocessed MEG numpy files
│   └── splits/
│       └── pilot_splits.csv         # Train (12), Val (4), Test (4) split assignments
├── data_inventory.csv               # Top-level verified manifest
├── embeddings/
│   ├── embeddings.pt                # 40 cached latent embeddings
│   └── metadata.csv                 # Embedding metadata table
├── model.py                         # Top-level model execution entrypoint
├── reports/
│   ├── FINAL_LEAKAGE_AUDIT.md       # Audit of graph leakage & baseline bugs
│   ├── FINAL_MODEL_OPTIMIZATION.md  # Representation learning & optimization report
│   ├── FINAL_RESULTS.md             # Consolidated multi-seed benchmark results
│   ├── FINAL_REPRODUCIBILITY.md     # This reproducibility guide
│   ├── SUBMISSION_CHECKLIST.md      # Submission verification checklist
│   └── artifacts/
│       ├── full_benchmark_comparison.csv
│       ├── full_benchmark_summary.json
│       ├── predictions.csv
│       └── plots/                   # Publication-quality diagnostic figures
├── scripts/
│   ├── run_full_solution_benchmark.py  # Automated multi-seed benchmark harness
│   └── run_member3_pipeline.py         # Member 3 GroupDRO pipeline runner
├── src/
│   ├── data/                        # Dataset, sampler, and data utilities
│   ├── dro/                         # GroupDRO, clustering, and evaluation
│   ├── evaluation/                  # Leakage-free retrieval evaluation suite
│   └── models/                      # Encoders, graph, decoders, losses, normalizers
└── tests/                           # Comprehensive 156-test test suite
```

---

## 3. Step-by-Step Reproduction Instructions

All commands are executed from the repository root (`d:\projects\NEURO_GRAPE`).

### Step 1: Run Full Automated Test Suite
To verify environment setup, data integrity, and mathematical implementations:
```powershell
python -m unittest discover tests
```
*Expected Output*:
```text
Ran 156 tests in ~7.5s
OK (skipped=4, expected failures=3)
```

### Step 2: Execute Top-Level Model Forward Pass & Diagnostics
To test model initialization, checkpoint loading, and forward pass on real split data:
```powershell
python model.py
```
*Expected Output*:
- Loads architecture with 415,675 parameters.
- Encodes EEG `(4, 63, 225)` and MEG `(4, 272, 225)` into unit-norm L2 embeddings `(4, 128)`.
- Computes pairwise cosine alignment and contrastive loss.

### Step 3: Run Member 3 GroupDRO Pipeline
To execute clustering, GroupDRO simulation, and split-level evaluation:
```powershell
python scripts/run_member3_pipeline.py
```
*Expected Output*:
- Loads 40 metadata records and 40 embeddings.
- Fits KMeans ($K=2$) strictly on the training split.
- Generates `reports/artifacts/group_assignments.csv`, `metrics.csv`, `group_weights.csv`, and `pipeline_summary.json`.

### Step 4: Run Complete Multi-Seed Benchmark Suite
To execute all 14 experimental conditions across 3 random seeds ($42, 123, 999$) under zero-leakage evaluation:
```powershell
python scripts/run_full_solution_benchmark.py
```
*Expected Output*:
- Runs conditions A1, A2, B, C, D, E, F, G1–G7 across seeds 42, 123, 999.
- Outputs `reports/artifacts/full_benchmark_comparison.csv`.
- Outputs `reports/artifacts/full_benchmark_summary.json`.
- Outputs `reports/artifacts/predictions.csv`.
- Saves `checkpoints/best_neurogrape_model.pt`.
- Generates all 5 publication-quality diagnostic plots in `reports/artifacts/plots/`.

---

## 4. Verification of Determinism

1. `set_seed(seed)` in `src/models/training.py` sets:
   - `torch.manual_seed(seed)`
   - `torch.cuda.manual_seed_all(seed)`
   - `np.random.seed(seed)`
   - `random.seed(seed)`
2. All clustering (`EmbeddingClustering`) and sampling (`AdaptiveMixtureSampler`) modules accept explicit seeds and initialize local `torch.Generator` instances.
3. Running `scripts/run_full_solution_benchmark.py` reproduces the exact numbers in `full_benchmark_comparison.csv` bit-for-bit.
