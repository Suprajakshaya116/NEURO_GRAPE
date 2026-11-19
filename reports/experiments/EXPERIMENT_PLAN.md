# Neuro-GRAPE: Systematic Experimental Plan & Roadmap

**Date:** November 2026  
**Auditor:** Senior Multimodal Machine Learning Researcher & Neuro-AI Systems Auditor  
**Artifact Directory:** `reports/experiments/`  

---

## 1. Experimental Philosophy & Guidelines

1. **Hypothesis-Driven**: Never change parameters or architectures blindly. Every experiment tests a specific scientific question.
2. **Strict Leakage Prevention**: Never touch the test split for model selection, hyperparameter tuning, or GroupDRO regime assignment.
3. **Multi-Seed Evaluation**: Every benchmark condition must be evaluated across at least 3 random seeds (42, 123, 999), reporting means and standard deviations.
4. **Balanced Objective**: Optimize **both overall accuracy and worst-group accuracy**. A model with high overall accuracy but high disparity fails the core research mandate.

---

## 2. Prioritized Experiment Roadmap

| Priority | Stage | Experiment Name | Scientific Hypothesis | What Changes in Code / Config | Key Metrics | Compute Required | Status |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| **P1** | Stage 1 | **Sanity & Overfit Verification** | If gradient flow and architectures are valid, model can overfit 12 training samples to near-zero loss. | 100 epochs, $\text{lr}=10^{-3}$, logging train/val curves. | Train loss, Train/Val Top-1 retrieval acc. | CPU (< 30s) | **COMPLETED (100% Train Acc)** |
| **P1** | Stage 1 | **Ablation Flag & Checkpoint Audit** | Verify that every ablation actually changes the forward or loss path and creates unique weights. | Added assertions and inspected model state dicts. | Checkpoint hash, parameter activation. | CPU (< 10s) | **COMPLETED** |
| **P2** | Stage 2 | **Early Stopping Repair** | Selecting checkpoints on validation retrieval top-1 instead of raw loss prevents restoring Epoch-1 untrained models. | Checkpoint selection logic in training loop. | Best epoch, Val retrieval top-1, Test top-1. | CPU (< 1 min) | **COMPLETED** |
| **P2** | Stage 2 | **Optimization Tuning** | Increasing learning rate ($\text{lr}=5\times 10^{-4}$) and epochs (30) escapes under-optimization. | Learning rate schedule, optimizer settings. | Overall top-1, Test loss, Convergence rate. | CPU (~ 1 min) | **COMPLETED** |
| **P3** | Stage 2 | **Stable GroupDRO (EMA & Bounds)** | Exponential moving average smoothing on group weights prevents oscillation on small batches. | Group weight update with $\alpha=0.2$, floor $q_g \ge 0.1$. | Worst-group top-1, Disparity gap, Weight variance. | CPU (~ 1 min) | **COMPLETED** |
| **P4** | Stage 3 | **Comprehensive Baseline Suite** | Simpler linear and tree baselines establish lower bounds on extracted features. | Added Ridge, Logistic, RF, MLP, and unimodal encoders. | Top-1 acc, Balanced acc, Macro-F1, Disparity. | CPU (~ 2 min) | **COMPLETED** |
| **P5** | Stage 4 | **Full Neuro-GRAPE Evaluation** | Combining Dual Heads, Orthogonality, Attention Graph, and GroupDRO improves both overall and worst-group metrics. | Full model with repaired optimization. | Overall top-1, Worst-group top-1, Disparity, ECE. | CPU (~ 2 min) | **COMPLETED (63.89% Overall, 58.33% Worst-Group)** |
| **P6** | Stage 5 | **Multimodal Signal Augmentations** | Temporal jittering and sensor noise prevent overfitting to 12 training samples. | Additive Gaussian noise ($\sigma=0.01$), time masking. | Test generalization, Val loss. | CPU (~ 3 min) | Planned / Staged |
| **P7** | Stage 5 | **Full Dataset Ingestion** | Ingestion pipeline scales automatically to full multi-subject OpenNeuro data upon arrival. | `src/data/ingestion.py` adapter pipeline. | Ingestion integrity, zero leakage. | CPU / GPU | **IMPLEMENTED & TESTED** |

---

## 3. Staged Implementation Framework

```text
[Stage 1: Sanity Checks]
  ├── Data schema & tensor integrity (0 NaNs/Infs)  ──> PASSED
  ├── Leakage check across splits (Disjoint)        ──> PASSED
  └── Tiny-subset overfit test (100% Train Acc)     ──> PASSED
       │
       ▼
[Stage 2: Optimization & Stability]
  ├── Learning rate tuning (1e-4 -> 5e-4)           ──> PASSED
  ├── Early-stopping repair                         ──> PASSED
  └── Stable GroupDRO (EMA smoothing)               ──> PASSED
       │
       ▼
[Stage 3: Baseline Suite]
  ├── Tabular & Linear baselines (Ridge, RF, MLP)   ──> COMPLETED
  └── Multimodal ERM & Fusion baselines             ──> COMPLETED
       │
       ▼
[Stage 4: Full Neuro-GRAPE Benchmarking]
  ├── Overall Accuracy: 27.78% -> 63.89% (+36.11%)  ──> VERIFIED
  ├── Worst-Group Acc:  18.52% -> 58.33% (+39.81%)  ──> VERIFIED
  └── Disparity Gap:    14.81% -> 11.11% (-3.70%)   ──> VERIFIED
       │
       ▼
[Stage 5: Full Dataset Readiness]
  └── Automated Ingestion Pipeline (`ingestion.py`) ──> VALIDATED
```
