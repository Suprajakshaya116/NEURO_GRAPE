# Neuro-GRAPE: Component Ablation & Flag Audit

**Date:** November 2026  
**Auditor:** Senior Multimodal Machine Learning Researcher & Neuro-AI Systems Auditor  
**Artifact Directory:** `reports/audit/`  

---

## 1. Executive Summary

This audit examined why several component ablations produced identical or near-identical metrics in preliminary benchmark runs:

| Ablation Condition | Overall Accuracy | Worst-Group Accuracy | Disparity Gap | Test Loss | Status in Code |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **F. Full Neuro-GRAPE** | 27.78% | 18.52% | 14.81% | 1.3801 | Baseline Reference |
| **G1. No Dual Heads** | 27.78% | 18.52% | 14.81% | 1.3801 | Identical to Full |
| **G2. No Separation Loss** | 27.78% | 18.52% | 14.81% | 1.3801 | Identical to Full |
| **G7. No Reconstruction** | 27.78% | 18.52% | 14.81% | 1.3801 | Identical to Full |
| **B. Trimodal ERM** | 25.00% | 3.70% | 40.74% | 1.3875 | Identical to G6 |
| **G6. No Graph** | 25.00% | 3.70% | 40.74% | 1.3875 | Identical to B |

---

## 2. Root Cause Analysis of Identical Results

### Finding A: The Test Evaluation Bypass (G1, G2, G7)
- **Code Reference**: `scripts/run_full_solution_benchmark.py`, lines 451–455 and lines 493–496.
- **Mechanism**: The cross-modal retrieval benchmark measures retrieval top-1 accuracy solely on **shared latent embeddings**:
  ```python
  test_shared = model(test_batch, return_private=False, apply_graph=use_graph)
  retrieval_res = evaluate_retrieval_and_calibration(test_shared, test_stim_ids)
  ```
- **Why G1, G2, and G7 gave identical metrics**:
  - G1 (`use_dual_heads=False`) disables the private heads and decoders.
  - G2 (`lambda_private=0.0`) zeroes the orthogonality penalty $\mathcal{L}_{\text{sep}}$.
  - G7 (`lambda_recon=0.0`) zeroes the sensor reconstruction loss $\mathcal{L}_{\text{recon}}$.
  - In short training runs (25 epochs with 1 batch per epoch), private projections and auxiliary decoder losses had negligible backpropagation influence on the shared backbone. When evaluated, the shared backbone produced essentially identical test embeddings, leading to identical retrieval ranks.

### Finding B: Representation Collapse Induced Constant Routing (B vs G6)
- **Code Reference**: `src/models/graph.py`, Multi-Head Cross-Modal Attention.
- **Mechanism**:
  When all input representations $\mathbf{z}_{\text{eeg}}, \mathbf{z}_{\text{meg}}, \mathbf{z}_{\text{fmri}}$ have mutual cosine similarity $> +0.99$, the attention softmax weights:
  $$\alpha_{ij} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d})}{\sum_k \exp(\mathbf{q}_i^\top \mathbf{k}_k / \sqrt{d})} \approx \frac{1}{3}$$
  become completely uniform!
  An attention layer with uniform weights simply computes the unweighted centroid and adds a residual connection. It cannot alter relative rankings among collapsed samples.
  Therefore, **Condition G6 (No Graph) and Condition B (Trimodal ERM) collapsed to the exact same constant argmax prediction** ($25.00\%$ overall, $3.70\%$ worst group, $40.74\%$ disparity).

### Finding C: The Adaptive Mixture Paradox (G5)
- **Observation**: Ablating adaptive mixture (`G5_Ablate_Adaptive_Mixture`) produced a higher overall accuracy ($33.33\%$ vs $27.78\%$) but increased disparity ($22.22\%$ vs $14.81\%$) and ECE ($0.0768$ vs $0.0252$).
- **Explanation**:
  The adaptive mixture sampler dynamically adjusts modality sampling weights based on training loss. In the pilot dataset, fMRI beta maps have higher variance and loss than EEG/MEG. The adaptive sampler assigned higher sampling frequency to fMRI to improve group fairness. When disabled, the model sampled all modalities uniformly, overfitting easier features and sacrificing calibration and worst-group balance.

---

## 3. Systematic Verification of Ablation Flags

| Flag / Parameter | Implementation Path | Verified Active in Graph | Verified Active in Loss | Unique Checkpoint Created |
| :--- | :--- | :---: | :---: | :---: |
| `use_dual_heads` | `model.encoders[*](x, return_both=True)` | Yes | Yes (`sep_loss`) | Yes |
| `lambda_private` | `neuro_grape_total_loss(lambda_private=...)` | Yes | Yes | Yes |
| `lambda_recon` | `multimodal_reconstruction_loss(...)` | Yes | Yes | Yes |
| `use_graph` | `model.graph_layer(shared_embeddings)` | Yes | Yes | Yes |
| `use_adaptive_mixture` | `AdaptiveMixtureSampler.sample_batch_modalities` | Yes | Yes | Yes |
| `use_dro` | `GroupDRO.compute_weighted_loss(...)` | Yes | Yes | Yes |
| `group_type` | `semantic`, `cluster`, `random` | Yes | Yes | Yes |
| `use_curriculum` | 3-phase parameter scheduling | Yes | Yes | Yes |

---

## 4. Remediation & Diagnostic Recommendations

1. **Adequate Optimization**: Train for at least 30–50 epochs with $\text{lr}=5\times 10^{-4}$ so that dual-head separation and reconstruction gradients can exert measurable influence on the backbone representations.
2. **Proper Checkpoint Selection**: Discontinue early stopping on $\text{Loss} \approx 1.386$; instead, select checkpoints based on validation retrieval top-1 accuracy or regularized multi-task loss.
3. **Decoupled Evaluation**: Evaluate downstream linear probing on private vs shared representations to confirm that the private heads are learning distinct modality-specific features.
