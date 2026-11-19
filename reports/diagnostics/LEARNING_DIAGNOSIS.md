# Neuro-GRAPE: Learning Diagnostics & Failure Mode Analysis

**Date:** November 2026  
**Auditor:** Senior Multimodal Machine Learning Researcher & Neuro-AI Systems Auditor  
**Artifact Directory:** `reports/diagnostics/`  
**Figures Directory:** `reports/diagnostics/figures/`  

---

## 1. Executive Summary

This diagnostic investigation resolved the central mystery of the preliminary benchmark:  
**Why was overall classification/retrieval accuracy hovering between 22% and 27% (essentially random guessing for a 4-class retrieval task)?**

Through controlled experiments, weight-distance tracking, intermediate activation hooks, and a tiny-subset overfit experiment, we established:

1. **The System Can Learn (Gradient Flow is Intact)**:
   - When trained with an appropriate learning rate ($\text{lr}=10^{-3}$) without premature early stopping, the network successfully overfit the 12 training samples:
   - **Training Loss**: Dropped from $2.4963$ to **$0.0006$**.
   - **Training Top-1 Retrieval Accuracy**: Reached **$100.0\%$**.
   - **Validation Top-1 Retrieval Accuracy**: Reached **$50.0\%$** (2× random chance).
2. **The Root Causes of Poor Benchmark Results**:
   - **Premature Early-Stopping Collapse**: Restored an essentially untrained Epoch-1 model.
   - **Positive-Cone Representation Collapse**: Intermediate activations produced embeddings with mean off-diagonal cosine similarities $> +0.997$.
   - **Constant Argmax Predictor**: The network predicted the same gallery sample for every query, achieving $1/4 = 25.0\%$ overall accuracy by coincidence while producing $0.0\%$ worst-group accuracy.

---

## 2. Systematic Evaluation of 13 Potential Hypotheses

| Hypothesis | Description | Diagnosis | Verdict |
| :--- | :--- | :--- | :---: |
| **A. Data Quality** | Corrupted tensors, excessive noise, or numerical instability | Verified 0 NaNs/Infs; physical signal units confirmed ($\mu\text{V}$, $\text{fT}$, beta). | **Refuted** |
| **B. Label Alignment** | Stimulus IDs mismatched across CSV rows or modalities | Verified matching `stimulus_id` associative indexing in `NeuroGrapeDataset`. | **Refuted** |
| **C. Modality Alignment** | Attempting direct sample-wise time synchronization | Modalities originate from separate experiments; alignment is purely associative over stimulus identity. | **Confirmed / Handled** |
| **D. Model Architecture** | Insufficient expressive capacity or wrong tensor rank handling | 415,675 parameters easily overfit 12 training samples ($100\%$ accuracy). | **Refuted** |
| **E. Optimization** | Learning rate too small, single-batch steps insufficient, early stopping trap | $\text{lr}=10^{-4}$ with 25 epochs (25 steps) was severely under-optimized; early stopping restored Epoch 1. | **PRIMARY CAUSE** |
| **F. Overfitting** | Network memorizing 12 training samples without generalization | With only 12 training stimuli, generalization gap is large; requires regularization and GroupDRO. | **Contributing Factor** |
| **G. Underfitting** | Model unable to minimize training objective | Model easily minimizes training loss ($2.49 \rightarrow 0.0006$). | **Refuted** |
| **H. Distribution Shift** | Disjoint test concepts differ from training concepts | Zero-shot stimulus generalization creates expected natural domain gap. | **Contributing Factor** |
| **I. Group Imbalance** | Living / Animate classes represent only 20–25% of data | Severe class/group imbalance penalizes worst-group performance under standard ERM. | **Confirmed** |
| **J. Class Imbalance** | Unequal exemplar distribution per concept | Verified 1 exemplar per concept in pilot. | **Refuted** |
| **K. Incorrect Evaluation** | Evaluating wrong tensor keys or metric bugs | Retrieval correctly evaluated on 4-stimulus test set; random chance is mathematically 25%. | **Confirmed** |
| **L. Implementation Bugs** | Checkpoint saving/restoring errors, frozen weights | Early-stopping logic selected Epoch 1; LayerNorm applied intra-sample rather than inter-sample. | **PRIMARY CAUSE** |
| **M. Insufficient Data** | 12 training stimuli is too small for a 415k parameter foundation model | Current pilot data limits statistical power; full dataset will provide needed sample density. | **Confirmed Reality** |

---

## 3. The Controlled Overfit Test

To determine whether the network architecture can learn cross-modal alignments, we conducted a controlled 100-epoch overfit experiment on the training set:

- **Initial Train Loss**: $2.4963$
- **Final Train Loss**: **$0.0006$** (99.98% loss reduction)
- **Train Cross-Modal Top-1 Accuracy**: **$100.0\%$**
- **Val Cross-Modal Top-1 Accuracy**: **$50.0\%$** (vs 25% chance baseline)

![Overfit Learning Curves](file:///d:/projects/NEURO_GRAPE/reports/diagnostics/figures/overfit_learning_curves.png)

### Key Insights from Learning Curves:
1. The loss curve decreases monotonically to near-zero, proving that gradient backpropagation and tensor connections are fully functional.
2. The network achieves $100\%$ train retrieval accuracy by epoch 40.
3. Validation retrieval accuracy peaks at $50.0\%$ and remains above chance throughout training.

---

## 4. The Positive-Cone Collapse & Constant Argmax Signature

Using activation hooks on `AdaptiveAvgPool` and linear projection heads, we measured the similarity of representations before and after projection:

```text
Modality: EEG
  Mean Off-Diagonal Cosine Similarity: +0.9977
  Similarity Matrix Row Variance:       0.000001
  Test Query Predictions:              [2, 2, 2, 2]  (Target: [0, 1, 2, 3])

Modality: MEG
  Mean Off-Diagonal Cosine Similarity: +0.9979
  Similarity Matrix Row Variance:       0.000001
  Test Query Predictions:              [0, 0, 0, 0]  (Target: [0, 1, 2, 3])

Modality: fMRI
  Mean Off-Diagonal Cosine Similarity: +0.9999
  Similarity Matrix Row Variance:       0.000000
  Test Query Predictions:              [1, 1, 1, 1]  (Target: [0, 1, 2, 3])
```

### Mathematical Explanation:
When pairwise cosine similarities are $\ge 0.997$, every sample vector points into the same quadrant of the hypersphere. For any query $\mathbf{q}$, the inner product $\mathbf{q}^\top \mathbf{k}_j$ is maximized by the gallery vector $\mathbf{k}_j$ that has the largest component along the cone's central axis.

Thus, **every query in the batch selects the exact same gallery sample $j^*$**.  
Because exactly one query in the batch actually has ground-truth index $j^*$, that single query is correct, yielding:
$$\text{Top-1 Accuracy} = \frac{1}{4} = 25.00\%$$

When evaluated under semantic subgroups (Animate vs Inanimate):
- The group containing $j^*$ gets $1/2 = 50.0\%$ accuracy.
- The group not containing $j^*$ gets $0/2 = 0.0\%$ accuracy.
- Worst-group accuracy is **$0.0\%$**, and disparity is **$50.0\%$**.

This precisely replicates the preliminary table for `GroupDRO Semantic`!

---

## 5. Summary of Recommended Fixes

1. **Replace Early Stopping on Raw Loss**: Stop relying on minimum validation loss when batch size is 4. Use validation retrieval top-1 accuracy or a moving average with patience.
2. **Increase Effective Steps**: Train for 30–50 epochs with $\text{lr}=5\times 10^{-4}$ and cosine learning rate decay.
3. **Decouple Representations**: Add batch-centering or a small uniform dispersion loss to prevent the positive-cone collapse.
4. **EMA Smoothing in GroupDRO**: Apply exponential moving average smoothing to group weights to prevent unstable weight oscillations on small batches.
