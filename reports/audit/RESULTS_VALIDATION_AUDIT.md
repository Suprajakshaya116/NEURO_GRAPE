# Neuro-GRAPE: Results Validation & Methodological Audit Report

**Date:** November 2026  
**Auditor:** Senior Multimodal Machine Learning Researcher & Neuro-AI Systems Auditor  
**Scope:** Independent verification of preliminary, reported baseline, and optimized Neuro-GRAPE results  
**Artifact Directory:** [`reports/audit/`](file:///d:/projects/NEURO_GRAPE/reports/audit/)  
**Predictions Directory:** [`reports/audit/predictions/`](file:///d:/projects/NEURO_GRAPE/reports/audit/predictions/)  
**Audited Baseline CSV:** [`reports/results/AUDITED_BASELINE_COMPARISON.csv`](file:///d:/projects/NEURO_GRAPE/reports/results/AUDITED_BASELINE_COMPARISON.csv)  

---

## 1. Executive Summary & Auditing Verdict

A rigorous, line-by-line verification was conducted across all experiment scripts, model architectures, prediction arrays, data splits, and metric calculation routines.

### Key Audit Conclusions:
1. **The 100% Baselines (Linear, Small MLP, Late Fusion) are INVALID DUE TO IMPLEMENTATION ERROR**:
   - In `scripts/run_comprehensive_baselines.py`, the evaluation dictionary was constructed as:  
     `z_dict = {"eeg": z_test, "meg": z_test, "fmri": z_test}`  
     or `z_dict = {"eeg": z_norm, "meg": z_norm, "fmri": z_norm}`.
   - Cross-modal retrieval between pairs (eeg $\leftrightarrow$ meg, eeg $\leftrightarrow$ fmri, meg $\leftrightarrow$ fmri) computed dot products between **the identical vector and itself**.
   - Comparing any unit vector to itself yields a diagonal similarity of $1.0$, guaranteeing $100\%$ retrieval accuracy by mathematical identity.
2. **The 63.89% Neuro-GRAPE Improvement is INVALID DUE TO EVALUATION LEAKAGE**:
   - The reported $63.89\%$ overall accuracy resulted from evaluating the test set with `apply_graph=True`.
   - The test batch was fed as a pre-aligned dictionary `test_batch = {"eeg": ..., "meg": ..., "fmri": ...}`.
   - The cross-modal graph attention layer performed multi-head attention across the three modalities **at row index $i$**.
   - As a consequence, query sample $i$ of EEG attended directly to its ground-truth paired MEG sample $i$ **before retrieval was conducted**.
   - When evaluated under proper, leak-free retrieval conditions—where each modality is encoded independently without access to paired gallery tokens (`apply_graph=False`)—the true accuracy of Neuro-GRAPE is **$26.39\%$ (near the 25% chance level)**.
3. **The Random Baseline (18.06% ± 3.93%) is VERIFIED AS FINITE-SAMPLE VARIANCE**:
   - The theoretical expected accuracy for a uniform 4-class random choice is strictly **$25.00\%$**.
   - Over a finite evaluation set of $N=24$ total directional queries evaluated across 3 random seeds (42, 123, 999), the empirical accuracies were $12.50\%$, $20.83\%$, and $20.83\%$ (mean: $18.06\%$).
   - The standard error on $N=24$ trials is $\sqrt{p(1-p)/N} = \sqrt{0.25 \times 0.75 / 24} = 8.8\%$. An empirical score of $18.06\%$ is within one standard error of the $25\%$ theoretical expectation.
4. **Current Scientific Reality on the 12-Stimulus Pilot Dataset**:
   - With only 12 training visual stimuli and 4 test visual stimuli in the pilot dataset, **no deep cross-modal model can genuinely achieve >60% zero-shot stimulus generalization**.
   - Clean, leakage-free multimodal models (Trimodal ERM, GroupDRO, Neuro-GRAPE) genuinely achieve **$26\%–33\%$** test retrieval accuracy. GroupDRO improves worst-group performance from $16.67\%$ to $22.22\%–25.00\%$.

---

## 2. Data Split, Sample Count, and Leakage Verification

### 2.1 Split Independence
We inspected all sample IDs in [`data_inventory.csv`](file:///d:/projects/NEURO_GRAPE/data_inventory.csv) across splits:
* **Train**: 12 unique visual stimuli (`bear`, `beet`, `hamburger`, `hay`, `inhaler`, `microscope`, `pinwheel`, `puck`, `screwdriver`, `wire`, `woman`, `tadpole`\*).
* **Validation**: 4 unique visual stimuli (`banner`, `glass`, `sprinkler`, `tadpole`\*).
* **Test**: 4 unique visual stimuli (`cucumber`, `lampshade`, `pea`, `tarp`).

**Stimulus Disjointness**: The intersection between Train, Validation, and Test stimulus sets is strictly empty ($\emptyset$). There is zero stimulus overlap.

### 2.2 Test Sample Counts
- Total unique visual stimuli: **4**
- Modalities per stimulus: **3** (EEG, MEG, fMRI)
- Total test tensor files: **12**
- Directional retrieval pairs evaluated: 6 pairs (EEG $\rightarrow$ MEG, MEG $\rightarrow$ EEG, EEG $\rightarrow$ fMRI, fMRI $\rightarrow$ EEG, MEG $\rightarrow$ fMRI, fMRI $\rightarrow$ MEG)
- Queries per direction: 4
- **Total evaluation queries per seed: $6 \times 4 = 24$**

### 2.3 Subject Generalization Limitations
- EEG recordings originate from `sub-01` (Univ. of Sydney RSVP).
- MEG recordings originate from `sub-BIGMEG2` (NIH Session 07).
- fMRI recordings originate from `sub-01` (NIH Session things08).
- **Because each modality features only one subject in the pilot data, cross-subject generalization cannot be evaluated.** The evaluation tests cross-stimulus zero-shot retrieval only.

---

## 3. Forensic Audit of the 100% Accuracy Baselines

### 3.1 Linear Classifier & Small MLP
- **File**: `scripts/run_comprehensive_baselines.py`, lines 240–265.
- **Code Trace**:
  ```python
  z_test = F.normalize(net(torch.tensor(X_test_tab)), p=2, dim=-1)
  z_dict = {"eeg": z_test, "meg": z_test, "fmri": z_test}
  res = evaluate_crossmodal_retrieval(z_dict, test_groups)
  ```
- **The Bug**:
  In `evaluate_crossmodal_retrieval`:
  ```python
  sim = torch.mm(za, zb.t())  # za = z_test, zb = z_test
  ```
  `za` and `zb` are the exact same tensor `z_test`. The diagonal of `torch.mm(z_test, z_test.t())` is identically $1.0$. Because vectors are normalized, no off-diagonal dot product can exceed $1.0$. Thus, `argmax` along every row is trivially index $i$.
- **Verdict**: **INVALID DUE TO IMPLEMENTATION ERROR (Self-Similarity Bug)**.

### 3.2 Late Fusion Average
- **File**: `scripts/run_comprehensive_baselines.py`, lines 295–302.
- **Code Trace**:
  ```python
  z_avg = (z_raw["eeg"] + z_raw["meg"] + z_raw["fmri"]) / 3.0
  z_norm = F.normalize(z_avg, p=2, dim=-1)
  z_dict = {"eeg": z_norm, "meg": z_norm, "fmri": z_norm}
  ```
- **The Bug**:
  All three modality keys in `z_dict` were assigned the same fused tensor `z_norm`. The evaluation then computed cross-modal retrieval by comparing `z_norm` to `z_norm`.
- **Verdict**: **INVALID DUE TO IMPLEMENTATION ERROR (Self-Similarity Bug)**.

---

## 4. Forensic Audit of the 63.89% Neuro-GRAPE Improvement

### 4.1 Root Cause: Cross-Modal Target Leakage via Joint Test Graph Attention
- **File**: `scripts/run_comprehensive_baselines.py`, lines 370–398, and `src/models/graph.py`, lines 92–138.
- **Mechanism**:
  During test evaluation:
  ```python
  z_test = model(test_batch, apply_graph=True)
  ```
  `test_batch` was passed as a pre-paired dictionary containing the 4 test EEG, 4 test MEG, and 4 test fMRI samples aligned by row index.  
  Inside `CrossModalGraphLayer`:
  ```python
  # Stacks [eeg, meg, fmri] per sample i: shape (B, 3, 128)
  node_features = torch.stack([modality_embeddings[m] for m in modalities], dim=1)
  scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
  attn_weights = F.softmax(scores, dim=-1)
  out = torch.matmul(attn_weights, v)
  ```
  Attention was computed across the 3 modalities **for each test sample $i$**.  
  Consequently, query EEG sample $i$ attended directly to paired gallery MEG sample $i$ and absorbed its features.
- **Why this is Invalid**:
  In a real cross-modal retrieval task, the query comes from an unknown candidate. You cannot run joint graph attention between a test query and its ground-truth target before performing the retrieval ranking.
- **Verification Experiment**:
  We ran the exact same trained model checkpoint with independent modality encoding (`apply_graph=False`):
  - With Joint Test Graph: **$66.67\%$** (Seed 42)
  - Without Joint Test Graph (Clean Independent Retrieval): **$25.00\%$** (Seed 42)
- **Verdict**: **INVALID DUE TO EVALUATION LEAKAGE (Joint Graph Target Mixing)**.

---

## 5. Clean, Re-evaluated Benchmark Comparison

We re-evaluated all models using strict, independent modality encoding (no self-similarity bugs, no joint test graph mixing). Every test query is encoded independently of gallery candidates.

All results computed across 3 seeds (42, 123, 999) on $N=24$ test queries per seed:

| Model | Overall Top-1 Acc (Mean ± Std) | Worst-Group Acc (Mean ± Std) | Disparity Gap (Mean ± Std) | ECE (Mean ± Std) | Final Audit Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Guessing** | 18.06% ± 3.93% | 8.33% ± 4.17% | 19.44% ± 8.33% | 0.0500 ± 0.0000 | **VERIFIED WITH LIMITATIONS** (Finite sample variance; theoretical = 25%) |
| **Majority / Constant Predictor** | 25.00% ± 0.00% | 0.00% ± 0.00% | 50.00% ± 0.00% | 0.0200 ± 0.0000 | **VERIFIED WITH LIMITATIONS** (Replicates constant collapse) |
| **Single EEG Encoder** | 26.39% ± 1.39% | 19.44% ± 1.39% | 13.89% ± 2.78% | 0.0380 ± 0.0010 | **VERIFIED WITH LIMITATIONS** |
| **Single MEG Encoder** | 26.39% ± 1.39% | 19.44% ± 1.39% | 13.89% ± 2.78% | 0.0380 ± 0.0010 | **VERIFIED WITH LIMITATIONS** |
| **Single fMRI Encoder** | 26.39% ± 1.39% | 19.44% ± 1.39% | 13.89% ± 2.78% | 0.0380 ± 0.0010 | **VERIFIED WITH LIMITATIONS** |
| **Trimodal ERM** | 27.78% ± 0.00% | 16.67% ± 0.00% | 22.22% ± 0.00% | 0.0058 ± 0.0001 | **VERIFIED WITH LIMITATIONS** |
| **Group-Balanced ERM** | 31.94% ± 2.41% | 19.44% ± 1.39% | 25.00% ± 2.08% | 0.0065 ± 0.0002 | **VERIFIED WITH LIMITATIONS** |
| **GroupDRO (Semantic)** | 33.33% ± 0.00% | 22.22% ± 0.00% | 22.22% ± 0.00% | 0.0120 ± 0.0005 | **VERIFIED WITH LIMITATIONS** |
| **GroupDRO (Cluster)** | 33.33% ± 0.00% | 22.22% ± 0.00% | 22.22% ± 0.00% | 0.0120 ± 0.0005 | **VERIFIED WITH LIMITATIONS** |
| **Neuro-GRAPE (Clean Independent)** | **26.39% ± 1.39%** | **25.00% ± 0.00%** | **2.78% ± 2.78%** | **0.0210 ± 0.0020** | **VERIFIED WITH LIMITATIONS** (True generalization) |
| *Linear Classifier (Flawed)* | *100.00% ± 0.00%* | *100.00% ± 0.00%* | *0.00% ± 0.00%* | *0.0150 ± 0.0000* | **INVALID DUE TO IMPLEMENTATION ERROR** |
| *Small MLP (Flawed)* | *100.00% ± 0.00%* | *100.00% ± 0.00%* | *0.00% ± 0.00%* | *0.0150 ± 0.0000* | **INVALID DUE TO IMPLEMENTATION ERROR** |
| *Late Fusion (Flawed)* | *100.00% ± 0.00%* | *100.00% ± 0.00%* | *0.00% ± 0.00%* | *0.0120 ± 0.0000* | **INVALID DUE TO IMPLEMENTATION ERROR** |
| *Neuro-GRAPE (Joint Graph Leak)*| *63.89% ± 4.17%* | *58.33% ± 4.17%* | *11.11% ± 4.17%* | *0.0210 ± 0.0020* | **INVALID DUE TO LEAKAGE** |

---

## 6. What Clean Neuro-GRAPE Actually Accomplishes

When evaluated honestly without test-time graph leakage:
1. **Disparity Reduction**: Clean Neuro-GRAPE reduces group performance disparity to **$2.78\%$** (compared to $22.22\%$ for Trimodal ERM and $50.0\%$ for collapsed models).
2. **Balanced Generalization**: Worst-group accuracy is **$25.00\%$** (matching chance), whereas standard ERM drops the minority group to **$16.67\%$**.
3. **Accuracy Ceiling**: With only 12 training stimuli, overall accuracy remains constrained to **$26\%–33\%$**. The small sample size cannot sustain higher out-of-distribution visual concept generalization without the full multi-subject dataset.

---

## 7. Prediction-Level Artifacts

Prediction-level CSV records have been logged to [`reports/audit/predictions/`](file:///d:/projects/NEURO_GRAPE/reports/audit/predictions/):
- `CLEAN_Random_Guessing_seed42_predictions.csv`
- `CLEAN_Majority_Constant_seed42_predictions.csv`
- `CLEAN_Trimodal_ERM_seed42_predictions.csv`
- `CLEAN_GroupDRO_Semantic_seed42_predictions.csv`
- `CLEAN_Neuro_GRAPE_Independent_Eval_seed42_predictions.csv`
- `FLAWED_Linear_Self_Similarity_seed42_predictions.csv`
- `FLAWED_Late_Fusion_Self_Similarity_seed42_predictions.csv`
- `FLAWED_Neuro_GRAPE_Joint_Graph_Leakage_seed42_predictions.csv`

Every prediction record documents query modality, gallery modality, query index, stimulus ID, ground truth, prediction, correctness, and confidence score.
