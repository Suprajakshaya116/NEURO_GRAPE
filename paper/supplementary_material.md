# Supplementary Material: Neuro-GRAPE
## Group-Robust Adaptive Data Mixture Optimization for Multimodal Neural Representation Learning

**PixelPioneers Research Group**  
**Date:** November 2026  
**Companion Document to:** *Neuro-GRAPE: Group-Robust Adaptive Data Mixture Optimization for Multimodal Neural Representation Learning* (IEEE Format Manuscript)

---

## S1. Detailed Architectural Specifications & Parameter Counts

Table S1 provides a complete breakdown of layer types, input/output tensor dimensions, activation functions, and parameter counts across all modules of the Neuro-GRAPE architecture.

**Table S1: Detailed Layer-by-Layer Architectural Specifications.**

| Module | Component / Layer | Input Dimensions | Output Dimensions | Kernel / Stride | Activation / Norm | Parameter Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **EEG Encoder** | Conv1d Block 1 | $(B, 63, 225)$ | $(B, 32, 113)$ | $k=7, s=2, p=3$ | BatchNorm1d, GELU, Dropout1d ($p=0.2$) | 14,112 |
| | Conv1d Block 2 | $(B, 32, 113)$ | $(B, 64, 57)$ | $k=5, s=2, p=2$ | BatchNorm1d, GELU, Dropout1d ($p=0.2$) | 10,240 |
| | Conv1d Block 3 | $(B, 64, 57)$ | $(B, 128, 29)$ | $k=3, s=2, p=1$ | BatchNorm1d, GELU | 24,576 |
| | AdaptiveAvgPool1d | $(B, 128, 29)$ | $(B, 128, 1)$ | - | LayerNorm | 256 |
| | Shared Head ($\mathbf{z}_s$) | $(B, 128)$ | $(B, 128)$ | Linear | LayerNorm, $L_2$ norm | 16,640 |
| | Private Head ($\mathbf{z}_p$) | $(B, 128)$ | $(B, 128)$ | Linear | LayerNorm, $L_2$ norm | 16,640 |
| | Private Decoder | $(B, 128)$ | $(B, 63)$ | Linear | - | 8,127 |
| **MEG Encoder** | Conv1d Block 1 | $(B, 272, 225)$ | $(B, 32, 113)$ | $k=7, s=2, p=3$ | BatchNorm1d, GELU, Dropout1d ($p=0.2$) | 60,928 |
| | Conv1d Block 2 | $(B, 32, 113)$ | $(B, 64, 57)$ | $k=5, s=2, p=2$ | BatchNorm1d, GELU, Dropout1d ($p=0.2$) | 10,240 |
| | Conv1d Block 3 | $(B, 64, 57)$ | $(B, 128, 29)$ | $k=3, s=2, p=1$ | BatchNorm1d, GELU | 24,576 |
| | AdaptiveAvgPool1d | $(B, 128, 29)$ | $(B, 128, 1)$ | - | LayerNorm | 256 |
| | Shared Head ($\mathbf{z}_s$) | $(B, 128)$ | $(B, 128)$ | Linear | LayerNorm, $L_2$ norm | 16,640 |
| | Private Head ($\mathbf{z}_p$) | $(B, 128)$ | $(B, 128)$ | Linear | LayerNorm, $L_2$ norm | 16,640 |
| | Private Decoder | $(B, 128)$ | $(B, 272)$ | Linear | - | 35,088 |
| **fMRI Encoder** *(Archived)* | Conv3d Block 1--3 | $(B, 1, 96, 96, 60)$ | $(B, 64)$ | $k=3, s=2$ | BatchNorm3d, GELU | 69,712 |
| | Shared/Private Heads | $(B, 64)$ | $(B, 128)$ | Linear | LayerNorm, $L_2$ norm | 16,896 |
| **Graph Layer** | Multi-Head Attention | $(B, 2, 128)$ | $(B, 2, 128)$ | $h=4, d_k=32$ | LayerNorm | 66,048 |
| **Total Model** | **Active Pipeline** | - | - | - | - | **415,675 Parameters** |

---

## S2. Benchmark Stimulus Vocabulary & Semantic Partitions

The THINGS-data pilot benchmark consists of $N=20$ harmonized visual stimulus concepts. Table S2 documents the exact concept assignment across partitions and semantic cognitive categories.

**Table S2: Visual Object Concepts in the THINGS-data Pilot Benchmark.**

| Partition | Concept Index | Concept Identifier | Semantic Category | Biological Subgroup | Modality Availability |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Training** | 1 | `bear` | Animal | Animate | EEG, MEG |
| | 2 | `beet` | Food / Plant | Inanimate | EEG, MEG |
| | 3 | `hamburger` | Food | Inanimate | EEG, MEG |
| | 4 | `hay` | Plant / Agriculture | Inanimate | EEG, MEG |
| | 5 | `inhaler` | Tool / Medical | Inanimate | EEG, MEG |
| | 6 | `microscope` | Tool / Science | Inanimate | EEG, MEG |
| | 7 | `pinwheel` | Toy / Object | Inanimate | EEG, MEG |
| | 8 | `puck` | Sport / Object | Inanimate | EEG, MEG |
| | 9 | `screwdriver` | Tool | Inanimate | EEG, MEG |
| | 10 | `boa` | Animal / Reptile | Animate | EEG, MEG |
| | 11 | `bread` | Food | Inanimate | EEG, MEG |
| | 12 | `butterfly` | Animal / Insect | Animate | EEG, MEG |
| **Validation** | 13 | `banner` | Object | Inanimate | EEG, MEG |
| | 14 | `glass` | Object / Container | Inanimate | EEG, MEG |
| | 15 | `sprinkler` | Tool / Agriculture | Inanimate | EEG, MEG |
| | 16 | `tadpole` | Animal / Amphibian | Animate | EEG, MEG |
| **Test** | 17 | `tadpole_02s.jpg` | Animal / Amphibian | Animate | EEG, MEG |
| | 18 | `tarp_02s.jpg` | Material / Cover | Inanimate | EEG, MEG |
| | 19 | `wire_02s.jpg` | Construction / Tool | Inanimate | EEG, MEG |
| | 20 | `woman_02s.jpg` | Person / Human | Animate | EEG, MEG |

### S2.1 Split Statistics
* **Zero Stimulus Overlap:** $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{val}} = \emptyset$, $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{test}} = \emptyset$, $\mathcal{C}_{\text{val}} \cap \mathcal{C}_{\text{test}} = \emptyset$.
* **Balanced Biological Taxonomy:** The test partition contains an exact 50/50 balance between Animate (`tadpole`, `woman`) and Inanimate (`tarp`, `wire`) categories.

---

## S3. Cross-Modal Evaluation Protocol

For any test concept $c \in \mathcal{C}_{\text{test}}$, evaluation measures zero-shot retrieval across both cross-modal directions. Let $\mathcal{Q}$ denote the query modality and $\mathcal{G}$ denote the gallery modality ($\mathcal{Q} \neq \mathcal{G}$).

The directional evaluation pairs are:
$$\mathcal{D} = \{ (\text{EEG} \rightarrow \text{MEG}), (\text{MEG} \rightarrow \text{EEG}) \}$$
Evaluating $|\mathcal{C}_{\text{test}}| = 4$ concepts across $|\mathcal{D}| = 2$ directional pairs yields:
$$N_{\text{trials}} = 4 \times 2 = 8 \text{ directional queries per seed (24 trials across 3 seeds)}$$

Top-1 retrieval is successful if and only if:
$$\arg\max_{c' \in \mathcal{C}_{\text{test}}} \left( \frac{\mathbf{z}_{s}^{(\mathcal{Q})}(c) \cdot \mathbf{z}_{s}^{(\mathcal{G})}(c')}{\|\mathbf{z}_{s}^{(\mathcal{Q})}(c)\|_2 \|\mathbf{z}_{s}^{(\mathcal{G})}(c')\|_2} \right) = c$$

---

## S4. Complete Hyperparameter Configuration

Table S3 documents all hyperparameter settings utilized across preprocessing, training, and evaluation.

**Table S3: Complete Hyperparameter Settings.**

| Hyperparameter Category | Symbol / Key | Description | Configured Value |
| :--- | :---: | :--- | :---: |
| **Optimization** | $\eta$ | AdamW initial learning rate | $1 \times 10^{-3}$ |
| | $\beta_1, \beta_2$ | AdamW momentum coefficients | $0.90, 0.999$ |
| | $\lambda_{\text{decay}}$ | Weight decay regularization | $1 \times 10^{-4}$ |
| | $B$ | Training batch size (concepts) | 12 |
| | $E_{\text{max}}$ | Maximum training epochs | 30 |
| | - | Model selection criterion | Validation InfoNCE Loss |
| **Regularization** | $p$ | Spatial / temporal dropout probability | 0.20 |
| **Representation** | $D$ | Shared latent embedding dimension | 128 |
| | $\tau$ | InfoNCE temperature scaling | 0.25 |
| | $\lambda_{\text{sep}}$ | Orthogonal separation loss weight | 0.50 |
| | $\lambda_{\text{recon}}$ | Private sensor reconstruction loss weight | 0.20 |
| **Graph Attention** | $h$ | Number of attention heads | 4 |
| | $d_k$ | Key / query projection dimension | 32 |
| | $p_{\text{dropout}}$ | Attention dropout probability | 0.10 |
| **GroupDRO** | $\eta_q$ | Exponentiated gradient ascent step size | 0.01 |
| | $\beta_{\text{EMA}}$ | Group loss EMA smoothing factor | 0.90 |
| | $G$ | Number of evaluated groups / regimes | 2 (Animate vs. Inanimate / K-Means Clusters) |
| **Adaptive Mixture** | $\gamma$ | Mixture exponentiated step size | 0.05 |
| | $\beta_{\text{mix}}$ | Modality loss EMA smoothing factor | 0.90 |
| | $\alpha_{\min}$ | Modality starvation allocation floor | 0.15 |

---

## S5. Convergence & Loss Trajectory Diagnostics

During training, standard unweighted ERM exhibits severe gradient oscillation and overfits the 12 training visual stimuli in fewer than 8 epochs. Under Neuro-GRAPE, the 3-phase curriculum and adaptive mixture dynamics smoothly regulate learning progress:

1. **Phase 1: Representation Discovery (Epochs 1–8):** Graph attention is disabled (`cur_apply_graph = False`) and GroupDRO is deactivated ($\lambda_{\text{robust}} = 0.0$), allowing modality encoders to learn stable 1D temporal representations.
2. **Phase 2: Robustness Focus (Epochs 9–18):** GroupDRO is activated ($\lambda_{\text{robust}} = 1.0$), dynamically reweighting lagging regimes. The adaptive mixture sampler begins tracking relative loss velocities.
3. **Phase 3: Graph Consolidation (Epochs 19–30):** Relational cross-modal graph attention is enabled (`cur_apply_graph = True`), refining manifold smoothness across modalities.

---

## S6. Reproducibility & Codebase Organization

The complete project codebase is structured in a modular hierarchy:

```
NEURO_GRAPE/
├── src/
│   ├── data/
│   │   ├── dataset.py            # Multimodal PyTorch dataset loader
│   │   ├── preprocessing.py      # Spatial & temporal filtering pipelines
│   │   └── adaptive_sampler.py   # Online loss-driven modality sampler
│   ├── models/
│   │   ├── encoders.py           # 1D CNN encoders with dual-head projections
│   │   ├── graph.py              # CrossModalGraphLayer multi-head attention
│   │   ├── losses.py             # InfoNCE, separation, and reconstruction
│   │   └── normalization.py      # Preprocessing normalizer fitted on train
│   ├── dro/
│   │   ├── group_dro.py          # Minimax GroupDRO optimizer engine
│   │   └── clustering.py         # Latent KMeans regime discovery
│   └── evaluation/
│       └── retrieval.py          # Certified leakage-free evaluation suite
├── tests/                        # 156 unit and integration tests (all passing)
├── paper/
│   ├── neuro_grape_research_paper.md   # Complete scientific paper manuscript
│   ├── neuro_grape_research_paper.tex  # IEEEtran compilable LaTeX paper
│   ├── references.bib                  # BibTeX bibliographic entries
│   └── supplementary_material.md       # Architectural & split appendix
├── reports/
│   ├── FINAL_LEAKAGE_AUDIT.md          # Forensic leakage and bug elimination audit
│   ├── FINAL_MODEL_OPTIMIZATION.md     # Diagnostic optimization report
│   ├── FINAL_RESULTS.md                # Consolidated multi-seed benchmark report
│   ├── FINAL_REPRODUCIBILITY.md        # Reproduction guide
│   ├── SUBMISSION_CHECKLIST.md         # Point-by-point audit sign-off
│   └── artifacts/
│       ├── full_benchmark_comparison.csv
│       ├── full_benchmark_summary.json
│       ├── predictions.csv             # Traceable predictions across test queries
│       └── plots/                      # Publication-quality figures
├── scripts/
│   ├── run_full_solution_benchmark.py  # Multi-seed 14-condition benchmark harness
│   └── run_member3_pipeline.py         # Member 3 pipeline execution runner
└── model.py                            # Standalone runnable verification entrypoint
```

### Reproducible Execution Commands

1. **Run Full Test Suite (156 Tests)**:
   ```powershell
   python -m unittest discover tests
   ```
2. **Execute Model Forward Pass & Latent Projections**:
   ```powershell
   python model.py
   ```
3. **Execute Full 14-Condition Multi-Seed Benchmark Suite**:
   ```powershell
   python scripts/run_full_solution_benchmark.py
   ```
4. **Execute Member 3 GroupDRO Pipeline**:
   ```powershell
   python scripts/run_member3_pipeline.py
   ```


---

## S7. Scaled Multi-Concept Dataset Specifications & Partitions

The scaled benchmark evaluates Neuro-GRAPE on the full multimodal visual concept dataset:
- **Total Dataset Size**: 2,804 paired trials across 1,499 unique visual object concepts.
- **EEG Dimension**: $(2804, 63, 257)$ float32 (181.6 MB).
- **MEG Dimension**: $(2804, 310, 257)$ float32 (893.6 MB).
- **Partitioning Protocol**: Strictly concept-disjoint splitting (no semantic concept appears in more than one partition):
  - **Train Split**: 1,954 trials (1,049 concepts, 69.7%).
  - **Validation Split**: 424 trials (225 concepts, 15.1%).
  - **Test Split**: 426 trials (225 concepts, 15.2%).
- **Chance Retrieval Baseline**: With 426 test candidates, random guessing chance is $1/426 \approx 0.2347\%$.

---

## S8. Geometric Centering & Hubness Diagnostics

In high-dimensional multimodal embedding spaces, embeddings suffer from the "cone effect" where modality-specific mean vectors dominate cosine similarity.
- **Gallery Centering**: $\tilde{\mathbf{z}}_g = \frac{\mathbf{z}_g - \boldsymbol{\mu}_g}{\|\mathbf{z}_g - \boldsymbol{\mu}_g\|_2}$.
- **Dual Mean-Centering**: Centers both query $\mathbf{z}_q$ and gallery $\mathbf{z}_g$ before computing dot products.
- **Hubness Metrics**:
  - *Hubness Skewness*: Skewness of the distribution of $k$-occurrences (number of times each gallery item appears as rank-1 nearest neighbor).
  - *Max Hub Occurrences*: The highest number of queries falsely captured by a single gallery item. Dual Mean-Centering and CSLS cut max hub occurrences from $89.5 \to 33.0$ on `GroupDRO_Cluster` and $61.5 \to 38.0$ on `Full_Neuro_GRAPE`.

---

## S9. Neural Data Augmentation Specifications

To suppress sample-level memorization without changing the neural architecture:
- **Sensor Channel Dropout**: Randomly drops $15\%$ of channels per sample ($p=0.15$).
- **Temporal Cutout**: Randomly zeros out contiguous $15\%$ temporal slices ($35\text{--}40\text{ ms}$).
- **Gaussian Jitter**: Zero-mean additive Gaussian noise $\mathcal{N}(0, \sigma^2)$ with $\sigma=0.02$.
- **Optimizer Regularization**: AdamW weight decay set to $10^{-2}$.
- **Certified Evaluation**: Augmentations are deactivated during validation and test passes (`apply_graph=False`, clean unaugmented data).
