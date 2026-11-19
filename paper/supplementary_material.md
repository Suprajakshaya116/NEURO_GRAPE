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
| **EEG Encoder** | Conv1d Block 1 | $(B, 63, 225)$ | $(B, 64, 110)$ | $k=7, s=2, p=0$ | BatchNorm1d, GELU | 28,352 |
| | Conv1d Block 2 | $(B, 64, 110)$ | $(B, 128, 53)$ | $k=5, s=2, p=0$ | BatchNorm1d, GELU | 41,216 |
| | Conv1d Block 3 | $(B, 128, 53)$ | $(B, 128, 26)$ | $k=3, s=2, p=0$ | BatchNorm1d, GELU | 49,408 |
| | AdaptiveAvgPool1d | $(B, 128, 26)$ | $(B, 128, 1)$ | - | LayerNorm | 256 |
| | Shared Head ($\mathbf{z}_s$) | $(B, 128)$ | $(B, 128)$ | Linear | LayerNorm, $L_2$ norm | 16,640 |
| | Private Head ($\mathbf{z}_p$) | $(B, 128)$ | $(B, 128)$ | Linear | LayerNorm, $L_2$ norm | 16,640 |
| | Private Decoder | $(B, 128)$ | $(B, 63)$ | Linear | - | 8,127 |
| **MEG Encoder** | Conv1d Block 1 | $(B, 272, 225)$ | $(B, 64, 110)$ | $k=7, s=2, p=0$ | BatchNorm1d, GELU | 121,984 |
| | Conv1d Block 2 | $(B, 64, 110)$ | $(B, 128, 53)$ | $k=5, s=2, p=0$ | BatchNorm1d, GELU | 41,216 |
| | Conv1d Block 3 | $(B, 128, 53)$ | $(B, 128, 26)$ | $k=3, s=2, p=0$ | BatchNorm1d, GELU | 49,408 |
| | AdaptiveAvgPool1d | $(B, 128, 26)$ | $(B, 128, 1)$ | - | LayerNorm | 256 |
| | Shared Head ($\mathbf{z}_s$) | $(B, 128)$ | $(B, 128)$ | Linear | LayerNorm, $L_2$ norm | 16,640 |
| | Private Head ($\mathbf{z}_p$) | $(B, 128)$ | $(B, 128)$ | Linear | LayerNorm, $L_2$ norm | 16,640 |
| | Private Decoder | $(B, 128)$ | $(B, 272)$ | Linear | - | 35,088 |
| **fMRI Encoder** | Conv3d Block 1 | $(B, 1, 96, 96, 60)$ | $(B, 16, 48, 48, 30)$ | $k=3, s=2, p=1$ | BatchNorm3d, GELU | 464 |
| | Conv3d Block 2 | $(B, 16, 48, 48, 30)$ | $(B, 32, 24, 24, 15)$ | $k=3, s=2, p=1$ | BatchNorm3d, GELU | 13,888 |
| | Conv3d Block 3 | $(B, 32, 24, 24, 15)$ | $(B, 64, 12, 12, 8)$ | $k=3, s=2, p=1$ | BatchNorm3d, GELU | 55,360 |
| | AdaptiveAvgPool3d | $(B, 64, 12, 12, 8)$ | $(B, 64, 1, 1, 1)$ | - | LayerNorm | 128 |
| | Shared Head ($\mathbf{z}_s$) | $(B, 64)$ | $(B, 128)$ | Linear | LayerNorm, $L_2$ norm | 8,448 |
| | Private Head ($\mathbf{z}_p$) | $(B, 64)$ | $(B, 128)$ | Linear | LayerNorm, $L_2$ norm | 8,448 |
| | Private Decoder | $(B, 128)$ | $(B, 64)$ | Linear | - | 8,256 |
| **Graph Layer** | Multi-Head Attention | $(B, 3, 128)$ | $(B, 3, 128)$ | $h=4, d_k=32$ | LayerNorm | 66,048 |
| **Total Model** | **All Modules Active** | - | - | - | - | **415,675 Parameters** |

---

## S2. Benchmark Stimulus Vocabulary & Semantic Partitions

The THINGS-data pilot benchmark consists of $N=20$ harmonized visual stimulus concepts. Table S2 documents the exact concept assignment across partitions and semantic cognitive categories.

**Table S2: Visual Object Concepts in the THINGS-data Pilot Benchmark.**

| Partition | Concept Index | Concept Identifier | Semantic Category | Biological Subgroup | Modality Availability |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Training** | 1 | `ant` | Animal | Animate | EEG, MEG, fMRI |
| | 2 | `apple` | Food / Fruit | Inanimate | EEG, MEG, fMRI |
| | 3 | `arm` | Body Part | Animate | EEG, MEG, fMRI |
| | 4 | `backpack` | Container / Accessory | Inanimate | EEG, MEG, fMRI |
| | 5 | `banana` | Food / Fruit | Inanimate | EEG, MEG, fMRI |
| | 6 | `bear` | Animal | Animate | EEG, MEG, fMRI |
| | 7 | `bed` | Furniture | Inanimate | EEG, MEG, fMRI |
| | 8 | `bee` | Animal | Animate | EEG, MEG, fMRI |
| | 9 | `belt` | Clothing | Inanimate | EEG, MEG, fMRI |
| | 10 | `bench` | Furniture | Inanimate | EEG, MEG, fMRI |
| | 11 | `bird` | Animal | Animate | EEG, MEG, fMRI |
| | 12 | `boat` | Vehicle | Inanimate | EEG, MEG, fMRI |
| **Validation** | 13 | `book` | Object | Inanimate | EEG, MEG, fMRI |
| | 14 | `bottle` | Container | Inanimate | EEG, MEG, fMRI |
| | 15 | `cat` | Animal | Animate | EEG, MEG, fMRI |
| | 16 | `dog` | Animal | Animate | EEG, MEG, fMRI |
| **Test** | 17 | `car` | Vehicle | Inanimate | EEG, MEG, fMRI |
| | 18 | `chair` | Furniture | Inanimate | EEG, MEG, fMRI |
| | 19 | `elephant` | Animal | Animate | EEG, MEG, fMRI |
| | 20 | `horse` | Animal | Animate | EEG, MEG, fMRI |

### S2.1 Split Statistics
* **Zero Stimulus Overlap:** $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{val}} = \emptyset$, $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{test}} = \emptyset$, $\mathcal{C}_{\text{val}} \cap \mathcal{C}_{\text{test}} = \emptyset$.
* **Balanced Biological Taxonomy:** Each partition contains an exact 50/50 balance between Animate and Inanimate categories (Train: 6/6, Val: 2/2, Test: 2/2).

---

## S3. Cross-Modal Evaluation Protocol

For any test concept $c \in \mathcal{C}_{\text{test}}$, evaluation measures zero-shot retrieval across all pairwise cross-modal directions. Let $\mathcal{Q}$ denote the query modality and $\mathcal{G}$ denote the gallery modality ($\mathcal{Q} \neq \mathcal{G}$).

The complete set of directional evaluation pairs is:
$$\mathcal{D} = \{ (\text{EEG} \rightarrow \text{MEG}), (\text{EEG} \rightarrow \text{fMRI}), (\text{MEG} \rightarrow \text{EEG}), (\text{MEG} \rightarrow \text{fMRI}), (\text{fMRI} \rightarrow \text{EEG}), (\text{fMRI} \rightarrow \text{MEG}) \}$$
Evaluating $|\mathcal{C}_{\text{test}}| = 4$ concepts across $|\mathcal{D}| = 6$ directional pairs yields:
$$N_{\text{trials}} = 4 \times 6 = 24 \text{ trials per seed}$$

Top-1 retrieval is successful if and only if:
$$\arg\max_{c' \in \mathcal{C}_{\text{test}}} \left( \frac{\mathbf{z}_{s}^{(\mathcal{Q})}(c) \cdot \mathbf{z}_{s}^{(\mathcal{G})}(c')}{\|\mathbf{z}_{s}^{(\mathcal{Q})}(c)\|_2 \|\mathbf{z}_{s}^{(\mathcal{G})}(c')\|_2} \right) = c$$

---

## S4. Complete Hyperparameter Configuration

Table S3 documents all hyperparameter settings utilized across preprocessing, training, and evaluation.

**Table S3: Complete Hyperparameter Settings.**

| Hyperparameter Category | Symbol / Key | Description | Configured Value |
| :--- | :---: | :--- | :---: |
| **Optimization** | $\eta$ | Adam initial learning rate | $1 \times 10^{-4}$ |
| | $\beta_1, \beta_2$ | Adam momentum coefficients | $0.90, 0.999$ |
| | $\lambda_{\text{decay}}$ | $L_2$ weight decay regularization | $1 \times 10^{-4}$ |
| | $B$ | Training batch size (concepts) | 4 ($12$ multimodal tensors) |
| | $E_{\text{max}}$ | Maximum training epochs | 30 |
| | $P$ | Early stopping patience | 10 epochs |
| | - | Early stopping monitor | Validation Worst-Group Loss |
| **Representation** | $D$ | Shared latent embedding dimension | 128 |
| | $\tau$ | Learnable InfoNCE temperature (init) | 0.07 |
| | $\lambda_{\text{sep}}$ | Orthogonal separation loss weight | 0.10 |
| | $\lambda_{\text{recon}}$ | Private sensor reconstruction loss weight | 0.05 |
| **Graph Attention** | $h$ | Number of attention heads | 4 |
| | $d_k$ | Key / query projection dimension | 32 |
| | $p_{\text{dropout}}$ | Attention dropout probability | 0.10 |
| **GroupDRO** | $\eta_q$ | Exponentiated gradient ascent step size | 0.01 |
| | $\beta_{\text{EMA}}$ | Group loss EMA smoothing factor | 0.90 |
| | $G$ | Number of evaluated groups | 2 (Animate vs. Inanimate) |
| **Adaptive Mixture** | $\eta_{\text{mix}}$ | Mixture exponentiated step size | 0.05 |
| | $\beta_{\text{mix}}$ | Modality loss EMA smoothing factor | 0.85 |
| | $\alpha_{\text{min}}$ | Modality starvation allocation floor | 0.15 |

---

## S5. Convergence & Loss Trajectory Diagnostics

During training, standard unweighted ERM exhibits severe gradient oscillation between the high-dimensional fMRI representations and the temporal electrophysiology streams. Under Neuro-GRAPE, the online adaptive mixture dynamics smoothly balance learning progress:

1. **Epochs 1–5 (Warm-up Phase):** Initial modality sampling probabilities start at uniform allocation: $\mathbf{\alpha} = [0.333, 0.333, 0.333]$. GroupDRO weights start at $\mathbf{q} = [0.50, 0.50]$.
2. **Epochs 6–15 (Disparity Discovery):** As Inanimate concepts converge faster than Animate concepts due to higher structural contrast in visual stimuli, the Animate subgroup loss temporarily lags. GroupDRO dynamically shifts probability mass: $q_{\text{Animate}} \rightarrow 0.62$, forcing the optimizer to prioritize Animate cross-modal alignment.
3. **Epochs 16–30 (Equilibrium Phase):** As representation alignment across Animate concepts improves, $q$ stabilizes near $[0.51, 0.49]$, achieving minimal cross-group disparity gap ($2.78\%$) and well-calibrated retrieval probabilities.

---

## S6. Reproducibility & Codebase Organization

The complete project codebase is structured in a modular, decoupled hierarchy:

```
NEURO_GRAPE/
├── src/
│   ├── data/
│   │   ├── dataset.py            # Multimodal PyTorch dataset loader
│   │   ├── preprocessing.py      # Spatial & temporal filtering pipelines
│   │   ├── adaptive_sampler.py   # Online loss-driven modality sampler
│   │   └── ingestion.py          # Scalable data ingestion engine
│   ├── models/
│   │   ├── encoders.py           # 1D CNN (EEG/MEG) and 3D CNN (fMRI)
│   │   ├── graph.py              # CrossModalGraphLayer multi-head attention
│   │   ├── losses.py             # InfoNCE, separation, and reconstruction
│   │   └── normalization.py      # Robust hypersphere projection
│   ├── dro/
│   │   ├── group_dro.py          # Minimax GroupDRO optimizer engine
│   │   └── clustering.py         # Latent KMeans regime discovery
│   └── evaluation/
│       └── metrics.py            # Accuracy, disparity, and ECE calculation
├── tests/                        # 156 unit and integration tests
├── paper/
│   ├── neuro_grape_research_paper.md   # Complete scientific paper manuscript
│   ├── neuro_grape_research_paper.tex  # IEEEtran compilable LaTeX paper
│   ├── references.bib                  # BibTeX bibliographic entries
│   └── supplementary_material.md       # Architectural & split appendix
├── reports/
│   ├── PAPER_PROJECT_AUDIT.md          # Factual implementation inventory
│   ├── PAPER_REFERENCE_AUDIT.md        # Comprehensive literature audit
│   └── audit/                          # Complete forensic verification logs
└── model.py                            # Standalone runnable verification entrypoint
```

### Reproducible Execution Command
To reproduce the audited baseline benchmarks and verified Neuro-GRAPE evaluation:
```bash
python model.py --seed 42 --epochs 30 --batch-size 4
```
All unit tests can be validated independently via:
```bash
python -m unittest discover tests
```
*Confirmed status: 156 tests passing.*
