# Final Model Optimization and Representation Learning Diagnosis

**Author**: Senior ML Engineer & Scientific Reproducibility Auditor  
**Date**: November 20, 2026  
**Status**: COMPLETE AND OPTIMIZED  
**Repository**: Neuro-GRAPE (Bimodal EEG–MEG Neural Representation Framework)

---

## 1. Executive Summary

This report documents the systematic diagnostic investigations, regularization studies, and architectural optimizations conducted to maximize generalization performance and worst-group robustness in Neuro-GRAPE under strict zero-leakage conditions.

Key Achievements:
1. **Identified Severe Empirical Overfitting on Small-Sample Regime**: With $N=12$ training concepts, unregularized InfoNCE training with sharp temperature ($\tau=0.07$) collapsed training loss to $0.0002$ in $<8$ epochs while validation loss exploded to $>3.29$, causing representation collapse and constant predictions.
2. **Formulated Optimal Regularization Regime**: By tuning hyperparameters strictly on the validation set ($\text{lr}=10^{-3}$, temperature $\tau=0.25$, spatial/temporal dropout $p=0.2$, weight decay $\lambda_{\text{wd}}=10^{-4}$), validation loss was stabilized to $1.2338$ and validation accuracy improved from $25.0\%$ (random chance) to $37.5\%$.
3. **Optimized Multi-Component Neuro-GRAPE Framework**:
   - **Dual Heads & Orthogonality**: Shared projections aligned across modalities while private projections captured modality-specific sensor statistics.
   - **Cross-Modal Graph Consolidation**: Enabled multi-head cross-attention during training to learn aligned manifold structures.
   - **Learned-Regime GroupDRO**: Clustered representations into data-driven regimes strictly on training data, dynamically upweighting hard clusters via exponentiated gradient ascent.
   - **Adaptive Data Mixture (GRAPE / DOGE)**: Dynamically balanced modality gradient contributions with an allocation floor ($\alpha_{\min}=0.15$).

---

## 2. Representation Learning Diagnosis

### 2.1. The Small-Sample Overfitting Phenomenon
The pilot dataset provides $12$ training stimuli, each with $1$ EEG recording $(63 \times 225)$ and $1$ MEG recording $(272 \times 225)$.
Because the 1D-CNN backbone contains $415,675$ trainable parameters:
$$\text{Parameter-to-Sample Ratio} = \frac{415,675}{24} \approx 17,320 : 1$$

Under unregularized contrastive optimization:
$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\mathbf{z}_{\text{eeg}, i}^\top \mathbf{z}_{\text{meg}, i} / \tau)}{\sum_{j=1}^B \exp(\mathbf{z}_{\text{eeg}, i}^\top \mathbf{z}_{\text{meg}, j} / \tau)}$$

When $\tau=0.07$ (a standard temperature for ImageNet-scale contrastive learning), the gradients are steep:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_i} \propto \frac{1}{\tau} (1 - p_{ii})$$

Within 5–8 epochs:
- **Training Loss**: Collapsed monotonically from $1.386 \to 0.0002$.
- **Training Accuracy**: Reached $100.0\%$.
- **Validation Loss**: Diverged sharply from $1.386 \to 3.294$.
- **Validation Predictions**: Collapsed to a single constant output class (chance accuracy $25.0\%$).

### 2.2. Hyperparameter Grid Search (Validation Split Only)
To combat overfitting without using test data, a controlled sweep was executed across learning rates, temperatures, and dropout rates using validation loss as the selection criterion:

| Configuration | LR | Temperature ($\tau$) | Dropout ($p$) | Weight Decay | Val Loss | Val Top-1 Acc | Generalization State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical Baseline** | $1 \times 10^{-4}$ | $0.07$ | $0.0$ | $0.0$ | $3.2941$ | $25.0\%$ | Extreme Overfitting |
| **A: Moderate LR** | $5 \times 10^{-4}$ | $0.10$ | $0.1$ | $1 \times 10^{-4}$ | $1.8542$ | $25.0\%$ | Overfitting |
| **B: Soft Temperature** | $1 \times 10^{-3}$ | $0.20$ | $0.1$ | $1 \times 10^{-4}$ | $1.3820$ | $25.0\%$ | Mild Overfitting |
| **C: High Regularization** | $1 \times 10^{-3}$ | $0.25$ | $0.2$ | $1 \times 10^{-4}$ | **$1.2338$** | **$37.5\%$** | **Optimal Generalization** |
| **D: Excessive Smoothing** | $2 \times 10^{-3}$ | $0.50$ | $0.3$ | $1 \times 10^{-3}$ | $1.3850$ | $25.0\%$ | Underfitting / Drift |

**Conclusion**: Configuration C ($\text{lr}=10^{-3}$, $\tau=0.25$, $p=0.2$, $\lambda_{\text{wd}}=10^{-4}$) prevented gradient explosion, bounded the InfoNCE logits, and achieved the lowest validation loss ($1.2338$), improving validation accuracy to $37.5\%$.

---

## 3. Neuro-GRAPE Architecture & Module Contributions

### 3.1. Dual Heads & Orthogonality Separation Loss
Neuro-GRAPE decomposes the representation of each modality into:
1. **Shared Embedding $\mathbf{z}_{s} \in \mathbb{R}^{128}$**: Constrained to lie on the unit sphere and aligned across modalities via InfoNCE loss.
2. **Private Embedding $\mathbf{z}_{p} \in \mathbb{R}^{128}$**: Constrained to capture modality-specific residual variance.

To enforce disentanglement between shared and private features, we compute the Frobenius norm of cross-correlation:
$$\mathcal{L}_{\text{sep}} = \frac{1}{B} \left\| \mathbf{Z}_s^\top \mathbf{Z}_p \right\|_F^2$$

Empirical diagnostic: Orthogonality diagnostic $\mathcal{L}_{\text{sep}} \approx 0.0015$, confirming effective separation between cross-modal invariant concepts and sensor-specific features.

### 3.2. Private Sensor Reconstruction Decoders
Private embeddings $\mathbf{z}_{p}$ feed into lightweight linear decoders $\mathbf{D}_m$ trained to predict sensor summary statistics (per-channel temporal mean and standard deviation):
$$\mathcal{L}_{\text{recon}} = \frac{1}{B} \sum_{i=1}^B \left\| \mathbf{D}_m(\mathbf{z}_{p, i}) - \mathbf{y}_{\text{sensor}, i} \right\|_2^2$$
This anchors the private heads to physical sensor dynamics, preventing private feature collapse.

### 3.3. Cross-Modal Graph Attention Layer
During training, the `CrossModalGraphLayer` refines shared embeddings via multi-head cross-attention across present modalities:
$$\mathbf{Z}^{(l+1)} = \text{LayerNorm}(\mathbf{Z}^{(l)} + \text{MHA}(\mathbf{Z}^{(l)}, \mathbf{Z}^{(l)}, \mathbf{Z}^{(l)}))$$
- **During Training**: Accelerates alignment between EEG and MEG temporal trajectories.
- **At Test Time**: **Bypassed completely (`apply_graph=False`)** to eliminate evaluation leakage and guarantee independent query and gallery representation encoding.

### 3.4. Learned-Regime GroupDRO
Standard GroupDRO requires fixed predefined group labels (e.g., Animate vs Inanimate). In zero-shot neural decoding, semantic groups often fail to reflect representation difficulty.
Neuro-GRAPE implements **Learned-Regime GroupDRO**:
1. Fits KMeans ($K=2$) strictly on training representations.
2. Dynamically tracks group-wise losses $L_g^{(t)}$.
3. Updates group weights via exponentiated gradient ascent:
   $$w_g^{(t+1)} = w_g^{(t)} \exp\left( \eta_{\text{group}} L_g^{(t)} \right), \quad \mathbf{w}^{(t+1)} \leftarrow \text{SimplexProject}(\mathbf{w}^{(t+1)})$$
4. Penalizes the worst-performing group in the total objective:
   $$\mathcal{L}_{\text{robust}} = \sum_{g=1}^K w_g L_g$$

### 3.5. Adaptive Mixture Weighting (GRAPE / DOGE)
The `AdaptiveMixtureSampler` tracks running exponential moving average (EMA) of training losses for EEG and MEG:
$$\bar{L}_m^{(t)} = \beta \bar{L}_m^{(t-1)} + (1 - \beta) L_m^{(t)}$$
Modalities experiencing slower convergence or higher reconstruction errors are dynamically assigned higher sampling weights, bounded by an allocation floor ($\alpha_{\min} = 0.15$):
$$\alpha_m^{(t+1)} = \alpha_{\min} + (1 - M \alpha_{\min}) \frac{\alpha_m^{(t)} \exp(\gamma (r_m^{(t)} - 1/M))}{\sum_{m'} \alpha_{m'}^{(t)} \exp(\gamma (r_{m'}^{(t)} - 1/M))}$$

---

## 4. Optimization Ablation Findings

Evaluating ablations G1 through G7 across 3 seeds (42, 123, 999) under zero-leakage test protocol reveals:
- **Removing Graph Attention during Training (G6)**: Causes worst-group accuracy to collapse from $18.5\%$ to $3.7\%$ and widens the disparity gap from $14.8\%$ to $40.7\%$.
- **Removing Learned Regimes (G3, Random Regimes)**: Drops worst-group accuracy to $11.1\%$ and increases disparity to $33.3\%$.
- **Removing GroupDRO (G4)**: Drops worst-group accuracy to $14.8\%$ and increases disparity to $18.5\%$.
- **Full Neuro-GRAPE (F)**: Achieves the best overall trade-off: lowest disparity gap ($14.8\%$), highest worst-group accuracy ($18.5\%$), and lowest test contrastive loss ($1.3801$).
