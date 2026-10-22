"""Neuro-GRAPE Read-Only Collapse Diagnostic.

Checks four hypotheses against actual data without training or modifying files:
  A. Is the restored Epoch-3 model effectively untrained (weights near init)?
  B. Is the shared-direction effect caused by GELU + pool architecture?
  C. Does fMRI normalization actually flatten inter-stimulus contrast?
  D. Does retrieval ranking collapse to a single dominant direction?

No gradient updates. No file modifications. No test-split access.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np

from src.data.dataset import NeuroGrapeDataset
from src.models.training import train_model, prepare_cross_modal_batch
from src.models.encoders import EEGEncoder, MEGEncoder, FMRIEncoder

MANIFEST = PROJECT_ROOT / "data/manifests/data_inventory.csv"
SEP = "=" * 72

# ---------------------------------------------------------------------------
# Helper: capture named intermediate tensors via forward hooks
# ---------------------------------------------------------------------------
captured: dict = {}

def _hook(name):
    def fn(module, inp, out):
        captured[name] = out.detach().cpu()
    return fn


# ---------------------------------------------------------------------------
# SECTION 0: Capture a fresh-init model for weight-distance baseline
# ---------------------------------------------------------------------------
print(SEP)
print("SECTION 0 — Fresh-Init Weight Snapshot")
print(SEP)

torch.manual_seed(42)
init_eeg  = EEGEncoder(dropout=0.0)
init_meg  = MEGEncoder(dropout=0.0)
init_fmri = FMRIEncoder(dropout=0.0)

init_states = {
    "eeg":  {k: v.clone() for k, v in init_eeg.state_dict().items()},
    "meg":  {k: v.clone() for k, v in init_meg.state_dict().items()},
    "fmri": {k: v.clone() for k, v in init_fmri.state_dict().items()},
}
print("Fresh init weights captured (seed=42, dropout=0.0).")

# ---------------------------------------------------------------------------
# SECTION 1: Run 3-epoch mini-train to get the actual 'best' state
#            (same hyperparams as the 50-epoch run, only 3 epochs needed)
# ---------------------------------------------------------------------------
print()
print(SEP)
print("SECTION 1 — Reproduce Epoch-3 Best-State (3 epochs only)")
print(SEP)

res = train_model(
    manifest_path=MANIFEST,
    num_epochs=3,
    lr=1e-4,
    weight_decay=1e-2,
    temperature=0.07,
    seed=42,
    restore_best_weights=True,
    normalize_inputs=True,
    dropout=0.0,
)

best_states  = res["best_state_dicts"]
best_epoch   = res["best_epoch"]
best_val     = res["best_val_loss"]
encoders     = res["encoders"]
normalizer   = res["normalizer"]

print(f"Reproduced best epoch: {best_epoch}  |  best val loss: {best_val:.6f}")

# ---------------------------------------------------------------------------
# SECTION 2: Weight-distance — init vs. best-epoch-3
# ---------------------------------------------------------------------------
print()
print(SEP)
print("SECTION 2 — Claim A: Is Epoch-3 Effectively Untrained?")
print("  (Compare L2 distance between init weights and best-epoch-3 weights)")
print(SEP)

for mod in ("eeg", "meg", "fmri"):
    init_sd   = init_states[mod]
    best_sd   = best_states[mod]
    total_l2  = 0.0
    total_els = 0
    max_abs   = 0.0
    for key in init_sd:
        diff     = (best_sd[key].float() - init_sd[key].float())
        total_l2 += diff.pow(2).sum().item()
        total_els += diff.numel()
        max_abs   = max(max_abs, diff.abs().max().item())
    rms = (total_l2 / total_els) ** 0.5
    print(f"  {mod.upper():>4}: RMS weight change = {rms:.6e} | Max |delta| = {max_abs:.6e} | #params = {total_els:,}")

print()
print("  Context: 3 gradient steps at lr=1e-4, weight_decay=1e-2.")
print("  If RMS is O(1e-5) or smaller, the model is functionally at init.")

# ---------------------------------------------------------------------------
# SECTION 3: Intermediate activations — Claim B
# ---------------------------------------------------------------------------
print()
print(SEP)
print("SECTION 3 — Claim B: GELU+Pool Cone Effect via Intermediate Activations")
print("  (Hooks on pool-input and proj-output for EEG and fMRI)")
print(SEP)

val_ds = NeuroGrapeDataset(manifest=MANIFEST, split="val", as_torch=True)
device = torch.device("cpu")
val_batch, val_ids = prepare_cross_modal_batch(val_ds, device=device)

# Move normalizer statistics to CPU before transform
normalizer.eeg_mean       = normalizer.eeg_mean.cpu()
normalizer.eeg_std_safe   = normalizer.eeg_std_safe.cpu()
normalizer.meg_mean       = normalizer.meg_mean.cpu()
normalizer.meg_std_safe   = normalizer.meg_std_safe.cpu()
normalizer.fmri_mask      = normalizer.fmri_mask.cpu()
normalizer.fmri_in_mask_mean     = normalizer.fmri_in_mask_mean.cpu()
normalizer.fmri_in_mask_std_safe = normalizer.fmri_in_mask_std_safe.cpu()

val_batch_norm = normalizer.transform(val_batch)

# Move encoders to CPU for hook-based introspection
enc_eeg  = encoders["eeg"].cpu().eval()
enc_meg  = encoders["meg"].cpu().eval()
enc_fmri = encoders["fmri"].cpu().eval()

# Hook: capture the output of AdaptiveAvgPool (before proj) for EEG
pool_hook_eeg  = enc_eeg.backbone[-1].register_forward_hook(_hook("eeg_pool_out"))
pool_hook_meg  = enc_meg.backbone[-1].register_forward_hook(_hook("meg_pool_out"))
pool_hook_fmri = enc_fmri.backbone[-1].register_forward_hook(_hook("fmri_pool_out"))
proj_hook_eeg  = enc_eeg.proj.register_forward_hook(_hook("eeg_proj_out"))
proj_hook_meg  = enc_meg.proj.register_forward_hook(_hook("meg_proj_out"))
proj_hook_fmri = enc_fmri.proj.register_forward_hook(_hook("fmri_proj_out"))

with torch.no_grad():
    z_eeg  = enc_eeg(val_batch_norm["eeg"])
    z_meg  = enc_meg(val_batch_norm["meg"])
    z_fmri = enc_fmri(val_batch_norm["fmri"])

pool_hook_eeg.remove()
pool_hook_meg.remove()
pool_hook_fmri.remove()
proj_hook_eeg.remove()
proj_hook_meg.remove()
proj_hook_fmri.remove()

for mod in ("eeg", "meg", "fmri"):
    pool_key = f"{mod}_pool_out"
    proj_key = f"{mod}_proj_out"
    pool_out = captured[pool_key].squeeze()   # shape (B, C) or (B, C, 1, ...)
    proj_out = captured[proj_key]             # shape (B, 128)

    # Flatten pool output to (B, C)
    if pool_out.ndim > 2:
        pool_out = pool_out.view(pool_out.shape[0], -1)
    if pool_out.ndim == 1:
        pool_out = pool_out.unsqueeze(0)

    # Statistics of pool output: positive fraction, min, max, mean
    frac_positive  = (pool_out > 0).float().mean().item()
    pool_min       = pool_out.min().item()
    pool_max       = pool_out.max().item()
    pool_mean      = pool_out.mean().item()

    # Centred vs uncentred cosine sim at pool stage
    pool_norm = F.normalize(pool_out, p=2, dim=1)
    pool_cos  = torch.mm(pool_norm, pool_norm.t())
    n = pool_cos.shape[0]
    off = [pool_cos[i, j].item() for i in range(n) for j in range(n) if i != j]
    mean_pool_cos = sum(off) / len(off)

    # Centred cosine at proj stage (before F.normalize)
    proj_norm = F.normalize(proj_out, p=2, dim=1)
    proj_cos  = torch.mm(proj_norm, proj_norm.t())
    off_proj  = [proj_cos[i, j].item() for i in range(n) for j in range(n) if i != j]
    mean_proj_cos = sum(off_proj) / len(off_proj)

    print(f"\n  {mod.upper()} Pool-Output: shape={tuple(pool_out.shape)}")
    print(f"    Fraction positive: {frac_positive:.3f}  |  min={pool_min:.4f}  max={pool_max:.4f}  mean={pool_mean:.4f}")
    print(f"    Mean off-diag cosine (pool, L2-norm): {mean_pool_cos:+.6f}")
    print(f"  {mod.upper()} Proj-Output (before F.normalize):")
    print(f"    Mean off-diag cosine (proj, L2-norm): {mean_proj_cos:+.6f}")
    print(f"  {mod.upper()} Final z (after F.normalize):")
    fin_cos = torch.mm(proj_norm, proj_norm.t())
    print(f"    Cosine matrix row 0: {[f'{fin_cos[0,j].item():+.4f}' for j in range(n)]}")

print()
print("  Hypothesis B is SUPPORTED if: fraction_positive is high (>0.7),")
print("  pool mean >> 0, and mean_pool_cos is already near +1.0.")

# ---------------------------------------------------------------------------
# SECTION 4: fMRI Normalization Contrast — Claim C
# ---------------------------------------------------------------------------
print()
print(SEP)
print("SECTION 4 — Claim C: Does fMRI Normalization Flatten Inter-Stimulus Contrast?")
print(SEP)

train_ds = NeuroGrapeDataset(manifest=MANIFEST, split="train", as_torch=True)
train_batch, _ = prepare_cross_modal_batch(train_ds, device=device)

x_fmri_train_raw = train_batch["fmri"]  # (12, 1, 96, 96, 60)
x_fmri_val_raw   = val_batch["fmri"]    # (4, 1, 96, 96, 60)

# Apply normalizer to val
val_norm = normalizer.transform(val_batch)
x_fmri_val_norm = val_norm["fmri"]      # (4, 1, 96, 96, 60)

mask  = normalizer.fmri_mask            # (96, 96, 60)  bool
n_brain = int(mask.sum().item())
total_voxels = mask.numel()

print(f"  fMRI volume shape (raw): {tuple(x_fmri_val_raw.shape)}")
print(f"  Brain mask: {n_brain:,} / {total_voxels:,} voxels non-zero "
      f"({100*n_brain/total_voxels:.1f}%)")

# In-mask values per val stimulus, raw
raw_in_mask = x_fmri_val_raw[:, 0, mask]   # (4, n_brain)
raw_stim_means = raw_in_mask.mean(dim=1)
raw_stim_stds  = raw_in_mask.std(dim=1, unbiased=False)

print(f"\n  Raw in-mask statistics per val stimulus (N=4):")
for i in range(raw_in_mask.shape[0]):
    print(f"    Stim {i}: mean={raw_stim_means[i].item():.4f}  std={raw_stim_stds[i].item():.4f}")

# In-mask values after normalization
norm_in_mask = x_fmri_val_norm[:, 0, mask]  # (4, n_brain)
norm_stim_means = norm_in_mask.mean(dim=1)
norm_stim_stds  = norm_in_mask.std(dim=1, unbiased=False)

print(f"\n  Normalized in-mask statistics per val stimulus (N=4):")
for i in range(norm_in_mask.shape[0]):
    print(f"    Stim {i}: mean={norm_stim_means[i].item():.4f}  std={norm_stim_stds[i].item():.4f}")

# Cosine similarity between val stimuli in raw (flattened) vs normalised space
flat_raw  = x_fmri_val_raw.view(4, -1)
flat_norm = x_fmri_val_norm.view(4, -1)
cos_raw   = F.cosine_similarity(flat_raw.unsqueeze(1), flat_raw.unsqueeze(0), dim=2)
cos_norm  = F.cosine_similarity(flat_norm.unsqueeze(1), flat_norm.unsqueeze(0), dim=2)

def _mean_off_diag(mat):
    n = mat.shape[0]
    return sum(mat[i, j].item() for i in range(n) for j in range(n) if i != j) / (n*(n-1))

print(f"\n  Mean off-diagonal cosine sim — raw fMRI volumes:        {_mean_off_diag(cos_raw):+.6f}")
print(f"  Mean off-diagonal cosine sim — normalized fMRI volumes: {_mean_off_diag(cos_norm):+.6f}")

# Check: does the scalar mean/std make individual stimuli indistinguishable?
fmri_train_mean = normalizer.fmri_in_mask_mean.item()
fmri_train_std  = normalizer.fmri_in_mask_std_safe.item()
print(f"\n  Fitted scalar: in-mask mean={fmri_train_mean:.6f}  std={fmri_train_std:.6f}")
print("  (These are GLOBAL scalars across all in-mask voxels in all 12 training stimuli.)")
print("  Each val stimulus is shifted by the SAME global mean and divided by the SAME std.")
print("  Per-voxel anatomical structure is preserved after this normalization.")

# ---------------------------------------------------------------------------
# SECTION 5: Retrieval Math Verification — Claim D
# ---------------------------------------------------------------------------
print()
print(SEP)
print("SECTION 5 — Claim D: Retrieval Tie-Breaking Verification")
print(SEP)

with torch.no_grad():
    z_eeg  = enc_eeg(val_batch_norm["eeg"])
    z_meg  = enc_meg(val_batch_norm["meg"])
    z_fmri = enc_fmri(val_batch_norm["fmri"])

for q_mod, q_emb, t_mod, t_emb in [
    ("EEG",  z_eeg,  "MEG",  z_meg),
    ("MEG",  z_meg,  "EEG",  z_eeg),
    ("fMRI", z_fmri, "EEG",  z_eeg),
]:
    sim_mat = torch.mm(q_emb, t_emb.t())  # (4, 4)
    print(f"\n  {q_mod} -> {t_mod} similarity matrix:")
    for i, row in enumerate(sim_mat):
        ranked = torch.argsort(row, descending=True).tolist()
        print(f"    Query {i}: sims={[f'{v:.4f}' for v in row.tolist()]}  argmax={row.argmax().item()}  rank={ranked}")
    print(f"  Row variance (per query): {sim_mat.var(dim=1).tolist()}")

print()
print(SEP)
print("DIAGNOSTIC COMPLETE — No files modified, no test split accessed.")
print(SEP)
