"""Neuro-GRAPE LayerNorm Diagnostic — Read-Only, No Training.

Measures the effect of the newly inserted nn.LayerNorm on embedding cosine
similarity using:
  A. Fresh-init encoders + synthetic random inputs.
  B. Fresh-init encoders + real (normalizer-preprocessed) validation inputs.

NOTE: No trained model checkpoint is available in-memory; this script does NOT
call train_model(). All measurements are on untrained encoders only.
Measurements are clearly labelled UNTRAINED throughout.

No files are modified. Only train and val splits are accessed. Test split is
never loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from src.data.dataset import NeuroGrapeDataset
from src.models.encoders import EEGEncoder, FMRIEncoder, MEGEncoder
from src.models.normalization import TrimodalNormalizer
from src.models.training import prepare_cross_modal_batch

MANIFEST = PROJECT_ROOT / "data/manifests/data_inventory.csv"
SEP = "=" * 72


def mean_off_diag_cosine(z: torch.Tensor) -> float:
    """Mean cosine similarity of all off-diagonal pairs in (N, D) embedding."""
    cos = torch.mm(z, z.t())
    n = cos.shape[0]
    return sum(cos[i, j].item() for i in range(n) for j in range(n) if i != j) / (n * (n - 1))


def row_variance(sim: torch.Tensor) -> list:
    return sim.var(dim=1).tolist()


# ---------------------------------------------------------------------------
# SECTION 0: Verify LayerNorm is present and correctly wired
# ---------------------------------------------------------------------------
print(SEP)
print("SECTION 0 — Structural Verification: pre_norm attribute")
print(SEP)
for cls in (EEGEncoder, MEGEncoder, FMRIEncoder):
    enc = cls()
    has_attr = hasattr(enc, "pre_norm")
    is_ln    = isinstance(enc.pre_norm, torch.nn.LayerNorm) if has_attr else False
    shape    = enc.pre_norm.normalized_shape if is_ln else "N/A"
    print(f"  {cls.__name__:15s}: has pre_norm={has_attr}, "
          f"is LayerNorm={is_ln}, normalized_shape={shape}")


# ---------------------------------------------------------------------------
# SECTION 1: UNTRAINED encoders — synthetic random inputs
# ---------------------------------------------------------------------------
print()
print(SEP)
print("SECTION 1 — UNTRAINED Encoders — Synthetic Random Inputs")
print("  (Baseline: without LayerNorm, EEG=+0.982, MEG=+0.972, fMRI=+0.999)")
print(SEP)

torch.manual_seed(0)
for name, enc, x in [
    ("EEG",  EEGEncoder().eval(),  torch.randn(8, 63, 225)),
    ("MEG",  MEGEncoder().eval(),  torch.randn(8, 272, 225)),
    ("fMRI", FMRIEncoder().eval(), torch.randn(4, 1, 96, 96, 60)),
]:
    with torch.no_grad():
        z = enc(x)
    cos = mean_off_diag_cosine(z)
    print(f"  {name:5s} mean off-diag cosine (UNTRAINED, synthetic): {cos:+.6f}")


# ---------------------------------------------------------------------------
# SECTION 2: UNTRAINED encoders — real validation inputs + train normalizer
# ---------------------------------------------------------------------------
print()
print(SEP)
print("SECTION 2 — UNTRAINED Encoders — Real Val Inputs (train-fitted normalizer)")
print("  Normalizer fitted on training split (N=12). Validation split (N=4) only.")
print(SEP)

# Load and normalize training data to fit normalizer
train_ds = NeuroGrapeDataset(manifest=MANIFEST, split="train", as_torch=True)
val_ds   = NeuroGrapeDataset(manifest=MANIFEST, split="val",   as_torch=True)

cpu = torch.device("cpu")
train_batch, train_ids = prepare_cross_modal_batch(train_ds, device=cpu)
val_batch,   val_ids   = prepare_cross_modal_batch(val_ds,   device=cpu)

normalizer = TrimodalNormalizer()
normalizer.fit(train_batch)

val_norm = normalizer.transform(val_batch)

print(f"  Val stimuli: {val_ids['eeg']}")
print()

torch.manual_seed(0)
encs = {
    "eeg":  EEGEncoder().cpu().eval(),
    "meg":  MEGEncoder().cpu().eval(),
    "fmri": FMRIEncoder().cpu().eval(),
}

with torch.no_grad():
    z = {mod: encs[mod](val_norm[mod]) for mod in ("eeg", "meg", "fmri")}

for mod in ("eeg", "meg", "fmri"):
    emb = z[mod]
    cos = mean_off_diag_cosine(emb)
    norms = emb.norm(p=2, dim=1).tolist()
    print(f"  {mod.upper():5s} mean off-diag cosine (UNTRAINED, real val): {cos:+.6f}")
    print(f"        L2 norms: {[f'{v:.6f}' for v in norms]}")


# ---------------------------------------------------------------------------
# SECTION 3: Cross-modal similarity matrices (UNTRAINED, real val inputs)
# ---------------------------------------------------------------------------
print()
print(SEP)
print("SECTION 3 — Cross-Modal Similarity Matrices (UNTRAINED, real val inputs)")
print(SEP)

directions = [("eeg", "meg"), ("meg", "eeg"), ("eeg", "fmri"),
              ("fmri", "eeg"), ("meg", "fmri"), ("fmri", "meg")]

with torch.no_grad():
    for q_mod, t_mod in directions:
        sim = torch.mm(z[q_mod], z[t_mod].t())
        preds = torch.argmax(sim, dim=1).tolist()
        correct = sum(p == i for i, p in enumerate(preds))
        rv = row_variance(sim)
        print(f"  {q_mod.upper():>4} -> {t_mod.upper():<4}: "
              f"preds={preds}  correct={correct}/4  "
              f"row_var=[{', '.join(f'{v:.2e}' for v in rv)}]")

print()
print(SEP)
print("DIAGNOSTIC COMPLETE — No training, no file modifications, no test split accessed.")
print(SEP)
