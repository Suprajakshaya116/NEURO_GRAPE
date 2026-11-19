import os
import shutil
from pathlib import Path
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# 1. Remove fMRI directories
fmri_dirs = [
    REPO_ROOT / "data" / "processed" / "fmri",
    REPO_ROOT / "data" / "processed" / "fmri_beta",
    REPO_ROOT / "dattta" / "data" / "processed" / "fmri",
    REPO_ROOT / "dattta" / "data" / "processed" / "fmri_beta",
]
for d in fmri_dirs:
    if d.exists():
        shutil.rmtree(d)
        print(f"Removed directory: {d}")

# 2. Filter manifests
manifest_paths = [
    REPO_ROOT / "data_inventory.csv",
    REPO_ROOT / "data" / "manifests" / "data_inventory.csv",
    REPO_ROOT / "dattta" / "data" / "manifests" / "data_inventory.csv",
]
for mp in manifest_paths:
    if mp.exists():
        df = pd.read_csv(mp)
        orig_len = len(df)
        df_clean = df[df["modality"] != "fmri"].copy()
        df_clean.to_csv(mp, index=False)
        mods = df_clean["modality"].unique().tolist()
        print(f"Updated {mp}: {orig_len} -> {len(df_clean)} rows. Modalities: {mods}")

# 3. Filter splits
splits_paths = [
    REPO_ROOT / "data" / "splits" / "pilot_splits.csv",
    REPO_ROOT / "dattta" / "data" / "splits" / "pilot_splits.csv",
]
for sp in splits_paths:
    if sp.exists():
        df = pd.read_csv(sp)
        orig_len = len(df)
        if "modality" in df.columns:
            df_clean = df[df["modality"] != "fmri"].copy()
            df_clean.to_csv(sp, index=False)
            print(f"Updated {sp}: {orig_len} -> {len(df_clean)} rows")
        else:
            print(f"{sp} columns: {df.columns.tolist()}")

# 4. Filter embeddings
meta_p = REPO_ROOT / "embeddings" / "metadata.csv"
emb_p = REPO_ROOT / "embeddings" / "embeddings.pt"
if meta_p.exists() and emb_p.exists():
    meta = pd.read_csv(meta_p)
    embs = torch.load(emb_p, map_location="cpu")
    if "modality" in meta.columns:
        keep_mask = (meta["modality"] != "fmri").values
        meta_clean = meta[keep_mask].copy()
        if isinstance(embs, dict):
            embs_clean = {k: v for k, v in embs.items() if not k.startswith("fmri_")}
        else:
            embs_clean = embs[keep_mask]
        meta_clean.to_csv(meta_p, index=False)
        torch.save(embs_clean, emb_p)
        print(f"Updated embeddings: {len(meta)} -> {len(meta_clean)} entries")

print("fMRI removal completed successfully.")
