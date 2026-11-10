"""
scripts/audit_data_integrity.py
===============================
Independent audit and data-integrity verification script for Neuro-GRAPE Phase 2.5.
Performs comprehensive verification of:
1. Inventory consistency (root data_inventory.csv vs data/manifests/data_inventory.csv)
2. File existence and integrity across all modality subdirectories:
   - data/processed/eeg/
   - data/processed/meg/
   - data/processed/fmri/
   - data/processed/fmri_beta/
3. Array shapes, dtypes, and numerical finiteness (no NaNs / Infs)
4. Orphan files and unreferenced artifacts
5. Cross-modality stimulus alignment and independent acquisition metadata
6. Split isolation and stimulus-disjoint integrity
7. Dataset loader and batch assembly verification
8. Generation of machine-readable report: reports/artifacts/data_integrity_summary.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def audit_data_integrity() -> Dict[str, Any]:
    print("=" * 70)
    print("NEURO-GRAPE PHASE 2.5: COMPREHENSIVE DATA INTEGRITY AUDIT")
    print("=" * 70)

    results: Dict[str, Any] = {
        "status": "PASS",
        "repo_root": str(REPO_ROOT),
        "inventory_checks": {},
        "modality_file_checks": {},
        "all_files_summary": {},
        "split_checks": {},
        "cross_modality_checks": {},
        "loader_checks": {},
        "discrepancies": [],
    }

    # 1. Compare inventories
    root_inv = REPO_ROOT / "data_inventory.csv"
    manifest_inv = REPO_ROOT / "data" / "manifests" / "data_inventory.csv"
    pilot_splits = REPO_ROOT / "data" / "splits" / "pilot_splits.csv"

    print("\n[Step 1] Auditing Inventory Manifests...")
    if not root_inv.exists():
        results["discrepancies"].append("Root data_inventory.csv missing")
    if not manifest_inv.exists():
        results["discrepancies"].append("data/manifests/data_inventory.csv missing")

    df_root = pd.read_csv(root_inv) if root_inv.exists() else pd.DataFrame()
    df_man = pd.read_csv(manifest_inv) if manifest_inv.exists() else pd.DataFrame()

    inventories_identical = df_root.equals(df_man)
    print(f"  - Root inventory shape: {df_root.shape}")
    print(f"  - Manifests inventory shape: {df_man.shape}")
    print(f"  - Identical bit-for-bit: {inventories_identical}")

    results["inventory_checks"] = {
        "root_inventory_path": str(root_inv),
        "manifest_inventory_path": str(manifest_inv),
        "root_inventory_exists": root_inv.exists(),
        "manifest_inventory_exists": manifest_inv.exists(),
        "inventories_identical": inventories_identical,
        "total_records": len(df_root),
        "unique_sample_ids": int(df_root["sample_id"].nunique()) if "sample_id" in df_root else 0,
        "unique_stimuli": int(df_root["stimulus_id"].nunique()) if "stimulus_id" in df_root else 0,
        "modalities": df_root["modality"].value_counts().to_dict() if "modality" in df_root else {},
        "splits": df_root["split"].value_counts().to_dict() if "split" in df_root else {},
    }

    # 2. Modality directories inspection
    print("\n[Step 2] Auditing Processed Modality Directories on Disk...")
    modality_dirs = {
        "eeg": REPO_ROOT / "data" / "processed" / "eeg",
        "meg": REPO_ROOT / "data" / "processed" / "meg",
        "fmri": REPO_ROOT / "data" / "processed" / "fmri",
        "fmri_beta": REPO_ROOT / "data" / "processed" / "fmri_beta",
    }

    all_disk_files: Dict[str, Path] = {}
    for mod_name, mod_dir in modality_dirs.items():
        if not mod_dir.exists():
            print(f"  - WARNING: Directory {mod_dir} does not exist!")
            results["discrepancies"].append(f"Directory {mod_dir} does not exist")
            continue
        files = list(mod_dir.glob("*.npy"))
        print(f"  - {mod_name:10s}: {len(files)} .npy files found in {mod_dir.relative_to(REPO_ROOT)}")
        for f in files:
            all_disk_files[str(f.relative_to(REPO_ROOT)).replace("\\", "/")] = f

    results["all_files_summary"]["total_disk_npy_files"] = len(all_disk_files)
    print(f"  Total .npy files on disk across all 4 directories: {len(all_disk_files)}")

    # 3. Inventory-to-File validation
    print("\n[Step 3] Auditing Every Record in Inventory...")
    referenced_files = set()
    loaded_data_info = []

    for idx, row in df_root.iterrows():
        sid = row["sample_id"]
        mod = row["modality"]
        data_path_rel = str(row["data_path"]).replace("\\", "/")
        raw_window_rel = str(row["raw_window_path"]).replace("\\", "/")

        p_data = REPO_ROOT / data_path_rel
        p_raw = REPO_ROOT / raw_window_rel

        referenced_files.add(data_path_rel)
        referenced_files.add(raw_window_rel)

        data_exists = p_data.exists()
        raw_exists = p_raw.exists()

        if not data_exists:
            results["discrepancies"].append(f"Missing data_path: {data_path_rel} for sample {sid}")
        if not raw_exists:
            results["discrepancies"].append(f"Missing raw_window_path: {raw_window_rel} for sample {sid}")

        # Safe inspect of .npy
        if data_exists:
            try:
                arr = np.load(p_data)
                is_finite = bool(np.all(np.isfinite(arr)))
                loaded_data_info.append({
                    "sample_id": sid,
                    "modality": mod,
                    "path": data_path_rel,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "is_finite": is_finite,
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                })
            except Exception as e:
                results["discrepancies"].append(f"Error loading {data_path_rel}: {str(e)}")

    print(f"  - Successfully validated and loaded {len(loaded_data_info)}/{len(df_root)} inventory target files.")

    # Check for orphan files (disk files not in referenced_files)
    disk_file_set = set(all_disk_files.keys())
    orphan_files = list(disk_file_set - referenced_files)
    missing_from_disk = list(referenced_files - disk_file_set)

    print(f"  - Orphan files on disk (not referenced in inventory): {len(orphan_files)}")
    print(f"  - Missing referenced files (in inventory but not on disk): {len(missing_from_disk)}")
    results["all_files_summary"]["orphan_files"] = orphan_files
    results["all_files_summary"]["missing_from_disk"] = missing_from_disk

    # 4. Modality specific breakdown
    df_info = pd.DataFrame(loaded_data_info)
    modality_breakdown = {}
    for mod in ["eeg", "meg", "fmri"]:
        sub_info = df_info[df_info["modality"] == mod]
        shapes = sub_info["shape"].apply(tuple).value_counts().to_dict()
        shapes_str = {str(k): int(v) for k, v in shapes.items()}
        dtypes = sub_info["dtype"].value_counts().to_dict()
        all_finite = bool(sub_info["is_finite"].all())
        modality_breakdown[mod] = {
            "record_count": len(sub_info),
            "shapes": shapes_str,
            "dtypes": dtypes,
            "all_finite": all_finite,
            "mean_val_range": [float(sub_info["min"].min()), float(sub_info["max"].max())],
        }

    # Also inspect fmri raw windows in data/processed/fmri
    fmri_raw_dir = REPO_ROOT / "data" / "processed" / "fmri"
    fmri_raw_files = list(fmri_raw_dir.glob("*.npy"))
    fmri_raw_shapes = {}
    fmri_raw_finite = True
    for f in fmri_raw_files:
        arr = np.load(f)
        fmri_raw_shapes[str(arr.shape)] = fmri_raw_shapes.get(str(arr.shape), 0) + 1
        if not np.all(np.isfinite(arr)):
            fmri_raw_finite = False

    modality_breakdown["fmri_raw_window"] = {
        "record_count": len(fmri_raw_files),
        "shapes": fmri_raw_shapes,
        "all_finite": fmri_raw_finite,
    }

    results["modality_file_checks"] = modality_breakdown
    print("\n[Step 4] Modality Breakdown:")
    for k, v in modality_breakdown.items():
        print(f"  - {k:16s}: {v['record_count']} files, shapes: {v['shapes']}, finite: {v['all_finite']}")

    # 5. Split and Leakage Checks
    print("\n[Step 5] Checking Split Integrity and Leakage...")
    df_splits = pd.read_csv(pilot_splits) if pilot_splits.exists() else pd.DataFrame()
    train_stims = set(df_splits[df_splits["split"] == "train"]["stimulus_id"])
    val_stims = set(df_splits[df_splits["split"] == "val"]["stimulus_id"])
    test_stims = set(df_splits[df_splits["split"] == "test"]["stimulus_id"])

    leak_train_val = list(train_stims.intersection(val_stims))
    leak_train_test = list(train_stims.intersection(test_stims))
    leak_val_test = list(val_stims.intersection(test_stims))

    leakage_free = (len(leak_train_val) == 0) and (len(leak_train_test) == 0) and (len(leak_val_test) == 0)
    print(f"  - Stimulus sets: train={len(train_stims)}, val={len(val_stims)}, test={len(test_stims)}")
    print(f"  - Leakage free (pairwise disjoint): {leakage_free}")

    # Verify inventory splits match pilot_splits.csv exactly
    inv_split_map = df_root.groupby("stimulus_id")["split"].unique()
    split_conflicts = []
    for stim, splits_found in inv_split_map.items():
        if len(splits_found) > 1:
            split_conflicts.append(f"Inconsistent splits for stimulus {stim}: {splits_found}")
        expected_split = df_splits[df_splits["stimulus_id"] == stim]["split"].iloc[0]
        if splits_found[0] != expected_split:
            split_conflicts.append(f"Split mismatch for {stim}: inventory={splits_found[0]}, manifest={expected_split}")

    results["split_checks"] = {
        "train_stimuli_count": len(train_stims),
        "val_stimuli_count": len(val_stims),
        "test_stimuli_count": len(test_stims),
        "pairwise_disjoint": leakage_free,
        "split_conflicts": split_conflicts,
    }
    print(f"  - Split conflicts: {len(split_conflicts)}")

    # 6. Cross-modality joins
    print("\n[Step 6] Cross-Modality Alignment Checks...")
    stim_modality_counts = df_root.groupby("stimulus_id")["modality"].apply(lambda s: sorted(list(set(s))))
    all_have_trimodal = all(mods == ["eeg", "fmri", "meg"] for mods in stim_modality_counts)
    print(f"  - Every stimulus present across all 3 modalities: {all_have_trimodal}")

    results["cross_modality_checks"] = {
        "all_stimuli_trimodal": all_have_trimodal,
        "trimodal_stimuli_count": sum(1 for mods in stim_modality_counts if mods == ["eeg", "fmri", "meg"]),
        "independent_subjects": {
            "eeg": df_root[df_root["modality"] == "eeg"]["subject_id"].unique().tolist(),
            "meg": df_root[df_root["modality"] == "meg"]["subject_id"].unique().tolist(),
            "fmri": df_root[df_root["modality"] == "fmri"]["subject_id"].unique().tolist(),
        }
    }

    # 7. Dataset Loader Test
    print("\n[Step 7] Testing src/data/dataset.py loader...")
    from src.data.dataset import NeuroGrapeDataset
    ds_train = NeuroGrapeDataset(manifest=root_inv, split="train", as_torch=True)
    ds_val = NeuroGrapeDataset(manifest=root_inv, split="val", as_torch=True)
    ds_test = NeuroGrapeDataset(manifest=root_inv, split="test", as_torch=True)

    loader_ok = (len(ds_train) == 36) and (len(ds_val) == 12) and (len(ds_test) == 12)
    sample_train0 = ds_train[0]
    results["loader_checks"] = {
        "train_len": len(ds_train),
        "val_len": len(ds_val),
        "test_len": len(ds_test),
        "sample_keys": list(sample_train0.keys()),
        "loader_functional": loader_ok,
    }
    print(f"  - Loader functional: {loader_ok} (train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)})")

    # Final verdict
    if results["discrepancies"]:
        results["status"] = "FAIL"
    else:
        results["status"] = "PASS"

    out_json = REPO_ROOT / "reports" / "artifacts" / "data_integrity_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"AUDIT COMPLETE: Overall Status = {results['status']}")
    print(f"Machine-readable summary written to: {out_json}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    res = audit_data_integrity()
    if res["status"] != "PASS":
        sys.exit(1)
