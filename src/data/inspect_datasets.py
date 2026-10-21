
from pathlib import Path
import csv
import json
import platform
import shutil
import os
from collections import Counter

# Change this path only if your dataset location differs.
DATA_ROOT = Path.home() / "Downloads" / "BrainFLORA"

DATASETS = {
    "EEG": DATA_ROOT / "Dataset1" / "ds003825-1.1.0",
    "MEG": DATA_ROOT / "Dataset2" / "ds004212-2.0.0",
    "fMRI": DATA_ROOT / "Dataset3" / "ds004192-1.0.7",
}

RECORDING_EXTENSIONS = {
    ".fif", ".edf", ".bdf", ".set", ".vhdr", ".eeg",
    ".dat", ".nii", ".gz", ".con", ".sqd", ".ds"
}

METADATA_EXTENSIONS = {".tsv", ".csv", ".json", ".yaml", ".yml"}


def inspect_dataset(name, root):
    print(f"\n{'=' * 65}")
    print(f"{name} DATASET")
    print(f"Path: {root}")

    if not root.exists():
        print("STATUS: PATH NOT FOUND")
        return

    print("STATUS: FOUND")

    extension_counts = Counter()
    file_count = 0
    total_bytes = 0
    examples = []

    # Walk directories without opening large recording files.
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "__pycache__"}
        ]

        for filename in files:
            path = Path(current_root) / filename
            file_count += 1
            extension = path.suffix.lower() or "[no extension]"
            extension_counts[extension] += 1

            try:
                size = path.stat().st_size
                total_bytes += size
            except OSError:
                size = 0

            if extension in RECORDING_EXTENSIONS and len(examples) < 10:
                examples.append((str(path), size))

    print(f"Total files: {file_count}")
    print(f"Total file size: {total_bytes / (1024**3):.2f} GiB")

    print("\nFile types:")
    for extension, count in extension_counts.most_common():
        print(f"  {extension}: {count}")

    print("\nExample recording files:")
    if examples:
        for path, size in examples:
            print(f"  {path} ({size / (1024**2):.2f} MiB)")
    else:
        print("  No files with the configured recording extensions found.")


def main():
    print("NEURO-GRAPE DATASET INVENTORY")
    print(f"Operating system: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Dataset root: {DATA_ROOT}")

    disk = shutil.disk_usage(DATA_ROOT if DATA_ROOT.exists() else Path.home())
    print(f"Disk free: {disk.free / (1024**3):.2f} GiB")

    for name, path in DATASETS.items():
        inspect_dataset(name, path)


if __name__ == "__main__":
    main()
