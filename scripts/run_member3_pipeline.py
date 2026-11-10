"""CLI Runner for Neuro-GRAPE Member 3 Pipeline.

Usage:
    python scripts/run_member3_pipeline.py [--config configs/groupdro_config.yaml] [--output reports/artifacts]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dro.pipeline import run_member3_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Neuro-GRAPE Member 3 GroupDRO & Evaluation Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/groupdro_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/artifacts",
        help="Directory to save artifacts and logs",
    )
    args = parser.parse_args()

    print(f"=== Neuro-GRAPE Member 3 Pipeline ===")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")

    summary = run_member3_pipeline(config_or_path=args.config, output_dir=args.output)
    print("\nPipeline finished successfully!")
    print(f"Artifacts generated at: {summary['output_dir']}")
    for k, v in summary["artifacts"].items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
