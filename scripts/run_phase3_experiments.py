"""
scripts/run_phase3_experiments.py
=================================
Automated execution script for Neuro-GRAPE Member 3 Phase 3 Large-Scale Experiments.

Orchestrates:
1. Multi-seed (42, 123, 456) Baseline vs GroupDRO runs under:
   - Semantic Subgroup Partitioning (Animate vs Inanimate)
   - Learned Cluster Partitioning (K=2 Unsupervised Latent Regimes)
2. Strict Leakage Prevention:
   - Normalization statistics fitted strictly on training data
   - Group weights updated strictly from training losses
   - Checkpoint selection guided strictly by validation loss
   - Final evaluation conducted strictly on the held-out test split
3. Generation of all required individual and comparative artifacts:
   - metrics.csv, group_metrics.csv, group_losses.csv, group_weights.csv, group_assignments.csv / .parquet
   - Publication-quality diagnostic plots (loss trajectories, weight trajectories, robustness disparity gaps)
   - Machine-readable benchmark summaries (JSON & CSV)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dro.experiment_runner import run_and_save_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_comparison_plots(
    experiments_dir: Path,
    summary_dir: Path,
) -> List[str]:
    """Render publication-quality evaluation figures comparing Baseline and GroupDRO."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []

    # 1. Plot Weight Trajectories for GroupDRO Semantic
    g_weights_file = experiments_dir / "groupdro_semantic" / "group_weights.csv"
    if g_weights_file.exists():
        df_w = pd.read_csv(g_weights_file)
        plt.figure(figsize=(8, 5))
        for col in df_w.columns:
            if col != "epoch":
                plt.plot(df_w["epoch"], df_w[col], marker="o", label=f"Group: {col}")
        plt.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Uniform Weight (0.50)")
        plt.title("GroupDRO Dynamic Weight Trajectory (Semantic Groups — Seed 42)", fontsize=13, fontweight="bold")
        plt.xlabel("Training Epoch", fontsize=11)
        plt.ylabel("Distribution Weight $q_k$", fontsize=11)
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=True, facecolor="white", loc="best")
        plt.tight_layout()
        plot_path = summary_dir / "weight_trajectories_semantic.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        generated_plots.append(str(plot_path))

    # 2. Plot Loss Trajectories: Baseline vs GroupDRO Semantic
    b_losses_file = experiments_dir / "baseline_semantic" / "group_losses.csv"
    g_losses_file = experiments_dir / "groupdro_semantic" / "group_losses.csv"
    if b_losses_file.exists() and g_losses_file.exists():
        df_b = pd.read_csv(b_losses_file)
        df_g = pd.read_csv(g_losses_file)
        plt.figure(figsize=(9, 5))
        # Animate and Inanimate curves
        plt.plot(df_b["epoch"], df_b["animate"], "r--", label="Baseline (ERM): Animate")
        plt.plot(df_b["epoch"], df_b["inanimate"], "b--", label="Baseline (ERM): Inanimate")
        plt.plot(df_g["epoch"], df_g["animate"], "r-", linewidth=2, label="GroupDRO: Animate (Targeted)")
        plt.plot(df_g["epoch"], df_g["inanimate"], "b-", linewidth=2, label="GroupDRO: Inanimate")
        plt.title("Training Loss Trajectories by Subgroup (Seed 42)", fontsize=13, fontweight="bold")
        plt.xlabel("Training Epoch", fontsize=11)
        plt.ylabel("Contrastive Loss", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=True, facecolor="white", loc="best")
        plt.tight_layout()
        plot_path = summary_dir / "loss_trajectories_semantic.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        generated_plots.append(str(plot_path))

    # 3. Bar Chart: Robustness Gap Comparison across conditions
    comp_csv = summary_dir / "benchmark_comparison.csv"
    if comp_csv.exists():
        df_comp = pd.read_csv(comp_csv)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(df_comp))
        width = 0.35

        ax.bar(x - width/2, df_comp["test_worst_group_loss_mean"], width, label="Worst-Group Loss", color="#d95f02", alpha=0.85)
        ax.bar(x + width/2, df_comp["test_overall_loss_mean"], width, label="Overall Average Loss", color="#1b9e77", alpha=0.85)

        ax.set_ylabel("Loss (Lower is Better)", fontsize=11)
        ax.set_title("Test Set Generalization: Worst-Group vs Overall Loss", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df_comp["experiment"], fontsize=10)
        ax.legend(frameon=True, facecolor="white")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plot_path = summary_dir / "worst_group_vs_overall_loss.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        generated_plots.append(str(plot_path))

    return generated_plots


def run_all_experiments() -> Dict[str, Any]:
    print("=" * 75)
    print("NEURO-GRAPE MEMBER 3 — PHASE 3: LARGE-SCALE BENCHMARKING EXPERIMENTS")
    print("=" * 75)

    base_exp_dir = REPO_ROOT / "reports" / "artifacts" / "experiments"
    summary_dir = base_exp_dir / "summary_comparison"
    base_exp_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    seeds = (42, 123, 456)
    num_epochs = 25
    lr = 1e-4
    group_lr = 0.05

    experiment_definitions = [
        {
            "name": "baseline_semantic",
            "mode": "baseline",
            "strategy": "semantic",
            "out": base_exp_dir / "baseline_semantic",
        },
        {
            "name": "groupdro_semantic",
            "mode": "groupdro",
            "strategy": "semantic",
            "out": base_exp_dir / "groupdro_semantic",
        },
        {
            "name": "baseline_cluster",
            "mode": "baseline",
            "strategy": "cluster",
            "out": base_exp_dir / "baseline_cluster",
        },
        {
            "name": "groupdro_cluster",
            "mode": "groupdro",
            "strategy": "cluster",
            "out": base_exp_dir / "groupdro_cluster",
        },
    ]

    all_summaries: List[Dict[str, Any]] = []

    for exp in experiment_definitions:
        print(f"\n>>> Running Experiment: {exp['name']} ({exp['mode']} | {exp['strategy']}) across seeds {seeds}...")
        summary = run_and_save_experiment(
            mode=exp["mode"],
            group_strategy=exp["strategy"],
            output_dir=exp["out"],
            seeds=seeds,
            num_epochs=num_epochs,
            lr=lr,
            group_lr=group_lr,
        )
        summary["experiment"] = exp["name"]
        all_summaries.append(summary)

    # Compile Benchmark Comparison Table
    df_comparison = pd.DataFrame(all_summaries)
    comp_csv = summary_dir / "benchmark_comparison.csv"
    df_comparison.to_csv(comp_csv, index=False)
    print(f"\n[Artifact] Benchmark comparison table written to: {comp_csv}")

    comp_json = summary_dir / "comparison_summary.json"
    with open(comp_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[Artifact] Benchmark comparison summary written to: {comp_json}")

    # Generate Evaluation Figures
    print("\n[Plots] Generating publication-quality diagnostic plots...")
    generated_plots = generate_comparison_plots(base_exp_dir, summary_dir)
    for p in generated_plots:
        print(f"  - Plot created: {p}")

    print("\n" + "=" * 75)
    print("ALL PHASE 3 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    return {
        "status": "SUCCESS",
        "comparison_table": df_comparison.to_dict(orient="records"),
        "plots": generated_plots,
    }


if __name__ == "__main__":
    run_all_experiments()
