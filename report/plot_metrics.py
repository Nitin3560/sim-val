"""Generate SimVal plots from gap report JSON files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_ssim_timeline(
    report_path: str,
    output_dir: str = "outputs/plots",
) -> str | None:
    report = _load_report(report_path)
    scores = report.get("camera", {}).get("ssim_per_frame", [])
    if not scores:
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scenario_id = report["scenario_id"]
    values = np.array(scores, dtype=np.float64)
    mean = float(np.mean(values))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values, linewidth=1.2, color="#4A90D9", label="SSIM per frame")
    ax.axhline(mean, color="#E24B4A", linewidth=1.5, linestyle="--", label=f"Mean={mean:.3f}")
    ax.axhline(0.7, color="#F5A623", linewidth=1.0, linestyle=":", label="0.7 reference")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("SSIM")
    ax.set_title(f"Camera Rendering Fidelity - {scenario_id}")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = str(Path(output_dir) / f"ssim_timeline_{scenario_id}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_metrics_comparison(
    report_paths: list[str],
    output_dir: str = "outputs/plots",
) -> str:
    reports = [_load_report(path) for path in report_paths]
    if not reports:
        raise ValueError("No reports provided")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    labels = [r["scenario_id"].replace("_", "\n") for r in reports]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Gazebo vs Isaac Sim RTX Camera Fidelity", fontsize=12, y=1.01)

    _bar_panel(
        axes[0],
        x,
        [r["camera"]["psnr_mean_db"] for r in reports],
        labels,
        "PSNR (dB) higher is better",
        "#4A90D9",
        30.0,
    )
    _bar_panel(
        axes[1],
        x,
        [r["camera"]["ssim_mean"] for r in reports],
        labels,
        "SSIM higher is better",
        "#7B68EE",
        0.7,
        ylim=(0.0, 1.05),
    )
    _bar_panel(
        axes[2],
        x,
        [r["camera"]["lpips_mean"] for r in reports],
        labels,
        "LPIPS lower is better",
        "#E24B4A",
        0.2,
    )

    fig.tight_layout()
    out = str(Path(output_dir) / "metrics_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_authority_comparison(
    report_paths: list[str],
    output_dir: str = "outputs/plots",
) -> str | None:
    reports = [
        report for report in (_load_report(path) for path in report_paths)
        if report.get("ekf_propagation")
    ]
    if not reports:
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    labels = [r["scenario_id"].replace("_", "\n") for r in reports]
    gazebo_vals = [r["ekf_propagation"]["gazebo_authority_mean"] for r in reports]
    isaac_vals = [r["ekf_propagation"]["isaac_authority_mean"] for r in reports]
    risks = [r["ekf_propagation"]["calibration_risk"] for r in reports]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, gazebo_vals, width, label="Gazebo", color="#4A90D9", alpha=0.85)
    ax.bar(x + width / 2, isaac_vals, width, label="Isaac Sim RTX", color="#1D9E75", alpha=0.85)
    for idx, risk in enumerate(risks):
        color = {"LOW": "#1D9E75", "MEDIUM": "#F5A623", "HIGH": "#E24B4A"}.get(risk, "gray")
        ax.text(x[idx], max(gazebo_vals[idx], isaac_vals[idx]) + 0.02, risk, ha="center", color=color)

    ax.axhline(0.5, color="#E24B4A", linewidth=1.0, linestyle="--", label="0.5 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean authority_scale")
    ax.set_title("TwinGuard Authority Scale Calibration")
    ax.set_ylim(0.0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = str(Path(output_dir) / "authority_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_all_plots(
    reports_dir: str = "outputs/reports",
    plots_dir: str = "outputs/plots",
) -> dict[str, list[str]]:
    report_paths = sorted(Path(reports_dir).glob("gap_report_*.json"))
    if not report_paths:
        return {}

    timelines = [
        path for path in (plot_ssim_timeline(str(report), plots_dir) for report in report_paths)
        if path
    ]
    metrics = [plot_metrics_comparison([str(path) for path in report_paths], plots_dir)]
    authority_plot = plot_authority_comparison([str(path) for path in report_paths], plots_dir)
    authority = [authority_plot] if authority_plot else []

    return {
        "ssim_timelines": timelines,
        "metrics_comparison": metrics,
        "authority_comparison": authority,
    }


def _bar_panel(
    ax: Any,
    x: np.ndarray,
    values: list[float],
    labels: list[str],
    title: str,
    color: str,
    threshold: float,
    ylim: tuple[float, float] | None = None,
) -> None:
    ax.bar(x, values, width=0.5, color=color, alpha=0.85)
    ax.axhline(threshold, color="#F5A623", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)


def _load_report(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
