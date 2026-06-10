"""Evaluation plots. Matplotlib/Agg only; runs headless"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402


def plot_loss_curves(history: dict[str, list[float]], steps: list[int], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in history.items():
        ax.plot(steps, values, label=name, linewidth=1.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("Training losses")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_identity_histogram(
    swap_vs_source, impostor_scores, threshold: float, far_target: float, out_path: str | Path
) -> Path:
    #Similarity histogram with the calibrated verification threshold marked
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(impostor_scores, bins=50, alpha=0.5, density=True, label="impostor pairs")
    ax.hist(swap_vs_source, bins=50, alpha=0.6, density=True, label="swap vs source")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"threshold @ FAR={far_target:.1%}")
    ax.set_xlabel("cosine similarity (held-out ArcFace)")
    ax.set_ylabel("density")
    ax.set_title("Identity preservation vs verification threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_pareto(strengths: list[float], id_scores: list[float], attribute_errors: list[float],out_path: str | Path,) -> Path:
    #Identity similarity vs attribute error as injection strength is swept
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(attribute_errors, id_scores, "-o", linewidth=1.5)
    for s, x, y in zip(strengths, attribute_errors, id_scores):
        ax.annotate(f"α={s:.2f}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("attribute error (pose, degrees) — lower is better")
    ax.set_ylabel("identity similarity to source — higher is better")
    ax.set_title("Identity vs attribute trade-off (Pareto front)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_comparison_bars(metrics_by_model: dict[str, dict[str, float]], out_path: str | Path) -> Path:
    #Grouped bar chart comparing models across the metric suite. normalize values
    
    out_path = Path(out_path)
    models = list(metrics_by_model.keys())
    metric_names = list(next(iter(metrics_by_model.values())).keys())
    x = range(len(metric_names))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        offsets = [xi + i * width for xi in x]
        values = [metrics_by_model[model][m] for m in metric_names]
        ax.bar(offsets, values, width, label=model)
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(metric_names, rotation=20, ha="right")
    ax.set_ylabel("score (normalised, higher = better)")
    ax.set_title("Model comparison across the metric suite")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path



