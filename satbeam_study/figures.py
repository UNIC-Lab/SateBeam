"""Compact, final-size figures generated only from saved metric rows."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import METHOD_COLORS


def metric_overview(rows: Sequence[Mapping[str, object]], output_dir: Path,
                    name: str = "fig_smoke_overview") -> None:
    import matplotlib.pyplot as plt

    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(str(row["method"]))
    f1 = []
    rmse = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        f1_values = [float(row["f1"]) for row in subset if np.isfinite(float(row["f1"]))]
        rmse_values = [float(row["query_rmse_db"]) for row in subset
                       if np.isfinite(float(row["query_rmse_db"]))]
        f1.append(float(np.mean(f1_values)) if f1_values else np.nan)
        rmse.append(float(np.mean(rmse_values)) if rmse_values else np.nan)
    colors = [METHOD_COLORS.get(method, "#555555") for method in methods]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), constrained_layout=True)
    positions = np.arange(len(methods))
    axes[0].bar(positions, f1, color=colors, width=0.75)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("F1", fontsize=8)
    axes[1].bar(positions, rmse, color=colors, width=0.75)
    axes[1].set_ylabel("Query RMSE (dB)", fontsize=8)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=6)
        ax.tick_params(labelsize=7)
        ax.text(0.5, -0.50, tag, transform=ax.transAxes, ha="center", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def condition_curves(rows: Sequence[Mapping[str, object]], output_dir: Path,
                     x_key: str, name: str) -> None:
    import matplotlib.pyplot as plt

    methods = sorted({str(row["method"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), constrained_layout=True)
    for method in methods:
        members = [row for row in rows if row["method"] == method]
        xs = sorted({float(row[x_key]) for row in members})
        f1, rmse = [], []
        for x in xs:
            chunk = [row for row in members if float(row[x_key]) == x]
            f = [float(row["f1"]) for row in chunk if np.isfinite(float(row["f1"]))]
            r = [float(row["query_rmse_db"]) for row in chunk if np.isfinite(float(row["query_rmse_db"]))]
            f1.append(np.mean(f) if f else np.nan); rmse.append(np.mean(r) if r else np.nan)
        axes[0].plot(xs, f1, marker="o", ms=2.5, lw=1.0,
                     color=METHOD_COLORS.get(method), label=method)
        axes[1].plot(xs, rmse, marker="o", ms=2.5, lw=1.0,
                     color=METHOD_COLORS.get(method), label=method)
    axes[0].set_ylabel("F1", fontsize=8); axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("Query RMSE (dB)", fontsize=8)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.set_xlabel(x_key.replace("_", " "), fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25, linewidth=0.5)
        ax.text(0.5, -0.34, tag, transform=ax.transAxes, ha="center", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)),
                   fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.13))
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
