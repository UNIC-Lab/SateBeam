"""Tabular summaries and confidence intervals from per-scene rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy import stats


def write_rows(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    arrays = {}
    for key in keys:
        values = [row.get(key, np.nan) for row in rows]
        try:
            arrays[key] = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            arrays[key] = np.asarray(values, dtype=str)
    np.savez_compressed(output.with_suffix(".npz"), **arrays)


def _ci(values: Iterable[float]) -> tuple[float, float, float, int]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))])
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, np.nan, np.nan, 1
    half = float(stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr)))
    return mean, mean - half, mean + half, len(arr)


def summarize(rows: Sequence[Mapping[str, object]], output_dir: Path) -> List[Dict[str, object]]:
    groups: Dict[tuple, List[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["condition"], row["method"]), []).append(row)
    metrics = ["f1", "precision", "recall", "exact_support", "query_rmse_db",
               "query_corr_db", "signal_nmse", "center_error_deg", "width_mae_deg",
               "k_abs_error", "runtime_s", "solver_failures"]
    summary = []
    for (condition, method), members in sorted(groups.items()):
        item: Dict[str, object] = {"condition": condition, "method": method,
                                  "trials": len(members)}
        for metric in metrics:
            mean, lo, hi, n = _ci(row.get(metric, np.nan) for row in members)
            item[f"{metric}_mean"] = mean; item[f"{metric}_lo"] = lo
            item[f"{metric}_hi"] = hi; item[f"{metric}_n"] = n
        runtimes = np.asarray([float(row["runtime_s"]) for row in members])
        item["runtime_s_median"] = float(np.median(runtimes))
        item["runtime_s_p90"] = float(np.quantile(runtimes, 0.90))
        summary.append(item)
    write_rows(summary, output_dir / "summary.csv")
    lines = ["| Condition | Method | n | F1 (95% CI) | Query RMSE dB (95% CI) | Runtime median/P90 s |",
             "|---|---|---:|---:|---:|---:|"]
    for item in summary:
        def fmt(metric: str) -> str:
            value = item[f"{metric}_mean"]
            if not np.isfinite(value):
                return "N/A"
            return f"{value:.3f} [{item[f'{metric}_lo']:.3f}, {item[f'{metric}_hi']:.3f}]"
        lines.append(f"| {item['condition']} | {item['method']} | {item['trials']} | "
                     f"{fmt('f1')} | {fmt('query_rmse_db')} | "
                     f"{item['runtime_s_median']:.3f}/{item['runtime_s_p90']:.3f} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_manifest(output_dir: Path, payload: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
