"""Rebuild key SateBeam tables and figures from frozen publication data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "publication"

COLORS = {
    "SateBeam": "#D1495B",
    "OMP-Grid": "#2F6690",
    "Lasso-Grid": "#2A9D8F",
    "Orbit-NNLS": "#8C6D31",
    "Peak": "#7B7F8C",
    "Kriging": "#D6A21E",
    "DeepRM-TD": "#7656A5",
    "SoftImpute": "#C06C84",
    "Oracle-Support": "#252525",
}

DISPLAY = {
    "SateBeam": "SateBeam", "OMP-Grid": "OMP", "Lasso-Grid": "Lasso",
    "Orbit-NNLS": "Orbit-NNLS", "Peak": "Peak", "Kriging": "Kriging",
    "DeepRM-TD": "DeepRM-TD", "SoftImpute": "SoftImpute",
    "Oracle-Support": "Oracle support",
}


def configure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.28, "figure.facecolor": "white",
        "pdf.fonttype": 42,
    })


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(map(cell, headers)) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_tables(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(DATA / "main_summary.csv")
    order = [name for name in ["SateBeam", "OMP-Grid", "Lasso-Grid", "Orbit-NNLS",
                                "Peak", "DeepRM-TD", "Kriging", "SoftImpute",
                                "Oracle-Support"] if name in set(main["method"])]
    main = main.set_index("method").loc[order].reset_index()
    rows = []
    for _, row in main.iterrows():
        f1 = "--" if pd.isna(row.f1_mean) else f"{row.f1_mean:.3f}"
        exact = "--" if pd.isna(row.exact_support_mean) else f"{row.exact_support_mean:.2f}"
        center = "--" if pd.isna(row.center_error_deg_mean) else f"{row.center_error_deg_mean:.2f}"
        width = "--" if pd.isna(row.width_mae_deg_mean) else f"{row.width_mae_deg_mean:.2f}"
        rows.append([DISPLAY[row.method], f1, exact, f"{row.query_rmse_db_mean:.3f}",
                     center, width, f"{row.runtime_median:.2f}/{row.runtime_p90:.2f}"])
    (output / "main_comparison.md").write_text(markdown_table(
        ["Method", "F1", "Exact", "RMSE (dB)", "Center (deg)", "HPBW (deg)",
         "Median/P90 (s)"], rows), encoding="utf-8")

    mismatch = pd.read_csv(DATA / "mismatch_summary.csv")
    rows = []
    labels = {"legacy_sinc": "Sinc", "elliptic_r1p5_rot30": "Ellipse",
              "shadow_4db": "Shadowing 4 dB", "blockage_10pct": "Blockage 10%"}
    for condition in labels:
        part = mismatch[mismatch.condition.eq(condition)]
        values = {(row.method, row.metric): row["mean"] for _, row in part.iterrows()}
        rows.append([labels[condition], f"{values[('SateBeam', 'f1')]:.3f}",
                     f"{values[('OMP-Grid', 'f1')]:.3f}",
                     f"{values[('SateBeam', 'query_rmse_db')]:.3f}",
                     f"{values[('OMP-Grid', 'query_rmse_db')]:.3f}"])
    (output / "robustness.md").write_text(markdown_table(
        ["Condition", "SateBeam F1", "OMP F1", "SateBeam RMSE", "OMP RMSE"], rows),
        encoding="utf-8")

    scaling = pd.read_csv(DATA / "scaling_summary.csv")
    rows = []
    for axis in ("n_candidates", "n_active"):
        part = scaling[scaling[axis].notna()]
        for value in sorted(part[axis].unique()):
            group = part[part[axis].eq(value)]
            for method in sorted(group.method.unique()):
                members = group[group.method.eq(method)]
                metrics = {row.metric: row["mean"] for _, row in members.iterrows()}
                rows.append([axis, int(value), DISPLAY.get(method, method),
                             f"{metrics.get('f1', np.nan):.3f}",
                             f"{metrics.get('query_rmse_db', np.nan):.3f}",
                             f"{metrics.get('runtime_s', np.nan):.2f}"])
    (output / "scaling.md").write_text(markdown_table(
        ["Sweep", "Value", "Method", "F1", "RMSE (dB)", "Runtime (s)"], rows),
        encoding="utf-8")


def main_figure(output: Path) -> None:
    frame = pd.read_csv(DATA / "main_summary.csv")
    order = [name for name in ["SateBeam", "OMP-Grid", "Lasso-Grid", "Orbit-NNLS",
                                "Peak", "DeepRM-TD", "Kriging", "SoftImpute",
                                "Oracle-Support"] if name in set(frame.method)]
    frame = frame.set_index("method").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [1, 1.2]})
    source = frame[frame.f1_mean.notna()].copy()
    y = np.arange(len(source))[::-1]
    for yi, (_, row) in zip(y, source.iterrows()):
        color = COLORS[row.method]
        axes[0].errorbar(row.f1_mean, yi + 0.08,
                         xerr=[[row.f1_mean-row.f1_lo], [row.f1_hi-row.f1_mean]],
                         fmt="o", color=color, capsize=2)
        axes[0].scatter(row.exact_support_mean, yi - 0.08, marker="D",
                        facecolor="white", edgecolor=color)
    axes[0].set_yticks(y, [DISPLAY[m] for m in source.method])
    axes[0].set_xlim(0, 1.03); axes[0].set_xlabel("Source-identification score")
    offsets = {
        "SateBeam": (5, 9), "Oracle-Support": (-76, 8), "OMP-Grid": (5, 5),
        "DeepRM-TD": (5, 5), "Lasso-Grid": (5, 5), "Orbit-NNLS": (5, 5),
        "Peak": (5, 5), "Kriging": (5, 5), "SoftImpute": (5, 5),
    }
    for _, row in frame.iterrows():
        axes[1].scatter(row.runtime_median, row.query_rmse_db_mean, s=34,
                        color=COLORS[row.method], label=DISPLAY[row.method])
        axes[1].annotate(DISPLAY[row.method], (row.runtime_median, row.query_rmse_db_mean),
                         xytext=offsets[row.method], textcoords="offset points", fontsize=6)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Median runtime (s)"); axes[1].set_ylabel("Query RMSE (dB)")
    fig.tight_layout()
    save(fig, output, "fig_main_performance_frontier")


def mismatch_figure(output: Path) -> None:
    frame = pd.read_csv(DATA / "mismatch_summary.csv")
    conditions = ["legacy_sinc", "elliptic_r1p5_rot30", "shadow_4db", "blockage_10pct"]
    labels = ["Sinc", "Ellipse", "Shadowing", "Blockage"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8))
    for axis, metric, xlabel in zip(axes, ("f1", "query_rmse_db"), ("F1", "Query RMSE (dB)")):
        for method, offset in (("SateBeam", -0.08), ("OMP-Grid", 0.08)):
            part = frame[(frame.metric.eq(metric)) & frame.method.eq(method)].set_index("condition")
            means = np.array([part.loc[c, "mean"] for c in conditions])
            lo = np.array([part.loc[c, "lo"] for c in conditions])
            hi = np.array([part.loc[c, "hi"] for c in conditions])
            y = np.arange(len(conditions)) + offset
            axis.errorbar(means, y, xerr=np.vstack([means-lo, hi-means]), fmt="o",
                          capsize=2, color=COLORS[method], label=DISPLAY[method])
        axis.set_yticks(np.arange(len(labels)), labels); axis.invert_yaxis(); axis.set_xlabel(xlabel)
    axes[0].legend(frameon=False, ncol=2, loc="lower center")
    fig.tight_layout()
    save(fig, output, "fig_mismatch_dumbbell")


def scaling_figure(output: Path) -> None:
    frame = pd.read_csv(DATA / "scaling_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0))
    for method in ("SateBeam", "OMP-Grid"):
        for column, metric in enumerate(("f1", "runtime_s")):
            part = frame[(frame.method.eq(method)) & frame.n_candidates.notna()
                         & frame.metric.eq(metric)].sort_values("n_candidates")
            if len(part):
                axes[0, column].plot(part.n_candidates, part["mean"], "o-",
                                     color=COLORS[method], label=DISPLAY[method])
    for column, metric in enumerate(("f1", "query_rmse_db")):
        part = frame[(frame.method.eq("SateBeam")) & frame.n_active.notna()
                     & frame.metric.eq(metric)].sort_values("n_active")
        axes[1, column].plot(part.n_active, part["mean"], "o-", color=COLORS["SateBeam"])
    axes[0, 0].set_ylabel("F1"); axes[0, 1].set_ylabel("Runtime (s)")
    axes[1, 0].set_ylabel("F1"); axes[1, 1].set_ylabel("Query RMSE (dB)")
    axes[0, 0].set_xlabel("Candidates N"); axes[0, 1].set_xlabel("Candidates N")
    axes[1, 0].set_xlabel("Active sources K"); axes[1, 1].set_xlabel("Active sources K")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save(fig, output, "fig_scalability")


def audit_figure(output: Path) -> None:
    frame = pd.read_csv(DATA / "solver_revision_audit.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    image = np.vstack([frame.exact_support_v3, frame.exact_support_v1, frame.exact_support_omp])
    axes[0].imshow(image, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    axes[0].set_yticks([0, 1, 2], ["SateBeam", "Original", "OMP"])
    axes[0].set_xlabel("Paired scenario index")
    for column, color, label in (("query_rmse_db_v3", COLORS["SateBeam"], "SateBeam"),
                                 ("query_rmse_db_v1", "#6B7280", "Original"),
                                 ("query_rmse_db_omp", COLORS["OMP-Grid"], "OMP")):
        values = np.sort(frame[column].to_numpy(float))
        axes[1].step(values, np.arange(1, len(values)+1)/len(values), where="post",
                     color=color, label=label)
    axes[1].set_xscale("log"); axes[1].set_xlabel("Query RMSE (dB)"); axes[1].set_ylabel("CDF")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, output, "fig_solver_revision_audit")


def grid_identifiability_figure(output: Path) -> None:
    grid = pd.read_csv(DATA / "grid_v3.csv")
    ident = pd.read_csv(DATA / "identifiability_v3.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    for method in ("SateBeam", "OMP-Grid"):
        for column, metric in enumerate(("f1", "query_rmse_db")):
            part = grid[(grid.method.eq(method)) & grid.metric.eq(metric)].sort_values("grid_step_deg")
            axes[0, column].plot(part.grid_step_deg, part["mean"], "o-",
                                 color=COLORS[method], label=DISPLAY[method])
    condition_order = ["coh_q10_ratio00db", "coh_q90_ratio00db",
                       "coh_q10_ratio06db", "coh_q90_ratio06db"]
    labels = ["Low, 0 dB", "High, 0 dB", "Low, 6 dB", "High, 6 dB"]
    y = np.arange(len(condition_order))
    for method, offset in (("SateBeam", -0.08), ("OMP-Grid", 0.08)):
        for column, metric in enumerate(("f1", "query_rmse_db")):
            part = ident[(ident.method.eq(method)) & ident.metric.eq(metric)].set_index("condition")
            means = np.array([part.loc[c, "mean"] for c in condition_order])
            lo = np.array([part.loc[c, "lo"] for c in condition_order])
            hi = np.array([part.loc[c, "hi"] for c in condition_order])
            axes[1, column].errorbar(means, y+offset, xerr=np.vstack([means-lo, hi-means]),
                                     fmt="o", capsize=2, color=COLORS[method])
    axes[0, 0].set_ylabel("F1"); axes[0, 1].set_ylabel("Query RMSE (dB)")
    axes[0, 0].set_xlabel("Grid step (deg)"); axes[0, 1].set_xlabel("Grid step (deg)")
    axes[0, 0].invert_xaxis(); axes[0, 1].invert_xaxis()
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_yticks(y, labels); axes[1, 1].set_yticks(y, [])
    axes[1, 0].invert_yaxis(); axes[1, 1].invert_yaxis()
    axes[1, 0].set_xlabel("F1"); axes[1, 1].set_xlabel("Query RMSE (dB)")
    fig.tight_layout()
    save(fig, output, "fig_grid_identifiability")


def matrix_figure(csv_name: str, row_key: str, output: Path, stem: str) -> None:
    frame = pd.read_csv(DATA / csv_name)
    fields = ["gt_dbm", "satebeam_dbm", "omp_dbm", "deeprm_td_dbm", "kriging_dbm"]
    labels = ["GT", "SateBeam", "OMP", "DeepRM-TD", "Kriging"]
    rows = list(dict.fromkeys(frame[row_key].tolist()))
    vmin = float(frame[fields].min().min()); vmax = float(frame[fields].max().max())
    fig, axes = plt.subplots(len(rows), len(fields), figsize=(7.2, 1.35*len(rows)), squeeze=False)
    image = None
    for i, row_value in enumerate(rows):
        part = frame[frame[row_key].eq(row_value)]
        gt_beams = [column for column in part.columns
                    if column.startswith("gt_beam_candidate_") and part[column].notna().any()]
        for j, field in enumerate(fields):
            pivot = part.pivot(index="north_km", columns="east_km", values=field).sort_index()
            image = axes[i, j].imshow(pivot.to_numpy(), origin="lower", cmap="RdBu_r",
                                      vmin=vmin, vmax=vmax, interpolation="bilinear",
                                      aspect="equal")
            for beam_field in gt_beams:
                beam = part.pivot(index="north_km", columns="east_km",
                                  values=beam_field).sort_index()
                axes[i, j].contour(beam.to_numpy(), levels=[0.5], colors=["#FFD54F"],
                                   linewidths=0.9, linestyles="dashed")
            estimate_prefix = ({"satebeam_dbm": "satebeam_beam_candidate_",
                                "omp_dbm": "omp_beam_candidate_"}).get(field)
            if estimate_prefix is not None:
                estimate_beams = [column for column in part.columns
                                  if column.startswith(estimate_prefix) and
                                  part[column].notna().any()]
                for beam_field in estimate_beams:
                    beam = part.pivot(index="north_km", columns="east_km",
                                      values=beam_field).sort_index()
                    axes[i, j].contour(beam.to_numpy(), levels=[0.5],
                                       colors=["#00B8D4"], linewidths=0.65)
            if field != "gt_dbm" and "is_measurement" in part:
                query = part.is_measurement.eq(0).to_numpy()
                error = part[field].to_numpy()[query] - part.gt_dbm.to_numpy()[query]
                rmse = float(np.sqrt(np.mean(error ** 2)))
                axes[i, j].text(0.96, 0.04, f"{rmse:.2f}",
                                transform=axes[i, j].transAxes, ha="right",
                                va="bottom", fontsize=6.5,
                                bbox={"boxstyle": "round,pad=0.10", "fc": "white",
                                      "ec": "none", "alpha": 0.74})
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([]); axes[i, j].grid(False)
            if i == 0: axes[i, j].set_title(labels[j], fontsize=8)
            if j == 0:
                if row_key == "scenario" and str(row_value).startswith("scene_"):
                    label = f"Scenario {int(str(row_value).split('_')[-1]):02d}"
                elif row_key == "snapshot" and "time_s" in part:
                    label = f"t = {float(part.time_s.iloc[0]):g} s"
                else:
                    label = str(row_value)
                axes[i, j].set_ylabel(label, fontsize=7)
    axes[0, 1].legend(
        handles=[Line2D([0], [0], color="#FFD54F", lw=1.0, ls="--", label="GT beams"),
                 Line2D([0], [0], color="#00B8D4", lw=0.8, ls="-", label="Estimated beams")],
        loc="upper left", fontsize=5.5, frameon=True, framealpha=0.78,
        borderpad=0.22, handlelength=1.5, labelspacing=0.20)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.07, top=0.95, wspace=0.03, hspace=0.05)
    if image is not None:
        bar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012)
        bar.set_label("Total received power (dBm)")
    save(fig, output, stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reproduced")
    args = parser.parse_args()
    root = ROOT / args.output
    figures = root / "figures"; tables = root / "tables"
    configure()
    write_tables(tables)
    main_figure(figures)
    mismatch_figure(figures)
    scaling_figure(figures)
    audit_figure(figures)
    grid_identifiability_figure(figures)
    matrix_figure("rm_scenario_matrix.csv", "scenario", figures, "fig_rm_scenario_matrix")
    matrix_figure("rm_temporal_matrix.csv", "snapshot", figures, "fig_rm_temporal_matrix")
    print(f"Reproduced tables: {tables}")
    print(f"Reproduced figures: {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
