"""Verify frozen SateBeam inputs and the headline manuscript-facing claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from satbeam_study.physics import physical_validation  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def check_hashes() -> dict[str, str]:
    manifest = ROOT / "data" / "checksums.sha256"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing checksum manifest: {manifest}")
    checked: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        # Rebuilding from path parts keeps the manifest platform-independent.
        path = ROOT.joinpath(*relative.replace("\\", "/").split("/"))
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input is missing: {relative}")
        actual = sha256(path)
        if actual != expected.upper():
            raise RuntimeError(f"Checksum mismatch for {relative}: {actual} != {expected}")
        checked[relative] = actual
    return checked


def value(frame: pd.DataFrame, method: str, column: str) -> float:
    row = frame.loc[frame["method"].eq(method)]
    if len(row) != 1:
        raise AssertionError(f"Expected one {method!r} row, found {len(row)}")
    return float(row.iloc[0][column])


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: {actual} differs from {expected} by more than {tolerance}")


def verify_claims() -> dict[str, object]:
    data = ROOT / "data" / "publication"
    main = pd.read_csv(data / "main_summary.csv")
    audit = pd.read_csv(data / "solver_revision_audit.csv")
    mismatch = pd.read_csv(data / "mismatch_summary.csv")
    scaling = pd.read_csv(data / "scaling_summary.csv")
    grid = pd.read_csv(data / "grid_v3.csv")
    ident = pd.read_csv(data / "identifiability_v3.csv")

    assert_close(value(main, "SateBeam", "f1_mean"), 1.0, 1e-12, "SateBeam F1")
    assert_close(value(main, "SateBeam", "exact_support_mean"), 1.0, 1e-12,
                 "SateBeam exact support")
    assert_close(value(main, "SateBeam", "query_rmse_db_mean"), 0.050093411573223785,
                 1e-12, "SateBeam query RMSE")
    assert_close(value(main, "OMP-Grid", "f1_mean"), 0.9747354497354498, 1e-12, "OMP F1")
    assert_close(value(main, "OMP-Grid", "query_rmse_db_mean"), 0.6554714752236447,
                 1e-12, "OMP query RMSE")

    if len(audit) != 30 or not np.allclose(audit["exact_support_v3"], 1.0):
        raise AssertionError("The 30-scenario revised-solver audit is incomplete or non-exact")

    for condition in sorted(mismatch["condition"].unique()):
        part = mismatch[mismatch["condition"].eq(condition)]
        for metric, direction in (("f1", "higher"), ("query_rmse_db", "lower")):
            rows = part[part["metric"].eq(metric)].set_index("method")
            proposed = float(rows.loc["SateBeam", "mean"])
            omp = float(rows.loc["OMP-Grid", "mean"])
            if direction == "higher" and proposed < omp - 1e-12:
                raise AssertionError(f"Mismatch F1 ordering failed for {condition}")
            if direction == "lower" and proposed > omp + 1e-12:
                raise AssertionError(f"Mismatch RMSE ordering failed for {condition}")

    for step in sorted(grid["grid_step_deg"].unique()):
        rows = grid[(grid["grid_step_deg"].eq(step))
                    & grid["metric"].eq("query_rmse_db")].set_index("method")
        if float(rows.loc["SateBeam", "mean"]) >= float(rows.loc["OMP-Grid", "mean"]):
            raise AssertionError(f"Grid RMSE ordering failed at {step} degrees")

    for condition in sorted(ident["condition"].unique()):
        part = ident[ident["condition"].eq(condition)]
        for metric, direction in (("f1", "higher"), ("query_rmse_db", "lower")):
            rows = part[part["metric"].eq(metric)].set_index("method")
            proposed = float(rows.loc["SateBeam", "mean"])
            omp = float(rows.loc["OMP-Grid", "mean"])
            if direction == "higher" and proposed < omp - 1e-12:
                raise AssertionError(f"Coherence F1 ordering failed for {condition}")
            if direction == "lower" and proposed > omp + 1e-12:
                raise AssertionError(f"Coherence RMSE ordering failed for {condition}")

    n64 = scaling[(scaling["method"].eq("SateBeam"))
                  & scaling["n_candidates"].eq(64.0)
                  & scaling["metric"].eq("f1")]
    k8 = scaling[(scaling["method"].eq("SateBeam"))
                 & scaling["n_active"].eq(8.0)
                 & scaling["metric"].eq("f1")]
    assert_close(float(n64.iloc[0]["mean"]), 0.9527777777777778, 1e-12, "N=64 F1")
    assert_close(float(k8.iloc[0]["mean"]), 0.9441176470588235, 1e-12, "K=8 F1")

    physical = physical_validation()
    if float(physical["max_identity_error"]) > 1e-8:
        raise AssertionError(f"Physical identities failed: {physical}")

    tle = ROOT / "data" / "tle" / "catalog_starlink_20260911.tle"
    lines = [line for line in tle.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = sum(line.startswith("1 ") for line in lines)
    if records != 10_713:
        raise AssertionError(f"Expected 10,713 TLE records, found {records}")

    return {
        "tle_records": records,
        "main_scenarios": int(len(audit)),
        "satebeam_f1": value(main, "SateBeam", "f1_mean"),
        "satebeam_exact_support": value(main, "SateBeam", "exact_support_mean"),
        "satebeam_query_rmse_db": value(main, "SateBeam", "query_rmse_db_mean"),
        "omp_f1": value(main, "OMP-Grid", "f1_mean"),
        "omp_query_rmse_db": value(main, "OMP-Grid", "query_rmse_db_mean"),
        "mismatch_conditions": sorted(mismatch["condition"].unique().tolist()),
        "physical_validation": physical,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="verification")
    args = parser.parse_args()
    hashes = check_hashes()
    claims = verify_claims()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "passed", "checked_files": hashes, "claims": claims}
    (output / "verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# SateBeam release verification",
        "",
        "Status: **PASSED**",
        "",
        f"- Frozen files checked: {len(hashes)}",
        f"- TLE records: {claims['tle_records']}",
        f"- Main paired scenarios: {claims['main_scenarios']}",
        f"- SateBeam F1 / exact support: {claims['satebeam_f1']:.3f} / "
        f"{claims['satebeam_exact_support']:.2f}",
        f"- SateBeam / OMP query RMSE: {claims['satebeam_query_rmse_db']:.3f} / "
        f"{claims['omp_query_rmse_db']:.3f} dB",
        "- Physical identity checks: passed",
    ]
    (output / "verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("SateBeam release verification PASSED")
    print(output / "verification_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
