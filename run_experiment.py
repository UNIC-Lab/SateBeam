"""Run the reviewer-response SateBeam experiment suites.

The smoke test uses a tiny real-TLE subset and exercises simulation, fitting,
raw-data saving, metrics, and plotting end to end.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from satbeam_study.config import StudyConfig
from satbeam_study.data import OrbitSceneFactory, save_scene, simulate_scene
from satbeam_study.figures import metric_overview
from satbeam_study.methods import run_method
from satbeam_study.metrics import evaluate, save_fit
from satbeam_study.physics import physical_validation
from satbeam_study.reporting import summarize, write_manifest, write_rows


CORE_METHODS = ["SateBeam", "OMP-Grid", "MP-Grid", "Lasso-Grid",
                "Orbit-NNLS", "Peak", "IDW", "Kriging", "SoftImpute",
                "DeepRM-TD", "Oracle-Support"]
FAST_METHODS = ["SateBeam", "OMP-Grid", "Orbit-NNLS", "Peak", "IDW",
                "Kriging", "Oracle-Support"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--suite", choices=["physical", "validation", "calibration", "main", "mismatch",
                                             "identifiability", "scaling", "selection", "grid", "all"],
                        default="validation")
    parser.add_argument("--trials", type=int, default=0,
                        help="override trials per condition; zero uses suite default")
    parser.add_argument("--output", default="results/satebeam_revision")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON_VALUE",
                        help="override one StudyConfig field")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="run only named conditions from the selected suite")
    parser.add_argument("--scene-start", type=int, default=0)
    return parser.parse_args()


def condition_specs(suite: str) -> List[Tuple[str, Dict[str, object]]]:
    if suite == "validation":
        return [("matched_four_by_60s", {"truth_pattern": "sinc", "n_snapshots": 4,
                                         "snapshot_step_s": 60.0, "n_candidates": 16}),
                ("single_snapshot", {"truth_pattern": "circular", "n_snapshots": 1,
                                      "snapshot_step_s": 0.0, "n_candidates": 16}),
                ("four_by_30s", {"truth_pattern": "circular", "n_snapshots": 4,
                                  "snapshot_step_s": 30.0, "n_candidates": 16}),
                ("four_by_60s", {"truth_pattern": "circular", "n_snapshots": 4,
                                  "snapshot_step_s": 60.0, "n_candidates": 16}),
                ("six_by_30s", {"truth_pattern": "circular", "n_snapshots": 6,
                                 "snapshot_step_s": 30.0, "n_candidates": 16})]
    if suite == "main":
        return [(f"eirp_{eirp:02d}dbw", {"truth_pattern": "circular", "eirp_dbw": float(eirp)})
                for eirp in (30, 36, 42)]
    if suite == "calibration":
        return [("null_max_scan", {"n_active": 0, "selection_rule": "bic"})]
    if suite == "mismatch":
        return [
            ("circular_nominal", {"truth_pattern": "circular"}),
            ("legacy_sinc", {"truth_pattern": "sinc"}),
            ("elliptic_r1p5_rot30", {"truth_pattern": "circular", "ellipse_axis_ratio": 1.5,
                                     "ellipse_rotation_deg": 30.0}),
            ("array_tapered", {"truth_pattern": "array"}),
            ("shadow_4db", {"shadow_sigma_db": 4.0}),
            ("calibration_1db", {"calibration_sigma_db": 1.0}),
            ("blockage_10pct", {"blockage_fraction": 0.10}),
            ("background_gradient", {"background_gradient": 0.75}),
            ("rician_k15db", {"rician_k_db": 15.0}),
            ("doppler_error_10khz", {"doppler_error_sigma_hz": 10_000.0}),
        ]
    if suite == "identifiability":
        return [(f"coh_q{int(q * 100):02d}_ratio{ratio:02d}db",
                 {"n_active": 2, "active_pair_quantile": q,
                  "active_power_ratio_db": float(ratio)})
                for q in (0.10, 0.50, 0.90) for ratio in (0, 6, 12)]
    if suite == "scaling":
        specs = [(f"N{n:02d}", {"n_candidates": n, "n_active": min(4, n),
                                 "snapshot_step_s": 30.0, "min_elevation_deg": 10.0})
                 for n in (8, 16, 24, 32, 64)]
        specs += [(f"K{k:02d}", {"n_active": k, "order_cap": 14}) for k in (1, 2, 4, 8, 12)]
        specs += [(f"M{m:03d}", {"n_measurements": m}) for m in (50, 100, 160, 320)]
        return specs
    if suite == "selection":
        rules = [("bic", {}), ("aic", {"selection_rule": "aic"})]
        rules += [(f"margin_bic_{margin:g}", {"selection_rule": "margin_bic",
                                               "selection_margin": float(margin)})
                  for margin in (2, 4, 8)]
        specs = []
        for k in (0, 4, 8):
            for name, values in rules:
                specs.append((f"K{k}_{name}", {"n_active": k, **values}))
        return specs
    if suite == "grid":
        return [(f"grid_{step:g}deg_w{widths}", {"grid_step_deg": float(step),
                                                  "grid_width_count": int(widths),
                                                  "n_candidates": 8,
                                                  "n_measurements": 100,
                                                  "order_cap": 6})
                for step, widths in ((4, 3), (2, 5), (1, 5))]
    if suite == "physical":
        return []
    raise ValueError(suite)


def suite_defaults(suite: str) -> Tuple[int, Sequence[str]]:
    if suite == "validation":
        return 5, FAST_METHODS
    if suite == "main":
        return 30, CORE_METHODS
    if suite == "calibration":
        return 50, ["SateBeam"]
    if suite == "mismatch":
        return 20, FAST_METHODS
    if suite == "identifiability":
        return 20, ["SateBeam", "OMP-Grid", "Oracle-Support"]
    if suite == "scaling":
        return 10, ["SateBeam", "OMP-Grid", "Orbit-NNLS", "IDW"]
    if suite == "selection":
        return 20, ["SateBeam"]
    if suite == "grid":
        return 10, ["SateBeam", "OMP-Grid", "Lasso-Grid"]
    return 0, []


def failure_row(scene: Mapping[str, object], cfg: StudyConfig, method: str,
                condition: str, error: BaseException) -> Dict[str, object]:
    return {
        "scene_id": scene["scene_id"], "epoch_iso": scene["epoch_iso"],
        "condition": condition, "method": method, "runtime_s": np.nan,
        "solver_failures": 1, "query_rmse_db": np.nan, "query_corr_db": np.nan,
        "f1": np.nan, "precision": np.nan, "recall": np.nan,
        "exact_support": np.nan, "signal_nmse": np.nan,
        "center_error_deg": np.nan, "width_mae_deg": np.nan,
        "k_abs_error": np.nan, "error": f"{type(error).__name__}: {error}",
        "n_candidates": cfg.n_candidates, "n_active": cfg.n_active,
        "n_measurements": cfg.n_measurements,
    }


def run_condition(base_cfg: StudyConfig, condition: str, overrides: Mapping[str, object],
                  methods: Sequence[str], trials: int, root: Path,
                  scene_start: int = 0) -> List[Dict[str, object]]:
    cfg = base_cfg.evolve(**dict(overrides))
    factory = OrbitSceneFactory(cfg)
    condition_dir = root / condition
    rows: List[Dict[str, object]] = []
    started = time.perf_counter()
    for scene_id in range(scene_start, scene_start + trials):
        scene = simulate_scene(cfg, factory, scene_id)
        save_scene(scene, condition_dir / "raw" / f"scene_{scene_id:04d}.npz")
        for method in methods:
            try:
                result = run_method(scene, cfg, method)
                save_fit(result, condition_dir / "fits" / method / f"scene_{scene_id:04d}.npz")
                row = evaluate(scene, result, cfg)
                row["condition"] = condition
                rows.append(row)
            except Exception as error:
                rows.append(failure_row(scene, cfg, method, condition, error))
                error_dir = condition_dir / "errors"; error_dir.mkdir(parents=True, exist_ok=True)
                (error_dir / f"{method}_scene_{scene_id:04d}.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
        write_rows(rows, condition_dir / "metrics_partial.csv")
        print(f"  {condition}: scene {scene_id} ({len({r['scene_id'] for r in rows})}/{trials})", flush=True)
    write_rows(rows, condition_dir / "metrics.csv")
    write_manifest(condition_dir, {"condition": condition, "overrides": dict(overrides),
                                   "config": cfg.to_dict(), "orbit": factory.manifest(),
                                   "methods": list(methods), "trials": trials,
                                   "scene_start": scene_start,
                                   "elapsed_s": time.perf_counter() - started})
    return rows


def environment() -> Dict[str, object]:
    import scipy, sklearn
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else ""
        torch_version = torch.__version__
    except Exception:
        cuda = False; gpu = ""; torch_version = "unavailable"
    return {"python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "torch": torch_version, "cuda": cuda, "gpu": gpu}


def main() -> int:
    args = parse_args()
    base_cfg = StudyConfig(use_gpu=not args.cpu_only)
    command_overrides = {}
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set requires KEY=JSON_VALUE, received {item!r}")
        key, value = item.split("=", 1)
        try:
            command_overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            command_overrides[key] = value
    if command_overrides:
        base_cfg = base_cfg.evolve(**command_overrides)
    if args.smoke:
        base_cfg = base_cfg.evolve(n_candidates=6, n_active=2, n_measurements=36,
                                   grid_size=12, n_snapshots=2, snapshot_step_s=30.0,
                                   candidate_maxiter=8, joint_maxiter=10,
                                   grid_step_deg=4.0, grid_width_count=3, order_cap=4)
    root = Path(args.output) / ("smoke" if args.smoke else args.suite)
    root.mkdir(parents=True, exist_ok=True)
    checks = physical_validation()
    write_manifest(root, {"environment": environment(), "base_config": base_cfg.to_dict(),
                          "physical_validation": checks})
    if checks["max_identity_error"] > 1e-8:
        raise RuntimeError(f"Physical validation failed: {checks}")
    print("Physical validation PASSED", json.dumps(checks, sort_keys=True), flush=True)
    if args.suite == "physical":
        print("Physical-only suite complete.", flush=True)
        return 0

    suites = (["validation", "calibration", "main", "mismatch", "identifiability",
               "scaling", "selection", "grid"]
              if args.suite == "all" else [args.suite])
    all_rows: List[Dict[str, object]] = []
    for suite in suites:
        default_trials, default_methods = suite_defaults(suite)
        trials = args.trials or default_trials
        methods = args.methods or list(default_methods)
        specs = condition_specs(suite)
        if args.conditions:
            requested = set(args.conditions)
            specs = [item for item in specs if item[0] in requested]
            missing = requested - {item[0] for item in specs}
            if missing:
                raise ValueError(f"Unknown conditions for {suite}: {sorted(missing)}")
        if args.smoke:
            trials = 1
            methods = args.methods or ["SateBeam", "OMP-Grid", "IDW", "DeepRM-TD"]
            specs = [("real_tle_smoke", {"truth_pattern": "circular"})]
        suite_rows = []
        for condition, overrides in specs:
            suite_rows.extend(run_condition(base_cfg, condition, overrides, methods, trials,
                                            root / suite if args.suite == "all" else root,
                                            scene_start=args.scene_start))
        all_rows.extend(suite_rows)
        if args.smoke:
            break
    write_rows(all_rows, root / "all_metrics.csv")
    summarize(all_rows, root)
    metric_overview(all_rows, root / "figures",
                    "fig_satebeam_smoke" if args.smoke else f"fig_satebeam_{args.suite}_overview")
    if args.smoke:
        print("Smoke test PASSED", flush=True)
    else:
        print(f"Suite {args.suite} PASSED", flush=True)
    print(f"Results: {root.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
