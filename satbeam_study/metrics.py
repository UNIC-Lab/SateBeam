"""Independent-query metrics, identifiability diagnostics, and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from .config import StudyConfig
from .physics import center_vector, template


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def identifiability(scene: Mapping[str, object], cfg: StudyConfig) -> Dict[str, float]:
    active = [int(v) for v in scene["active_ids"]]
    dirs = np.asarray(scene["satellite_directions"])
    separations = [
        np.rad2deg(np.arccos(np.clip(dirs[i] @ dirs[j], -1.0, 1.0)))
        for p, i in enumerate(active) for j in active[p + 1:]
    ]
    train_features = scene["features_train"]
    truth_params = scene["params"]
    truth_atoms = {}
    for idx in active:
        item = truth_params[idx]
        par = np.asarray([item["offset_az_deg"], item["offset_el_deg"], item["width_deg"]])
        atom = template(train_features[idx], par, cfg, pattern=scene["truth_pattern"],
                        axis_ratio=cfg.ellipse_axis_ratio,
                        rotation_deg=cfg.ellipse_rotation_deg)
        atom = atom - np.mean(atom)
        norm = np.linalg.norm(atom)
        truth_atoms[idx] = atom / norm if norm > 1e-18 else np.full_like(atom, np.nan)
    active_coh = [abs(float(truth_atoms[i] @ truth_atoms[j]))
                  for p, i in enumerate(active) for j in active[p + 1:]
                  if np.all(np.isfinite(truth_atoms[i])) and np.all(np.isfinite(truth_atoms[j]))]
    nominal_width = 0.5 * (cfg.truth_width_min_deg + cfg.truth_width_max_deg)
    active_inactive = []
    for i in active:
        if not np.all(np.isfinite(truth_atoms[i])):
            continue
        for j in range(cfg.n_candidates):
            if j in active:
                continue
            atom = template(train_features[j], np.asarray([0.0, 0.0, nominal_width]), cfg)
            atom -= np.mean(atom)
            norm = np.linalg.norm(atom)
            if norm > 1e-18:
                active_inactive.append(abs(float(truth_atoms[i] @ (atom / norm))))
    return {
        "min_satellite_separation_deg": float(min(separations)) if separations else float("nan"),
        "max_active_coherence": float(max(active_coh)) if active_coh else float("nan"),
        "max_active_inactive_coherence": float(max(active_inactive)) if active_inactive else float("nan"),
    }


def evaluate(scene: Mapping[str, object], result: Mapping[str, object],
             cfg: StudyConfig) -> Dict[str, object]:
    query = np.asarray(scene["query_spatial_idx"], dtype=int)
    truth_total = np.asarray(scene["mean_total_aggregate_mw"])[query]
    truth_signal = np.asarray(scene["signal_aggregate_mw"])[query]
    pred_total = np.asarray(result["pred_total_aggregate_mw"])[query]
    valid = np.isfinite(pred_total) & (pred_total > 0)
    if not np.all(valid):
        total_rmse_db = float("nan")
        total_corr = float("nan")
    else:
        truth_db = 10.0 * np.log10(np.maximum(truth_total, 1e-300))
        pred_db = 10.0 * np.log10(np.maximum(pred_total, 1e-300))
        total_rmse_db = float(np.sqrt(np.mean((pred_db - truth_db) ** 2)))
        total_corr = _safe_corr(pred_db, truth_db)

    row: Dict[str, object] = {
        "scene_id": int(scene["scene_id"]),
        "epoch_iso": scene["epoch_iso"],
        "method": result["method"],
        "runtime_s": float(result.get("runtime_s", np.nan)),
        "solver_failures": int(result.get("solver_failures", 0)),
        "query_rmse_db": total_rmse_db,
        "query_corr_db": total_corr,
        "realized_cn_db": float(10.0 * np.log10(
            max(float(np.mean(scene["signal_aggregate_mw"])), 1e-300)
            / max(float(np.mean(scene["background_aggregate_mw"])) + float(scene["noise_total_mw"]), 1e-300)
        )),
        "n_visible": int(scene["n_visible"]),
        "n_candidates": cfg.n_candidates,
        "n_active": len(scene["active_ids"]),
        "n_measurements": cfg.n_measurements,
        "n_query": len(query),
        "truth_pattern": scene["truth_pattern"],
        "compute_device": result.get("compute_device", result.get("device", "cpu")),
    }
    row.update(identifiability(scene, cfg))

    if bool(result.get("map_only", False)):
        row.update({"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                    "exact_support": float("nan"), "k_est": float("nan"),
                    "k_abs_error": float("nan"), "signal_nmse": float("nan"),
                    "center_error_deg": float("nan"), "width_mae_deg": float("nan"),
                    "background_error_db": float("nan"), "matched_sources": 0})
        return row

    truth_ids = set(int(v) for v in scene["active_ids"])
    selected_ids = set(int(v) for v in result.get("selected_ids", []))
    matched = sorted(truth_ids & selected_ids)
    tp = len(matched)
    precision = tp / len(selected_ids) if selected_ids else (1.0 if not truth_ids else 0.0)
    recall = tp / len(truth_ids) if truth_ids else (1.0 if not selected_ids else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    pred_signal = np.asarray(result["pred_signal_aggregate_mw"])[query]
    denominator = float(truth_signal @ truth_signal)
    signal_nmse = (float(np.sum((pred_signal - truth_signal) ** 2)) / denominator
                   if denominator > 0 else float("nan"))
    center_errors = []
    width_errors = []
    for idx in matched:
        truth = scene["params"][idx]
        estimate = result["estimates"][idx]
        feature = scene["features_all"][idx]
        center_az = float(np.asarray(feature["center_az_deg"]).reshape(-1)[0])
        center_off = float(np.asarray(feature["center_offnadir_deg"]).reshape(-1)[0])
        c_true = center_vector(center_az, center_off,
                               truth["offset_az_deg"], truth["offset_el_deg"])
        c_est = center_vector(center_az, center_off,
                              estimate["offset_az_deg"], estimate["offset_el_deg"])
        center_errors.append(np.rad2deg(np.arccos(np.clip(c_true @ c_est, -1.0, 1.0))))
        width_errors.append(abs(float(estimate["width_deg"]) - float(truth["width_deg"])))
    true_bg = float(np.mean(scene["background_aggregate_mw"]))
    est_bg = float(result.get("background_mw", np.nan)) * max(int(cfg.n_subbands), 1)
    background_error = (10.0 * np.log10(max(est_bg, 1e-300) / max(true_bg, 1e-300))
                        if np.isfinite(est_bg) else float("nan"))
    row.update({
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "exact_support": float(selected_ids == truth_ids),
        "k_est": len(selected_ids), "k_abs_error": abs(len(selected_ids) - len(truth_ids)),
        "signal_nmse": signal_nmse,
        "center_error_deg": float(np.mean(center_errors)) if center_errors else float("nan"),
        "width_mae_deg": float(np.mean(width_errors)) if width_errors else float("nan"),
        "background_error_db": float(background_error), "matched_sources": len(matched),
        "candidate_time_s": float(result.get("candidate_time_s", 0.0)),
        "refine_time_s": float(result.get("refine_time_s", 0.0)),
        "optimizer_nfev": int(result.get("optimizer_nfev", 0)),
        "dictionary_atoms": int(result.get("dictionary_atoms", 0)),
        "dictionary_bytes": int(result.get("dictionary_bytes", 0)),
        "dictionary_time_s": float(result.get("dictionary_time_s", 0.0)),
    })
    return row


def save_fit(result: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    estimates = result.get("estimates", {})
    trace = result.get("score_trace", [])
    payload = {
        "method": np.asarray(result["method"]),
        "map_only": np.asarray(bool(result.get("map_only", False))),
        "pred_total_mw": np.asarray(result["pred_total_mw"]),
        "runtime_s": np.asarray(result.get("runtime_s", np.nan)),
        "solver_failures": np.asarray(result.get("solver_failures", 0)),
        "compute_device": np.asarray(result.get("compute_device", result.get("device", "cpu"))),
        "estimates_json": np.asarray(json.dumps(estimates, sort_keys=True)),
        "trace_json": np.asarray(json.dumps(trace, sort_keys=True)),
    }
    if "pred_signal_mw" in result:
        payload["pred_signal_mw"] = np.asarray(result["pred_signal_mw"])
    if "pred_total_aggregate_mw" in result:
        payload["pred_total_aggregate_mw"] = np.asarray(result["pred_total_aggregate_mw"])
    if "pred_signal_aggregate_mw" in result:
        payload["pred_signal_aggregate_mw"] = np.asarray(result["pred_signal_aggregate_mw"])
    if "selected_ids" in result:
        payload["selected_ids"] = np.asarray(result["selected_ids"], dtype=int)
    for key in ("background_mw", "candidate_time_s", "refine_time_s", "optimizer_nfev",
                "dictionary_atoms", "dictionary_bytes", "dictionary_time_s"):
        if key in result:
            payload[key] = np.asarray(result[key])
    np.savez_compressed(path, **payload)
