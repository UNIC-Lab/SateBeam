"""Authentic-orbit scene construction and controlled RSS simulation."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from satbeam.orbits import TLECatalog

from .config import StudyConfig
from .physics import candidate_feature, ground_grid, lla_to_ecef, template


GOLDEN_FRACTION = 0.6180339887498949


def subset_feature(feature: Mapping[str, object], indices: np.ndarray) -> Dict[str, object]:
    out: Dict[str, object] = {}
    n = len(np.asarray(feature["base_mw"]))
    for key, value in feature.items():
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == n:
            out[key] = value[indices]
        else:
            out[key] = value
    return out


class OrbitSceneFactory:
    """Load one frozen catalog and draw deterministic feasible candidate rosters."""

    def __init__(self, cfg: StudyConfig):
        self.cfg = cfg
        path = Path(cfg.catalog_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen orbit catalog missing: {path}. Download it before running experiments."
            )
        self.catalog = TLECatalog(cfg.catalog_url, path)
        self.catalog.load(False)
        self.base_epoch = dt.datetime.fromisoformat(cfg.base_epoch_iso)
        if self.base_epoch.tzinfo is None:
            self.base_epoch = self.base_epoch.replace(tzinfo=dt.timezone.utc)

    def epoch(self, scene_id: int) -> dt.datetime:
        seconds = 60.0 * self.cfg.epoch_window_min
        return self.base_epoch + dt.timedelta(
            seconds=seconds * ((int(scene_id) * GOLDEN_FRACTION) % 1.0)
        )

    def candidate_states(self, scene_id: int, n_candidates: Optional[int] = None,
                         epoch: Optional[dt.datetime] = None) -> Tuple[List[List[Dict[str, float]]], int]:
        cfg = self.cfg
        when = epoch or self.epoch(scene_id)
        requested = int(n_candidates if n_candidates is not None else cfg.n_candidates)
        times = [when + dt.timedelta(seconds=t * cfg.snapshot_step_s)
                 for t in range(cfg.n_snapshots)]
        visible = self.catalog.visible_indices(
            times, cfg.center_lat_deg, cfg.center_lon_deg,
            cfg.min_elevation_deg, cfg.steering_limit_deg,
            cfg.altitude_min_km, cfg.altitude_max_km,
        )
        n_visible = int(visible.size)
        if n_visible < requested:
            raise RuntimeError(
                f"Scene {scene_id}: requested N={requested}, only {n_visible} satellites pass geometry"
            )
        rng = np.random.default_rng(cfg.random_seed + 100_000 + int(scene_id))
        chosen = np.sort(rng.choice(visible, size=requested, replace=False))
        states = self.catalog.state_series(times, chosen)
        return states, n_visible

    def manifest(self) -> Dict[str, object]:
        return {
            "catalog_path": str(Path(self.cfg.catalog_path).resolve()),
            "catalog_url": self.cfg.catalog_url,
            "catalog_records": self.catalog.size,
            "base_epoch_iso": self.base_epoch.isoformat(),
        }


def _correlated_shadow(cfg: StudyConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.shadow_sigma_db <= 0:
        return np.zeros(cfg.grid_size * cfg.grid_size)
    spacing = cfg.region_side_km / max(cfg.grid_size - 1, 1)
    sigma_px = max(cfg.shadow_corr_km / max(spacing, 1e-9) / np.sqrt(2.0), 0.25)
    field = gaussian_filter(rng.normal(size=(cfg.grid_size, cfg.grid_size)), sigma=sigma_px,
                            mode="reflect")
    field -= field.mean()
    field /= max(field.std(), 1e-12)
    return (cfg.shadow_sigma_db * field).ravel()


def _blockage_mask(cfg: StudyConfig, xy_km: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fraction = float(np.clip(cfg.blockage_fraction, 0.0, 1.0))
    if fraction <= 0:
        return np.zeros(len(xy_km), dtype=bool)
    center = rng.uniform(-0.35, 0.35, size=2) * cfg.region_side_km
    radius2 = np.sum((xy_km - center) ** 2, axis=1)
    count = max(1, int(round(fraction * len(xy_km))))
    selected = np.argpartition(radius2, count - 1)[:count]
    mask = np.zeros(len(xy_km), dtype=bool)
    mask[selected] = True
    return mask


def _rician_average_gain(k_db: float, n_points: int, looks: int,
                         rng: np.random.Generator) -> np.ndarray:
    if not np.isfinite(k_db) or k_db >= 60.0:
        return np.ones(n_points)
    k = 10.0 ** (k_db / 10.0)
    los = np.sqrt(k / (k + 1.0))
    scatter = np.sqrt(1.0 / (2.0 * (k + 1.0)))
    real = rng.normal(size=(n_points, looks))
    imag = rng.normal(size=(n_points, looks))
    return np.mean(np.abs(los + scatter * (real + 1j * imag)) ** 2, axis=1)


def simulate_scene(cfg: StudyConfig, factory: OrbitSceneFactory, scene_id: int,
                   active_ids: Optional[Sequence[int]] = None,
                   truth_pattern: Optional[str] = None) -> Dict[str, object]:
    """Generate a complete grid, hide measurements, and retain independent queries."""
    rng = np.random.default_rng(cfg.random_seed + int(scene_id))
    epoch = factory.epoch(scene_id)
    state_series, n_visible = factory.candidate_states(scene_id, cfg.n_candidates, epoch)
    lats, lons, xy = ground_grid(cfg)
    times = [epoch + dt.timedelta(seconds=t * cfg.snapshot_step_s)
             for t in range(cfg.n_snapshots)]
    catalog_ids = np.asarray([state["catalog_index"] for state in state_series[0]], dtype=int)
    delta_s = 0.5
    before = factory.catalog.state_series(
        [value - dt.timedelta(seconds=delta_s) for value in times], catalog_ids
    )
    after = factory.catalog.state_series(
        [value + dt.timedelta(seconds=delta_s) for value in times], catalog_ids
    )
    features_by_time = []
    for t, states in enumerate(state_series):
        current = []
        for candidate, state in enumerate(states):
            before_feature = candidate_feature(before[t][candidate], lats, lons, cfg)
            after_feature = candidate_feature(after[t][candidate], lats, lons, cfg)
            radial_mps = (after_feature["range_km"] - before_feature["range_km"]) * 1000.0 / (2.0 * delta_s)
            current.append(candidate_feature(state, lats, lons, cfg, radial_mps))
        features_by_time.append(current)
    n_grid = len(lats)
    if cfg.n_measurements >= n_grid:
        raise ValueError("n_measurements must be smaller than the grid to preserve a query set")
    order = rng.permutation(n_grid)
    train_grid_idx = np.sort(order[:cfg.n_measurements])
    query_grid_idx = np.sort(order[cfg.n_measurements:])
    train_spatial_idx = np.concatenate([train_grid_idx + t * n_grid for t in range(cfg.n_snapshots)])
    query_spatial_idx = np.concatenate([query_grid_idx + t * n_grid for t in range(cfg.n_snapshots)])
    n_subbands = max(int(cfg.n_subbands), 1)
    train_idx = np.concatenate([np.arange(index * n_subbands, (index + 1) * n_subbands)
                                for index in train_spatial_idx])
    query_idx = np.concatenate([np.arange(index * n_subbands, (index + 1) * n_subbands)
                                for index in query_spatial_idx])

    features = []
    for candidate in range(cfg.n_candidates):
        per_time = [features_by_time[t][candidate] for t in range(cfg.n_snapshots)]
        merged: Dict[str, object] = {}
        for key in per_time[0]:
            first = per_time[0][key]
            if isinstance(first, np.ndarray):
                merged[key] = np.concatenate([np.asarray(item[key]) for item in per_time], axis=0)
            elif key in {"center_az_deg", "center_offnadir_deg"}:
                merged[key] = np.concatenate([
                    np.full(n_grid, float(item[key])) for item in per_time
                ])
            else:
                merged[key] = first
        features.append(merged)

    k = min(cfg.n_active, cfg.n_candidates)
    if active_ids is not None:
        active = np.asarray(active_ids, dtype=int)
    elif k == 2 and 0.0 <= cfg.active_pair_quantile <= 1.0:
        nominal_width = 0.5 * (cfg.truth_width_min_deg + cfg.truth_width_max_deg)
        nominal = []
        for feature in features:
            atom = template(feature, np.asarray([0.0, 0.0, nominal_width]), cfg)
            atom = atom[train_idx] - np.mean(atom[train_idx])
            nominal.append(atom / max(float(np.linalg.norm(atom)), 1e-300))
        pairs = [(abs(float(nominal[i] @ nominal[j])), i, j)
                 for i in range(cfg.n_candidates) for j in range(i + 1, cfg.n_candidates)]
        pairs.sort()
        rank = int(round(cfg.active_pair_quantile * (len(pairs) - 1)))
        active = np.asarray(sorted(pairs[rank][1:]), dtype=int)
    else:
        active = np.sort(rng.choice(cfg.n_candidates, size=k, replace=False))
    if np.any(active < 0) or np.any(active >= cfg.n_candidates):
        raise ValueError("active_ids outside candidate roster")

    pattern = truth_pattern or cfg.truth_pattern
    params: Dict[int, Dict[str, float]] = {}
    n_spatial = cfg.n_snapshots * n_grid
    n_field = n_spatial * n_subbands
    signal_components = np.zeros((len(active), n_field), dtype=float)
    blockage = _blockage_mask(cfg, xy, rng)
    blockage_spatial = np.tile(blockage, cfg.n_snapshots)
    blockage_full = np.repeat(blockage_spatial, n_subbands)
    for pos, idx in enumerate(active):
        offset_az = float(rng.uniform(-cfg.truth_center_bound_deg, cfg.truth_center_bound_deg))
        offset_el = float(rng.uniform(-cfg.truth_center_bound_deg, cfg.truth_center_bound_deg))
        width = float(rng.uniform(cfg.truth_width_min_deg, cfg.truth_width_max_deg))
        if len(active) == 2 and cfg.active_power_ratio_db >= 0:
            tx_offset = float((0.5 if pos == 0 else -0.5) * cfg.active_power_ratio_db)
        else:
            tx_offset = float(rng.normal(0.0, cfg.tx_offset_sigma_db))
        doppler_offset = float(rng.normal(0.0, cfg.doppler_error_sigma_hz))
        pars = np.asarray([offset_az, offset_el, width])
        component = template(
            features[int(idx)], pars, cfg, pattern=pattern,
            axis_ratio=cfg.ellipse_axis_ratio,
            rotation_deg=cfg.ellipse_rotation_deg,
            doppler_offset_hz=doppler_offset,
        ) * 10.0 ** (tx_offset / 10.0)
        shadow_db = np.repeat(np.tile(_correlated_shadow(cfg, rng), cfg.n_snapshots),
                              n_subbands)
        component *= 10.0 ** (shadow_db / 10.0)
        if np.any(blockage_full):
            component[blockage_full] *= 10.0 ** (-cfg.blockage_loss_db / 10.0)
        signal_components[pos] = component
        params[int(idx)] = {
            "offset_az_deg": offset_az,
            "offset_el_deg": offset_el,
            "width_deg": width,
            "tx_offset_db": tx_offset,
            "doppler_offset_hz": doppler_offset,
        }

    signal = signal_components.sum(axis=0)
    signal_aggregate = signal.reshape(n_spatial, n_subbands).sum(axis=1)
    normalized_x = np.tile(xy[:, 0] / max(0.5 * cfg.region_side_km, 1e-12), cfg.n_snapshots)
    background_aggregate = cfg.noise_mw * cfg.background_ratio * (
        1.0 + cfg.background_gradient * normalized_x
    )
    background_aggregate = np.maximum(background_aggregate, 0.0)
    background = np.repeat(background_aggregate[:, None] / n_subbands,
                           n_subbands, axis=1).reshape(-1)

    # Average powers from mutually uncorrelated sources. Rician fluctuations are
    # applied per source; receiver noise plus background is averaged separately.
    looks = max(int(cfg.n_looks), 1)
    mean_train = cfg.noise_subband_mw + background[train_idx] + signal[train_idx]
    if cfg.rician_k_db >= 60.0:
        # Independent coded satellite waveforms plus receiver noise give a
        # Gamma-distributed averaged energy with the total mean power.
        observed = rng.gamma(shape=looks, scale=mean_train / looks)
    else:
        measured_signal = np.zeros(len(train_spatial_idx) * n_subbands)
        for pos in range(len(active)):
            fading = _rician_average_gain(cfg.rician_k_db, len(train_spatial_idx), looks, rng)
            component = signal_components[pos].reshape(n_spatial, n_subbands)[train_spatial_idx]
            measured_signal += (component * fading[:, None]).reshape(-1)
        noise_background = rng.gamma(
            shape=looks, scale=(cfg.noise_subband_mw + background[train_idx]) / looks
        )
        observed = measured_signal + noise_background
    calibration_db = rng.normal(0.0, cfg.calibration_sigma_db, len(train_spatial_idx))
    observed *= np.repeat(10.0 ** (calibration_db / 10.0), n_subbands)
    observations_aggregate = observed.reshape(len(train_spatial_idx), n_subbands).sum(axis=1)
    mean_total = cfg.noise_subband_mw + background + signal
    mean_total_aggregate = mean_total.reshape(n_spatial, n_subbands).sum(axis=1)

    states_mid = state_series[cfg.n_snapshots // 2]
    sat_ecef = np.asarray([
        lla_to_ecef(s["lat"], s["lon"], s["altitude_km"]) for s in states_mid
    ])
    region_ecef = lla_to_ecef(cfg.center_lat_deg, cfg.center_lon_deg, 0.0)
    sat_dirs = sat_ecef - region_ecef
    sat_dirs /= np.maximum(np.linalg.norm(sat_dirs, axis=1, keepdims=True), 1e-12)

    return {
        "scene_id": int(scene_id),
        "epoch_iso": epoch.isoformat(),
        "states": states_mid,
        "state_series": state_series,
        "n_visible": n_visible,
        "lats": lats,
        "lons": lons,
        "xy_km": xy,
        "train_idx": train_idx,
        "query_idx": query_idx,
        "train_spatial_idx": train_spatial_idx,
        "query_spatial_idx": query_spatial_idx,
        "train_grid_idx": train_grid_idx,
        "query_grid_idx": query_grid_idx,
        "n_grid": n_grid,
        "features_all": features,
        "features_train": [subset_feature(f, train_spatial_idx) for f in features],
        "active_ids": active,
        "params": params,
        "signal_components_mw": signal_components,
        "signal_mw": signal,
        "signal_aggregate_mw": signal_aggregate,
        "background_mw": background,
        "background_aggregate_mw": background_aggregate,
        "noise_mw": float(cfg.noise_subband_mw),
        "noise_total_mw": float(cfg.noise_mw),
        "mean_total_mw": mean_total,
        "mean_total_aggregate_mw": mean_total_aggregate,
        "observations_mw": observed,
        "observations_aggregate_mw": observations_aggregate,
        "calibration_db": calibration_db,
        "blockage_mask": blockage_full,
        "satellite_directions": sat_dirs,
        "truth_pattern": pattern,
        "config": cfg.to_dict(),
    }


def save_scene(scene: Mapping[str, object], path: Path) -> None:
    """Save every numeric truth input needed to recompute metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    states = scene["states"]
    state_series = scene["state_series"]
    params = scene["params"]
    np.savez_compressed(
        path,
        scene_id=np.asarray(scene["scene_id"]),
        epoch_iso=np.asarray(scene["epoch_iso"]),
        n_visible=np.asarray(scene["n_visible"]),
        lats=scene["lats"], lons=scene["lons"], xy_km=scene["xy_km"],
        train_idx=scene["train_idx"], query_idx=scene["query_idx"],
        train_spatial_idx=scene["train_spatial_idx"], query_spatial_idx=scene["query_spatial_idx"],
        train_grid_idx=scene["train_grid_idx"], query_grid_idx=scene["query_grid_idx"],
        active_ids=scene["active_ids"],
        observations_mw=scene["observations_mw"],
        observations_aggregate_mw=scene["observations_aggregate_mw"],
        signal_components_mw=scene["signal_components_mw"],
        signal_mw=scene["signal_mw"], signal_aggregate_mw=scene["signal_aggregate_mw"],
        background_mw=scene["background_mw"],
        background_aggregate_mw=scene["background_aggregate_mw"],
        mean_total_mw=scene["mean_total_mw"],
        mean_total_aggregate_mw=scene["mean_total_aggregate_mw"],
        noise_mw=np.asarray(scene["noise_mw"]), noise_total_mw=np.asarray(scene["noise_total_mw"]),
        calibration_db=scene["calibration_db"],
        blockage_mask=scene["blockage_mask"], satellite_directions=scene["satellite_directions"],
        state_catalog_index=np.asarray([s["catalog_index"] for s in states], dtype=int),
        state_lat=np.asarray([s["lat"] for s in states]),
        state_lon=np.asarray([s["lon"] for s in states]),
        state_altitude_km=np.asarray([s["altitude_km"] for s in states]),
        state_names=np.asarray([s["name"] for s in states]),
        state_series_catalog_index=np.asarray([[s["catalog_index"] for s in snapshot]
                                               for snapshot in state_series], dtype=int),
        state_series_lat=np.asarray([[s["lat"] for s in snapshot] for snapshot in state_series]),
        state_series_lon=np.asarray([[s["lon"] for s in snapshot] for snapshot in state_series]),
        state_series_altitude_km=np.asarray([[s["altitude_km"] for s in snapshot]
                                             for snapshot in state_series]),
        config_json=np.asarray(json.dumps(scene["config"], sort_keys=True)),
        params_json=np.asarray(json.dumps(params, sort_keys=True)),
        truth_pattern=np.asarray(scene["truth_pattern"]),
    )
