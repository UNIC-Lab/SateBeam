"""Geometry, link-budget, antenna, and analytic validation routines."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import j1, ndtr

from satbeam.geometry import elevation_deg, lla_to_ecef, look_angles, wrap_deg

from .config import StudyConfig


EARTH_RADIUS_KM = 6378.137
SPEED_OF_LIGHT = 299_792_458.0
SINC_HPBW_FACTOR = 0.8858929413789047
CIRCULAR_HALF_POWER_X = brentq(lambda x: (2.0 * j1(x) / x) ** 2 - 0.5, 1.0, 3.0)


def ground_grid(cfg: StudyConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular local east/north grid mapped to latitude and longitude."""
    half = 0.5 * cfg.region_side_km
    axis = np.linspace(-half, half, cfg.grid_size)
    east, north = np.meshgrid(axis, axis)
    lat = cfg.center_lat_deg + north.ravel() / 111.32
    lon = cfg.center_lon_deg + east.ravel() / (
        111.32 * np.cos(np.deg2rad(cfg.center_lat_deg))
    )
    xy = np.column_stack((east.ravel(), north.ravel()))
    return lat, lon, xy


def body_vectors(az_deg: np.ndarray, offnadir_deg: np.ndarray) -> np.ndarray:
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(offnadir_deg, dtype=float))
    return np.column_stack((np.sin(el) * np.cos(az),
                            np.sin(el) * np.sin(az),
                            np.cos(el)))


def center_vector(center_az_deg: float, center_offnadir_deg: float,
                  offset_az_deg: float, offset_el_deg: float) -> np.ndarray:
    az = np.deg2rad(center_az_deg + offset_az_deg)
    el = np.deg2rad(center_offnadir_deg + offset_el_deg)
    return np.asarray([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)])


def local_beam_coordinates(feature: Mapping[str, np.ndarray], offset_az_deg: float,
                           offset_el_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact separation and gnomonic coordinates around a beam center."""
    q = feature["body_vectors"]
    az0 = np.deg2rad(np.asarray(feature["center_az_deg"], dtype=float) + offset_az_deg)
    el0 = np.deg2rad(np.asarray(feature["center_offnadir_deg"], dtype=float) + offset_el_deg)
    if az0.ndim == 0:
        az0 = np.full(len(q), float(az0)); el0 = np.full(len(q), float(el0))
    c = np.column_stack((np.sin(el0) * np.cos(az0),
                         np.sin(el0) * np.sin(az0), np.cos(el0)))
    e_az = np.column_stack((-np.sin(az0), np.cos(az0), np.zeros_like(az0)))
    e_el = np.column_stack((np.cos(el0) * np.cos(az0),
                            np.cos(el0) * np.sin(az0), -np.sin(el0)))
    dot_c = np.clip(np.sum(q * c, axis=1), -1.0, 1.0)
    x = np.rad2deg(np.arctan2(np.sum(q * e_az, axis=1), dot_c))
    y = np.rad2deg(np.arctan2(np.sum(q * e_el, axis=1), dot_c))
    theta = np.rad2deg(np.arccos(dot_c))
    return x, y, theta


def sinc_gain(x_deg: np.ndarray, y_deg: np.ndarray, width_deg: float) -> np.ndarray:
    scale = SINC_HPBW_FACTOR / max(float(width_deg), 1e-6)
    return np.sinc(scale * x_deg) ** 2 * np.sinc(scale * y_deg) ** 2


def circular_gain(theta_deg: np.ndarray, width_deg: float) -> np.ndarray:
    """Uniform circular aperture, parameterized by full HPBW."""
    half = np.deg2rad(max(float(width_deg), 1e-6) / 2.0)
    ka = CIRCULAR_HALF_POWER_X / max(np.sin(half), 1e-12)
    z = ka * np.sin(np.deg2rad(np.asarray(theta_deg, dtype=float)))
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-12
    out[nz] = (2.0 * j1(z[nz]) / z[nz]) ** 2
    return np.maximum(out, 1e-12)


def array_gain(x_deg: np.ndarray, y_deg: np.ndarray, width_deg: float) -> np.ndarray:
    """Independent tapered rectangular-array factor used only as mismatch truth."""
    n_elem = max(8, int(round(102.0 / max(width_deg, 1.0))))
    weights = np.hamming(n_elem)
    weights /= weights.sum()

    def cut(angle_deg: np.ndarray) -> np.ndarray:
        phase = np.pi * np.sin(np.deg2rad(angle_deg))
        idx = np.arange(n_elem) - 0.5 * (n_elem - 1)
        af = np.exp(1j * phase[:, None] * idx[None, :]) @ weights
        return np.abs(af) ** 2

    return np.maximum(cut(np.asarray(x_deg)) * cut(np.asarray(y_deg)), 1e-12)


def antenna_gain(feature: Mapping[str, np.ndarray], offset_az_deg: float,
                 offset_el_deg: float, width_deg: float, pattern: str,
                 axis_ratio: float = 1.0, rotation_deg: float = 0.0) -> np.ndarray:
    x, y, theta = local_beam_coordinates(feature, offset_az_deg, offset_el_deg)
    if axis_ratio != 1.0 or rotation_deg != 0.0:
        a = np.deg2rad(rotation_deg)
        xr = np.cos(a) * x + np.sin(a) * y
        yr = -np.sin(a) * x + np.cos(a) * y
        theta = np.hypot(xr / max(axis_ratio, 1e-6), yr * max(axis_ratio, 1e-6))
        x, y = xr / max(axis_ratio, 1e-6), yr * max(axis_ratio, 1e-6)
    if pattern == "sinc":
        return sinc_gain(x, y, width_deg)
    if pattern == "circular":
        return circular_gain(theta, width_deg)
    if pattern == "array":
        return array_gain(x, y, width_deg)
    raise ValueError(f"Unknown pattern: {pattern}")


def candidate_feature(state: Mapping[str, float], lats: np.ndarray, lons: np.ndarray,
                      cfg: StudyConfig,
                      radial_velocity_mps: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    az, offnadir, ranges = look_angles(state["lat"], state["lon"], state["altitude_km"], lats, lons)
    center_az, center_off, _ = look_angles(
        state["lat"], state["lon"], state["altitude_km"],
        np.asarray([cfg.center_lat_deg]), np.asarray([cfg.center_lon_deg])
    )
    sat = lla_to_ecef(state["lat"], state["lon"], state["altitude_km"])
    ground = lla_to_ecef(lats, lons, 0.0)
    elev = elevation_deg(sat[None, :], ground)
    fspl = 20.0 * np.log10(np.maximum(ranges, 1e-9)) + 20.0 * np.log10(cfg.frequency_ghz) + 92.45
    atm = cfg.zenith_atmospheric_loss_db / np.maximum(np.sin(np.deg2rad(np.maximum(elev, 3.0))), 0.05)
    received_dbm = (cfg.eirp_dbw + cfg.receiver_gain_dbi - fspl - atm
                    - cfg.polarization_loss_db + 30.0)
    doppler = (np.zeros_like(ranges) if radial_velocity_mps is None else
               -cfg.frequency_ghz * 1e9 * np.asarray(radial_velocity_mps) / SPEED_OF_LIGHT)
    return {
        "az_deg": az,
        "offnadir_deg": offnadir,
        "range_km": ranges,
        "elevation_deg": elev,
        "body_vectors": body_vectors(az, offnadir),
        "center_az_deg": float(center_az[0]),
        "center_offnadir_deg": float(center_off[0]),
        "base_mw": 10.0 ** (received_dbm / 10.0),
        "doppler_hz": doppler,
        "catalog_index": int(state["catalog_index"]),
        "sat_lat": float(state["lat"]),
        "sat_lon": float(state["lon"]),
        "sat_altitude_km": float(state["altitude_km"]),
    }


def spectral_bin_weights(doppler_hz: np.ndarray, cfg: StudyConfig,
                         offset_hz: float = 0.0) -> np.ndarray:
    """Fraction of a Gaussian source spectrum captured by each subband."""
    n_subbands = max(int(cfg.n_subbands), 1)
    if n_subbands == 1:
        return np.ones((len(np.asarray(doppler_hz).reshape(-1)), 1))
    edges = np.linspace(-0.5 * cfg.bandwidth_hz, 0.5 * cfg.bandwidth_hz, n_subbands + 1)
    sigma = max(cfg.source_spectral_fwhm_hz / 2.354820045, 1.0)
    center = np.asarray(doppler_hz, dtype=float).reshape(-1, 1) + float(offset_hz)
    return ndtr((edges[1:][None, :] - center) / sigma) - ndtr(
        (edges[:-1][None, :] - center) / sigma
    )


def template(feature: Mapping[str, np.ndarray], params: np.ndarray, cfg: StudyConfig,
             pattern: str | None = None, axis_ratio: float = 1.0,
             rotation_deg: float = 0.0, doppler_offset_hz: float = 0.0) -> np.ndarray:
    gain = antenna_gain(feature, float(params[0]), float(params[1]), float(params[2]),
                        pattern or cfg.estimator_pattern, axis_ratio, rotation_deg)
    spatial = feature["base_mw"] * gain
    weights = spectral_bin_weights(feature.get("doppler_hz", np.zeros_like(spatial)),
                                   cfg, doppler_offset_hz)
    return (spatial[:, None] * weights).reshape(-1)


def physical_validation() -> Dict[str, float]:
    """Analytic identities that must hold before stochastic experiments."""
    d1, d2 = 500.0, 1000.0
    f1, f2 = 10.0, 20.0
    fspl_d = 20.0 * np.log10(d2 / d1)
    fspl_f = 20.0 * np.log10(f2 / f1)
    center = circular_gain(np.asarray([0.0]), 8.0)[0]
    half = circular_gain(np.asarray([4.0]), 8.0)[0]
    sinc_half = sinc_gain(np.asarray([4.0]), np.asarray([0.0]), 8.0)[0]
    return {
        "distance_doubling_db": float(fspl_d),
        "frequency_doubling_db": float(fspl_f),
        "circular_center_gain": float(center),
        "circular_half_power_gain": float(half),
        "sinc_half_power_gain": float(sinc_half),
        "max_identity_error": float(max(abs(fspl_d - 6.020599913),
                                         abs(fspl_f - 6.020599913),
                                         abs(center - 1.0), abs(half - 0.5),
                                         abs(sinc_half - 0.5))),
    }
