"""Geometry, link budget and antenna patterns.

These functions are the physical layer of the experiment: they turn a satellite
sub-point and a set of ground receivers into look angles, a free-space path
loss, and an antenna gain.  Nothing here knows about estimators or baselines.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np

EARTH_RADIUS_KM = 6378.137

PATTERNS = ("itu3gpp", "itu_s1528", "sinc")


def wrap_deg(delta: np.ndarray | float) -> np.ndarray:
    """Wrap angles to [-180, 180) degrees."""
    return np.rad2deg(np.angle(np.exp(1j * np.deg2rad(delta))))


def lla_to_ecef(lat_deg: np.ndarray | float, lon_deg: np.ndarray | float,
                alt_km: np.ndarray | float = 0.0) -> np.ndarray:
    """Spherical-Earth ECEF coordinates, sufficient for link-budget geometry."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    radius = EARTH_RADIUS_KM + np.asarray(alt_km)
    return np.stack((radius * np.cos(lat) * np.cos(lon),
                     radius * np.cos(lat) * np.sin(lon),
                     radius * np.sin(lat)), axis=-1)


def ecef_to_lla(ecef: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`lla_to_ecef`, broadcast over any leading axes."""
    ecef = np.asarray(ecef, dtype=float)
    radius = np.linalg.norm(ecef, axis=-1)
    lat = np.rad2deg(np.arcsin(np.clip(ecef[..., 2] / np.maximum(radius, 1e-12), -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(ecef[..., 1], ecef[..., 0]))
    return lat, lon, radius - EARTH_RADIUS_KM


def offnadir_deg(sat_ecef: np.ndarray, target_ecef: np.ndarray) -> np.ndarray:
    """Off-nadir angle to one target, broadcast over any leading axes.

    Used to screen a whole catalogue against the region of interest before any
    per-receiver geometry is computed.
    """
    sat = np.asarray(sat_ecef, dtype=float)
    vec = np.asarray(target_ecef, dtype=float) - sat
    ranges = np.linalg.norm(vec, axis=-1)
    nadir = -sat / np.maximum(np.linalg.norm(sat, axis=-1, keepdims=True), 1e-12)
    cosine = np.sum(vec * nadir, axis=-1) / np.maximum(ranges, 1e-12)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def elevation_deg(sat_ecef: np.ndarray, target_ecef: np.ndarray) -> np.ndarray:
    """Elevation of the satellite above the local horizon at ``target_ecef``.

    Off-nadir angle alone does not imply visibility: a satellite above the
    antipode sees the Earth's centre at nadir and so would pass an off-nadir
    test while being far below the horizon.  Elevation is the test that
    actually decides whether a link exists.
    """
    target = np.asarray(target_ecef, dtype=float)
    up = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-12)
    vec = np.asarray(sat_ecef, dtype=float) - target
    ranges = np.linalg.norm(vec, axis=-1)
    sine = np.sum(vec * up, axis=-1) / np.maximum(ranges, 1e-12)
    return np.rad2deg(np.arcsin(np.clip(sine, -1.0, 1.0)))


def look_angles(sat_lat: float, sat_lon: float, sat_alt_km: float,
                gs_lats: np.ndarray, gs_lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return satellite-body azimuth, off-nadir angle, and range in km."""
    sat = lla_to_ecef(sat_lat, sat_lon, sat_alt_km)
    gs = lla_to_ecef(gs_lats, gs_lons, 0.0)
    vectors = gs - sat
    ranges = np.linalg.norm(vectors, axis=1)
    unit = vectors / np.maximum(ranges[:, None], 1e-12)
    nadir = -sat / np.linalg.norm(sat)
    north = np.array([0.0, 0.0, 1.0])
    x_body = north - np.dot(north, nadir) * nadir
    x_body /= np.linalg.norm(x_body)
    y_body = np.cross(nadir, x_body)
    az = np.rad2deg(np.arctan2(unit @ y_body, unit @ x_body)) % 360.0
    offnadir = np.rad2deg(np.arccos(np.clip(unit @ nadir, -1.0, 1.0)))
    return az, offnadir, ranges


def fspl_db(range_km: np.ndarray, frequency_ghz: float) -> np.ndarray:
    """Free-space path loss with distance in km and frequency in GHz."""
    return 20.0 * np.log10(np.maximum(range_km, 1e-9)) + 20.0 * np.log10(frequency_ghz) + 92.45


def elliptic_offaxis(delta_az: np.ndarray, delta_el: np.ndarray,
                     bw_az: float, bw_el: float, rotation_deg: float = 0.0) -> np.ndarray:
    """Off-axis angle normalised by an elliptical, rotated 3 dB contour."""
    angle = np.deg2rad(rotation_deg)
    u = np.cos(angle) * delta_az + np.sin(angle) * delta_el
    v = -np.sin(angle) * delta_az + np.cos(angle) * delta_el
    return np.sqrt((u / max(bw_az, 1e-3)) ** 2 + (v / max(bw_el, 1e-3)) ** 2)


def antenna_gain_db(delta_az: np.ndarray, delta_el: np.ndarray, bw_az: float,
                    bw_el: float, pattern: str = "itu3gpp", rotation_deg: float = 0.0,
                    sidelobe_db: float = -25.0) -> np.ndarray:
    """Evaluate normalized ITU/3GPP-style or sinc-squared antenna patterns.

    ``itu3gpp`` follows the widely used 3GPP satellite antenna approximation
    ``G(theta) = -min(12(theta/theta_3dB)^2, A_m)`` in dB, with an elliptical
    off-axis angle.  ``itu_s1528`` adds a logarithmic sidelobe roll-off and a
    floor inspired by ITU-R S.1528 family patterns.  The ``sinc`` option is
    retained only as a deliberately mismatched legacy generator.
    """
    d_az = wrap_deg(delta_az)
    d_el = np.asarray(delta_el)
    if pattern == "sinc":
        a_az = 0.886 * np.pi / np.deg2rad(max(bw_az, 1e-3))
        a_el = 0.886 * np.pi / np.deg2rad(max(bw_el, 1e-3))
        gain = np.sinc(a_az * np.deg2rad(d_az) / np.pi) ** 2
        gain *= np.sinc(a_el * np.deg2rad(d_el) / np.pi) ** 2
        return 10.0 * np.log10(np.maximum(gain, 10.0 ** (sidelobe_db / 10.0)))
    theta = elliptic_offaxis(d_az, d_el, bw_az, bw_el, rotation_deg)
    if pattern == "itu_s1528":
        # Smooth mainlobe plus logarithmic sidelobe decay and a finite floor.
        main = -np.minimum(12.0 * theta ** 2, 25.0)
        side = -25.0 - 20.0 * np.log10(np.maximum(theta, 1.0))
        return np.where(theta <= 1.0, main, np.maximum(side, -45.0))
    if pattern != "itu3gpp":
        raise ValueError(f"Unknown antenna pattern: {pattern}")
    return -np.minimum(12.0 * theta ** 2, 30.0)


def antenna_gain_linear(*args, **kwargs) -> np.ndarray:
    return 10.0 ** (antenna_gain_db(*args, **kwargs) / 10.0)


def pattern_template(feature: Mapping[str, np.ndarray], cfg: object,
                     offset_az: float, offset_el: float, bw_az: float, bw_el: float,
                     rotation_deg: float = 0.0,
                     pattern: Optional[str] = None) -> np.ndarray:
    """Unit-amplitude received power template for one satellite hypothesis."""
    return feature["base_mw"] * antenna_gain_linear(
        feature["az"] - (feature["center_az"] + offset_az),
        feature["el"] - (feature["center_el"] + offset_el),
        bw_az, bw_el, pattern=pattern or cfg.pattern_estimator, rotation_deg=rotation_deg)
