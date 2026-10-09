"""Authentic TLE ingestion and bulk SGP4 propagation.

Only a few tens of satellites in a Starlink-sized catalogue illuminate a given
region at a given instant: the region is visible from a spherical cap of about
seven degrees' radius, roughly 0.4% of the sphere.  Screening therefore has to
run over the whole catalogue rather than an arbitrary slice of it, so this
module propagates every record at once with ``SatrecArray`` and keeps the
satellites that actually see the region.

Positions are propagated in TEME and rotated to ECEF through the Greenwich mean
sidereal angle.  Geodetic subtleties are deliberately ignored: the rest of the
study uses a spherical Earth, and using a geodetic sub-point here would make
the geometry inconsistent with the link budget that consumes it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import ecef_to_lla, elevation_deg, lla_to_ecef, offnadir_deg

DEFAULT_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
EARTH_RADIUS_KM = 6378.137

SatState = Dict[str, float]
StateSeries = List[List[SatState]]

# A fixed epoch keeps every run reproducible; trials vary the geometry by
# drawing an offset inside a window rather than by using the wall clock.
REFERENCE_EPOCH = dt.datetime(2026, 3, 21, 12, 0, 0, tzinfo=dt.timezone.utc)


def default_epoch() -> dt.datetime:
    return REFERENCE_EPOCH


def snapshot_times(epoch: dt.datetime, n_snapshots: int, step_s: float) -> List[dt.datetime]:
    return [epoch + dt.timedelta(seconds=i * step_s) for i in range(n_snapshots)]


def _julian(times: Sequence[dt.datetime]) -> Tuple[np.ndarray, np.ndarray]:
    """Split UTC datetimes into the whole-day and fractional parts SGP4 wants."""
    from sgp4.api import jday

    jd = np.empty(len(times))
    fr = np.empty(len(times))
    for i, when in enumerate(times):
        t = when.astimezone(dt.timezone.utc)
        jd[i], fr[i] = jday(t.year, t.month, t.day, t.hour, t.minute,
                            t.second + t.microsecond * 1e-6)
    return jd, fr


def gmst_rad(jd_ut1: np.ndarray) -> np.ndarray:
    """Greenwich mean sidereal angle (Vallado's ``gstime``)."""
    t = (np.asarray(jd_ut1, dtype=float) - 2451545.0) / 36525.0
    seconds = (-6.2e-6 * t ** 3 + 0.093104 * t ** 2
               + (876600.0 * 3600.0 + 8640184.812866) * t + 67310.54841)
    return np.deg2rad((seconds % 86400.0) / 240.0)


def teme_to_ecef(teme_km: np.ndarray, jd_ut1: np.ndarray) -> np.ndarray:
    """Rotate TEME positions ``(n_sat, n_time, 3)`` into ECEF."""
    theta = gmst_rad(jd_ut1)[None, :]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x, y, z = teme_km[..., 0], teme_km[..., 1], teme_km[..., 2]
    return np.stack((cos_t * x + sin_t * y, -sin_t * x + cos_t * y, z), axis=-1)


class TLECatalog:
    """A TLE catalogue that can be propagated and screened in bulk."""

    def __init__(self, tle_url: str = DEFAULT_TLE_URL,
                 cache_path: Path = Path("results_satbeam/tle/catalog.tle"),
                 max_records: int = 0):
        self.tle_url = tle_url
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = int(max_records)
        self.names: List[str] = []
        self._array = None
        self._propagation_cache: Dict[Tuple[str, ...], np.ndarray] = {}

    # -- loading ---------------------------------------------------------
    def load(self, force_download: bool = False) -> int:
        from sgp4.api import Satrec, SatrecArray

        if force_download or not self.cache_path.exists():
            import requests

            response = requests.get(self.tle_url, timeout=60)
            response.raise_for_status()
            self.cache_path.write_text(response.text, encoding="utf-8")
        records = self._parse(self.cache_path.read_text(encoding="utf-8"), self.max_records)
        if not records:
            raise RuntimeError(f"No valid TLE records were found in {self.cache_path}")
        satrecs = []
        self.names = []
        for name, line1, line2 in records:
            try:
                satrecs.append(Satrec.twoline2rv(line1, line2))
            except Exception:
                continue
            self.names.append(name)
        if not satrecs:
            raise RuntimeError("No TLE record could be parsed by SGP4")
        self._array = SatrecArray(satrecs)
        self._propagation_cache.clear()
        return len(satrecs)

    @staticmethod
    def _parse(text: str, max_records: int = 0) -> List[Tuple[str, str, str]]:
        """Accept both two-line and three-line (named) TLE files."""
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        records: List[Tuple[str, str, str]] = []
        i = 0
        while i + 1 < len(lines) and (max_records <= 0 or len(records) < max_records):
            if lines[i].startswith("1 "):
                name = f"SAT-{len(records):05d}"
                line1, line2 = lines[i], lines[i + 1]
                i += 2
            elif i + 2 < len(lines):
                name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
                i += 3
            else:
                break
            if line1.startswith("1 ") and line2.startswith("2 "):
                records.append((name.strip(), line1, line2))
        return records

    @property
    def size(self) -> int:
        return len(self.names)

    # -- propagation -----------------------------------------------------
    def propagate(self, times: Sequence[dt.datetime]) -> np.ndarray:
        """ECEF positions in km, shape ``(n_sat, n_time, 3)``.

        Records that SGP4 rejects at a requested epoch (decayed or numerically
        unstable elements) come back as NaN and are dropped by the screening.
        """
        if self._array is None:
            self.load()
        key = tuple(t.isoformat() for t in times)
        cached = self._propagation_cache.get(key)
        if cached is not None:
            return cached
        jd, fr = _julian(times)
        errors, teme, _ = self._array.sgp4(jd, fr)
        ecef = teme_to_ecef(np.asarray(teme, dtype=float), jd + fr)
        ecef[np.asarray(errors) != 0] = np.nan
        # Keep at most a handful of cached epochs: a scaling sweep would
        # otherwise retain one array per condition for the whole run.
        if len(self._propagation_cache) > 8:
            self._propagation_cache.clear()
        self._propagation_cache[key] = ecef
        return ecef

    # -- screening -------------------------------------------------------
    def visible_indices(self, times: Sequence[dt.datetime], center_lat: float, center_lon: float,
                        min_elevation_deg: float, offaxis_limit_deg: float,
                        alt_min_km: float = 200.0, alt_max_km: float = 2000.0) -> np.ndarray:
        """Catalogue indices that illuminate the region at every snapshot.

        A satellite qualifies when it stays above ``min_elevation_deg`` at the
        region centre, can steer there within ``offaxis_limit_deg``, and reports
        a plausible LEO altitude.  The altitude test matters in practice: a
        public catalogue carries decayed and stale elements that SGP4 happily
        propagates to absurd distances.
        """
        ecef = self.propagate(times)
        target = lla_to_ecef(center_lat, center_lon, 0.0)
        elevation = elevation_deg(ecef, target)
        theta = offnadir_deg(ecef, target)
        altitude = np.linalg.norm(ecef, axis=-1) - EARTH_RADIUS_KM
        healthy = np.all(np.isfinite(elevation) & np.isfinite(theta), axis=1)
        healthy &= np.all((altitude >= alt_min_km) & (altitude <= alt_max_km), axis=1)
        return np.flatnonzero(healthy
                              & np.all(elevation >= min_elevation_deg, axis=1)
                              & np.all(theta <= offaxis_limit_deg, axis=1))

    def state_series(self, times: Sequence[dt.datetime], indices: Sequence[int]) -> StateSeries:
        """Sub-satellite points for ``indices``, one list per snapshot."""
        ecef = self.propagate(times)[np.asarray(indices, dtype=int)]
        lat, lon, alt = ecef_to_lla(ecef)
        series: StateSeries = []
        for t_idx in range(len(times)):
            series.append([{"id": new_id, "name": self.names[int(cat_id)],
                            "lat": float(lat[new_id, t_idx]), "lon": float(lon[new_id, t_idx]),
                            "altitude_km": float(alt[new_id, t_idx]),
                            "catalog_index": int(cat_id)}
                           for new_id, cat_id in enumerate(indices)])
        return series


def visible_state_series(catalog: TLECatalog, times: Sequence[dt.datetime], cfg: object,
                         max_candidates: Optional[int] = None,
                         rng: Optional[np.random.Generator] = None) -> Tuple[StateSeries, int]:
    """States for the satellites illuminating the region, and how many there were.

    When more satellites are visible than the experiment asks for, the roster is
    sampled at random rather than truncated, so the candidate set does not
    inherit the catalogue ordering.
    """
    indices = catalog.visible_indices(times, cfg.region_center_lat, cfg.region_center_lon,
                                      cfg.min_elevation_deg, cfg.offaxis_limit_deg,
                                      cfg.sat_alt_min_km, cfg.sat_alt_max_km)
    n_visible = int(indices.size)
    if max_candidates is not None and n_visible > max_candidates:
        picker = rng if rng is not None else np.random.default_rng(0)
        indices = np.sort(picker.choice(indices, size=int(max_candidates), replace=False))
    return catalog.state_series(times, indices), n_visible
