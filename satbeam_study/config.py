"""Single source of truth for the revised SateBeam experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict


@dataclass(frozen=True)
class StudyConfig:
    # Archived orbit input and reference campaign.
    catalog_path: str = "data/tle/catalog_starlink_20260911.tle"
    catalog_url: str = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    base_epoch_iso: str = "2026-09-10T10:07:05+00:00"
    epoch_window_min: float = 90.0
    center_lat_deg: float = 40.0
    center_lon_deg: float = -100.0
    region_side_km: float = 200.0
    grid_size: int = 32
    n_measurements: int = 160
    n_snapshots: int = 4
    snapshot_step_s: float = 60.0
    n_candidates: int = 16
    n_active: int = 4
    min_elevation_deg: float = 20.0
    steering_limit_deg: float = 65.0
    min_support_fraction: float = 0.35
    altitude_min_km: float = 200.0
    altitude_max_km: float = 2000.0

    # Link budget. EIRP is the peak-direction power integrated over bandwidth.
    frequency_ghz: float = 12.0
    bandwidth_hz: float = 1.0e6
    n_subbands: int = 33
    source_spectral_fwhm_hz: float = 50_000.0
    doppler_error_sigma_hz: float = 0.0
    noise_figure_db: float = 5.0
    receiver_gain_dbi: float = 10.0
    eirp_dbw: float = 36.0
    zenith_atmospheric_loss_db: float = 0.20
    polarization_loss_db: float = 0.5
    background_ratio: float = 1.0
    n_looks: int = 64

    # Truth and mismatch controls.
    truth_pattern: str = "circular"
    estimator_pattern: str = "circular"
    truth_width_min_deg: float = 4.0
    truth_width_max_deg: float = 12.0
    fit_width_min_deg: float = 4.0
    fit_width_max_deg: float = 12.0
    truth_center_bound_deg: float = 3.0
    fit_center_bound_deg: float = 6.0
    tx_offset_sigma_db: float = 0.0
    amplitude_min_db: float = -3.0
    amplitude_max_db: float = 3.0
    background_max_ratio: float = 5.0
    ellipse_axis_ratio: float = 1.0
    ellipse_rotation_deg: float = 0.0
    shadow_sigma_db: float = 0.0
    shadow_corr_km: float = 20.0
    calibration_sigma_db: float = 0.0
    blockage_fraction: float = 0.0
    blockage_loss_db: float = 20.0
    background_gradient: float = 0.0
    rician_k_db: float = 99.0
    active_pair_quantile: float = -1.0
    active_power_ratio_db: float = -1.0

    # SateBeam and grid methods.
    order_cap: int = 8
    selection_rule: str = "bic"  # bic | aic | margin_bic | calibrated
    selection_margin: float = 0.0
    calibrated_threshold: float = 0.0
    candidate_starts: int = 2
    candidate_maxiter: int = 35
    joint_maxiter: int = 120
    refine_each_step: bool = False
    joint_shortlist: int = 3
    optimizer_ftol: float = 1e-8
    grid_step_deg: float = 2.0
    grid_span_deg: float = 6.0
    grid_width_count: int = 5
    lasso_alphas: int = 8
    lasso_maxiter: int = 10000
    lasso_cross_validate: bool = False
    lasso_alpha_relative: float = 1e-3
    screening_iterations: int = 160
    screening_lambda_relative: float = 0.05
    torch_joint_iterations: int = 240
    torch_joint_lr: float = 0.03
    peak_mad_scale: float = 3.0
    use_gpu: bool = True
    gpu_min_dictionary_elements: int = 500_000

    random_seed: int = 20260911

    @property
    def noise_dbm(self) -> float:
        return -174.0 + 10.0 * __import__("math").log10(self.bandwidth_hz) + self.noise_figure_db

    @property
    def noise_mw(self) -> float:
        return 10.0 ** (self.noise_dbm / 10.0)

    @property
    def noise_subband_mw(self) -> float:
        return self.noise_mw / max(self.n_subbands, 1)

    @property
    def max_order(self) -> int:
        n_observations = self.n_measurements * self.n_snapshots
        dimension_cap = max(0, (n_observations - 3) // 4)
        return min(self.order_cap, self.n_candidates, dimension_cap)

    def evolve(self, **overrides: Any) -> "StudyConfig":
        known = {item.name for item in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown StudyConfig fields: {sorted(unknown)}")
        return replace(self, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


METHOD_COLORS = {
    "SateBeam": "#D1495B",
    "OMP-Refine": "#E07A3F",
    "OMP-Grid": "#2F6690",
    "MP-Grid": "#6C5B9A",
    "Lasso-Grid": "#2A9D8F",
    "Orbit-NNLS": "#8C6D31",
    "Peak": "#7B7F8C",
    "IDW": "#4FA3B5",
    "Kriging": "#D6A21E",
    "SoftImpute": "#C06C84",
    "DeepRM-TD": "#7656A5",
    "Oracle-Support": "#252525",
}
