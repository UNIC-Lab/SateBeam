"""SateBeam and matched-protocol comparison methods."""

from __future__ import annotations

import time
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import lsq_linear, minimize, nnls
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Lasso, LassoCV

from .config import StudyConfig
from .physics import (CIRCULAR_HALF_POWER_X, local_beam_coordinates,
                      spectral_bin_weights, template)


SOURCE_METHODS = {
    "SateBeam", "OMP-Refine", "OMP-Grid", "MP-Grid", "Lasso-Grid",
    "Orbit-NNLS", "Peak", "Oracle-Support",
}
MAP_METHODS = {"IDW", "Kriging", "SoftImpute", "DeepRM-TD"}


def _fit_positive(templates: Sequence[np.ndarray], y: np.ndarray, cfg: StudyConfig,
                  allow_zero_last: bool = False
                  ) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Profile bounded source amplitudes and a bounded nonnegative background."""
    columns = [np.ones_like(y)] + [np.asarray(item, dtype=float) for item in templates]
    design = np.column_stack(columns)
    scales = np.maximum(np.linalg.norm(design, axis=0), 1e-300)
    amplitude_lo = 10.0 ** (cfg.amplitude_min_db / 10.0)
    amplitude_hi = 10.0 ** (cfg.amplitude_max_db / 10.0)
    source_lower = [amplitude_lo] * len(templates)
    if allow_zero_last and source_lower:
        source_lower[-1] = 0.0
    lower = np.asarray([0.0] + source_lower)
    upper = np.asarray([cfg.background_max_ratio * cfg.noise_subband_mw] + [amplitude_hi] * len(templates))
    result = lsq_linear(design / scales, np.asarray(y, dtype=float),
                        bounds=(lower * scales, upper * scales),
                        method="trf", tol=1e-10, max_iter=500)
    coef = result.x / scales
    prediction = design @ coef
    residual = np.asarray(y, dtype=float) - prediction
    return coef[1:], float(coef[0]), prediction, float(residual @ residual)


def _score(rss0: float, rss1: float, n: int, cfg: StudyConfig) -> float:
    improvement = n * np.log(max(rss0, 1e-300) / max(rss1, 1e-300))
    rule = cfg.selection_rule.lower()
    if rule == "bic":
        return float(improvement - 4.0 * np.log(max(n, 2)))
    if rule == "aic":
        return float(improvement - 8.0)
    if rule == "margin_bic":
        return float(improvement - 4.0 * np.log(max(n, 2)) - cfg.selection_margin)
    if rule == "calibrated":
        return float(improvement - cfg.calibrated_threshold)
    raise ValueError(f"Unknown selection rule: {cfg.selection_rule}")


def _candidate_starts(feature: Mapping[str, object], residual: np.ndarray,
                      cfg: StudyConfig) -> List[np.ndarray]:
    middle = 0.5 * (cfg.fit_width_min_deg + cfg.fit_width_max_deg)
    amplitude_hi = 10.0 ** (cfg.amplitude_max_db / 10.0)
    coarse_values = np.linspace(-cfg.fit_center_bound_deg, cfg.fit_center_bound_deg, 3)
    coarse_widths = np.linspace(cfg.fit_width_min_deg, cfg.fit_width_max_deg, 3)
    coarse_best = None
    residual_scale = max(float(np.linalg.norm(residual)), 1e-300)
    normalized_residual = residual / residual_scale
    for offset_az in coarse_values:
        for offset_el in coarse_values:
            for width in coarse_widths:
                pars = np.asarray([offset_az, offset_el, width])
                atom = template(feature, pars, cfg)
                amp = np.clip(float(atom @ normalized_residual)
                              / max(float(atom @ atom), 1e-300), 0.0, amplitude_hi)
                error = normalized_residual - amp * atom
                value = float(error @ error)
                if coarse_best is None or value < coarse_best[0]:
                    coarse_best = (value, pars)
    assert coarse_best is not None
    starts = [coarse_best[1]]
    x, y, _ = local_beam_coordinates(feature, 0.0, 0.0)
    n_subbands = max(int(cfg.n_subbands), 1)
    spatial_residual = residual.reshape(-1, n_subbands).sum(axis=1)
    peak = int(np.argmax(spatial_residual))
    starts.append(np.asarray([
        np.clip(x[peak], -cfg.fit_center_bound_deg, cfg.fit_center_bound_deg),
        np.clip(y[peak], -cfg.fit_center_bound_deg, cfg.fit_center_bound_deg),
        middle,
    ]))
    starts.append(np.asarray([0.0, 0.0, middle]))
    if cfg.candidate_starts >= 3:
        starts.append(np.asarray([0.0, 0.0, cfg.fit_width_min_deg]))
    return starts[:max(1, cfg.candidate_starts)]


def _fit_candidate(feature: Mapping[str, object], residual: np.ndarray,
                   cfg: StudyConfig) -> Tuple[np.ndarray, np.ndarray, float, int, bool]:
    bounds = [(-cfg.fit_center_bound_deg, cfg.fit_center_bound_deg),
              (-cfg.fit_center_bound_deg, cfg.fit_center_bound_deg),
              (cfg.fit_width_min_deg, cfg.fit_width_max_deg)]
    best = None
    total_nfev = 0
    all_success = True
    scale = max(float(np.linalg.norm(residual)), 1e-300)
    normalized_residual = residual / scale
    for start in _candidate_starts(feature, residual, cfg):
        def objective(x: np.ndarray) -> float:
            atom = template(feature, x, cfg)
            norm = max(float(atom @ atom), 1e-300)
            amp = np.clip(float(atom @ normalized_residual) / norm, 0.0,
                          10.0 ** (cfg.amplitude_max_db / 10.0))
            error = normalized_residual - amp * atom
            return float(error @ error)

        result = minimize(
            objective, start, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": cfg.candidate_maxiter, "ftol": cfg.optimizer_ftol},
        )
        total_nfev += int(result.nfev)
        all_success &= bool(result.success)
        atom = template(feature, result.x, cfg)
        amp = np.clip(float(atom @ residual) / max(float(atom @ atom), 1e-300), 0.0,
                      10.0 ** (cfg.amplitude_max_db / 10.0))
        error = residual - amp * atom
        value = float(error @ error)
        if best is None or value < best[0]:
            best = (value, np.asarray(result.x, dtype=float), atom, amp)
    assert best is not None
    return best[1], best[2], float(best[3]), total_nfev, all_success


def _joint_refine(features: Sequence[Mapping[str, object]], y: np.ndarray,
                  selected: Sequence[int], params: Sequence[np.ndarray], cfg: StudyConfig
                  ) -> Tuple[List[np.ndarray], np.ndarray, float, np.ndarray, float, Dict[str, object]]:
    if not selected:
        amps, bg, pred, rss = _fit_positive([], y, cfg)
        return [], amps, bg, pred, rss, {"nfev": 0, "success": True}
    bounds = []
    for _ in selected:
        bounds.extend([(-cfg.fit_center_bound_deg, cfg.fit_center_bound_deg),
                       (-cfg.fit_center_bound_deg, cfg.fit_center_bound_deg),
                       (cfg.fit_width_min_deg, cfg.fit_width_max_deg)])
    scale = max(float(np.linalg.norm(y)), 1e-300)

    def evaluate(flat: np.ndarray):
        current = [np.asarray(flat[3 * i:3 * i + 3]) for i in range(len(selected))]
        atoms = [template(features[idx], par, cfg) / scale for idx, par in zip(selected, current)]
        amps, bg, pred, rss = _fit_positive(atoms, y / scale, cfg)
        return current, atoms, amps / scale, bg * scale, pred * scale, rss

    def objective(flat: np.ndarray) -> float:
        current = [np.asarray(flat[3 * i:3 * i + 3]) for i in range(len(selected))]
        atoms = [template(features[idx], par, cfg) for idx, par in zip(selected, current)]
        return _fit_positive(atoms, y, cfg)[3] / (scale * scale)

    x0 = np.concatenate(params)
    initial_atoms = [template(features[idx], par, cfg) for idx, par in zip(selected, params)]
    initial_amps, initial_bg, initial_pred, initial_rss = _fit_positive(initial_atoms, y, cfg)
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": cfg.joint_maxiter, "ftol": cfg.optimizer_ftol})
    refined = [np.asarray(result.x[3 * i:3 * i + 3]) for i in range(len(selected))]
    atoms = [template(features[idx], par, cfg) for idx, par in zip(selected, refined)]
    amps, bg, pred, rss = _fit_positive(atoms, y, cfg)
    if not np.isfinite(rss) or rss > initial_rss * (1.0 + 1e-10):
        return list(params), initial_amps, initial_bg, initial_pred, initial_rss, {
            "nfev": int(result.nfev), "success": False, "reverted": True,
        }
    return refined, amps, bg, pred, rss, {
        "nfev": int(result.nfev), "nit": int(result.nit),
        "success": bool(result.success), "message": str(result.message), "reverted": False,
    }


def _gamma_joint_refine(features: Sequence[Mapping[str, object]], observations: np.ndarray,
                        selected: Sequence[int], params: Sequence[np.ndarray], cfg: StudyConfig
                        ) -> Tuple[List[np.ndarray], np.ndarray, float, np.ndarray, float, Dict[str, object]]:
    """Joint maximum likelihood for averaged-energy (Gamma) observations."""
    if selected and cfg.use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                return _torch_gamma_joint_refine(features, observations, selected, params, cfg)
        except Exception:
            pass
    noise = float(cfg.noise_subband_mw)
    scale = max(float(np.mean(observations)), noise, 1e-300)
    y = np.asarray(observations, dtype=float) / scale
    noise_n = noise / scale
    amplitude_lo = 10.0 ** (cfg.amplitude_min_db / 10.0)
    amplitude_hi = 10.0 ** (cfg.amplitude_max_db / 10.0)
    bg_hi = cfg.background_max_ratio * noise / scale
    initial_atoms = [template(features[idx], par, cfg) for idx, par in zip(selected, params)]
    initial_amps, initial_bg, _, _ = _fit_positive(initial_atoms, observations - noise, cfg)
    x0: List[float] = []
    bounds: List[Tuple[float, float]] = []
    for par, amp in zip(params, initial_amps):
        x0.extend([float(par[0]), float(par[1]), float(par[2]), float(amp)])
        bounds.extend([(-cfg.fit_center_bound_deg, cfg.fit_center_bound_deg),
                       (-cfg.fit_center_bound_deg, cfg.fit_center_bound_deg),
                       (cfg.fit_width_min_deg, cfg.fit_width_max_deg),
                       (amplitude_lo, amplitude_hi)])
    x0.append(float(np.clip(initial_bg / scale, 0.0, bg_hi)))
    bounds.append((0.0, bg_hi))

    def mean_power(flat: np.ndarray) -> np.ndarray:
        mu = np.full_like(y, noise_n + flat[-1])
        for position, idx in enumerate(selected):
            par = flat[4 * position:4 * position + 3]
            amp = flat[4 * position + 3]
            mu += amp * template(features[idx], par, cfg) / scale
        return np.maximum(mu, 1e-12)

    def objective(flat: np.ndarray) -> float:
        mu = mean_power(flat)
        return float(cfg.n_looks * np.mean(np.log(mu) + y / mu))

    result = minimize(objective, np.asarray(x0), method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": max(cfg.joint_maxiter, 60),
                               "ftol": min(cfg.optimizer_ftol, 1e-10)})
    mu = mean_power(result.x) * scale
    # Twice the negative log likelihood, omitting terms independent of mu.
    deviance = float(2.0 * cfg.n_looks * np.sum(np.log(mu) + observations / mu))
    refined = []
    amps = []
    for position, _ in enumerate(selected):
        refined.append(np.asarray(result.x[4 * position:4 * position + 3]))
        amps.append(float(result.x[4 * position + 3]))
    background = float(result.x[-1] * scale)
    return refined, np.asarray(amps), background, mu, deviance, {
        "nfev": int(result.nfev), "nit": int(result.nit), "success": bool(result.success),
        "message": str(result.message), "objective": float(result.fun),
    }


def _torch_gamma_joint_refine(features: Sequence[Mapping[str, object]], observations: np.ndarray,
                              selected: Sequence[int], params: Sequence[np.ndarray], cfg: StudyConfig
                              ) -> Tuple[List[np.ndarray], np.ndarray, float, np.ndarray, float, Dict[str, object]]:
    """GPU auto-differentiated Gamma maximum likelihood for circular beams."""
    import torch

    device = torch.device("cuda")
    dtype = torch.float32
    observations = np.asarray(observations, dtype=float)
    noise = float(cfg.noise_subband_mw)
    scale = max(float(np.mean(observations)), noise, 1e-300)
    y = torch.as_tensor(observations / scale, dtype=dtype, device=device)
    noise_n = noise / scale
    feature_tensors = []
    for idx in selected:
        feature = features[idx]
        weights = spectral_bin_weights(feature["doppler_hz"], cfg)
        feature_tensors.append({
            "q": torch.as_tensor(feature["body_vectors"], dtype=dtype, device=device),
            "center_az": torch.as_tensor(np.deg2rad(feature["center_az_deg"]), dtype=dtype, device=device),
            "center_el": torch.as_tensor(np.deg2rad(feature["center_offnadir_deg"]), dtype=dtype, device=device),
            "base": torch.as_tensor(feature["base_mw"] / scale, dtype=dtype, device=device),
            "weights": torch.as_tensor(weights, dtype=dtype, device=device),
        })
    initial_atoms = [template(features[idx], par, cfg) for idx, par in zip(selected, params)]
    initial_amps, initial_bg, _, _ = _fit_positive(initial_atoms, observations - noise, cfg)
    geom = torch.nn.Parameter(torch.as_tensor(np.asarray(params), dtype=dtype, device=device))
    amps = torch.nn.Parameter(torch.as_tensor(initial_amps, dtype=dtype, device=device))
    background = torch.nn.Parameter(torch.as_tensor(initial_bg / scale, dtype=dtype, device=device))
    optimizer = torch.optim.Adam([geom, amps, background], lr=cfg.torch_joint_lr)
    amplitude_lo = 10.0 ** (cfg.amplitude_min_db / 10.0)
    amplitude_hi = 10.0 ** (cfg.amplitude_max_db / 10.0)
    background_hi = cfg.background_max_ratio * noise / scale

    def mean_power() -> torch.Tensor:
        mu = torch.ones_like(y) * (noise_n + background)
        for position, feature in enumerate(feature_tensors):
            az0 = feature["center_az"] + torch.deg2rad(geom[position, 0])
            el0 = feature["center_el"] + torch.deg2rad(geom[position, 1])
            c = torch.stack((torch.sin(el0) * torch.cos(az0),
                             torch.sin(el0) * torch.sin(az0), torch.cos(el0)), dim=1)
            dot = torch.sum(feature["q"] * c, dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(dot)
            half = torch.deg2rad(geom[position, 2] / 2.0).clamp_min(1e-6)
            ka = CIRCULAR_HALF_POWER_X / torch.sin(half).clamp_min(1e-6)
            z = ka * torch.sin(theta)
            ratio = torch.where(torch.abs(z) < 1e-4,
                                1.0 - z * z / 8.0,
                                2.0 * torch.special.bessel_j1(z) / z)
            gain = torch.clamp(ratio * ratio, min=1e-12)
            component = (feature["base"] * gain)[:, None] * feature["weights"]
            mu = mu + amps[position] * component.reshape(-1)
        return torch.clamp(mu, min=1e-12)

    best_loss = float("inf"); best_state = None; used = 0
    for iteration in range(max(int(cfg.torch_joint_iterations), 1)):
        optimizer.zero_grad(set_to_none=True)
        mu = mean_power()
        loss = cfg.n_looks * torch.mean(torch.log(mu) + y / mu)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([geom, amps, background], max_norm=100.0)
        optimizer.step()
        with torch.no_grad():
            geom[:, 0].clamp_(-cfg.fit_center_bound_deg, cfg.fit_center_bound_deg)
            geom[:, 1].clamp_(-cfg.fit_center_bound_deg, cfg.fit_center_bound_deg)
            geom[:, 2].clamp_(cfg.fit_width_min_deg, cfg.fit_width_max_deg)
            amps.clamp_(amplitude_lo, amplitude_hi)
            background.clamp_(0.0, background_hi)
        value = float(loss.detach().cpu()); used = iteration + 1
        if value < best_loss:
            best_loss = value
            best_state = (geom.detach().clone(), amps.detach().clone(), background.detach().clone())
    if best_state is None:
        raise RuntimeError("Torch Gamma optimizer produced no finite iterate")
    with torch.no_grad():
        geom.copy_(best_state[0]); amps.copy_(best_state[1]); background.copy_(best_state[2])
        mu = mean_power() * scale
    mu_np = mu.detach().cpu().numpy().astype(float)
    refined_array = geom.detach().cpu().numpy().astype(float)
    amp_array = amps.detach().cpu().numpy().astype(float)
    background_value = float(background.detach().cpu()) * scale
    deviance = float(2.0 * cfg.n_looks * np.sum(np.log(mu_np) + observations / mu_np))
    return [row.copy() for row in refined_array], amp_array, background_value, mu_np, deviance, {
        "nfev": used, "nit": used, "success": True, "message": "Adam CUDA",
        "objective": best_loss, "device": "cuda",
    }


def satebeam_fit(scene: Mapping[str, object], cfg: StudyConfig,
                 forced_support: Sequence[int] | None = None) -> Dict[str, object]:
    start_total = time.perf_counter()
    features = scene["features_train"]
    y = np.asarray(scene["observations_mw"], dtype=float) - float(scene["noise_mw"])
    selected: List[int] = []
    params: List[np.ndarray] = []
    atoms: List[np.ndarray] = []
    amps, background, prediction, rss = _fit_positive([], y, cfg)
    trace: List[Dict[str, object]] = []
    candidate_time = 0.0
    refine_time = 0.0
    candidate_nfev = 0
    solver_failures = 0

    if forced_support is not None:
        selected = [int(idx) for idx in forced_support]
        design, labels = _grid_atoms(scene, cfg)
        params, screen_used = _spectral_unmix_initialization(
            scene, design, labels, selected, cfg
        )
        candidate_nfev += screen_used
        tick = time.perf_counter()
        params, amps, background, prediction, rss, refine_info = _gamma_joint_refine(
            features, np.asarray(scene["observations_mw"], dtype=float),
            selected, params, cfg
        )
        refine_time += time.perf_counter() - tick
        return _source_result(scene, cfg, selected, params, amps, background, trace,
                              start_total, candidate_time, refine_time, candidate_nfev,
                              solver_failures, refine_info, method="Oracle-Support")

    refine_info: Dict[str, object] = {"success": True, "nfev": 0}
    for iteration in range(cfg.max_order):
        candidates = [idx for idx in range(cfg.n_candidates) if idx not in selected]
        if not candidates:
            break
        provisional = []
        tick = time.perf_counter()
        residual = y - prediction
        for idx in candidates:
            x, atom, _, nfev, success = _fit_candidate(features[idx], residual, cfg)
            candidate_nfev += nfev
            solver_failures += int(not success)
            test_amps, test_bg, test_pred, test_rss = _fit_positive(
                atoms + [atom], y, cfg, allow_zero_last=True
            )
            amplitude_min = 10.0 ** (cfg.amplitude_min_db / 10.0)
            candidate_score = (_score(rss, test_rss, len(y), cfg)
                               if test_amps[-1] >= amplitude_min else -np.inf)
            provisional.append((candidate_score, idx, x, atom, test_amps, test_bg,
                                test_pred, test_rss))
        candidate_time += time.perf_counter() - tick
        provisional.sort(key=lambda item: item[0], reverse=True)
        best = provisional[0]
        if cfg.joint_shortlist > 0 and np.isfinite(best[0]):
            lookahead = []
            tick = time.perf_counter()
            for item in provisional[:min(cfg.joint_shortlist, len(provisional))]:
                _, idx, x, atom, _, _, _, _ = item
                refined, r_amps, r_bg, r_pred, r_rss, r_info = _joint_refine(
                    features, y, selected + [int(idx)], params + [x], cfg
                )
                solver_failures += int(not r_info.get("success", True))
                candidate_score = (_score(rss, r_rss, len(y), cfg)
                                   if r_amps[-1] >= amplitude_min else -np.inf)
                lookahead.append((candidate_score, idx, refined[-1],
                                  template(features[int(idx)], refined[-1], cfg),
                                  r_amps, r_bg, r_pred, r_rss, refined, r_info))
            refine_time += time.perf_counter() - tick
            lookahead.sort(key=lambda item: item[0], reverse=True)
            best = lookahead[0]
        assert best is not None
        trace.append({"iteration": iteration, "candidate": int(best[1]),
                      "score": float(best[0]), "rss0": float(rss), "rss1": float(best[7])})
        if best[0] <= 0.0:
            break
        if len(best) == 10:
            _, idx, x, atom, amps, background, prediction, rss, refined_all, refine_info = best
            selected.append(int(idx)); params = list(refined_all)
            atoms = [template(features[j], par, cfg) for j, par in zip(selected, params)]
        else:
            _, idx, x, atom, amps, background, prediction, rss = best
            selected.append(int(idx)); params.append(x); atoms.append(atom)
        if cfg.refine_each_step:
            tick = time.perf_counter()
            params, amps, background, prediction, rss, refine_info = _joint_refine(
                features, y, selected, params, cfg
            )
            refine_time += time.perf_counter() - tick
            atoms = [template(features[j], par, cfg) for j, par in zip(selected, params)]

    if selected and not cfg.refine_each_step and cfg.joint_shortlist <= 0:
        tick = time.perf_counter()
        params, amps, background, prediction, rss, refine_info = _joint_refine(
            features, y, selected, params, cfg
        )
        refine_time += time.perf_counter() - tick

    return _source_result(scene, cfg, selected, params, amps, background, trace,
                          start_total, candidate_time, refine_time, candidate_nfev,
                          solver_failures, refine_info, method="SateBeam")


def _nonnegative_lasso_screen(design: np.ndarray, target: np.ndarray,
                              cfg: StudyConfig) -> Tuple[np.ndarray, str, int]:
    """GPU FISTA screening used to rank satellite groups jointly."""
    norms = np.maximum(np.linalg.norm(design, axis=0), 1e-300)
    normalized = design / norms
    target_scale = max(float(np.linalg.norm(target)), 1e-300)
    y = target / target_scale
    iterations = max(int(cfg.screening_iterations), 1)
    if cfg.use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                A = torch.as_tensor(normalized, dtype=torch.float32, device=device)
                b = torch.as_tensor(y, dtype=torch.float32, device=device)
                vector = torch.ones(A.shape[1], dtype=torch.float32, device=device)
                vector /= torch.linalg.norm(vector)
                for _ in range(20):
                    vector = A.T @ (A @ vector)
                    vector /= torch.linalg.norm(vector).clamp_min(1e-12)
                lipschitz = float(torch.dot(vector, A.T @ (A @ vector)).item()) / len(y)
                step = 1.0 / max(lipschitz, 1e-8)
                alpha_max = float(torch.max(A.T @ b).item()) / len(y)
                threshold = step * alpha_max * cfg.lasso_alpha_relative
                x = torch.zeros(A.shape[1], dtype=torch.float32, device=device)
                z = x.clone(); momentum = 1.0
                used = 0
                for iteration in range(iterations):
                    gradient = A.T @ (A @ z - b) / len(y)
                    new_x = torch.relu(z - step * gradient - threshold)
                    new_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
                    z = new_x + ((momentum - 1.0) / new_momentum) * (new_x - x)
                    relative = float(torch.linalg.norm(new_x - x).item()) / max(
                        float(torch.linalg.norm(x).item()), 1e-8
                    )
                    x = new_x; momentum = new_momentum; used = iteration + 1
                    if iteration > 30 and relative < 1e-5:
                        break
                coefficient = x.detach().cpu().numpy() * target_scale / norms
                return coefficient, "cuda", used
        except Exception:
            pass
    model = Lasso(alpha=max(float(np.max(normalized.T @ y)) / len(y), 1e-12)
                  * cfg.lasso_alpha_relative,
                  positive=True, fit_intercept=False, max_iter=cfg.lasso_maxiter,
                  tol=1e-5, selection="cyclic")
    model.fit(normalized, y)
    return np.asarray(model.coef_) * target_scale / norms, "cpu", int(model.n_iter_)


def _doppler_group_screen(scene: Mapping[str, object], cfg: StudyConfig
                          ) -> Tuple[List[int], np.ndarray, int]:
    """Activity screening from ephemeris-derived Doppler trajectories only."""
    snapshots = cfg.n_snapshots
    subbands = max(int(cfg.n_subbands), 1)
    measurements = cfg.n_measurements
    observed = np.asarray(scene["observations_mw"]).reshape(snapshots, measurements, subbands)
    spectrum = observed.mean(axis=1)
    spectrum = spectrum - np.min(spectrum, axis=1, keepdims=True)
    y = spectrum.reshape(-1)
    columns = []
    for satellite in range(cfg.n_candidates):
        doppler = np.asarray(scene["features_train"][satellite]["doppler_hz"]).reshape(
            snapshots, measurements
        )
        weights = spectral_bin_weights(doppler.reshape(-1), cfg).reshape(
            snapshots, measurements, subbands
        ).mean(axis=1)
        for snapshot in range(snapshots):
            column = np.zeros((snapshots, subbands))
            column[snapshot] = weights[snapshot]
            columns.append(column.reshape(-1))
    design = np.column_stack(columns)
    norms = np.maximum(np.linalg.norm(design, axis=0), 1e-12)
    A = design / norms
    scale = max(float(np.linalg.norm(y)), 1e-12)
    b = y / scale
    lipschitz = float(np.linalg.norm(A, 2) ** 2 / max(len(b), 1))
    step = 1.0 / max(lipschitz, 1e-8)
    gradients = (A.T @ b).reshape(cfg.n_candidates, snapshots)
    lambda_max = float(np.max(np.linalg.norm(gradients, axis=1))) / max(len(b), 1)
    regularization = cfg.screening_lambda_relative * lambda_max
    x = np.zeros(A.shape[1]); z = x.copy(); momentum = 1.0
    used = 0
    for iteration in range(max(int(cfg.screening_iterations), 1)):
        gradient = A.T @ (A @ z - b) / max(len(b), 1)
        value = np.maximum(z - step * gradient, 0.0).reshape(cfg.n_candidates, snapshots)
        group_norm = np.linalg.norm(value, axis=1, keepdims=True)
        shrink = np.maximum(0.0, 1.0 - step * regularization / np.maximum(group_norm, 1e-12))
        new_x = (value * shrink).reshape(-1)
        new_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        new_z = new_x + ((momentum - 1.0) / new_momentum) * (new_x - x)
        relative = np.linalg.norm(new_x - x) / max(np.linalg.norm(x), 1e-8)
        x, z, momentum = new_x, new_z, new_momentum
        used = iteration + 1
        if iteration > 30 and relative < 1e-6:
            break
    coefficient = x.reshape(cfg.n_candidates, snapshots) * scale / norms.reshape(
        cfg.n_candidates, snapshots
    )
    scores = np.linalg.norm(coefficient, axis=1)
    ranking = np.argsort(scores)[::-1].tolist()
    return ranking, scores, used


def _full_group_screen(design: np.ndarray, target: np.ndarray, cfg: StudyConfig
                       ) -> Tuple[List[int], np.ndarray, np.ndarray, str, int]:
    """Group-Lasso screening over all beam atoms of every satellite."""
    atoms_per_group = design.shape[1] // cfg.n_candidates
    if atoms_per_group * cfg.n_candidates != design.shape[1]:
        raise ValueError("Dictionary columns do not form equal satellite groups")
    norms = np.maximum(np.linalg.norm(design, axis=0), 1e-300)
    normalized = design / norms
    scale = max(float(np.linalg.norm(target)), 1e-300)
    y_np = target / scale
    iterations = max(int(cfg.screening_iterations), 1)
    try:
        import torch
        if cfg.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            A = torch.as_tensor(normalized, dtype=torch.float32, device=device)
            y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
            vector = torch.ones(A.shape[1], device=device); vector /= torch.linalg.norm(vector)
            for _ in range(15):
                vector = A.T @ (A @ vector)
                vector /= torch.linalg.norm(vector).clamp_min(1e-12)
            lipschitz = float(torch.dot(vector, A.T @ (A @ vector)).item()) / len(y_np)
            step = 1.0 / max(lipschitz, 1e-8)
            gradient0 = (A.T @ y).reshape(cfg.n_candidates, atoms_per_group)
            lambda_max = float(torch.max(torch.linalg.norm(gradient0, dim=1)).item()) / len(y_np)
            regularization = cfg.screening_lambda_relative * lambda_max
            x = torch.zeros(A.shape[1], device=device); z = x.clone(); momentum = 1.0
            used = 0
            for iteration in range(iterations):
                gradient = A.T @ (A @ z - y) / len(y_np)
                value = torch.relu(z - step * gradient).reshape(cfg.n_candidates, atoms_per_group)
                group_norm = torch.linalg.norm(value, dim=1, keepdim=True).clamp_min(1e-12)
                shrink = torch.relu(1.0 - step * regularization / group_norm)
                new_x = (value * shrink).reshape(-1)
                new_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
                new_z = new_x + ((momentum - 1.0) / new_momentum) * (new_x - x)
                relative = float(torch.linalg.norm(new_x - x).item()) / max(
                    float(torch.linalg.norm(x).item()), 1e-8
                )
                x, z, momentum = new_x, new_z, new_momentum; used = iteration + 1
                if iteration > 30 and relative < 1e-5:
                    break
            coefficient = x.detach().cpu().numpy()
            scores = np.linalg.norm(coefficient.reshape(cfg.n_candidates, atoms_per_group), axis=1)
            return np.argsort(scores)[::-1].tolist(), scores, coefficient, "cuda", used
    except Exception:
        pass
    # CPU fallback uses the same proximal update.
    lipschitz = float(np.linalg.norm(normalized, 2) ** 2 / len(y_np))
    step = 1.0 / max(lipschitz, 1e-8)
    gradient0 = (normalized.T @ y_np).reshape(cfg.n_candidates, atoms_per_group)
    lambda_max = float(np.max(np.linalg.norm(gradient0, axis=1))) / len(y_np)
    regularization = cfg.screening_lambda_relative * lambda_max
    x = np.zeros(normalized.shape[1]); z = x.copy(); momentum = 1.0; used = 0
    for iteration in range(iterations):
        gradient = normalized.T @ (normalized @ z - y_np) / len(y_np)
        value = np.maximum(z - step * gradient, 0.0).reshape(cfg.n_candidates, atoms_per_group)
        group_norm = np.linalg.norm(value, axis=1, keepdims=True)
        shrink = np.maximum(0.0, 1.0 - step * regularization / np.maximum(group_norm, 1e-12))
        new_x = (value * shrink).reshape(-1)
        new_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        z = new_x + ((momentum - 1.0) / new_momentum) * (new_x - x)
        relative = np.linalg.norm(new_x - x) / max(np.linalg.norm(x), 1e-8)
        x, momentum = new_x, new_momentum; used = iteration + 1
        if iteration > 30 and relative < 1e-5:
            break
    scores = np.linalg.norm(x.reshape(cfg.n_candidates, atoms_per_group), axis=1)
    return np.argsort(scores)[::-1].tolist(), scores, x, "cpu", used


def satebeam_global_fit(scene: Mapping[str, object], cfg: StudyConfig) -> Dict[str, object]:
    """GPU overcomplete screening followed by backward Gamma-BIC pruning."""
    started = time.perf_counter()
    observations = np.asarray(scene["observations_mw"], dtype=float)
    y = observations - float(scene["noise_mw"])
    _, _, initial_bg, initial_mean, initial_deviance, initial_info = _gamma_joint_refine(
        scene["features_train"], observations, [], [], cfg
    )
    nnls_ranking, nnls_scores, nnls_coefficients = spectral_nnls_screen(scene, cfg)
    pool_support = nnls_ranking[:cfg.max_order]
    screen_iterations = cfg.n_snapshots * cfg.n_measurements
    device = "cuda" if cfg.use_gpu else "cpu"
    target = observations - float(scene["noise_mw"])
    build_started = time.perf_counter()
    design, labels = _grid_atoms(scene, cfg, support=pool_support)
    build_time = time.perf_counter() - build_started

    n = len(y)
    candidates = [{"k": 0, "ic": initial_deviance + np.log(n),
                   "support": [], "params": [], "amps": np.asarray([]),
                   "background": initial_bg, "deviance": initial_deviance,
                   "info": initial_info}]
    refine_time = 0.0
    solver_failures = 0
    support = list(pool_support)
    start_params, coordinate_evaluations = _spectral_unmix_initialization(
        scene, design, labels, support, cfg
    )
    while support:
        k = len(support)
        tick = time.perf_counter()
        refined, amps, background, mean_power, deviance, info = _gamma_joint_refine(
            scene["features_train"], observations, support, start_params, cfg
        )
        refine_time += time.perf_counter() - tick
        solver_failures += int(not info.get("success", True))
        ic = deviance + (4.0 * k + 1.0) * np.log(n)
        candidates.append({"k": k, "ic": float(ic), "support": list(support),
                           "params": [value.copy() for value in refined],
                           "amps": amps.copy(), "background": background,
                           "deviance": deviance, "info": info})
        # Delete the source whose removal produces the smallest likelihood loss
        # under the current joint fit, then reoptimize the reduced model.
        deletion_deviance = []
        for position, idx in enumerate(support):
            contribution = amps[position] * template(
                scene["features_train"][idx], refined[position], cfg
            )
            reduced_mean = np.maximum(mean_power - contribution, 1e-300)
            value = float(2.0 * cfg.n_looks * np.sum(
                np.log(reduced_mean) + observations / reduced_mean
            ))
            deletion_deviance.append(value)
        remove = int(np.argmin(deletion_deviance))
        support.pop(remove); refined.pop(remove)
        start_params = refined
    best = min(candidates, key=lambda item: item["ic"])
    trace = [{"k": item["k"], "ic": item["ic"], "deviance": item["deviance"],
              "support": item["support"]} for item in candidates]
    result = _source_result(
        scene, cfg, best["support"], best["params"], best["amps"], best["background"],
        trace, started, 0.0, refine_time, screen_iterations, solver_failures,
        best["info"], method="SateBeam",
    )
    result.update({"screening_device": device, "compute_device": device,
                   "screening_iterations": screen_iterations,
                   "nnls_screen_scores": nnls_scores,
                   "nnls_screen_coefficients": nnls_coefficients,
                   "coordinate_initialization_evaluations": coordinate_evaluations,
                   "dictionary_atoms": int(design.shape[1]),
                   "dictionary_bytes": int(design.nbytes),
                   "dictionary_time_s": float(build_time),
                   "screened_pool": pool_support})
    return result


def _source_result(scene: Mapping[str, object], cfg: StudyConfig, selected: Sequence[int],
                   params: Sequence[np.ndarray], amps: Sequence[float], background: float,
                   trace: Sequence[Mapping[str, object]], start_total: float,
                   candidate_time: float, refine_time: float, nfev: int,
                   failures: int, refine_info: Mapping[str, object], method: str) -> Dict[str, object]:
    all_features = scene["features_all"]
    signal = np.zeros(len(scene["mean_total_mw"]), dtype=float)
    estimates: Dict[int, Dict[str, float]] = {}
    for idx, par, amp in zip(selected, params, amps):
        signal += float(amp) * template(all_features[int(idx)], np.asarray(par), cfg)
        estimates[int(idx)] = {
            "offset_az_deg": float(par[0]), "offset_el_deg": float(par[1]),
            "width_deg": float(par[2]), "amplitude": float(amp),
        }
    total = float(scene["noise_mw"]) + float(background) + signal
    n_subbands = max(int(cfg.n_subbands), 1)
    signal_aggregate = signal.reshape(-1, n_subbands).sum(axis=1)
    total_aggregate = total.reshape(-1, n_subbands).sum(axis=1)
    return {
        "method": method,
        "map_only": False,
        "selected_ids": np.asarray(selected, dtype=int),
        "estimates": estimates,
        "background_mw": float(background),
        "pred_signal_mw": signal,
        "pred_total_mw": np.maximum(total, 1e-300),
        "pred_signal_aggregate_mw": signal_aggregate,
        "pred_total_aggregate_mw": np.maximum(total_aggregate, 1e-300),
        "score_trace": list(trace),
        "runtime_s": float(time.perf_counter() - start_total),
        "candidate_time_s": float(candidate_time),
        "refine_time_s": float(refine_time),
        "optimizer_nfev": int(nfev + int(refine_info.get("nfev", 0))),
        "solver_failures": int(failures + int(not refine_info.get("success", True))),
        "refine_info": dict(refine_info),
    }


def _grid_atoms(scene: Mapping[str, object], cfg: StudyConfig,
                nominal_only: bool = False,
                support: Sequence[int] | None = None) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
    features = scene["features_train"]
    if nominal_only:
        offsets = np.asarray([0.0])
        widths = np.asarray([0.5 * (cfg.truth_width_min_deg + cfg.truth_width_max_deg)])
    else:
        offsets = np.arange(-cfg.grid_span_deg, cfg.grid_span_deg + 1e-9, cfg.grid_step_deg)
        widths = np.linspace(cfg.fit_width_min_deg, cfg.fit_width_max_deg, cfg.grid_width_count)
    atoms = []
    labels: List[Tuple[int, np.ndarray]] = []
    candidate_ids = list(range(len(features))) if support is None else [int(value) for value in support]
    for idx in candidate_ids:
        feature = features[idx]
        for oa in offsets:
            for oe in offsets:
                for width in widths:
                    par = np.asarray([oa, oe, width], dtype=float)
                    atoms.append(template(feature, par, cfg))
                    labels.append((idx, par))
    return np.column_stack(atoms), labels


def _coordinate_grid_initialization(design: np.ndarray,
                                    labels: Sequence[Tuple[int, np.ndarray]],
                                    support: Sequence[int], target: np.ndarray,
                                    cfg: StudyConfig, passes: int = 3
                                    ) -> Tuple[List[np.ndarray], int]:
    """Alternating grid initialization conditioned on a candidate support."""
    if not support:
        return [], 0
    group_indices = {
        satellite: np.asarray([index for index, (sat, _) in enumerate(labels)
                               if sat == satellite], dtype=int)
        for satellite in support
    }
    norms = np.maximum(np.linalg.norm(design, axis=0), 1e-300)
    chosen = []
    for satellite in support:
        indices = group_indices[satellite]
        correlation = (design[:, indices] / norms[indices]).T @ target
        chosen.append(int(indices[int(np.argmax(correlation))]))
    evaluations = len(support)
    for _ in range(max(int(passes), 1)):
        atoms = [design[:, index] for index in chosen]
        amps, _, prediction, _ = _fit_positive(atoms, target, cfg)
        for position, satellite in enumerate(support):
            residual = target - prediction + amps[position] * atoms[position]
            indices = group_indices[satellite]
            correlation = (design[:, indices] / norms[indices]).T @ residual
            chosen[position] = int(indices[int(np.argmax(correlation))])
            atoms[position] = design[:, chosen[position]]
            amps, _, prediction, _ = _fit_positive(atoms, target, cfg)
            evaluations += 1
    return [np.asarray(labels[index][1], dtype=float) for index in chosen], evaluations


def _spectral_unmix_initialization(scene: Mapping[str, object], design: np.ndarray,
                                   labels: Sequence[Tuple[int, np.ndarray]],
                                   support: Sequence[int], cfg: StudyConfig
                                   ) -> Tuple[List[np.ndarray], int]:
    """Doppler-domain NNLS unmixing followed by per-source spatial grid matching."""
    if not support:
        return [], 0
    subbands = max(int(cfg.n_subbands), 1)
    n_spatial = cfg.n_snapshots * cfg.n_measurements
    observed = np.asarray(scene["observations_mw"]).reshape(n_spatial, subbands)
    observed = np.maximum(observed - cfg.noise_subband_mw, 0.0)
    weights = []
    for satellite in support:
        feature = scene["features_train"][satellite]
        weights.append(spectral_bin_weights(feature["doppler_hz"], cfg))
    source_power = np.zeros((len(support), n_spatial))
    for point in range(n_spatial):
        matrix = np.column_stack([item[point] for item in weights]
                                 + [np.ones(subbands)])
        coefficient, _ = nnls(matrix, observed[point], maxiter=1000)
        source_power[:, point] = coefficient[:-1]
    params = []
    evaluations = n_spatial
    for position, satellite in enumerate(support):
        indices = np.asarray([index for index, (sat, _) in enumerate(labels)
                              if sat == satellite], dtype=int)
        captured = np.maximum(weights[position].sum(axis=1), 1e-12)
        spatial_atoms = design[:, indices].reshape(n_spatial, subbands, len(indices)).sum(axis=1)
        spatial_atoms = spatial_atoms / captured[:, None]
        norms = np.maximum(np.linalg.norm(spatial_atoms, axis=0), 1e-300)
        correlation = (spatial_atoms / norms).T @ source_power[position]
        chosen = int(indices[int(np.argmax(correlation))])
        params.append(np.asarray(labels[chosen][1], dtype=float))
        evaluations += len(indices)
    return params, evaluations


def spectral_nnls_screen(scene: Mapping[str, object], cfg: StudyConfig
                         ) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """All-candidate Doppler NNLS at every space-time measurement."""
    subbands = max(int(cfg.n_subbands), 1)
    n_spatial = cfg.n_snapshots * cfg.n_measurements
    observed = np.asarray(scene["observations_mw"]).reshape(n_spatial, subbands)
    observed = np.maximum(observed - cfg.noise_subband_mw, 0.0)
    weights = [spectral_bin_weights(feature["doppler_hz"], cfg)
               for feature in scene["features_train"]]
    coefficients = np.zeros((cfg.n_candidates, n_spatial))
    for point in range(n_spatial):
        matrix = np.column_stack([item[point] for item in weights]
                                 + [np.ones(subbands)])
        value, _ = nnls(matrix, observed[point], maxiter=2000)
        coefficients[:, point] = value[:-1]
    scores = np.sqrt(np.mean(coefficients * coefficients, axis=1))
    ranking = np.argsort(scores)[::-1].tolist()
    return ranking, scores, coefficients


def _grid_greedy(scene: Mapping[str, object], cfg: StudyConfig, method: str,
                 nominal_only: bool = False,
                 gamma_selection: bool = False,
                 candidate_pool: Sequence[int] | None = None) -> Dict[str, object]:
    started = time.perf_counter()
    build_started = time.perf_counter()
    design, labels = _grid_atoms(scene, cfg, nominal_only, support=candidate_pool)
    build_time = time.perf_counter() - build_started
    norms = np.maximum(np.linalg.norm(design, axis=0), 1e-300)
    normalized = None
    use_cuda = False
    torch_design = None
    if cfg.use_gpu and design.size >= cfg.gpu_min_dictionary_elements:
        try:
            import torch
            if torch.cuda.is_available():
                torch_design = torch.as_tensor(design, dtype=torch.float32, device="cuda")
                torch_design /= torch.as_tensor(norms, dtype=torch.float32, device="cuda")
                use_cuda = True
        except Exception:
            torch_design = None
    if not use_cuda:
        normalized = design / norms
    y = np.asarray(scene["observations_mw"], dtype=float) - float(scene["noise_mw"])
    selected_atoms: List[int] = []
    selected_sats: List[int] = []
    continuous_params: List[np.ndarray] = []
    atoms: List[np.ndarray] = []
    amps, background, prediction, rss = _fit_positive([], y, cfg)
    def gamma_deviance(prediction_without_noise: np.ndarray) -> float:
        mean = np.maximum(float(scene["noise_mw"]) + prediction_without_noise, 1e-300)
        observations = np.asarray(scene["observations_mw"], dtype=float)
        return float(2.0 * cfg.n_looks * np.sum(np.log(mean) + observations / mean))
    deviance = gamma_deviance(prediction)
    trace = []
    for iteration in range(cfg.max_order):
        residual = y - prediction
        if use_cuda:
            import torch
            residual_gpu = torch.as_tensor(residual, dtype=torch.float32, device="cuda")
            correlations = (torch_design.T @ residual_gpu).cpu().numpy()
        else:
            assert normalized is not None
            correlations = normalized.T @ residual
        if selected_sats:
            used = set(selected_sats)
            correlations[[j for j, (sat, _) in enumerate(labels) if sat in used]] = -np.inf
        j = int(np.argmax(correlations))
        if not np.isfinite(correlations[j]) or correlations[j] <= 0:
            break
        atom = design[:, j]
        if method == "MP-Grid":
            amp = np.clip(float(atom @ residual) / max(float(atom @ atom), 1e-300), 0.0,
                          10.0 ** (cfg.amplitude_max_db / 10.0))
            test_pred = prediction + amp * atom
            test_rss = float(np.sum((y - test_pred) ** 2))
            test_amps = np.append(amps, amp); test_bg = background
        else:
            test_amps, test_bg, test_pred, test_rss = _fit_positive(
                atoms + [atom], y, cfg, allow_zero_last=True
            )
        amplitude_min = 10.0 ** (cfg.amplitude_min_db / 10.0)
        candidate_refined = None
        if gamma_selection:
            # The grid atom only proposes a satellite and an initial beam. Judge
            # model order after the continuous beam parameters have been jointly
            # optimized; scoring the coarse atom systematically rejects weak real
            # sources whose centers fall between grid points.
            candidate_refined, test_amps, test_bg, test_pred, test_rss, _ = _joint_refine(
                scene["features_train"], y,
                selected_sats + [int(labels[j][0])],
                continuous_params + [np.asarray(labels[j][1], dtype=float)], cfg,
            )
        if gamma_selection:
            rule = cfg.selection_rule.lower()
            if rule == "aic":
                penalty = 8.0
            elif rule == "margin_bic":
                penalty = 4.0 * np.log(len(y)) + cfg.selection_margin
            elif rule == "calibrated":
                penalty = cfg.calibrated_threshold
            else:
                penalty = 4.0 * np.log(len(y))
        if test_amps[-1] < amplitude_min:
            score = -np.inf
        elif gamma_selection:
            test_deviance = gamma_deviance(test_pred)
            score = deviance - test_deviance - penalty
        else:
            score = _score(rss, test_rss, len(y), cfg)
        trace.append({"iteration": iteration, "candidate": int(labels[j][0]),
                      "atom": j, "score": score, "rss0": rss, "rss1": test_rss})
        if score <= 0:
            break
        selected_atoms.append(j); selected_sats.append(int(labels[j][0])); atoms.append(atom)
        amps, background, prediction, rss = test_amps, test_bg, test_pred, test_rss
        if gamma_selection:
            assert candidate_refined is not None
            continuous_params = candidate_refined
            atoms = [template(scene["features_train"][satellite], value, cfg)
                     for satellite, value in zip(selected_sats, continuous_params)]
            deviance = gamma_deviance(prediction)

    if gamma_selection and selected_sats:
        # A source selected early can become redundant after later sources enter.
        # Backward Gamma-BIC pruning uses the final joint model and therefore
        # removes that path dependence without another experimental parameter.
        n = len(y)
        while len(selected_sats) > 1:
            current_ic = deviance + (4.0 * len(selected_sats) + 1.0) * np.log(n)
            deletion_candidates = []
            for position in range(len(selected_sats)):
                reduced_support = selected_sats[:position] + selected_sats[position + 1:]
                reduced_params = continuous_params[:position] + continuous_params[position + 1:]
                reduced_atoms = [
                    template(scene["features_train"][satellite], value, cfg)
                    for satellite, value in zip(reduced_support, reduced_params)
                ]
                reduced_amps, reduced_bg, reduced_pred, reduced_rss = _fit_positive(
                    reduced_atoms, y, cfg
                )
                reduced_deviance = gamma_deviance(reduced_pred)
                reduced_ic = reduced_deviance + (4.0 * len(reduced_support) + 1.0) * np.log(n)
                deletion_candidates.append((
                    reduced_ic, position, reduced_support, reduced_params,
                    reduced_amps, reduced_bg, reduced_pred, reduced_rss,
                ))
            best = min(deletion_candidates, key=lambda item: item[0])
            _, position, reduced_support, reduced_params, _, _, _, _ = best
            reduced_params, reduced_amps, reduced_bg, reduced_pred, reduced_rss, _ = _joint_refine(
                scene["features_train"], y, reduced_support, reduced_params, cfg
            )
            reduced_deviance = gamma_deviance(reduced_pred)
            reduced_ic = reduced_deviance + (4.0 * len(reduced_support) + 1.0) * np.log(n)
            if reduced_ic >= current_ic:
                break
            selected_sats, continuous_params = reduced_support, reduced_params
            amps, background = reduced_amps, reduced_bg
            prediction, rss, deviance = reduced_pred, reduced_rss, reduced_deviance
            atoms = [template(scene["features_train"][satellite], value, cfg)
                     for satellite, value in zip(selected_sats, continuous_params)]
            trace.append({"phase": "backward", "removed_position": int(position),
                          "score": float(current_ic - reduced_ic),
                          "deviance": float(deviance)})

        # Repair an early greedy mistake by temporarily removing each selected
        # source, proposing the strongest unselected residual atom, and comparing
        # the continuously refined replacements with the same Gamma-BIC.
        for _ in range(cfg.max_order):
            current_ic = deviance + (4.0 * len(selected_sats) + 1.0) * np.log(n)
            replacements = []
            selected_set = set(selected_sats)
            for position in range(len(selected_sats)):
                reduced_support = selected_sats[:position] + selected_sats[position + 1:]
                reduced_params = continuous_params[:position] + continuous_params[position + 1:]
                reduced_atoms = [
                    template(scene["features_train"][satellite], value, cfg)
                    for satellite, value in zip(reduced_support, reduced_params)
                ]
                _, _, reduced_pred, _ = _fit_positive(reduced_atoms, y, cfg)
                residual = y - reduced_pred
                if use_cuda:
                    import torch
                    residual_gpu = torch.as_tensor(residual, dtype=torch.float32, device="cuda")
                    correlations = (torch_design.T @ residual_gpu).cpu().numpy()
                else:
                    assert normalized is not None
                    correlations = normalized.T @ residual
                correlations[[j for j, (satellite, _) in enumerate(labels)
                              if satellite in selected_set]] = -np.inf
                atom_index = int(np.argmax(correlations))
                if (not np.isfinite(correlations[atom_index])
                        or correlations[atom_index] <= 0.0):
                    continue
                replacement = int(labels[atom_index][0])
                trial_support = reduced_support + [replacement]
                trial_params = reduced_params + [np.asarray(labels[atom_index][1], dtype=float)]
                trial_atoms = reduced_atoms + [design[:, atom_index]]
                trial_amps, trial_bg, trial_pred, trial_rss = _fit_positive(
                    trial_atoms, y, cfg, allow_zero_last=True
                )
                if trial_amps[-1] < amplitude_min:
                    continue
                trial_deviance = gamma_deviance(trial_pred)
                trial_ic = trial_deviance + (4.0 * len(trial_support) + 1.0) * np.log(n)
                replacements.append((
                    trial_ic, position, selected_sats[position], replacement,
                    trial_support, trial_params,
                ))
            if not replacements:
                break
            best = min(replacements, key=lambda item: item[0])
            screened_ic, position, removed, replacement, trial_support, trial_params = best
            if screened_ic >= current_ic:
                break
            trial_params, trial_amps, trial_bg, trial_pred, trial_rss, _ = _joint_refine(
                scene["features_train"], y, trial_support, trial_params, cfg
            )
            trial_deviance = gamma_deviance(trial_pred)
            best_ic = trial_deviance + (4.0 * len(trial_support) + 1.0) * np.log(n)
            if best_ic >= current_ic:
                break
            selected_sats, continuous_params = trial_support, trial_params
            amps, background = trial_amps, trial_bg
            prediction, rss, deviance = trial_pred, trial_rss, trial_deviance
            atoms = [template(scene["features_train"][satellite], value, cfg)
                     for satellite, value in zip(selected_sats, continuous_params)]
            trace.append({"phase": "swap", "removed_candidate": int(removed),
                          "candidate": int(replacement), "removed_position": int(position),
                          "score": float(current_ic - best_ic),
                          "deviance": float(deviance)})
    params = (continuous_params if gamma_selection else
              [labels[j][1] for j in selected_atoms])
    result = _source_result(scene, cfg, selected_sats, params, amps, background, trace,
                            started, 0.0, 0.0, 0, 0, {"success": True}, method)
    result["dictionary_atoms"] = int(design.shape[1])
    result["dictionary_bytes"] = int(design.nbytes)
    result["dictionary_time_s"] = float(build_time)
    result["compute_device"] = "cuda" if use_cuda else "cpu"
    return result


def omp_refine(scene: Mapping[str, object], cfg: StudyConfig) -> Dict[str, object]:
    base = _grid_greedy(scene, cfg, "OMP-Grid")
    started = time.perf_counter()
    support = [int(v) for v in base["selected_ids"]]
    params = [np.asarray([base["estimates"][idx]["offset_az_deg"],
                         base["estimates"][idx]["offset_el_deg"],
                         base["estimates"][idx]["width_deg"]]) for idx in support]
    y = np.asarray(scene["observations_mw"], dtype=float) - float(scene["noise_mw"])
    refined, amps, bg, _, _, info = _joint_refine(scene["features_train"], y, support, params, cfg)
    result = _source_result(scene, cfg, support, refined, amps, bg, base["score_trace"],
                            started, 0.0, 0.0, int(info.get("nfev", 0)),
                            int(not info.get("success", True)), info, "OMP-Refine")
    result["runtime_s"] += float(base["runtime_s"])
    result["dictionary_atoms"] = base["dictionary_atoms"]
    result["dictionary_bytes"] = base["dictionary_bytes"]
    result["dictionary_time_s"] = base["dictionary_time_s"]
    return result


def satebeam_final(scene: Mapping[str, object], cfg: StudyConfig) -> Dict[str, object]:
    """Doppler-aware grid OMP, BIC order selection, and continuous off-grid refinement."""
    screening_started = time.perf_counter()
    ranking, screening_scores, _ = spectral_nnls_screen(scene, cfg)
    # Retain a small scale-aware guard beyond the largest admissible model. The
    # logarithmic growth keeps the spatial dictionary bounded while allowing for
    # more near-ties in larger Doppler candidate sets (10 atoms groups at N=16,
    # 11 at N=64 for the default order cap).
    ratio = max(float(cfg.n_candidates) / max(float(cfg.order_cap), 1.0), 1.0)
    screen_guard = max(2, int(np.ceil(np.log2(ratio))))
    pool = ranking[:min(cfg.order_cap + screen_guard, cfg.n_candidates)]
    screening_time = time.perf_counter() - screening_started
    base = _grid_greedy(scene, cfg, "OMP-Grid", gamma_selection=True,
                        candidate_pool=pool)
    started = time.perf_counter()
    support = [int(value) for value in base["selected_ids"]]
    params = [np.asarray([base["estimates"][idx]["offset_az_deg"],
                         base["estimates"][idx]["offset_el_deg"],
                         base["estimates"][idx]["width_deg"]]) for idx in support]
    y = np.asarray(scene["observations_mw"], dtype=float) - float(scene["noise_mw"])
    refined, amps, background, _, _, info = _joint_refine(
        scene["features_train"], y, support, params, cfg
    )
    result = _source_result(
        scene, cfg, support, refined, amps, background, base["score_trace"],
        started, 0.0, 0.0, int(info.get("nfev", 0)),
        0, info, "SateBeam",
    )
    result["runtime_s"] += float(base["runtime_s"])
    result["dictionary_atoms"] = base["dictionary_atoms"]
    result["dictionary_bytes"] = base["dictionary_bytes"]
    result["dictionary_time_s"] = base["dictionary_time_s"]
    result["compute_device"] = base.get("compute_device", "cpu")
    result["selection_rule"] = cfg.selection_rule
    result["screened_pool"] = pool
    result["screening_scores"] = screening_scores
    result["screening_time_s"] = screening_time
    return result


def oracle_support_fit(scene: Mapping[str, object], cfg: StudyConfig) -> Dict[str, object]:
    """Information reference: true support, unknown continuous beam parameters."""
    base = _grid_greedy(scene, cfg, "OMP-Grid")
    started = time.perf_counter()
    support = [int(value) for value in scene["active_ids"]]
    y = np.asarray(scene["observations_mw"], dtype=float) - float(scene["noise_mw"])
    params = []
    for idx in support:
        if idx in base["estimates"]:
            item = base["estimates"][idx]
            params.append(np.asarray([item["offset_az_deg"], item["offset_el_deg"], item["width_deg"]]))
        else:
            value, _, _, _, _ = _fit_candidate(scene["features_train"][idx], y, cfg)
            params.append(value)
    refined, amps, background, _, _, info = _joint_refine(
        scene["features_train"], y, support, params, cfg
    )
    result = _source_result(scene, cfg, support, refined, amps, background, [], started,
                            0.0, 0.0, int(info.get("nfev", 0)),
                            0, info, "Oracle-Support")
    result["runtime_s"] += float(base["runtime_s"])
    result["compute_device"] = base.get("compute_device", "cpu")
    return result


def lasso_grid(scene: Mapping[str, object], cfg: StudyConfig) -> Dict[str, object]:
    started = time.perf_counter()
    design, labels = _grid_atoms(scene, cfg)
    norms = np.maximum(np.linalg.norm(design, axis=0), 1e-300)
    normalized = design / norms
    y = np.asarray(scene["observations_mw"], dtype=float) - float(scene["noise_mw"])
    background = max(0.0, float(np.percentile(y, 10.0)))
    target = y - background
    target_scale = max(float(np.linalg.norm(target)), 1e-300)
    yn = target / target_scale
    alpha_max = max(float(np.max(normalized.T @ yn)) / len(y), 1e-12)
    alphas = np.geomspace(alpha_max, alpha_max * 1e-4, cfg.lasso_alphas)
    lasso_device = "cpu"
    if not cfg.lasso_cross_validate and cfg.use_gpu:
        coef, lasso_device, lasso_iterations = _nonnegative_lasso_screen(design, target, cfg)
        model = None
    elif cfg.lasso_cross_validate:
        model = LassoCV(alphas=alphas, cv=3, positive=True, fit_intercept=False,
                        max_iter=cfg.lasso_maxiter, tol=1e-5, n_jobs=1)
    else:
        model = Lasso(alpha=alpha_max * cfg.lasso_alpha_relative, positive=True,
                      fit_intercept=False, max_iter=cfg.lasso_maxiter, tol=1e-5,
                      selection="cyclic")
    if model is not None:
        model.fit(normalized, yn)
        coef = np.asarray(model.coef_)
        lasso_iterations = int(np.max(np.atleast_1d(model.n_iter_)))
    best_by_sat: Dict[int, Tuple[float, int]] = {}
    for j, value in enumerate(coef):
        if value <= max(1e-6 * float(np.max(coef)), 1e-12):
            continue
        sat = int(labels[j][0])
        if sat not in best_by_sat or value > best_by_sat[sat][0]:
            best_by_sat[sat] = (float(value), j)
    ranked = sorted(best_by_sat.values(), reverse=True)[:cfg.max_order]
    chosen = [item[1] for item in ranked]
    atoms = [design[:, j] for j in chosen]
    amps, background, _, _ = _fit_positive(atoms, y, cfg)
    selected = [int(labels[j][0]) for j in chosen]
    params = [labels[j][1] for j in chosen]
    result = _source_result(scene, cfg, selected, params, amps, background, [], started,
                            0.0, 0.0, lasso_iterations, 0,
                            {"success": True}, "Lasso-Grid")
    selected_alpha = (float(model.alpha_ if hasattr(model, "alpha_") else model.alpha)
                      if model is not None else float(cfg.lasso_alpha_relative))
    result.update({"lasso_alpha": selected_alpha, "dictionary_atoms": design.shape[1],
                   "dictionary_bytes": design.nbytes, "compute_device": lasso_device})
    return result


def peak_fit(scene: Mapping[str, object], cfg: StudyConfig) -> Dict[str, object]:
    started = time.perf_counter()
    y = np.asarray(scene["observations_mw"], dtype=float) - float(scene["noise_mw"])
    width = 0.5 * (cfg.truth_width_min_deg + cfg.truth_width_max_deg)
    pars = [np.asarray([0.0, 0.0, width]) for _ in range(cfg.n_candidates)]
    atoms = [template(feature, par, cfg) for feature, par in zip(scene["features_train"], pars)]
    _, bg, pred0, _ = _fit_positive([], y, cfg)
    residual = y - pred0
    scores = np.asarray([float(atom @ residual) / max(float(np.linalg.norm(atom) * np.linalg.norm(residual)), 1e-300)
                         for atom in atoms])
    median = float(np.median(scores)); mad = 1.4826 * float(np.median(np.abs(scores - median)))
    threshold = median + cfg.peak_mad_scale * max(mad, 1e-12)
    selected = [int(i) for i in np.argsort(scores)[::-1]
                if scores[i] > threshold][:cfg.max_order]
    chosen_atoms = [atoms[i] for i in selected]
    amps, bg, _, _ = _fit_positive(chosen_atoms, y, cfg)
    result = _source_result(scene, cfg, selected, [pars[i] for i in selected], amps, bg, [],
                            started, 0.0, 0.0, 0, 0, {"success": True}, "Peak")
    result.update({"peak_threshold": threshold, "peak_scores": scores})
    return result


def _idw(scene: Mapping[str, object], cfg: StudyConfig) -> Dict[str, object]:
    started = time.perf_counter()
    xy = np.asarray(scene["xy_km"])
    train = np.asarray(scene["train_grid_idx"], dtype=int)
    query = np.asarray(scene["query_grid_idx"], dtype=int)
    observations = np.asarray(scene["observations_aggregate_mw"])
    values_all = 10.0 * np.log10(np.maximum(observations, 1e-300))
    delta = xy[query, None, :] - xy[train][None, :, :]
    distance2 = np.sum(delta * delta, axis=2)
    weights = 1.0 / np.maximum(distance2, 1e-8)
    fields = []
    for snapshot in range(cfg.n_snapshots):
        values = values_all[snapshot * cfg.n_measurements:(snapshot + 1) * cfg.n_measurements]
        prediction_dbm = (weights @ values) / weights.sum(axis=1)
        total = np.full(len(xy), np.nan)
        total[train] = observations[snapshot * cfg.n_measurements:(snapshot + 1) * cfg.n_measurements]
        total[query] = 10.0 ** (prediction_dbm / 10.0)
        fields.append(total)
    aggregate = np.concatenate(fields)
    return {"method": "IDW", "map_only": True, "pred_total_mw": aggregate,
            "pred_total_aggregate_mw": aggregate,
            "runtime_s": time.perf_counter() - started, "solver_failures": 0}


def _kriging(scene: Mapping[str, object], cfg: StudyConfig) -> Dict[str, object]:
    started = time.perf_counter()
    xy = np.asarray(scene["xy_km"]) / max(cfg.region_side_km, 1e-12)
    train = np.asarray(scene["train_grid_idx"], dtype=int)
    query = np.asarray(scene["query_grid_idx"], dtype=int)
    observations = np.asarray(scene["observations_aggregate_mw"])
    values_all = 10.0 * np.log10(np.maximum(observations, 1e-300))
    kernel = (ConstantKernel(1.0, constant_value_bounds="fixed")
              * Matern(0.15, length_scale_bounds="fixed", nu=1.5)
              + WhiteKernel(0.1, noise_level_bounds="fixed"))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None,
                                     random_state=cfg.random_seed)
    fields = []
    kernels = []
    for snapshot in range(cfg.n_snapshots):
        values = values_all[snapshot * cfg.n_measurements:(snapshot + 1) * cfg.n_measurements]
        model.fit(xy[train], values)
        prediction_dbm = model.predict(xy[query])
        total = np.full(len(xy), np.nan)
        total[train] = observations[snapshot * cfg.n_measurements:(snapshot + 1) * cfg.n_measurements]
        total[query] = 10.0 ** (prediction_dbm / 10.0)
        fields.append(total); kernels.append(str(model.kernel_))
    aggregate = np.concatenate(fields)
    return {"method": "Kriging", "map_only": True, "pred_total_mw": aggregate,
            "pred_total_aggregate_mw": aggregate,
            "runtime_s": time.perf_counter() - started, "solver_failures": 0,
            "kernel": kernels}


def _soft_impute(scene: Mapping[str, object], cfg: StudyConfig) -> Dict[str, object]:
    started = time.perf_counter()
    n = cfg.grid_size
    train = np.asarray(scene["train_grid_idx"], dtype=int)
    query = np.asarray(scene["query_grid_idx"], dtype=int)
    observations = np.asarray(scene["observations_aggregate_mw"])
    observed_dbm_all = 10.0 * np.log10(np.maximum(observations, 1e-300))
    mask = np.zeros(n * n, dtype=bool); mask[train] = True
    rank = min(8, max(2, n // 4))
    fields = []
    for snapshot in range(cfg.n_snapshots):
        observed_dbm = observed_dbm_all[snapshot * cfg.n_measurements:(snapshot + 1) * cfg.n_measurements]
        fill = np.full(n * n, float(np.mean(observed_dbm)))
        fill[train] = observed_dbm
        for _ in range(100):
            matrix = fill.reshape(n, n)
            u, s, vt = np.linalg.svd(matrix, full_matrices=False)
            reconstructed = (u[:, :rank] * s[:rank]) @ vt[:rank]
            updated = reconstructed.ravel(); updated[train] = observed_dbm
            if np.linalg.norm(updated - fill) / max(np.linalg.norm(fill), 1e-12) < 1e-6:
                fill = updated; break
            fill = updated
        total = np.full(n * n, np.nan)
        total[train] = observations[snapshot * cfg.n_measurements:(snapshot + 1) * cfg.n_measurements]
        total[query] = 10.0 ** (fill[query] / 10.0)
        fields.append(total)
    aggregate = np.concatenate(fields)
    return {"method": "SoftImpute", "map_only": True, "pred_total_mw": aggregate,
            "pred_total_aggregate_mw": aggregate,
            "runtime_s": time.perf_counter() - started, "solver_failures": 0,
            "rank": rank}


def _deeprm_td(scene: Mapping[str, object], cfg: StudyConfig) -> Dict[str, object]:
    """Documented 2-D adaptation of DeepRM's unsupervised neural TD stage."""
    import torch
    from torch import nn

    started = time.perf_counter()
    torch.manual_seed(cfg.random_seed + int(scene["scene_id"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = cfg.grid_size; rank = min(12, max(4, n // 2))
    coords = torch.linspace(-1.0, 1.0, n, device=device).unsqueeze(1)

    class FactorNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1, 32), nn.Sine() if hasattr(nn, "Sine") else nn.Tanh(),
                                     nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, rank))
        def forward(self, value):
            return self.net(value)

    train_grid = np.asarray(scene["train_grid_idx"], dtype=int)
    train = torch.as_tensor(train_grid, dtype=torch.long, device=device)
    observations = np.asarray(scene["observations_aggregate_mw"])
    observed_all = 10.0 * np.log10(np.maximum(observations, 1e-300))
    fields = []; total_iterations = 0
    for snapshot in range(cfg.n_snapshots):
        # Reinitialize the unsupervised factorization for each 2-D slice; no test
        # truth or neighboring snapshot is supplied to this adapted baseline.
        row_net, col_net = FactorNet().to(device), FactorNet().to(device)
        core = nn.Parameter(torch.eye(rank, device=device))
        optimizer = torch.optim.Adam(list(row_net.parameters()) + list(col_net.parameters()) + [core], lr=1e-2)
        values_np = observed_all[snapshot * cfg.n_measurements:(snapshot + 1) * cfg.n_measurements]
        mean, std = float(np.mean(values_np)), max(float(np.std(values_np)), 1e-6)
        values = torch.as_tensor((values_np - mean) / std, dtype=torch.float32, device=device)
        last = np.inf; iterations = 0
        for iteration in range(300):
            optimizer.zero_grad(set_to_none=True)
            u, v = row_net(coords), col_net(coords)
            field = u @ core @ v.T / rank
            pred = field.reshape(-1).index_select(0, train)
            smooth = ((field[1:] - field[:-1]) ** 2).mean() + ((field[:, 1:] - field[:, :-1]) ** 2).mean()
            loss = ((pred - values) ** 2).mean() + 2e-4 * smooth + 1e-5 * (core ** 2).mean()
            loss.backward(); optimizer.step()
            iterations = iteration + 1
            current = float(loss.detach().cpu())
            if iteration > 80 and abs(last - current) < 1e-7:
                break
            last = current
        with torch.no_grad():
            field = (row_net(coords) @ core @ col_net(coords).T / rank).cpu().numpy()
        prediction_dbm = field.ravel() * std + mean
        total = 10.0 ** (prediction_dbm / 10.0)
        total[train_grid] = observations[snapshot * cfg.n_measurements:(snapshot + 1) * cfg.n_measurements]
        fields.append(total); total_iterations += iterations
    aggregate = np.concatenate(fields)
    return {"method": "DeepRM-TD", "map_only": True, "pred_total_mw": aggregate,
            "pred_total_aggregate_mw": aggregate,
            "runtime_s": time.perf_counter() - started, "solver_failures": 0,
            "iterations": total_iterations, "rank": rank, "device": str(device),
            "adaptation": "2-D single-frequency neural tensor-decomposition stage"}


def run_method(scene: Mapping[str, object], cfg: StudyConfig, method: str) -> Dict[str, object]:
    if method == "SateBeam":
        return satebeam_final(scene, cfg)
    if method == "SateBeam-Global":
        return satebeam_global_fit(scene, cfg)
    if method == "SateBeam-Greedy":
        return satebeam_fit(scene, cfg)
    if method == "Oracle-Support":
        return oracle_support_fit(scene, cfg)
    if method == "Oracle-TruthInit":
        selected = [int(value) for value in scene["active_ids"]]
        params = [np.asarray([scene["params"][idx]["offset_az_deg"],
                              scene["params"][idx]["offset_el_deg"],
                              scene["params"][idx]["width_deg"]]) for idx in selected]
        started = time.perf_counter()
        refined, amps, background, _, _, info = _gamma_joint_refine(
            scene["features_train"], np.asarray(scene["observations_mw"]), selected, params, cfg
        )
        return _source_result(scene, cfg, selected, refined, amps, background, [], started,
                              0.0, 0.0, int(info.get("nfev", 0)),
                              int(not info.get("success", True)), info, "Oracle-TruthInit")
    if method == "OMP-Grid":
        return _grid_greedy(scene, cfg, method)
    if method == "MP-Grid":
        return _grid_greedy(scene, cfg, method)
    if method == "Orbit-NNLS":
        return _grid_greedy(scene, cfg, method, nominal_only=True)
    if method == "OMP-Refine":
        return omp_refine(scene, cfg)
    if method == "Lasso-Grid":
        return lasso_grid(scene, cfg)
    if method == "Peak":
        return peak_fit(scene, cfg)
    if method == "IDW":
        return _idw(scene, cfg)
    if method == "Kriging":
        return _kriging(scene, cfg)
    if method == "SoftImpute":
        return _soft_impute(scene, cfg)
    if method == "DeepRM-TD":
        return _deeprm_td(scene, cfg)
    raise ValueError(f"Unknown method: {method}")
