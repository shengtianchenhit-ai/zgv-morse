"""Generate full-wave weak-anisotropy splitting and remainder scalings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.optimize import brentq

from ..config import ReferenceConfig
from ..dispersion import RingAnchoredSpectralEvaluator
from ..elasticity import cubic_family, isotropic_tensor
from ..zgv import ZGVPoint, find_s1_zgv
from .common import load_stage_dependency, validate_stage_inputs, write_stage_artifact
from .sensitivity import compute_sensitivity_arrays


_PROFILE = MappingProxyType(
    {
        "smoke": MappingProxyType({"order": 9, "positive_count": 3}),
        "full": MappingProxyType({"order": 10, "positive_count": -1}),
    }
)


@dataclass(frozen=True, slots=True)
class _RadialPoint:
    theta: float
    radius: float
    omega: float
    frequency_uncertainty: float
    gradient_residual: float


def _evaluator(
    cfg: ReferenceConfig,
    exact: ZGVPoint,
    epsilon: float,
    order: int,
) -> RingAnchoredSpectralEvaluator:
    tensor = (
        isotropic_tensor(cfg.lam, cfg.mu)
        if epsilon == 0.0
        else cubic_family(cfg.lam, cfg.mu, cfg.delta, epsilon)[0]
    )
    return RingAnchoredSpectralEvaluator(
        tensor,
        rho=cfg.rho,
        half_thickness=cfg.h,
        k0=exact.kappa0,
        target_omega=exact.omega0,
        order=order,
        num_modes=12,
        angular_sectors=8,
    )


def _radial_stationary_point(
    evaluator: RingAnchoredSpectralEvaluator,
    theta: float,
    half_width: float,
) -> _RadialPoint:
    radial = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    tangent = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    radii = np.linspace(
        evaluator.k0 - half_width,
        evaluator.k0 + half_width,
        17,
    )

    def radial_gradient(radius: float) -> float:
        sample = evaluator(float(radius) * radial)
        return float(sample.gradient @ radial)

    gradients = np.array([radial_gradient(radius) for radius in radii])
    brackets = [
        (float(left), float(right))
        for left, right, value_left, value_right in zip(
            radii[:-1],
            radii[1:],
            gradients[:-1],
            gradients[1:],
            strict=True,
        )
        if value_left == 0.0 or np.signbit(value_left) != np.signbit(value_right)
    ]
    if not brackets:
        raise RuntimeError("radial group velocity does not bracket a stationary point")
    lower, upper = min(brackets, key=lambda pair: abs(0.5 * sum(pair) - evaluator.k0))
    root = brentq(
        radial_gradient,
        lower,
        upper,
        xtol=2.0e-13,
        rtol=8.0 * np.finfo(np.float64).eps,
    )
    sample = evaluator(root * radial)
    radial_residual = abs(float(sample.gradient @ radial))
    tangential_residual = abs(float(sample.gradient @ tangent))
    residual = float(np.hypot(radial_residual, tangential_residual))
    residual_limit = max(
        10.0 * sample.gradient_uncertainty,
        5.0e-10 * max(sample.omega / evaluator.k0, 1.0),
    )
    if residual > residual_limit:
        raise RuntimeError("radial stationary point fails the gradient-residual gate")
    return _RadialPoint(
        theta=float(theta),
        radius=float(root),
        omega=sample.omega,
        frequency_uncertainty=sample.frequency_uncertainty,
        gradient_residual=residual,
    )


def _sensitivity_arrays(
    cfg: ReferenceConfig,
    output_dir: Path,
    profile: str,
    input_artifacts: dict[str, dict[str, str]] | None = None,
) -> dict[str, np.ndarray]:
    required = {"theta", "V", "B", "V4"}
    arrays = load_stage_dependency(
        "sensitivity",
        cfg,
        output_dir,
        profile,
        required,
        input_artifacts,
    )
    if arrays is None:
        arrays, _diagnostics = compute_sensitivity_arrays(cfg, profile)
    theta = np.asarray(arrays["theta"], dtype=np.float64)
    V = np.asarray(arrays["V"], dtype=np.float64)
    B = np.asarray(arrays["B"], dtype=np.float64)
    if (
        theta.ndim != 1
        or theta.size < 16
        or V.shape != theta.shape
        or B.shape != theta.shape
        or not np.isfinite(theta).all()
        or not np.isfinite(V).all()
        or not np.isfinite(B).all()
    ):
        raise RuntimeError("sensitivity scaling arrays are invalid")
    return arrays


def _periodic_value(theta: np.ndarray, values: np.ndarray, query: float) -> float:
    extended_theta = np.concatenate((theta, [theta[0] + 2.0 * np.pi]))
    extended_values = np.concatenate((values, [values[0]]))
    result = float(
        np.interp(
            float(np.mod(query, 2.0 * np.pi)),
            extended_theta,
            extended_values,
        )
    )
    if not np.isfinite(result):
        raise RuntimeError("periodic sensitivity interpolation is non-finite")
    return result


def _log_slope(x: np.ndarray, y: np.ndarray, name: str) -> float:
    if x.ndim != 1 or y.shape != x.shape or x.size < 3:
        raise RuntimeError(f"{name} slope requires at least three aligned samples")
    if np.any(x <= 0.0) or np.any(y <= 0.0) or not np.isfinite(y).all():
        raise RuntimeError(f"{name} slope requires finite positive samples")
    logarithmic_x = np.log(x)
    logarithmic_y = np.log(y)
    centered_x = logarithmic_x - np.mean(logarithmic_x)
    denominator = float(centered_x @ centered_x)
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise RuntimeError(f"{name} slope has a singular logarithmic abscissa")
    slope = float(centered_x @ (logarithmic_y - np.mean(logarithmic_y)) / denominator)
    if not np.isfinite(slope):
        raise RuntimeError(f"{name} slope is non-finite")
    return slope


def _fourfold_orbit(theta: float) -> np.ndarray:
    return np.mod(theta + 0.5 * np.pi * np.arange(4), 2.0 * np.pi)


def _corrected_point(
    point: _RadialPoint,
    baseline: _RadialPoint,
    exact: ZGVPoint,
) -> tuple[float, float]:
    q = point.radius - baseline.radius
    omega = point.omega - baseline.omega + exact.omega0
    if not np.isfinite(q) or not np.isfinite(omega):
        raise RuntimeError("baseline-corrected full-wave point is non-finite")
    return float(q), float(omega)


def run(cfg: ReferenceConfig, output_dir: Path, profile: str) -> Path:
    """Run the registered full-wave perturbation-scaling stage."""

    config, directory, selected_profile = validate_stage_inputs(cfg, output_dir, profile)
    settings = _PROFILE[selected_profile]
    order = int(settings["order"])
    exact = find_s1_zgv(config, dps=60 if selected_profile == "smoke" else 80)
    half_width = config.annulus_fraction * exact.kappa0
    positive = np.array(
        [epsilon for epsilon in config.epsilon_values if epsilon > 0.0],
        dtype=np.float64,
    )
    count = int(settings["positive_count"])
    epsilon_values = positive if count < 0 else positive[:count]
    if epsilon_values.size < 3:
        raise RuntimeError("scaling profile requires at least three positive epsilon values")

    input_artifacts: dict[str, dict[str, str]] = {}
    sensitivity = _sensitivity_arrays(
        config,
        directory,
        selected_profile,
        input_artifacts,
    )
    sensitivity_theta = np.asarray(sensitivity["theta"], dtype=np.float64)
    V_values = np.asarray(sensitivity["V"], dtype=np.float64)
    B_values = np.asarray(sensitivity["B"], dtype=np.float64)
    axis_theta = 0.0
    diagonal_theta = 0.25 * np.pi
    V_axis = _periodic_value(sensitivity_theta, V_values, axis_theta)
    V_diagonal = _periodic_value(sensitivity_theta, V_values, diagonal_theta)
    B_axis = _periodic_value(sensitivity_theta, B_values, axis_theta)
    B_diagonal = _periodic_value(sensitivity_theta, B_values, diagonal_theta)

    isotropic = _evaluator(config, exact, 0.0, order)
    baseline_axis = _radial_stationary_point(isotropic, axis_theta, half_width)
    baseline_diagonal = _radial_stationary_point(isotropic, diagonal_theta, half_width)

    delta_full: list[float] = []
    delta_pred: list[float] = []
    q_min_full: list[float] = []
    q_min_pred: list[float] = []
    q_saddle_full: list[float] = []
    q_saddle_pred: list[float] = []
    omega_min_error: list[float] = []
    omega_saddle_error: list[float] = []
    maximum_frequency_uncertainty = 0.0
    maximum_gradient_residual = 0.0
    positive_roles: tuple[float, float] | None = None

    for epsilon in epsilon_values:
        evaluator = _evaluator(config, exact, float(epsilon), order)
        axis = _radial_stationary_point(evaluator, axis_theta, half_width)
        diagonal = _radial_stationary_point(evaluator, diagonal_theta, half_width)
        q_axis, omega_axis = _corrected_point(axis, baseline_axis, exact)
        q_diagonal, omega_diagonal = _corrected_point(
            diagonal,
            baseline_diagonal,
            exact,
        )
        records = (
            (axis_theta, q_axis, omega_axis, V_axis, B_axis, axis),
            (
                diagonal_theta,
                q_diagonal,
                omega_diagonal,
                V_diagonal,
                B_diagonal,
                diagonal,
            ),
        )
        minimum, saddle = sorted(records, key=lambda record: record[2])
        positive_roles = (minimum[0], saddle[0])
        minimum_prediction = exact.omega0 + epsilon * minimum[3]
        saddle_prediction = exact.omega0 + epsilon * saddle[3]
        delta_full.append(saddle[2] - minimum[2])
        delta_pred.append(saddle_prediction - minimum_prediction)
        q_min_full.append(minimum[1])
        q_saddle_full.append(saddle[1])
        q_min_pred.append(-epsilon * minimum[4] / exact.curvature_a)
        q_saddle_pred.append(-epsilon * saddle[4] / exact.curvature_a)
        omega_min_error.append(abs(minimum[2] - minimum_prediction))
        omega_saddle_error.append(abs(saddle[2] - saddle_prediction))
        maximum_frequency_uncertainty = max(
            maximum_frequency_uncertainty,
            axis.frequency_uncertainty,
            diagonal.frequency_uncertainty,
        )
        maximum_gradient_residual = max(
            maximum_gradient_residual,
            axis.gradient_residual,
            diagonal.gradient_residual,
        )

    epsilon_array = np.asarray(epsilon_values)
    delta_full_array = np.asarray(delta_full)
    delta_pred_array = np.asarray(delta_pred)
    q_min_array = np.asarray(q_min_full)
    q_saddle_array = np.asarray(q_saddle_full)
    q_min_pred_array = np.asarray(q_min_pred)
    q_saddle_pred_array = np.asarray(q_saddle_pred)
    omega_min_error_array = np.asarray(omega_min_error)
    omega_saddle_error_array = np.asarray(omega_saddle_error)
    maximum_remainder = np.maximum(omega_min_error_array, omega_saddle_error_array)

    slope_splitting = _log_slope(epsilon_array, delta_full_array, "splitting")
    radial_scale = np.maximum(np.abs(q_min_array), np.abs(q_saddle_array))
    slope_radial = _log_slope(epsilon_array, radial_scale, "radial shift")
    slope_remainder = _log_slope(epsilon_array, maximum_remainder, "frequency remainder")
    if abs(slope_splitting - 1.0) > 0.1:
        raise RuntimeError("full-wave splitting slope misses the registered linear gate")
    if abs(slope_radial - 1.0) > 0.1:
        raise RuntimeError("full-wave radial-shift slope misses the registered linear gate")
    if abs(slope_remainder - 2.0) > 0.2:
        raise RuntimeError("full-wave frequency-remainder slope misses the quadratic gate")

    if positive_roles is None:
        raise RuntimeError("positive-epsilon roles were not evaluated")
    reversal_epsilon = -float(epsilon_array[-1])
    negative = _evaluator(config, exact, reversal_epsilon, order)
    negative_axis = _radial_stationary_point(negative, axis_theta, half_width)
    negative_diagonal = _radial_stationary_point(negative, diagonal_theta, half_width)
    negative_axis_corrected = _corrected_point(negative_axis, baseline_axis, exact)
    negative_diagonal_corrected = _corrected_point(
        negative_diagonal,
        baseline_diagonal,
        exact,
    )
    negative_records = (
        (axis_theta, negative_axis_corrected[1]),
        (diagonal_theta, negative_diagonal_corrected[1]),
    )
    negative_minimum, negative_saddle = sorted(
        negative_records,
        key=lambda record: record[1],
    )
    if (
        abs(negative_minimum[0] - positive_roles[1]) > 1.0e-12
        or abs(negative_saddle[0] - positive_roles[0]) > 1.0e-12
    ):
        raise RuntimeError("epsilon sign reversal does not exchange axis/diagonal roles")

    arrays = {
        "epsilon": epsilon_array,
        "delta_omega_full": delta_full_array,
        "delta_omega_pred": delta_pred_array,
        "q_min_full": q_min_array,
        "q_min_pred": q_min_pred_array,
        "q_saddle_full": q_saddle_array,
        "q_saddle_pred": q_saddle_pred_array,
        "omega_min_error": omega_min_error_array,
        "omega_saddle_error": omega_saddle_error_array,
        "compensated_splitting": delta_full_array / epsilon_array,
        "compensated_q_min": q_min_array / epsilon_array,
        "compensated_q_saddle": q_saddle_array / epsilon_array,
        "compensated_frequency_error": maximum_remainder / epsilon_array**2,
        "role_reversal_theta_min": _fourfold_orbit(negative_minimum[0]),
        "role_reversal_theta_saddle": _fourfold_orbit(negative_saddle[0]),
        "slope_splitting": np.array(slope_splitting),
        "slope_radial": np.array(slope_radial),
        "slope_remainder": np.array(slope_remainder),
    }
    units = {
        "epsilon": "1",
        "delta_omega_full": "1",
        "delta_omega_pred": "1",
        "q_min_full": "1",
        "q_min_pred": "1",
        "q_saddle_full": "1",
        "q_saddle_pred": "1",
        "omega_min_error": "1",
        "omega_saddle_error": "1",
        "compensated_splitting": "Omega per epsilon",
        "compensated_q_min": "kappa per epsilon",
        "compensated_q_saddle": "kappa per epsilon",
        "compensated_frequency_error": "Omega per epsilon squared",
        "role_reversal_theta_min": "rad",
        "role_reversal_theta_saddle": "rad",
        "slope_splitting": "log-log slope",
        "slope_radial": "log-log slope",
        "slope_remainder": "log-log slope",
    }
    tolerances = {
        "maximum_frequency_uncertainty": maximum_frequency_uncertainty,
        "maximum_gradient_residual": maximum_gradient_residual,
        "isotropic_axis_radius_bias": baseline_axis.radius - exact.kappa0,
        "isotropic_diagonal_radius_bias": baseline_diagonal.radius - exact.kappa0,
        "slope_splitting_error": abs(slope_splitting - 1.0),
        "slope_radial_error": abs(slope_radial - 1.0),
        "slope_remainder_error": abs(slope_remainder - 2.0),
    }
    return write_stage_artifact(
        "scaling",
        config,
        directory,
        selected_profile,
        arrays,
        units,
        tolerances,
        input_artifacts=input_artifacts,
    )
