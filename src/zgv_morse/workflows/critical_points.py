"""Generate the resolution-checked weak-cubic Morse critical-point artifact."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..config import ReferenceConfig
from ..critical_points import Annulus, CriticalPoint, locate_critical_points, verify_annular_exhaustion
from ..dispersion import FrequencyGradient, RingAnchoredSpectralEvaluator
from ..elasticity import cubic_family, isotropic_tensor
from ..zgv import ZGVPoint, find_s1_zgv
from .common import load_stage_dependency, validate_stage_inputs, write_stage_artifact
from .sensitivity import compute_sensitivity_arrays


_PROFILE = MappingProxyType(
    {
        "smoke": MappingProxyType(
            {
                "epsilon": 0.08,
                "order": 9,
                "grid_nodes": 18,
                "hessian_step": 2.0e-3,
            }
        ),
        "full": MappingProxyType(
            {
                "epsilon": 0.02,
                "order": 10,
                "grid_nodes": 26,
                "hessian_step": 1.0e-3,
            }
        ),
    }
)


class _RotatedEvaluator:
    """Express one physical evaluator in a rotated Cartesian search frame."""

    def __init__(self, evaluator: RingAnchoredSpectralEvaluator, angle: float) -> None:
        self.evaluator = evaluator
        self.angle = float(angle)
        self.rotation = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ],
            dtype=np.float64,
        )

    def __call__(self, point: np.ndarray) -> FrequencyGradient:
        sample = self.evaluator(self.rotation @ np.asarray(point, dtype=np.float64))
        return FrequencyGradient(
            sample.omega,
            self.rotation.T @ sample.gradient,
            sample.frequency_uncertainty,
            sample.gradient_uncertainty,
        )

    def physical_points(self, points: list[CriticalPoint]) -> list[CriticalPoint]:
        result: list[CriticalPoint] = []
        for point in points:
            coordinates = self.rotation @ np.array([point.kx, point.ky])
            hessian = self.rotation @ point.hessian @ self.rotation.T
            eigenvalues = np.linalg.eigvalsh(hessian)
            theta = float(np.mod(np.arctan2(coordinates[1], coordinates[0]), 2.0 * np.pi))
            if min(theta, 2.0 * np.pi - theta) <= 5.0e-9:
                theta = 0.0
            result.append(
                CriticalPoint(
                    float(coordinates[0]),
                    float(coordinates[1]),
                    point.radius,
                    theta,
                    point.omega,
                    point.gradient_residual,
                    point.gradient_uncertainty,
                    hessian,
                    eigenvalues,
                    point.hessian_uncertainty,
                    point.kind,
                    point.morse_index,
                )
            )
        return sorted(result, key=lambda item: item.theta)


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


def _resolution_difference(
    coarse: list[CriticalPoint],
    fine: list[CriticalPoint],
    k0: float,
    evaluator: RingAnchoredSpectralEvaluator,
) -> tuple[float, float, float]:
    if len(coarse) != len(fine):
        raise RuntimeError("critical-point count changes under candidate-grid doubling")
    maximum_position = 0.0
    maximum_frequency = 0.0
    maximum_hessian = 0.0
    for coarse_point, fine_point in zip(coarse, fine, strict=True):
        if coarse_point.kind != fine_point.kind:
            raise RuntimeError("critical-point kind changes under candidate-grid doubling")
        position = float(
            np.hypot(
                coarse_point.kx - fine_point.kx,
                coarse_point.ky - fine_point.ky,
            )
        )
        position_uncertainty = (
            coarse_point.gradient_uncertainty
            / float(np.min(np.abs(coarse_point.hessian_eigenvalues)))
            + fine_point.gradient_uncertainty
            / float(np.min(np.abs(fine_point.hessian_eigenvalues)))
        )
        if position > max(5.0e-9 * k0, position_uncertainty):
            raise RuntimeError("critical-point position fails candidate-grid doubling")
        coarse_sample = evaluator(
            np.array([coarse_point.kx, coarse_point.ky], dtype=np.float64)
        )
        fine_sample = evaluator(
            np.array([fine_point.kx, fine_point.ky], dtype=np.float64)
        )
        positional_frequency_error = (
            0.5
            * max(
                float(np.max(np.abs(coarse_point.hessian_eigenvalues))),
                float(np.max(np.abs(fine_point.hessian_eigenvalues))),
            )
            * position**2
        )
        frequency = abs(coarse_point.omega - fine_point.omega)
        if frequency > (
            coarse_sample.frequency_uncertainty
            + fine_sample.frequency_uncertainty
            + positional_frequency_error
        ):
            raise RuntimeError("critical-point frequency fails candidate-grid doubling")
        hessian = float(
            np.max(
                np.abs(
                    coarse_point.hessian_eigenvalues
                    - fine_point.hessian_eigenvalues
                )
            )
        )
        if hessian > coarse_point.hessian_uncertainty + fine_point.hessian_uncertainty:
            raise RuntimeError("critical-point Hessian fails candidate-grid doubling")
        maximum_position = max(maximum_position, position)
        maximum_frequency = max(
            maximum_frequency,
            frequency,
        )
        maximum_hessian = max(maximum_hessian, hessian)
    return maximum_position, maximum_frequency, maximum_hessian


def _certified_points(
    evaluator: RingAnchoredSpectralEvaluator,
    annulus: Annulus,
    hessian_step: float,
    *,
    angular_offset: float = 0.0,
) -> tuple[list[CriticalPoint], dict[str, float]]:
    search_evaluator = (
        evaluator if angular_offset == 0.0 else _RotatedEvaluator(evaluator, angular_offset)
    )
    coarse = locate_critical_points(search_evaluator, annulus, 5, 16, hessian_step)
    fine = locate_critical_points(search_evaluator, annulus, 9, 32, hessian_step)
    if len(fine) != 8:
        raise RuntimeError("declared workflow requires eight resolved critical points")
    kinds = tuple(point.kind for point in fine)
    if kinds.count("minimum") != 4 or kinds.count("saddle") != 4:
        raise RuntimeError("registered annulus must contain four minima and four saddles")
    if any(kinds[index] == kinds[(index + 1) % len(kinds)] for index in range(len(kinds))):
        raise RuntimeError("minimum and saddle points must alternate in angular order")
    if sum(point.morse_index for point in fine) != 0:
        raise RuntimeError("critical-point Morse indices do not close")
    report = verify_annular_exhaustion(search_evaluator, annulus, fine, 32)
    if not report.boundary_is_noncritical or not report.index_closes:
        raise RuntimeError("annular boundary/index consistency check failed")
    if isinstance(search_evaluator, _RotatedEvaluator):
        coarse = search_evaluator.physical_points(coarse)
        fine = search_evaluator.physical_points(fine)
    position, frequency, hessian = _resolution_difference(
        coarse,
        fine,
        annulus.k0,
        evaluator,
    )
    return fine, {
        "candidate_grid_position_difference": position,
        "candidate_grid_frequency_difference": frequency,
        "candidate_grid_hessian_difference": hessian,
        "minimum_boundary_gradient": report.minimum_boundary_gradient,
        "maximum_boundary_gradient_uncertainty": report.maximum_gradient_uncertainty,
    }


def _sensitivity_arrays(
    cfg: ReferenceConfig,
    output_dir: Path,
    profile: str,
    input_artifacts: dict[str, dict[str, str]] | None = None,
) -> dict[str, np.ndarray]:
    required = {"theta", "V", "B", "V0", "V4"}
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
        raise RuntimeError("sensitivity prediction arrays are invalid")
    return arrays


def _periodic_interpolate(theta: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    angles = np.mod(np.asarray(query, dtype=np.float64), 2.0 * np.pi)
    extended_theta = np.concatenate((theta, [theta[0] + 2.0 * np.pi]))
    extended_values = np.concatenate((values, [values[0]]))
    result = np.interp(angles, extended_theta, extended_values)
    if not np.isfinite(result).all():
        raise RuntimeError("periodic sensitivity interpolation is non-finite")
    return result


def _verify_role_reversal(
    positive: list[CriticalPoint],
    negative: list[CriticalPoint],
) -> None:
    if len(positive) != len(negative):
        raise RuntimeError("epsilon sign reversal changes the critical-point count")
    for plus, minus in zip(positive, negative, strict=True):
        angular_difference = abs(plus.theta - minus.theta)
        angular_difference = min(angular_difference, 2.0 * np.pi - angular_difference)
        if angular_difference > 2.0e-6:
            raise RuntimeError("epsilon sign reversal changes the critical angles")
        expected = "saddle" if plus.kind == "minimum" else "minimum"
        if minus.kind != expected:
            raise RuntimeError("epsilon sign reversal does not swap minimum/saddle roles")


def _full_wave_grid(
    evaluator: RingAnchoredSpectralEvaluator,
    axis: np.ndarray,
) -> tuple[np.ndarray, float]:
    values = np.empty((axis.size, axis.size), dtype=np.float64)
    maximum_uncertainty = 0.0
    for row, ky in enumerate(axis):
        for column, kx in enumerate(axis):
            sample = evaluator(np.array([kx, ky], dtype=np.float64))
            values[row, column] = sample.omega
            maximum_uncertainty = max(
                maximum_uncertainty,
                sample.frequency_uncertainty,
            )
    if not np.isfinite(values).all():
        raise RuntimeError("full-wave Cartesian surface is non-finite")
    return values, maximum_uncertainty


def run(cfg: ReferenceConfig, output_dir: Path, profile: str) -> Path:
    """Run the registered weak-cubic critical-point stage."""

    config, directory, selected_profile = validate_stage_inputs(cfg, output_dir, profile)
    settings = _PROFILE[selected_profile]
    epsilon = float(settings["epsilon"])
    order = int(settings["order"])
    exact = find_s1_zgv(config, dps=60 if selected_profile == "smoke" else 80)
    annulus = Annulus(exact.kappa0, config.annulus_fraction * exact.kappa0)
    positive_evaluator = _evaluator(config, exact, epsilon, order)
    negative_evaluator = _evaluator(config, exact, -epsilon, order)
    points, positive_diagnostics = _certified_points(
        positive_evaluator,
        annulus,
        float(settings["hessian_step"]),
    )
    negative_points, negative_diagnostics = _certified_points(
        negative_evaluator,
        annulus,
        float(settings["hessian_step"]),
        angular_offset=np.pi / 16.0,
    )
    _verify_role_reversal(points, negative_points)

    input_artifacts: dict[str, dict[str, str]] = {}
    sensitivity = _sensitivity_arrays(
        config,
        directory,
        selected_profile,
        input_artifacts,
    )
    sensitivity_theta = np.asarray(sensitivity["theta"], dtype=np.float64)
    point_theta = np.array([point.theta for point in points], dtype=np.float64)
    V_at_points = _periodic_interpolate(
        sensitivity_theta,
        np.asarray(sensitivity["V"], dtype=np.float64),
        point_theta,
    )
    B_at_points = _periodic_interpolate(
        sensitivity_theta,
        np.asarray(sensitivity["B"], dtype=np.float64),
        point_theta,
    )
    q_pred = -epsilon * B_at_points / exact.curvature_a
    kappa_pred = exact.kappa0 + q_pred

    grid_axis = np.linspace(
        -annulus.outer_radius,
        annulus.outer_radius,
        int(settings["grid_nodes"]),
    )
    isotropic_evaluator = _evaluator(config, exact, 0.0, order)
    omega_iso_grid, isotropic_grid_uncertainty = _full_wave_grid(
        isotropic_evaluator,
        grid_axis,
    )
    omega_aniso_grid, anisotropic_grid_uncertainty = _full_wave_grid(
        positive_evaluator,
        grid_axis,
    )

    arrays = {
        "kx": np.array([point.kx for point in points]),
        "ky": np.array([point.ky for point in points]),
        "kappa": np.array([point.radius for point in points]),
        "theta": point_theta,
        "omega": np.array([point.omega for point in points]),
        "hessian_eigenvalues": np.stack(
            [point.hessian_eigenvalues for point in points]
        ),
        "morse_index": np.array([point.morse_index for point in points], dtype=np.int64),
        "kind": np.array([point.kind for point in points]),
        "kx_pred": kappa_pred * np.cos(point_theta),
        "ky_pred": kappa_pred * np.sin(point_theta),
        "omega_pred": exact.omega0 + epsilon * V_at_points,
        "gradient_residual": np.array([point.gradient_residual for point in points]),
        "kx_grid": grid_axis,
        "ky_grid": grid_axis,
        "omega_iso_grid": omega_iso_grid,
        "omega_aniso_grid": omega_aniso_grid,
    }
    units = {
        "kx": "1",
        "ky": "1",
        "kappa": "1",
        "theta": "rad",
        "omega": "1",
        "hessian_eigenvalues": "Omega per kappa squared",
        "morse_index": "integer",
        "kind": "classification",
        "kx_pred": "1",
        "ky_pred": "1",
        "omega_pred": "1",
        "gradient_residual": "Omega per kappa",
        "kx_grid": "1",
        "ky_grid": "1",
        "omega_iso_grid": "1",
        "omega_aniso_grid": "1",
    }
    minimum_hessian_ratio = min(
        float(np.min(np.abs(point.hessian_eigenvalues)))
        / max(point.hessian_uncertainty, np.finfo(np.float64).tiny)
        for point in (*points, *negative_points)
    )
    tolerances = {
        "epsilon": epsilon,
        "minimum_hessian_to_uncertainty": minimum_hessian_ratio,
        "maximum_gradient_residual": float(np.max(arrays["gradient_residual"])),
        "maximum_grid_frequency_uncertainty": max(
            isotropic_grid_uncertainty,
            anisotropic_grid_uncertainty,
        ),
        **{f"positive_{key}": value for key, value in positive_diagnostics.items()},
        **{f"negative_{key}": value for key, value in negative_diagnostics.items()},
    }
    return write_stage_artifact(
        "critical_points",
        config,
        directory,
        selected_profile,
        arrays,
        units,
        tolerances,
        input_artifacts=input_artifacts,
        extra_metadata={
            "sign_reversal_certificate": {
                "positive_epsilon": epsilon,
                "negative_epsilon": -epsilon,
                "positive_count": len(points),
                "negative_count": len(negative_points),
                "positive_kinds": [point.kind for point in points],
                "negative_kinds": [point.kind for point in negative_points],
                "positive_theta": [point.theta for point in points],
                "negative_theta": [point.theta for point in negative_points],
                "roles_exchanged_at_same_angles": True,
            }
        },
    )
