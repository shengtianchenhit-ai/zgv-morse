"""Generate directly measured spectral, quadrature, and robustness diagnostics."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.interpolate import CubicSpline

from ..config import ReferenceConfig
from ..critical_points import cartesian_hessian
from ..dispersion import RingAnchoredSpectralEvaluator
from ..elasticity import cubic_family, isotropic_tensor
from ..green_response import (
    BranchNodeSample,
    FullWaveRadialEvaluatorFactory,
    PolarBranchSurface,
    build_tracked_surface,
    integrate_branch_response,
)
from ..mode_tracking import (
    TrackedMode,
    mass_mac,
    relative_eigengap,
    seed_tracked_mode,
    track_mode,
)
from ..spectral_plate import assemble_plate_matrices, solve_plate_modes
from ..zgv import ZGVPoint, find_s1_zgv
from .common import load_stage_dependency, validate_stage_inputs, write_stage_artifact
from .sensitivity import compute_sensitivity_arrays


_PROFILE = MappingProxyType(
    {
        "smoke": MappingProxyType(
            {
                "orders": (6, 8, 10, 12),
                "surface_order": 9,
                "radial_resolution": (33, 65, 129),
                "reference_radial": 257,
                "sensitivity_step": (0.04, 0.02, 0.01, 0.005),
            }
        ),
        "full": MappingProxyType(
            {
                "orders": (6, 8, 10, 12, 14),
                "surface_order": 10,
                "radial_resolution": (65, 129, 257),
                "reference_radial": 513,
                "sensitivity_step": (0.02, 0.01, 0.005, 0.0025),
            }
        ),
    }
)

_ANGULAR_RESOLUTION = (16, 32, 64)
_REFERENCE_ANGULAR = 128
_RESPONSE_TIME = np.linspace(1500.0, 6000.0, 61)
_ROBUSTNESS_EPSILON = 0.02
_BOUNDARY_WEIGHT_TOLERANCE = 5.0e-3


class _CachedCubicSymmetryFactory:
    """Reuse exactly C4v-equivalent rays without changing any surface node."""

    def __init__(self, factory: FullWaveRadialEvaluatorFactory) -> None:
        self.factory = factory
        self.samples: dict[tuple[float, int, float], BranchNodeSample] = {}

    def __call__(self, theta: float, radial_direction: int):
        quadrant = float(np.mod(theta, 0.5 * np.pi))
        canonical = float(np.round(min(quadrant, 0.5 * np.pi - quadrant), 14))
        evaluator = self.factory(canonical, radial_direction)
        direction = np.array([np.cos(canonical), np.sin(canonical)])

        def evaluate(kxy: np.ndarray) -> BranchNodeSample:
            radius = float(np.linalg.norm(kxy))
            key = (canonical, radial_direction, float(np.round(radius, 14)))
            if key not in self.samples:
                self.samples[key] = evaluator(radius * direction)
            return self.samples[key]

        return evaluate


def _matrix_defect(matrix: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(matrix)), np.finfo(float).tiny)
    return float(np.linalg.norm(matrix - matrix.conj().T) / scale)


def _rms_complex(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def _subsurface(
    reference: PolarBranchSurface,
    radial_count: int,
    angular_count: int,
) -> PolarBranchSurface:
    radial_step, radial_remainder = divmod(reference.q.size - 1, radial_count - 1)
    angular_step, angular_remainder = divmod(reference.theta.size, angular_count)
    if radial_remainder or angular_remainder:
        raise RuntimeError("registered convergence grids are not nested")
    radial = slice(None, None, radial_step)
    angular = slice(None, None, angular_step)
    return PolarBranchSurface(
        reference.q[radial],
        reference.theta[angular],
        reference.omega[radial, angular],
        reference.amplitude[radial, angular],
        reference.frequency_error[radial, angular],
        reference.amplitude_error[radial, angular],
        reference.relative_eigengap[radial, angular],
    )


def _interpolation_error(
    coarse: PolarBranchSurface,
    reference: PolarBranchSurface,
) -> float:
    extended_theta = np.concatenate((coarse.theta, [2.0 * np.pi]))
    extended_omega = np.concatenate((coarse.omega, coarse.omega[:, :1]), axis=1)
    angular = CubicSpline(
        extended_theta,
        extended_omega,
        axis=1,
        bc_type="periodic",
    )(reference.theta)
    interpolated = CubicSpline(coarse.q, angular, axis=0)(reference.q)
    return float(np.max(np.abs(interpolated - reference.omega)))


def _direct_surface_diagnostics(
    cfg: ReferenceConfig,
    exact: ZGVPoint,
    settings: MappingProxyType,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    radial_resolution = np.asarray(settings["radial_resolution"], dtype=np.int64)
    angular_resolution = np.asarray(_ANGULAR_RESOLUTION, dtype=np.int64)
    reference_radial = int(settings["reference_radial"])
    tensor = cubic_family(
        cfg.lam,
        cfg.mu,
        cfg.delta,
        _ROBUSTNESS_EPSILON,
    )[0]
    evaluator = RingAnchoredSpectralEvaluator(
        tensor,
        rho=cfg.rho,
        half_thickness=cfg.h,
        k0=exact.kappa0,
        target_omega=exact.omega0,
        order=int(settings["surface_order"]),
        num_modes=12,
        angular_sectors=8,
    )
    physical_factory = FullWaveRadialEvaluatorFactory(evaluator)
    cached_factory = _CachedCubicSymmetryFactory(physical_factory)
    sigma = cfg.window_sigma_over_k0 * exact.kappa0
    maximum_window_sigma = max(cfg.window_sensitivity) * exact.kappa0
    q_half_width = 1.5 * maximum_window_sigma
    q = np.linspace(-q_half_width, q_half_width, reference_radial)
    theta = 2.0 * np.pi * np.arange(_REFERENCE_ANGULAR) / _REFERENCE_ANGULAR
    reference = build_tracked_surface(cached_factory, q, theta, exact.kappa0)
    maximum_relative_frequency_uncertainty = float(
        np.max(reference.frequency_error / reference.omega)
    )
    minimum_anisotropic_eigengap = float(np.min(reference.relative_eigengap))
    if minimum_anisotropic_eigengap <= 10.0 * maximum_relative_frequency_uncertainty:
        raise RuntimeError("anisotropic robustness surface eigengap is unresolved")
    levels = tuple(
        _subsurface(reference, int(radial), int(angular))
        for radial, angular in zip(
            radial_resolution,
            angular_resolution,
            strict=True,
        )
    )
    reference_response = integrate_branch_response(
        reference,
        _RESPONSE_TIME,
        exact.kappa0,
        exact.omega0,
        cfg.source_radius_over_h,
        sigma,
        chunk_size=1,
        phase_limit=cfg.phase_error_tolerance,
    )
    reference_scale = _rms_complex(reference_response.demodulated)
    quadrature_error = np.empty(len(levels))
    interpolation_error = np.empty(len(levels))
    phase_error = np.empty(len(levels))
    for index, level in enumerate(levels):
        response = integrate_branch_response(
            level,
            _RESPONSE_TIME,
            exact.kappa0,
            exact.omega0,
            cfg.source_radius_over_h,
            sigma,
            chunk_size=1,
            phase_limit=cfg.phase_error_tolerance,
        )
        quadrature_error[index] = (
            _rms_complex(response.demodulated - reference_response.demodulated)
            / reference_scale
        )
        interpolation_error[index] = _interpolation_error(level, reference)
        phase_error[index] = float(_RESPONSE_TIME[-1]) * (
            interpolation_error[index]
            + float(np.max(level.frequency_error))
            + float(np.max(reference.frequency_error))
        )
    if not np.all(np.diff(quadrature_error) < 0.0):
        raise RuntimeError("full-wave quadrature error does not decrease")
    if np.max(quadrature_error[1:] / quadrature_error[:-1]) > 0.5:
        raise RuntimeError("full-wave quadrature convergence rate is unresolved")
    if quadrature_error[-1] > 0.05:
        raise RuntimeError("finest full-wave quadrature error exceeds five percent")
    if not np.all(np.diff(interpolation_error) < 0.0):
        raise RuntimeError("full-wave interpolation error does not decrease")
    if np.max(interpolation_error[1:] / interpolation_error[:-1]) > 0.5:
        raise RuntimeError("full-wave interpolation convergence rate is unresolved")
    if phase_error[-1] > cfg.phase_error_tolerance:
        raise RuntimeError("finest registered interpolation exceeds the phase budget")

    source_width = cfg.source_radius_over_h * np.array([0.8, 1.0, 1.2])
    window_width = np.array(
        [
            cfg.window_sensitivity[0],
            cfg.window_sigma_over_k0,
            cfg.window_sensitivity[1],
        ]
    )
    response_sensitivity = np.empty((source_width.size, window_width.size))
    for row, source in enumerate(source_width):
        for column, window in enumerate(window_width):
            response = integrate_branch_response(
                reference,
                _RESPONSE_TIME,
                exact.kappa0,
                exact.omega0,
                float(source),
                float(window) * exact.kappa0,
                chunk_size=1,
                phase_limit=cfg.phase_error_tolerance,
            )
            response_sensitivity[row, column] = (
                _rms_complex(response.demodulated) / reference_scale
            )
    if not np.isclose(response_sensitivity[1, 1], 1.0, rtol=5.0e-13, atol=5.0e-13):
        raise RuntimeError("source/window baseline response is inconsistent")
    if np.ptp(response_sensitivity) <= 0.0:
        raise RuntimeError("source/window reruns do not resolve any response variation")
    maximum_window_sigma = float(window_width[-1] * exact.kappa0)
    boundary_window_weight = float(
        np.exp(-(float(np.max(np.abs(reference.q))) / maximum_window_sigma) ** 8)
    )
    if boundary_window_weight > _BOUNDARY_WEIGHT_TOLERANCE:
        raise RuntimeError("widest robustness window is truncated by the radial surface")
    diagnostics: dict[str, object] = {
        "reference_surface_shape": list(reference.omega.shape),
        "robustness_epsilon": _ROBUSTNESS_EPSILON,
        "response_time": _RESPONSE_TIME.tolist(),
        "maximum_reference_frequency_uncertainty": float(
            np.max(reference.frequency_error)
        ),
        "maximum_relative_frequency_uncertainty": (
            maximum_relative_frequency_uncertainty
        ),
        "minimum_anisotropic_relative_eigengap": minimum_anisotropic_eigengap,
        "response_sensitivity_measurement": (
            "direct full-wave response RMS-norm ratios over 61 registered times"
        ),
        "response_sensitivity_scope": (
            "source/window response-norm robustness only; not evidence for the "
            "crossover exponents or Morse theorem"
        ),
        "widest_window_boundary_weight": boundary_window_weight,
        "boundary_weight_tolerance": _BOUNDARY_WEIGHT_TOLERANCE,
        "widest_window_sigma_over_k0": float(window_width[-1]),
    }
    return (
        angular_resolution,
        radial_resolution,
        quadrature_error,
        interpolation_error,
        phase_error,
        response_sensitivity,
        diagnostics,
    )


def _tracked_frequency(
    cfg: ReferenceConfig,
    order: int,
    reference: TrackedMode,
    radius: float,
    angle: float,
    epsilon: float,
) -> float:
    tensor = cubic_family(cfg.lam, cfg.mu, cfg.delta, epsilon)[0]
    matrices = assemble_plate_matrices(
        radius * np.cos(angle),
        radius * np.sin(angle),
        tensor,
        cfg.rho,
        cfg.h,
        order=order,
    )
    modes = solve_plate_modes(matrices, 12)
    return track_mode(
        reference,
        modes,
        min_mac=0.8,
        predicted_eigenvalue=reference.eigenvalue,
    ).omega


def _periodic_value(theta: np.ndarray, values: np.ndarray, query: float) -> float:
    return float(
        np.interp(
            np.mod(query, 2.0 * np.pi),
            np.concatenate((theta, [theta[0] + 2.0 * np.pi])),
            np.concatenate((values, [values[0]])),
        )
    )


def _sensitivity_fd_errors(
    cfg: ReferenceConfig,
    output_dir: Path,
    profile: str,
    exact: ZGVPoint,
    settings: MappingProxyType,
    input_artifacts: dict[str, dict[str, str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    required = {"theta", "B", "V4"}
    sensitivity = load_stage_dependency(
        "sensitivity",
        cfg,
        output_dir,
        profile,
        required,
        input_artifacts,
    )
    if sensitivity is None:
        sensitivity, _diagnostics = compute_sensitivity_arrays(cfg, profile)
    theta = np.asarray(sensitivity["theta"], dtype=np.float64)
    B = np.asarray(sensitivity["B"], dtype=np.float64)
    V4 = float(sensitivity["V4"])
    angles = (0.0, 0.25 * np.pi)
    analytic_B = np.array([_periodic_value(theta, B, angle) for angle in angles])
    order = int(settings["surface_order"])
    references: list[TrackedMode] = []
    tensor = isotropic_tensor(cfg.lam, cfg.mu)
    for angle in angles:
        matrices = assemble_plate_matrices(
            exact.kappa0 * np.cos(angle),
            exact.kappa0 * np.sin(angle),
            tensor,
            cfg.rho,
            cfg.h,
            order=order,
        )
        modes = solve_plate_modes(matrices, 12)
        references.append(
            seed_tracked_mode(
                modes,
                int(np.argmin(abs(modes.omega - exact.omega0))),
            )
        )
    steps = np.asarray(settings["sensitivity_step"], dtype=np.float64)
    V4_fd_error = np.empty(steps.size)
    B_estimates = np.empty((steps.size, steps.size, len(angles)))
    for step_index, epsilon_step in enumerate(steps):
        V_fd = np.empty(2)
        for angle_index, (angle, reference) in enumerate(
            zip(angles, references, strict=True)
        ):
            plus = _tracked_frequency(
                cfg,
                order,
                reference,
                exact.kappa0,
                angle,
                float(epsilon_step),
            )
            minus = _tracked_frequency(
                cfg,
                order,
                reference,
                exact.kappa0,
                angle,
                -float(epsilon_step),
            )
            V_fd[angle_index] = (plus - minus) / (2.0 * epsilon_step)
            for radial_index, radial_step in enumerate(steps):
                mixed = (
                    _tracked_frequency(
                        cfg,
                        order,
                        reference,
                        exact.kappa0 + radial_step,
                        angle,
                        float(epsilon_step),
                    )
                    - _tracked_frequency(
                        cfg,
                        order,
                        reference,
                        exact.kappa0 + radial_step,
                        angle,
                        -float(epsilon_step),
                    )
                    - _tracked_frequency(
                        cfg,
                        order,
                        reference,
                        exact.kappa0 - radial_step,
                        angle,
                        float(epsilon_step),
                    )
                    + _tracked_frequency(
                        cfg,
                        order,
                        reference,
                        exact.kappa0 - radial_step,
                        angle,
                        -float(epsilon_step),
                    )
                )
                B_estimates[step_index, radial_index, angle_index] = mixed / (
                    4.0 * epsilon_step * radial_step
                )
        V4_fd = 0.5 * (V_fd[0] - V_fd[1])
        V4_fd_error[step_index] = abs(V4_fd - V4) / abs(V4)
    B_error_matrix = np.linalg.norm(
        B_estimates - analytic_B[np.newaxis, np.newaxis, :],
        axis=2,
    ) / np.linalg.norm(analytic_B)
    B_fd_error = np.diag(B_error_matrix)
    if not np.all(np.isfinite(B_error_matrix)):
        raise RuntimeError("independent B finite-difference sweep is non-finite")
    if np.max(V4_fd_error[-2:]) >= cfg.sensitivity_match_tolerance:
        raise RuntimeError("direct V4 finite-difference tail misses its tolerance")
    if np.max(B_error_matrix[-2:, -2:]) >= cfg.sensitivity_match_tolerance:
        raise RuntimeError("independent B finite-difference tail misses its tolerance")
    if min(V4_fd_error[-2:]) >= V4_fd_error[0]:
        raise RuntimeError("V4 finite-difference sweep does not converge")
    if min(B_error_matrix[-2:, -1]) >= B_error_matrix[0, -1]:
        raise RuntimeError("B epsilon-step sweep does not converge")
    if min(B_error_matrix[-1, -2:]) >= B_error_matrix[-1, 0]:
        raise RuntimeError("B radial-step sweep does not converge")
    diagnostics: dict[str, object] = {
        "epsilon_steps": steps.tolist(),
        "radial_steps": steps.tolist(),
        "relative_B_error_matrix": B_error_matrix.tolist(),
        "B_axis_estimate_matrix": B_estimates[:, :, 0].tolist(),
        "B_diagonal_estimate_matrix": B_estimates[:, :, 1].tolist(),
        "analytic_B_axis_diagonal": analytic_B.tolist(),
        "tail_maximum_relative_B_error": float(
            np.max(B_error_matrix[-2:, -2:])
        ),
        "sweep_design": (
            "independent tensor-product epsilon and radial centered differences"
        ),
    }
    return steps, V4_fd_error, B_fd_error, diagnostics


def run(cfg: ReferenceConfig, output_dir: Path, profile: str) -> Path:
    """Run the registered convergence and robustness stage."""

    config, directory, selected_profile = validate_stage_inputs(cfg, output_dir, profile)
    settings = _PROFILE[selected_profile]
    exact = find_s1_zgv(config, dps=60 if selected_profile == "smoke" else 80)
    tensor = isotropic_tensor(config.lam, config.mu)
    coarse_orders = np.asarray(settings["orders"], dtype=np.int64)
    polynomial_order = coarse_orders + 4
    omega0_error = np.empty(coarse_orders.size)
    kappa0_error = np.empty(coarse_orders.size)
    curvature_error = np.empty(coarse_orders.size)
    eigen_residual = np.empty(coarse_orders.size)
    hermitian_residual = np.empty(coarse_orders.size)
    mass_orthogonality = np.empty(coarse_orders.size)
    eigengap = np.empty(coarse_orders.size)

    for index, raw_order in enumerate(coarse_orders):
        coarse_order = int(raw_order)
        fine_order = coarse_order + 4
        evaluator = RingAnchoredSpectralEvaluator(
            tensor,
            rho=config.rho,
            half_thickness=config.h,
            k0=exact.kappa0,
            target_omega=exact.omega0,
            order=coarse_order,
            num_modes=12,
            angular_sectors=8,
        )
        anchor = evaluator(np.array([exact.kappa0, 0.0]))
        radial_shift = -anchor.gradient[0] / exact.curvature_a
        corrected = evaluator(np.array([exact.kappa0 + radial_shift, 0.0]))
        hessian, _uncertainty = cartesian_hessian(
            evaluator,
            np.array([exact.kappa0, 0.0]),
            2.0e-3,
        )
        omega0_error[index] = abs(corrected.omega - exact.omega0) / exact.omega0
        kappa0_error[index] = abs(radial_shift) / exact.kappa0
        curvature_error[index] = abs(hessian[0, 0] - exact.curvature_a) / exact.curvature_a

        matrices = assemble_plate_matrices(
            exact.kappa0,
            0.0,
            tensor,
            config.rho,
            config.h,
            order=fine_order,
        )
        modes = solve_plate_modes(matrices, 12)
        mode_index = int(np.argmin(abs(modes.omega - exact.omega0)))
        vector = modes.vectors[:, mode_index]
        eigenvalue = modes.eigenvalues[mode_index]
        left = matrices.stiffness @ vector
        right = eigenvalue * (matrices.mass @ vector)
        eigen_residual[index] = np.linalg.norm(left - right) / max(
            np.linalg.norm(left) + np.linalg.norm(right),
            np.finfo(float).tiny,
        )
        hermitian_residual[index] = max(
            _matrix_defect(matrices.stiffness),
            _matrix_defect(matrices.mass),
        )
        gram = modes.vectors.conj().T @ matrices.mass @ modes.vectors
        mass_orthogonality[index] = np.linalg.norm(gram - np.eye(gram.shape[0]))
        eigengap[index] = relative_eigengap(modes.eigenvalues, mode_index)

    if omega0_error[-1] >= config.isotropic_match_tolerance:
        raise RuntimeError("final spectral ZGV frequency misses its registered tolerance")
    if kappa0_error[-1] >= config.isotropic_match_tolerance:
        raise RuntimeError("final spectral ZGV wavenumber misses its registered tolerance")
    if curvature_error[-1] >= config.curvature_match_tolerance:
        raise RuntimeError("final spectral curvature misses its registered tolerance")
    if float(np.max(eigen_residual)) >= config.eigen_residual_tolerance:
        raise RuntimeError("spectral eigen residual misses its registered tolerance")
    if float(np.max(hermitian_residual)) >= 1.0e-12:
        raise RuntimeError("spectral Hermitian defect exceeds the registered gate")
    if float(np.max(mass_orthogonality)) >= 1.0e-10:
        raise RuntimeError("spectral mass orthogonality exceeds the registered gate")
    if float(np.min(eigengap)) <= 10.0 * max(
        float(np.max(eigen_residual)),
        np.finfo(float).eps,
    ):
        raise RuntimeError("spectral eigengap is unresolved against numerical error")

    (
        angular_resolution,
        radial_resolution,
        quadrature_error,
        interpolation_error,
        phase_error,
        response_sensitivity,
        direct_diagnostics,
    ) = _direct_surface_diagnostics(config, exact, settings)
    input_artifacts: dict[str, dict[str, str]] = {}
    sensitivity_step, V4_fd_error, B_fd_error, fd_diagnostics = _sensitivity_fd_errors(
        config,
        directory,
        selected_profile,
        exact,
        settings,
        input_artifacts,
    )

    tracking_kappa = np.linspace(exact.kappa0 - 0.05, exact.kappa0 + 0.05, 11)
    tracking_mac = np.empty(tracking_kappa.size)
    tracking_gap = np.empty(tracking_kappa.size)
    anchor_matrices = assemble_plate_matrices(
        exact.kappa0,
        0.0,
        tensor,
        config.rho,
        config.h,
        order=10,
    )
    anchor_modes = solve_plate_modes(anchor_matrices, 12)
    previous = seed_tracked_mode(
        anchor_modes,
        int(np.argmin(abs(anchor_modes.omega - exact.omega0))),
    )
    reference = previous
    for index, kappa in enumerate(tracking_kappa):
        matrices = assemble_plate_matrices(
            float(kappa),
            0.0,
            tensor,
            config.rho,
            config.h,
            order=10,
        )
        modes = solve_plate_modes(matrices, 12)
        tracked = track_mode(
            previous,
            modes,
            min_mac=0.8,
            predicted_eigenvalue=previous.eigenvalue,
        )
        tracking_mac[index] = mass_mac(reference.vector, tracked.vector, matrices.mass)
        tracking_gap[index] = tracked.eigengap
        previous = tracked
    if float(np.min(tracking_mac)) < 0.99:
        raise RuntimeError("tracked branch MAC falls below the registered robustness gate")
    if float(np.min(tracking_gap)) <= 10.0 * float(np.max(eigen_residual)):
        raise RuntimeError("tracked branch gap is unresolved against eigen residuals")

    source_width = config.source_radius_over_h * np.array([0.8, 1.0, 1.2])
    window_width = np.array(
        [
            config.window_sensitivity[0],
            config.window_sigma_over_k0,
            config.window_sensitivity[1],
        ]
    )
    arrays = {
        "polynomial_order": polynomial_order,
        "omega0_error": omega0_error,
        "kappa0_error": kappa0_error,
        "curvature_error": curvature_error,
        "eigen_residual": eigen_residual,
        "hermitian_residual": hermitian_residual,
        "mass_orthogonality": mass_orthogonality,
        "eigengap": eigengap,
        "angular_resolution": angular_resolution,
        "radial_resolution": radial_resolution,
        "quadrature_error": quadrature_error,
        "interpolation_error": interpolation_error,
        "phase_error": phase_error,
        "sensitivity_step": sensitivity_step,
        "V4_fd_error": V4_fd_error,
        "B_fd_error": B_fd_error,
        "tracking_kappa": tracking_kappa,
        "tracking_mac": tracking_mac,
        "tracking_gap": tracking_gap,
        "source_width": source_width,
        "window_width": window_width,
        "response_sensitivity": response_sensitivity,
    }
    units = {key: "1" for key in arrays}
    units.update(
        {
            "interpolation_error": "c_T/h",
            "phase_error": "rad",
            "tracking_kappa": "1",
            "source_width": "h",
            "window_width": "sigma/k0",
        }
    )
    tolerances = {
        "finest_quadrature_relative_error": float(quadrature_error[-1]),
        "finest_interpolation_frequency_error": float(interpolation_error[-1]),
        "finest_accumulated_phase_error": float(phase_error[-1]),
        "configured_phase_error": config.phase_error_tolerance,
        "minimum_eigengap": float(np.min(eigengap)),
        "maximum_eigen_residual": float(np.max(eigen_residual)),
        "configured_eigen_residual": config.eigen_residual_tolerance,
        "final_relative_omega0_error": float(omega0_error[-1]),
        "final_relative_kappa0_error": float(kappa0_error[-1]),
        "final_relative_curvature_error": float(curvature_error[-1]),
        "configured_isotropic_match": config.isotropic_match_tolerance,
        "configured_curvature_match": config.curvature_match_tolerance,
        "maximum_hermitian_residual": float(np.max(hermitian_residual)),
        "maximum_mass_orthogonality": float(np.max(mass_orthogonality)),
        "minimum_tracking_mac": float(np.min(tracking_mac)),
        "minimum_tracking_gap": float(np.min(tracking_gap)),
        "final_V4_fd_relative_error": float(V4_fd_error[-1]),
        "final_B_fd_relative_error": float(B_fd_error[-1]),
    }
    return write_stage_artifact(
        "convergence",
        config,
        directory,
        selected_profile,
        arrays,
        units,
        tolerances,
        input_artifacts=input_artifacts,
        extra_metadata={
            "direct_measurement_provenance": direct_diagnostics,
            "finite_difference_sweep": fd_diagnostics,
            "polynomial_order_semantics": (
                "reported fine GLL order p; each tracked evaluator compares p-4 and p"
            ),
            "no_synthetic_convergence_values": True,
        },
    )
