"""Generate resolution-checked full-wave Bessel and exact-Morse responses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.special import j0

from ..asymptotics import (
    UniformParameters,
    build_morse_contribution,
    morse_stationary_phase_response,
    scale_transition_response,
    uniform_bessel_response,
)
from ..config import ReferenceConfig
from ..critical_points import Annulus, locate_critical_points, verify_annular_exhaustion
from ..dispersion import RingAnchoredSpectralEvaluator
from ..elasticity import cubic_family, isotropic_tensor
from ..green_response import (
    COMPACT_TAPER_PLATEAU_SIGMA,
    DIRECT_RADIAL_SUPPORT_SIGMA,
    BranchNodeSample,
    FullWaveRadialEvaluatorFactory,
    build_tracked_surface,
    compact_radial_taper,
    integrate_branch_response,
    verify_registered_grid_convergence,
)
from ..validation import fit_power_law, operational_crossover_time, rms_envelope
from ..zgv import find_s1_zgv
from .common import load_stage_dependency, validate_stage_inputs, write_stage_artifact
from .sensitivity import compute_sensitivity_arrays


_PROFILE = MappingProxyType(
    {
        "smoke": MappingProxyType(
            {
                "epsilon": (0.01, 0.02, 0.08),
                "order": 9,
                "q_nodes": 129,
                "fixed_q_nodes": 129,
                "theta_nodes": 32,
                "fixed_theta_nodes": 64,
            }
        ),
        "full": MappingProxyType(
            {
                "epsilon": (0.005, 0.01, 0.02, 0.04, 0.08),
                "order": 10,
                "q_nodes": 129,
                "fixed_q_nodes": 129,
                "theta_nodes": 32,
                "fixed_theta_nodes": 64,
            }
        ),
    }
)

_EARLY_TAU_WINDOW = (0.10, 0.30)
_FIXED_EPSILON = 0.08
_MORSE_COMPARISON_START = 1500.0
_MORSE_COMPARISON_STOP = 10200.0
_MORSE_COHERENCE_THRESHOLD = 0.30
_UNIFORM_TAU_MAXIMUM = 2.0
_UNIFORM_EPSILON_MAXIMUM = 0.04
_MORSE_SEARCH_MARGIN_SIGMA = 0.05
_REGISTERED_MORSE_HALF_INTERVALS = 4


@dataclass(frozen=True, slots=True)
class _ResponseDomainControl:
    """Shared compact-support and stationary-search radial domain."""

    window_sigma: float
    taper_plateau_half_width: float
    direct_support_half_width: float
    morse_search_margin: float
    morse_search_half_width: float
    registered_annulus_half_width: float
    morse_radial_nodes: int

    def metadata(self, k0: float) -> dict[str, object]:
        """Return the dimensionless radial-domain control record for the sidecar."""

        return {
            "window": (
                "exp[-(q/window_sigma)^8] multiplied by a deterministic C-infinity compact taper"
            ),
            "compact_taper_plateau_abs_q_over_sigma": (
                self.taper_plateau_half_width / self.window_sigma
            ),
            "direct_support_abs_q_over_sigma": (self.direct_support_half_width / self.window_sigma),
            "direct_support_abs_q_over_k0": self.direct_support_half_width / k0,
            "endpoint_taper_value": 0.0,
            "endpoint_is_flat_to_all_derivative_orders": True,
            "registered_annulus_abs_q_over_k0": (self.registered_annulus_half_width / k0),
            "morse_search_margin_abs_q_over_sigma": (self.morse_search_margin / self.window_sigma),
            "morse_search_abs_q_over_sigma": (self.morse_search_half_width / self.window_sigma),
            "morse_search_abs_q_over_k0": self.morse_search_half_width / k0,
            "morse_search_inner_radius_over_k0": 1.0 - self.morse_search_half_width / k0,
            "morse_search_outer_radius_over_k0": 1.0 + self.morse_search_half_width / k0,
            "morse_search_strictly_contains_direct_support": (
                self.morse_search_half_width > self.direct_support_half_width
            ),
            "morse_search_radial_nodes": self.morse_radial_nodes,
            "morse_search_angular_nodes": 32,
            "morse_exhaustion_boundary_nodes": [16, 32],
        }


def _response_domain_control(
    *,
    k0: float,
    window_sigma: float,
    registered_annulus_half_width: float,
) -> _ResponseDomainControl:
    """Construct a search domain that strictly contains direct-response support."""

    values = {
        "k0": float(k0),
        "window_sigma": float(window_sigma),
        "registered_annulus_half_width": float(registered_annulus_half_width),
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("radial-domain scales must be finite and positive")
    plateau = COMPACT_TAPER_PLATEAU_SIGMA * values["window_sigma"]
    support = DIRECT_RADIAL_SUPPORT_SIGMA * values["window_sigma"]
    minimum_margin = _MORSE_SEARCH_MARGIN_SIGMA * values["window_sigma"]
    search = max(values["registered_annulus_half_width"], support + minimum_margin)
    if search >= values["k0"] or not search > support:
        raise ValueError(
            "Morse search must strictly contain compact support and retain a positive inner radius"
        )
    registered_spacing = values["registered_annulus_half_width"] / _REGISTERED_MORSE_HALF_INTERVALS
    half_intervals = int(np.ceil(search / registered_spacing))
    radial_nodes = 2 * half_intervals + 1
    return _ResponseDomainControl(
        values["window_sigma"],
        plateau,
        support,
        search - support,
        search,
        values["registered_annulus_half_width"],
        radial_nodes,
    )


def _compactly_tapered_node(
    node: BranchNodeSample,
    radial_offset: float,
    window_sigma: float,
) -> BranchNodeSample:
    """Apply the direct quadrature's compact taper to one Morse modal node."""

    if not isinstance(node, BranchNodeSample):
        raise TypeError("node must be a BranchNodeSample")
    taper = float(compact_radial_taper(np.array([radial_offset]), window_sigma)[0])
    return BranchNodeSample(
        node.omega,
        taper * node.amplitude,
        node.frequency_uncertainty,
        taper * node.amplitude_uncertainty,
        node.relative_eigengap,
    )


class _CachedCubicSymmetryFactory:
    """Evaluate one C4v irreducible angular wedge and reuse exact images."""

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


def _load_sensitivity(
    cfg: ReferenceConfig,
    output_dir: Path,
    profile: str,
    input_artifacts: dict[str, dict[str, str]] | None = None,
) -> dict[str, np.ndarray]:
    arrays = load_stage_dependency(
        "sensitivity",
        cfg,
        output_dir,
        profile,
        {"V0", "V4"},
        input_artifacts,
    )
    if arrays is None:
        arrays, _diagnostics = compute_sensitivity_arrays(cfg, profile)
    return arrays


def _isotropic_modal_amplitude(
    cfg: ReferenceConfig,
    k0: float,
    omega0: float,
    order: int,
) -> complex:
    evaluator = RingAnchoredSpectralEvaluator(
        isotropic_tensor(cfg.lam, cfg.mu),
        rho=cfg.rho,
        half_thickness=cfg.h,
        k0=k0,
        target_omega=omega0,
        order=order,
        num_modes=12,
        angular_sectors=8,
    )
    node = FullWaveRadialEvaluatorFactory(evaluator)(0.0, 1)(np.array([k0, 0.0]))
    return node.amplitude


def _measured_crossover_time(
    time: np.ndarray,
    scaled_response: np.ndarray,
    predicted_time: float,
) -> float:
    """Measure the first 0.9-envelope crossing in a preregistered bracket."""

    magnitude = np.abs(scaled_response)
    registered = (time >= 0.5 * predicted_time) & (time <= 1.5 * predicted_time)
    indices = np.flatnonzero(registered)
    for left_index, right_index in zip(indices[:-1], indices[1:], strict=True):
        left = float(magnitude[left_index] - 0.9)
        right = float(magnitude[right_index] - 0.9)
        if left >= 0.0 and right <= 0.0:
            fraction = left / (left - right)
            return float(time[left_index] + fraction * (time[right_index] - time[left_index]))
    raise RuntimeError("full-wave response does not resolve the registered crossover")


def _registered_fullwave_convergence(
    physical_factory: FullWaveRadialEvaluatorFactory,
    q: np.ndarray,
    theta: np.ndarray,
    time: np.ndarray,
    k0: float,
    carrier: float,
    source_radius: float,
    window_sigma: float,
    phase_limit: float,
):
    """Build three independent nested surfaces and check their responses."""

    q_fine = np.empty(2 * q.size - 1)
    q_fine[::2] = q
    q_fine[1::2] = 0.5 * (q[:-1] + q[1:])
    q_finest = np.empty(2 * q_fine.size - 1)
    q_finest[::2] = q_fine
    q_finest[1::2] = 0.5 * (q_fine[:-1] + q_fine[1:])
    theta_fine = 2.0 * np.pi * np.arange(2 * theta.size) / (2 * theta.size)
    theta_finest = 2.0 * np.pi * np.arange(4 * theta.size) / (4 * theta.size)
    surfaces = tuple(
        build_tracked_surface(
            _CachedCubicSymmetryFactory(physical_factory),
            q_level,
            theta_level,
            k0,
        )
        for q_level, theta_level in zip(
            (q, q_fine, q_finest),
            (theta, theta_fine, theta_finest),
            strict=True,
        )
    )
    responses = tuple(
        integrate_branch_response(
            surface,
            time,
            k0,
            carrier,
            source_radius,
            window_sigma,
            chunk_size=2,
            phase_limit=phase_limit,
        )
        for surface in surfaces
    )
    return verify_registered_grid_convergence(
        surfaces,
        responses,
        phase_limit=phase_limit,
    )


def _morse_coherence(time: np.ndarray, contributions: list) -> np.ndarray:
    """Return cancellation coherence using only the exact-Morse phasors."""

    coefficients: list[complex] = []
    for contribution in contributions:
        eigenvalues = np.linalg.eigvalsh(contribution.hessian)
        signature = int(np.count_nonzero(eigenvalues > 0.0) - np.count_nonzero(eigenvalues < 0.0))
        coefficients.append(
            contribution.amplitude
            * np.exp(-0.25j * np.pi * signature)
            / np.sqrt(abs(np.prod(eigenvalues)))
        )
    phasor_sum = sum(
        coefficient * np.exp(-1j * contribution.omega * time)
        for coefficient, contribution in zip(
            coefficients,
            contributions,
            strict=True,
        )
    )
    return np.abs(phasor_sum) / sum(map(abs, coefficients))


def _beat_period_fit(
    time: np.ndarray,
    response: np.ndarray,
    frequency_separation: float,
) -> tuple[float, np.ndarray, list[float], list[float], list[list[int]]]:
    """Fit measured one-beat RMS values without response-derived windows."""

    period = 2.0 * np.pi / frequency_separation
    centers: list[float] = []
    values: list[float] = []
    bins: list[list[int]] = []
    selected = np.zeros(time.shape, dtype=np.bool_)
    left = _MORSE_COMPARISON_START
    while left + period <= float(time[-1]):
        mask = (time >= left) & (time < left + period)
        if np.count_nonzero(mask) < 3:
            raise RuntimeError("fixed-epsilon beat window is under-resolved")
        selected |= mask
        centers.append(float(np.exp(np.mean(np.log(time[mask])))))
        values.append(float(np.sqrt(np.mean(np.abs(response[mask]) ** 2))))
        indices = np.flatnonzero(mask)
        bins.append([int(indices[0]), int(indices[-1] + 1)])
        left += period
    if len(centers) < 4:
        raise RuntimeError("fixed-epsilon slope requires four complete beat periods")
    slope = float(np.polyfit(np.log(centers), np.log(values), 1)[0])
    return slope, selected, centers, values, bins


def _beat_rms_envelope(
    time: np.ndarray,
    response: np.ndarray,
    frequency_separation: float,
) -> np.ndarray:
    """Return a moving one-beat RMS envelope for display and local slopes."""

    samples = max(3, int(round(2.0 * np.pi / frequency_separation / np.diff(time)[0])))
    if samples % 2 == 0:
        samples += 1
    samples = min(samples, time.size if time.size % 2 else time.size - 1)
    return rms_envelope(response, samples)


def run(cfg: ReferenceConfig, output_dir: Path, profile: str) -> Path:
    """Run the registered full-dispersion transition-response stage."""

    config, directory, selected_profile = validate_stage_inputs(cfg, output_dir, profile)
    settings = _PROFILE[selected_profile]
    exact = find_s1_zgv(config, dps=60 if selected_profile == "smoke" else 80)
    input_artifacts: dict[str, dict[str, str]] = {}
    sensitivity = _load_sensitivity(
        config,
        directory,
        selected_profile,
        input_artifacts,
    )
    V0 = float(sensitivity["V0"])
    V4 = float(sensitivity["V4"])
    if V4 == 0.0:
        raise RuntimeError("the registered cubic sensitivity V4 must be nonzero")
    epsilon = np.asarray(settings["epsilon"], dtype=np.float64)
    if epsilon.ndim != 1 or epsilon.size < 3 or np.any(epsilon <= 0.0):
        raise RuntimeError("Green workflow requires multiple positive epsilon values")
    if not np.any(epsilon == _FIXED_EPSILON):
        raise RuntimeError("Green workflow must include the registered fixed epsilon")
    order = int(settings["order"])
    window_sigma = config.window_sigma_over_k0 * exact.kappa0
    domain_control = _response_domain_control(
        k0=exact.kappa0,
        window_sigma=window_sigma,
        registered_annulus_half_width=config.annulus_fraction * exact.kappa0,
    )
    q = np.linspace(
        -domain_control.direct_support_half_width,
        domain_control.direct_support_half_width,
        int(settings["q_nodes"]),
    )
    theta = 2.0 * np.pi * np.arange(int(settings["theta_nodes"])) / int(settings["theta_nodes"])
    time = 400.0 + 25.0 * np.arange(505)

    modal_amplitude = _isotropic_modal_amplitude(
        config,
        exact.kappa0,
        exact.omega0,
        order,
    )
    source_weight = np.exp(-0.25 * config.source_radius_over_h**2 * exact.kappa0**2)
    annulus = Annulus(exact.kappa0, domain_control.morse_search_half_width)

    full_rows: list[np.ndarray] = []
    normal_rows: list[np.ndarray] = []
    morse_rows: list[np.ndarray] = []
    tau_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    scaled_rows: list[np.ndarray] = []
    envelope_rows: list[np.ndarray] = []
    rms_rows: list[np.ndarray] = []
    local_slope_rows: list[np.ndarray] = []
    omega_min: list[float] = []
    omega_saddle: list[float] = []
    phase_errors: list[float] = []
    crossover_times: list[float] = []
    collapse_errors: list[float] = []
    response_grid_errors: list[list[float]] = []
    nested_frequency_errors: list[list[float]] = []
    accumulated_phase_errors: list[list[float]] = []
    minimum_branch_eigengaps: list[float] = []
    maximum_relative_frequency_uncertainties: list[float] = []
    early_masks = np.zeros((epsilon.size, time.size), dtype=np.bool_)
    late_masks = np.zeros((epsilon.size, time.size), dtype=np.bool_)
    fixed_morse_error = np.nan
    fixed_morse_max_error = np.nan
    cancellation_region_normalized_rms = np.nan
    cancellation_fraction = np.nan
    cancellation_indices: list[int] = []
    fixed_morse_coherence: list[float] = []
    fixed_morse_normalization = np.nan
    late_fit_centers: list[float] = []
    late_fit_rms: list[float] = []
    late_fit_bins: list[list[int]] = []
    early_grid_slopes: list[float] = []
    late_grid_slopes: list[float] = []
    early_slope = np.nan
    late_slope = np.nan
    stationary_point_counts: list[int] = []
    morse_contribution_counts: list[int] = []
    boundary_is_noncritical: list[bool] = []
    gradient_index_closes: list[bool] = []
    minimum_boundary_gradients: list[float] = []
    maximum_boundary_gradient_uncertainties: list[float] = []
    fixed_morse_stationary_offsets_over_sigma: list[float] = []

    for row_index, epsilon_value in enumerate(epsilon):
        tensor = cubic_family(
            config.lam,
            config.mu,
            config.delta,
            float(epsilon_value),
        )[0]
        evaluator = RingAnchoredSpectralEvaluator(
            tensor,
            rho=config.rho,
            half_thickness=config.h,
            k0=exact.kappa0,
            target_omega=exact.omega0,
            order=order,
            num_modes=12,
            angular_sectors=8,
        )
        physical_factory = FullWaveRadialEvaluatorFactory(evaluator)
        carrier = exact.omega0 + float(epsilon_value) * V0
        q_row = (
            np.linspace(
                -domain_control.direct_support_half_width,
                domain_control.direct_support_half_width,
                int(settings["fixed_q_nodes"]),
            )
            if epsilon_value == _FIXED_EPSILON
            else q
        )
        theta_row = (
            2.0
            * np.pi
            * np.arange(int(settings["fixed_theta_nodes"]))
            / int(settings["fixed_theta_nodes"])
            if epsilon_value == _FIXED_EPSILON
            else theta
        )
        convergence = _registered_fullwave_convergence(
            physical_factory,
            q_row,
            theta_row,
            time,
            exact.kappa0,
            carrier,
            config.source_radius_over_h,
            window_sigma,
            config.phase_error_tolerance,
        )
        minimum_branch_eigengap = min(
            float(np.min(surface.relative_eigengap)) for surface in convergence.surfaces
        )
        maximum_relative_frequency_uncertainty = max(
            float(np.max(surface.frequency_error / surface.omega))
            for surface in convergence.surfaces
        )
        if minimum_branch_eigengap <= 10.0 * maximum_relative_frequency_uncertainty:
            raise RuntimeError(
                f"anisotropic branch eigengap is unresolved at epsilon={epsilon_value:.8g}"
            )
        direct = convergence.finest_response
        G_full_row = direct.analytic_signal()

        parameters = UniformParameters(
            exact.omega0,
            exact.kappa0,
            exact.curvature_a,
            float(epsilon_value),
            V0,
            V4,
            modal_amplitude * source_weight,
        )
        G_normal_form_row = uniform_bessel_response(time, parameters)
        scaled_row = scale_transition_response(time, G_full_row, parameters)
        tau_row = float(epsilon_value) * V4 * time
        target_row = j0(tau_row)

        points = locate_critical_points(
            evaluator,
            annulus,
            n_radial=domain_control.morse_radial_nodes,
            n_theta=32,
            hessian_step=2.0e-3 if order == 9 else 1.0e-3,
        )
        exhaustion = verify_annular_exhaustion(evaluator, annulus, points, 16)
        if (
            len(points) != 8
            or sum(point.kind == "minimum" for point in points) != 4
            or sum(point.kind == "saddle" for point in points) != 4
            or not exhaustion.boundary_is_noncritical
            or not exhaustion.index_closes
        ):
            raise RuntimeError(
                f"full-wave Morse set failed resolution checks at epsilon={epsilon_value:.8g}"
            )
        contributions = []
        for point in points:
            radial_direction = -1 if point.radius < exact.kappa0 else 1
            node = physical_factory(point.theta, radial_direction)(np.array([point.kx, point.ky]))
            tapered_node = _compactly_tapered_node(
                node,
                point.radius - exact.kappa0,
                window_sigma,
            )
            contributions.append(
                build_morse_contribution(
                    point,
                    tapered_node,
                    exact.kappa0,
                    config.source_radius_over_h,
                    window_sigma,
                )
            )
        stationary_point_counts.append(len(points))
        morse_contribution_counts.append(len(contributions))
        boundary_is_noncritical.append(exhaustion.boundary_is_noncritical)
        gradient_index_closes.append(exhaustion.index_closes)
        minimum_boundary_gradients.append(exhaustion.minimum_boundary_gradient)
        maximum_boundary_gradient_uncertainties.append(exhaustion.maximum_gradient_uncertainty)
        if len(contributions) != len(points):
            raise RuntimeError("every stationary point must feed the exact-Morse sum")
        if epsilon_value == _FIXED_EPSILON:
            fixed_morse_stationary_offsets_over_sigma = [
                (point.radius - exact.kappa0) / window_sigma for point in points
            ]
        morse = morse_stationary_phase_response(
            time,
            contributions,
            carrier,
            phase_limit=config.phase_error_tolerance,
        )
        G_morse_row = morse.analytic_signal()
        minimum_frequency = min(point.omega for point in points if point.kind == "minimum")
        saddle_frequency = min(point.omega for point in points if point.kind == "saddle")
        separation = abs(saddle_frequency - minimum_frequency)

        envelope_row = np.abs(G_full_row)
        rms_row = _beat_rms_envelope(time, direct.demodulated, separation)
        local_slope_row = np.gradient(
            np.log(np.maximum(rms_row, np.finfo(float).tiny)),
            np.log(time),
        )
        predicted_crossover = operational_crossover_time(float(epsilon_value), V4)
        measured_crossover = _measured_crossover_time(
            time,
            scaled_row,
            predicted_crossover,
        )
        uniform_mask = (time >= _MORSE_COMPARISON_START) & (np.abs(tau_row) <= _UNIFORM_TAU_MAXIMUM)
        if epsilon_value <= _UNIFORM_EPSILON_MAXIMUM:
            if np.count_nonzero(uniform_mask) < 10:
                raise RuntimeError("uniform comparison window is under-resolved")
            collapse_errors.append(
                float(np.max(np.abs(scaled_row[uniform_mask] - target_row[uniform_mask])))
            )
        else:
            collapse_errors.append(np.nan)

        if row_index == 0:
            early_mask = (
                (time >= _MORSE_COMPARISON_START)
                & (np.abs(tau_row) >= _EARLY_TAU_WINDOW[0])
                & (np.abs(tau_row) <= _EARLY_TAU_WINDOW[1])
            )
            early_fit = fit_power_law(time, envelope_row, early_mask)
            early_slope = early_fit.slope
            early_masks[row_index] = early_mask
            early_grid_slopes = [
                fit_power_law(
                    time,
                    np.abs(response.demodulated),
                    early_mask,
                ).slope
                for response in convergence.responses
            ]
            if abs(early_grid_slopes[-1] - early_grid_slopes[-2]) > 0.05:
                raise RuntimeError(f"early decay slope is not grid-converged: {early_grid_slopes}")

        if epsilon_value == _FIXED_EPSILON:
            grid_fit_records = [
                _beat_period_fit(time, response.demodulated, separation)
                for response in convergence.responses
            ]
            late_grid_slopes = [record[0] for record in grid_fit_records]
            (
                late_slope,
                late_mask,
                late_fit_centers,
                late_fit_rms,
                late_fit_bins,
            ) = grid_fit_records[-1]
            if abs(late_grid_slopes[-1] - late_grid_slopes[-2]) > 0.05:
                raise RuntimeError(f"late decay slope is not grid-converged: {late_grid_slopes}")
            late_masks[row_index] = late_mask
            coherence = _morse_coherence(time, contributions)
            comparison_base = (time >= _MORSE_COMPARISON_START) & (time <= _MORSE_COMPARISON_STOP)
            cancellation_mask = comparison_base & (coherence < _MORSE_COHERENCE_THRESHOLD)
            comparison_mask = comparison_base & ~cancellation_mask
            comparison_scale = float(
                np.sqrt(np.mean(np.abs(morse.demodulated[comparison_mask]) ** 2))
            )
            fixed_morse_normalization = comparison_scale
            complex_error = direct.demodulated - morse.demodulated
            fixed_morse_error = float(
                np.sqrt(np.mean(np.abs(complex_error[comparison_mask]) ** 2)) / comparison_scale
            )
            fixed_morse_max_error = float(
                np.max(np.abs(complex_error[comparison_mask])) / comparison_scale
            )
            cancellation_region_normalized_rms = float(
                np.sqrt(np.mean(np.abs(complex_error[cancellation_mask]) ** 2)) / comparison_scale
            )
            cancellation_fraction = float(
                np.count_nonzero(cancellation_mask) / np.count_nonzero(comparison_base)
            )
            cancellation_indices = np.flatnonzero(cancellation_mask).astype(int).tolist()
            fixed_morse_coherence = coherence.tolist()

        maximum_phase_error = max(
            float(np.max(convergence.accumulated_phase_errors)),
            morse.maximum_accumulated_phase_error,
        )
        if maximum_phase_error > config.phase_error_tolerance:
            raise RuntimeError("full-wave response exceeds the configured phase budget")

        full_rows.append(G_full_row)
        normal_rows.append(G_normal_form_row)
        morse_rows.append(G_morse_row)
        tau_rows.append(tau_row)
        target_rows.append(target_row)
        scaled_rows.append(scaled_row)
        envelope_rows.append(envelope_row)
        rms_rows.append(rms_row)
        local_slope_rows.append(local_slope_row)
        omega_min.append(minimum_frequency)
        omega_saddle.append(saddle_frequency)
        phase_errors.append(maximum_phase_error)
        crossover_times.append(measured_crossover)
        response_grid_errors.append(convergence.complex_response_errors.tolist())
        nested_frequency_errors.append(convergence.nested_frequency_errors.tolist())
        accumulated_phase_errors.append(convergence.accumulated_phase_errors.tolist())
        minimum_branch_eigengaps.append(minimum_branch_eigengap)
        maximum_relative_frequency_uncertainties.append(maximum_relative_frequency_uncertainty)

    finite_collapse_errors = np.asarray(collapse_errors)[np.isfinite(collapse_errors)]
    maximum_collapse_error = float(np.max(finite_collapse_errors))
    if maximum_collapse_error > 0.08:
        raise RuntimeError(
            "full-wave Bessel collapse error without fitted alignment "
            f"{maximum_collapse_error:.6g} exceeds 0.08"
        )
    if not np.isfinite(early_slope) or abs(early_slope + 0.5) > 0.05:
        raise RuntimeError("measured early-time full-wave slope misses -1/2")
    if not np.isfinite(late_slope) or abs(late_slope + 1.0) > 0.05:
        raise RuntimeError("measured fixed-epsilon full-wave slope misses -1")
    if (
        not np.isfinite(fixed_morse_error)
        or fixed_morse_error > 0.055
        or fixed_morse_max_error > 0.17
        or cancellation_region_normalized_rms > 0.09
    ):
        raise RuntimeError(
            "fixed-epsilon exact-Morse comparison fails its registered gate: "
            f"rms={fixed_morse_error:.6g}, max={fixed_morse_max_error:.6g}, "
            f"cancellation={cancellation_region_normalized_rms:.6g}"
        )
    fixed_index = int(np.flatnonzero(epsilon == _FIXED_EPSILON)[0])
    if (
        stationary_point_counts[fixed_index] != morse_contribution_counts[fixed_index]
        or not fixed_morse_stationary_offsets_over_sigma
    ):
        raise RuntimeError("fixed-epsilon Morse sum does not contain the complete search set")
    crossover_slope = float(np.polyfit(np.log(epsilon), np.log(crossover_times), 1)[0])
    if abs(crossover_slope + 1.0) > 0.10:
        raise RuntimeError("measured crossover times do not scale as epsilon^-1")

    time_step = float(time[1] - time[0])
    full_array = np.stack(full_rows)
    spectrum_omega = exact.omega0 + 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(time.size, time_step)
    )
    common_demodulated = full_array * np.exp(1j * exact.omega0 * time)[np.newaxis, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.ifft(common_demodulated, axis=1), axes=1))
    arrays = {
        "epsilon": epsilon,
        "time": time,
        "G_full": full_array,
        "G_normal_form": np.stack(normal_rows),
        "G_morse": np.stack(morse_rows),
        "tau": np.stack(tau_rows),
        "J0": np.stack(target_rows),
        "scaled_response": np.stack(scaled_rows),
        "envelope": np.stack(envelope_rows),
        "rms_envelope": np.stack(rms_rows),
        "local_slope": np.stack(local_slope_rows),
        "fit_window_early": early_masks,
        "fit_window_late": late_masks,
        "slope_early": np.array([early_slope]),
        "slope_late": np.array([late_slope]),
        "crossover_time": np.asarray(crossover_times),
        "spectrum_omega": spectrum_omega,
        "spectrum": spectrum,
        "omega_min": np.asarray(omega_min),
        "omega_saddle": np.asarray(omega_saddle),
        "phase_error": np.asarray(phase_errors),
    }
    units = {key: "1" for key in arrays}
    units.update(
        {
            "time": "h/c_T",
            "G_full": "response",
            "G_normal_form": "response",
            "G_morse": "response",
            "crossover_time": "h/c_T",
            "spectrum_omega": "c_T/h",
            "omega_min": "c_T/h",
            "omega_saddle": "c_T/h",
            "phase_error": "rad",
        }
    )
    tolerances = {
        "maximum_bessel_collapse_error": maximum_collapse_error,
        "bessel_collapse_error_by_epsilon": {
            f"{value:.8g}": float(error)
            for value, error in zip(epsilon, collapse_errors, strict=True)
            if np.isfinite(error)
        },
        "fixed_epsilon_morse_relative_rms_error": fixed_morse_error,
        "fixed_epsilon_morse_relative_max_error": fixed_morse_max_error,
        "fixed_epsilon_cancellation_region_normalized_rms": (cancellation_region_normalized_rms),
        "fixed_epsilon_cancellation_fraction": cancellation_fraction,
        "measured_early_slope_error": abs(early_slope + 0.5),
        "measured_late_slope_error": abs(late_slope + 1.0),
        "registered_early_slope_by_grid": early_grid_slopes,
        "registered_late_slope_by_grid": late_grid_slopes,
        "measured_crossover_log_slope": crossover_slope,
        "maximum_phase_error": float(np.max(phase_errors)),
        "configured_phase_error": config.phase_error_tolerance,
        "registered_complex_response_errors": response_grid_errors,
        "registered_nested_frequency_errors": nested_frequency_errors,
        "registered_accumulated_phase_errors": accumulated_phase_errors,
        "minimum_anisotropic_branch_eigengap_by_epsilon": minimum_branch_eigengaps,
        "maximum_relative_frequency_uncertainty_by_epsilon": (
            maximum_relative_frequency_uncertainties
        ),
    }
    return write_stage_artifact(
        "green",
        config,
        directory,
        selected_profile,
        arrays,
        units,
        tolerances,
        input_artifacts=input_artifacts,
        extra_metadata={
            "response_model": "tracked full-dispersion branch quadrature and exact full-wave Morse sum",
            "radial_domain_control": {
                **domain_control.metadata(exact.kappa0),
                "stationary_point_count_by_epsilon": {
                    f"{value:.8g}": count
                    for value, count in zip(
                        epsilon,
                        stationary_point_counts,
                        strict=True,
                    )
                },
                "morse_contribution_count_by_epsilon": {
                    f"{value:.8g}": count
                    for value, count in zip(
                        epsilon,
                        morse_contribution_counts,
                        strict=True,
                    )
                },
                "fixed_epsilon_stationary_offsets_over_sigma": (
                    fixed_morse_stationary_offsets_over_sigma
                ),
                "all_fixed_epsilon_stationary_points_included_in_morse_sum": True,
                "direct_and_morse_responses_use_identical_compact_taper": True,
                "boundary_is_noncritical_by_epsilon": {
                    f"{value:.8g}": certified
                    for value, certified in zip(
                        epsilon,
                        boundary_is_noncritical,
                        strict=True,
                    )
                },
                "gradient_index_closes_by_epsilon": {
                    f"{value:.8g}": certified
                    for value, certified in zip(
                        epsilon,
                        gradient_index_closes,
                        strict=True,
                    )
                },
                "minimum_boundary_gradient_by_epsilon": {
                    f"{value:.8g}": minimum
                    for value, minimum in zip(
                        epsilon,
                        minimum_boundary_gradients,
                        strict=True,
                    )
                },
                "maximum_boundary_gradient_uncertainty_by_epsilon": {
                    f"{value:.8g}": uncertainty
                    for value, uncertainty in zip(
                        epsilon,
                        maximum_boundary_gradient_uncertainties,
                        strict=True,
                    )
                },
            },
            "registered_grid_shapes": [
                [int(settings["q_nodes"]), int(settings["theta_nodes"])],
                [2 * int(settings["q_nodes"]) - 1, 2 * int(settings["theta_nodes"])],
                [4 * int(settings["q_nodes"]) - 3, 4 * int(settings["theta_nodes"])],
            ],
            "fixed_epsilon_registered_grid_shapes": [
                [
                    int(settings["fixed_q_nodes"]),
                    int(settings["fixed_theta_nodes"]),
                ],
                [
                    2 * int(settings["fixed_q_nodes"]) - 1,
                    2 * int(settings["fixed_theta_nodes"]),
                ],
                [
                    4 * int(settings["fixed_q_nodes"]) - 3,
                    4 * int(settings["fixed_theta_nodes"]),
                ],
            ],
            "slope_sources": {
                "early": "absolute envelope of the smallest-epsilon direct response",
                "late": "four exact-beat RMS values of the fixed-epsilon direct response",
            },
            "late_fit_evidence": {
                "epsilon": _FIXED_EPSILON,
                "geometric_time_centers": late_fit_centers,
                "direct_response_rms": late_fit_rms,
                "half_open_time_index_bins": late_fit_bins,
                "slope": late_slope,
            },
            "fixed_morse_cancellation_evidence": {
                "comparison_start_time": _MORSE_COMPARISON_START,
                "comparison_stop_time": _MORSE_COMPARISON_STOP,
                "coherence_threshold": _MORSE_COHERENCE_THRESHOLD,
                "coherence": fixed_morse_coherence,
                "cancellation_time_indices": cancellation_indices,
                "noncancellation_morse_rms_normalization": fixed_morse_normalization,
            },
            "spectrum_convention": (
                "common omega0 demodulation; inverse-DFT positive detuning; "
                "physical angular-frequency axis"
            ),
            "spectrum_role": (
                "qualitative display only; exact critical-point frequencies provide "
                "the quantitative feature separation"
            ),
            "uniform_rows": [
                float(value) for value in epsilon if value <= _UNIFORM_EPSILON_MAXIMUM
            ],
            "fixed_morse_row_excluded_from_first_order_uniform_gate": True,
            "fixed_morse_epsilon": _FIXED_EPSILON,
            "cubic_symmetry_cache": (
                "C4v covariance verified independently by Task14 tests; caches are "
                "separate across refinement levels"
            ),
            "no_fitted_amplitude_phase_frequency_or_time_shift": True,
        },
    )
