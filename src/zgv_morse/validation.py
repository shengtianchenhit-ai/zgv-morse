"""Independent spectral validation against exact isotropic Lamb-wave theory."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
import json
from numbers import Real
from pathlib import Path
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import j0

from .config import ReferenceConfig
from .elasticity import isotropic_tensor
from .mode_tracking import mass_mac, relative_eigengap, symmetric_lamb_parity_score
from .spectral_plate import (
    ModeSet,
    assemble_plate_matrices,
    assemble_wavevector_derivatives,
    solve_plate_modes,
)
from .zgv import ZGVPoint, find_s1_zgv


@dataclass(frozen=True, slots=True)
class IsotropicBenchmarkRow:
    """One converged spectral benchmark and its numerical diagnostics."""

    order: int
    elements: int
    k_zgv: float
    omega_zgv: float
    curvature: float
    relative_k_error: float
    relative_omega_error: float
    relative_curvature_error: float
    maximum_eigen_residual: float
    hermitian_defect: float
    mass_orthogonality_defect: float
    rotational_frequency_defect: float
    minimum_relative_eigengap: float

    def __post_init__(self) -> None:
        for name in ("order", "elements"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be a built-in integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        positive = (
            "k_zgv",
            "omega_zgv",
            "curvature",
            "minimum_relative_eigengap",
        )
        nonnegative = (
            "relative_k_error",
            "relative_omega_error",
            "relative_curvature_error",
            "maximum_eigen_residual",
            "hermitian_defect",
            "mass_orthogonality_defect",
            "rotational_frequency_defect",
        )
        for field in fields(self):
            if field.name in ("order", "elements"):
                continue
            value = _finite_real(getattr(self, field.name), field.name)
            if field.name in positive and value <= 0.0:
                raise ValueError(f"{field.name} must be positive")
            if field.name in nonnegative and value < 0.0:
                raise ValueError(f"{field.name} must be nonnegative")
            object.__setattr__(self, field.name, value)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        scalar = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validated_config(cfg: object) -> ReferenceConfig:
    if not isinstance(cfg, ReferenceConfig):
        raise TypeError("cfg must be a ReferenceConfig instance")
    cfg.validate()
    return cfg


def _validated_orders(orders: object) -> tuple[int, ...]:
    if not isinstance(orders, tuple):
        raise TypeError("orders must be a tuple of built-in integers")
    if not orders:
        raise ValueError("orders must be nonempty")
    if any(type(order) is not int for order in orders):
        raise TypeError("orders must contain built-in integers")
    if any(not 2 <= order <= 512 for order in orders):
        raise ValueError("orders must be between 2 and 512")
    if any(left >= right for left, right in zip(orders, orders[1:], strict=False)):
        raise ValueError("orders must be strictly increasing")
    return orders


def _validated_num_modes(num_modes: object) -> int:
    if type(num_modes) is not int:
        raise TypeError("num_modes must be a built-in integer")
    if num_modes < 18:
        raise ValueError("num_modes must be at least 18")
    return num_modes


def _symmetric_candidates(modes: ModeSet) -> np.ndarray:
    scores = np.array(
        [
            symmetric_lamb_parity_score(
                modes.vectors[:, index],
                modes.matrices.nodes,
                modes.matrices.mass,
            )
            for index in range(modes.omega.size)
        ],
        dtype=np.float64,
    )
    candidates = np.flatnonzero(scores > 0.5)
    if candidates.size == 0:
        raise RuntimeError("no symmetric Lamb mode was found in the reported spectrum")
    return candidates


def _benchmark_discretization(
    exact: ZGVPoint,
    stiffness: np.ndarray,
    cfg: ReferenceConfig,
    order: int,
    bounds: tuple[float, ...],
    num_modes: int,
) -> IsotropicBenchmarkRow:
    if 3 * (order * (len(bounds) - 1) + 1) < num_modes:
        raise ValueError("order and element count cannot provide num_modes eigenpairs")

    diagnostics: list[tuple[ModeSet, int]] = []

    def solve(kx: float, ky: float) -> ModeSet:
        matrices = assemble_plate_matrices(
            kx,
            ky,
            stiffness,
            cfg.rho,
            cfg.h,
            order=order,
            element_bounds=bounds,
        )
        return solve_plate_modes(matrices, num_modes=num_modes)

    center = solve(exact.kappa0, 0.0)
    center_candidates = _symmetric_candidates(center)
    seed_index = int(
        center_candidates[np.argmin(np.abs(center.omega[center_candidates] - exact.omega0))]
    )
    if abs(float(center.omega[seed_index]) - exact.omega0) / exact.omega0 > 0.05:
        raise RuntimeError("the reported spectrum does not contain the reference symmetric branch")
    seed = center.vectors[:, seed_index]
    diagnostics.append((center, seed_index))

    def select_radial_mode(modes: ModeSet) -> int:
        candidates = _symmetric_candidates(modes)
        overlaps = np.array(
            [mass_mac(seed, modes.vectors[:, index], modes.matrices.mass) for index in candidates],
            dtype=np.float64,
        )
        index = int(candidates[np.argmax(overlaps)])
        if float(np.max(overlaps)) < 0.5:
            raise RuntimeError("symmetric-branch MAC fell below 0.5 in the validation window")
        return index

    def branch_omega(kappa: float) -> float:
        modes = solve(float(kappa), 0.0)
        index = select_radial_mode(modes)
        diagnostics.append((modes, index))
        return float(modes.omega[index])

    minimum = minimize_scalar(
        branch_omega,
        bounds=(exact.kappa0 - 0.03, exact.kappa0 + 0.03),
        method="bounded",
        options={"xatol": 1.0e-11, "maxiter": 200},
    )
    if not minimum.success or not np.isfinite(minimum.x) or not np.isfinite(minimum.fun):
        message = getattr(minimum, "message", "non-finite result")
        raise RuntimeError(f"spectral ZGV minimization failed: {message}")
    if not exact.kappa0 - 0.03 < float(minimum.x) < exact.kappa0 + 0.03:
        raise RuntimeError("spectral ZGV minimum lies on the validation boundary")

    def branch_group_velocity(kappa: float) -> float:
        modes = solve(float(kappa), 0.0)
        index = select_radial_mode(modes)
        derivatives = assemble_wavevector_derivatives(
            float(kappa),
            0.0,
            stiffness,
            cfg.rho,
            cfg.h,
            order=order,
            element_bounds=bounds,
        )
        vector = modes.vectors[:, index]
        omega = float(modes.omega[index])
        if omega <= 0.0:
            raise RuntimeError("the validation branch has nonpositive frequency")
        numerator = np.vdot(vector, derivatives.dkx @ vector)
        scale = max(abs(numerator), 1.0)
        if abs(float(np.imag(numerator))) > 1.0e-11 * scale:
            raise RuntimeError("spectral group velocity has significant imaginary leakage")
        diagnostics.append((modes, index))
        return float(np.real(numerator) / (2.0 * omega))

    lower_kappa = exact.kappa0 - 0.03
    upper_kappa = exact.kappa0 + 0.03
    lower_velocity = branch_group_velocity(lower_kappa)
    upper_velocity = branch_group_velocity(upper_kappa)
    if not lower_velocity < 0.0 < upper_velocity:
        raise RuntimeError("spectral group velocity does not bracket a local minimum")
    k_zgv = float(
        brentq(
            branch_group_velocity,
            lower_kappa,
            upper_kappa,
            xtol=5.0e-15,
            rtol=8.0 * np.finfo(np.float64).eps,
        )
    )
    omega_zgv = branch_omega(k_zgv)

    def velocity_derivative_curvature(step: float) -> float:
        values = np.array(
            [branch_group_velocity(k_zgv + shift * step) for shift in (-2, -1, 1, 2)],
            dtype=np.float64,
        )
        return float((values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (12.0 * step))

    coarse = velocity_derivative_curvature(2.0e-3)
    fine = velocity_derivative_curvature(1.0e-3)
    curvature = (16.0 * fine - coarse) / 15.0
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise RuntimeError("spectral ZGV curvature is not finite and positive")
    if abs(fine - coarse) / curvature > 1.0e-5:
        raise RuntimeError("spectral ZGV curvature is not stable under step halving")

    fit_radius = 5.0e-3
    offsets = np.linspace(-fit_radius, fit_radius, 21)
    fit_frequencies = np.array(
        [branch_omega(k_zgv + float(offset)) for offset in offsets],
        dtype=np.float64,
    )
    scaled_offsets = offsets / fit_radius
    coefficients = np.polynomial.polynomial.polyfit(
        scaled_offsets,
        fit_frequencies,
        deg=6,
    )
    polynomial_curvature = float(2.0 * coefficients[2] / fit_radius**2)
    if (
        not np.isfinite(polynomial_curvature)
        or polynomial_curvature <= 0.0
        or abs(polynomial_curvature - curvature) / curvature > 1.0e-4
    ):
        raise RuntimeError("spectral ZGV curvature failed the local-polynomial cross-check")

    angle = np.pi / 7.0
    angled = solve(k_zgv * np.cos(angle), k_zgv * np.sin(angle))
    angled_candidates = _symmetric_candidates(angled)
    angled_index = int(
        angled_candidates[np.argmin(np.abs(angled.omega[angled_candidates] - omega_zgv))]
    )
    angled_omega = float(angled.omega[angled_index])
    diagnostics.append((angled, angled_index))

    return IsotropicBenchmarkRow(
        order=order,
        elements=len(bounds) - 1,
        k_zgv=k_zgv,
        omega_zgv=omega_zgv,
        curvature=curvature,
        relative_k_error=abs(k_zgv - exact.kappa0) / exact.kappa0,
        relative_omega_error=abs(omega_zgv - exact.omega0) / exact.omega0,
        relative_curvature_error=abs(curvature - exact.curvature_a) / exact.curvature_a,
        maximum_eigen_residual=max(float(modes.residuals[index]) for modes, index in diagnostics),
        hermitian_defect=max(float(modes.matrices.hermitian_defect) for modes, _ in diagnostics),
        mass_orthogonality_defect=max(
            float(modes.mass_orthogonality_defect) for modes, _ in diagnostics
        ),
        rotational_frequency_defect=abs(angled_omega - omega_zgv) / omega_zgv,
        minimum_relative_eigengap=min(
            relative_eigengap(modes.eigenvalues, index) for modes, index in diagnostics
        ),
    )


def run_isotropic_validation(
    cfg: ReferenceConfig,
    *,
    orders: tuple[int, ...] = (12, 16, 20, 24, 28),
    num_modes: int = 18,
) -> tuple[ZGVPoint, tuple[IsotropicBenchmarkRow, ...], IsotropicBenchmarkRow]:
    """Run single- and two-element convergence against exact Lamb theory."""

    config = _validated_config(cfg)
    validated_orders = _validated_orders(orders)
    mode_count = _validated_num_modes(num_modes)
    exact = find_s1_zgv(config, dps=80)
    stiffness = isotropic_tensor(config.lam, config.mu)
    rows = tuple(
        _benchmark_discretization(
            exact,
            stiffness,
            config,
            order,
            (-config.h, config.h),
            mode_count,
        )
        for order in validated_orders
    )
    split = _benchmark_discretization(
        exact,
        stiffness,
        config,
        validated_orders[-1],
        (-config.h, 0.0, config.h),
        mode_count,
    )
    return exact, rows, split


def _validate_exact(exact: object) -> ZGVPoint:
    if not isinstance(exact, ZGVPoint):
        raise TypeError("exact must be a ZGVPoint instance")
    for field in fields(exact):
        value = getattr(exact, field.name)
        if field.name == "branch_index":
            if type(value) is not int or value <= 0:
                raise ValueError("exact branch_index must be a positive integer")
        else:
            scalar = _finite_real(value, f"exact.{field.name}")
            if field.name in ("kappa0", "omega0", "curvature_a") and scalar <= 0.0:
                raise ValueError(f"exact.{field.name} must be positive")
            if field.name == "det_residual" and scalar < 0.0:
                raise ValueError("exact.det_residual must be nonnegative")
    return exact


def _validated_export_rows(
    rows: object,
    split: object,
) -> tuple[tuple[IsotropicBenchmarkRow, ...], IsotropicBenchmarkRow]:
    try:
        benchmark_rows = tuple(rows)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("benchmark rows must be a nonempty sequence") from error
    if not benchmark_rows:
        raise ValueError("benchmark rows must be nonempty")
    if any(not isinstance(row, IsotropicBenchmarkRow) for row in benchmark_rows):
        raise TypeError("benchmark rows must contain only IsotropicBenchmarkRow records")
    if any(row.elements != 1 for row in benchmark_rows):
        raise ValueError("single-element benchmark rows must have elements=1")
    if any(
        left.order >= right.order
        for left, right in zip(benchmark_rows, benchmark_rows[1:], strict=False)
    ):
        raise ValueError("benchmark row orders must be strictly increasing")
    if not isinstance(split, IsotropicBenchmarkRow):
        raise TypeError("two-element split must be an IsotropicBenchmarkRow")
    if split.elements != 2:
        raise ValueError("two-element split row must have elements=2")
    if split.order != benchmark_rows[-1].order:
        raise ValueError("two-element split must use the same order as the final row")
    return benchmark_rows, split


def write_isotropic_validation(
    exact: ZGVPoint,
    rows: tuple[IsotropicBenchmarkRow, ...],
    split: IsotropicBenchmarkRow,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Write deterministic finite JSON and CSV validation records."""

    exact_point = _validate_exact(exact)
    benchmark_rows, split_row = _validated_export_rows(rows, split)
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    if json_target.resolve() == csv_target.resolve():
        raise ValueError("JSON and CSV paths must be distinct")

    payload = {
        "exact": asdict(exact_point),
        "single_element": [asdict(row) for row in benchmark_rows],
        "two_element": asdict(split_row),
    }
    json_text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fieldnames = [field.name for field in fields(IsotropicBenchmarkRow)]

    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json_text, encoding="utf-8", newline="\n")
    with csv_target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in (*benchmark_rows, split_row))


def _metric_real_vector(values: object, name: str) -> np.ndarray:
    """Return an independent finite one-dimensional real array."""

    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite one-dimensional real array") from error
    if candidate.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if (
        candidate.size == 0
        or np.iscomplexobj(candidate)
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a nonempty real numeric array")
    try:
        result = np.array(candidate, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} entries must be finite")
    return result


def _metric_complex_vector(values: object, name: str) -> np.ndarray:
    """Return an independent finite one-dimensional numeric array."""

    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite one-dimensional numeric array") from error
    if candidate.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if (
        candidate.size == 0
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a nonempty numeric array")
    try:
        result = np.array(candidate, dtype=np.complex128, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} entries must be finite")
    return result


def _metric_boolean_mask(values: object) -> np.ndarray:
    """Return an independent one-dimensional mask without truth-value casting."""

    try:
        candidate = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError("mask must be a one-dimensional boolean array") from error
    if candidate.ndim != 1:
        raise ValueError("mask must be a one-dimensional boolean array")
    if candidate.size == 0 or candidate.dtype != np.dtype(np.bool_):
        raise TypeError("mask must be a nonempty boolean array")
    return np.array(candidate, dtype=np.bool_, copy=True)


def _stable_rms_nonnegative(values: np.ndarray, name: str) -> float:
    """Compute an RMS without squaring values at their original scale."""

    if values.ndim != 1 or values.size == 0 or np.any(values < 0.0):
        raise ValueError(f"{name} must be a nonempty nonnegative vector")
    scale = float(np.max(values))
    if not np.isfinite(scale):
        raise ValueError(f"{name} must be finite")
    if scale == 0.0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = np.divide(values, scale)
        rms = np.multiply(scale, np.sqrt(np.mean(np.square(normalized))))
    if not np.isfinite(rms):
        raise ValueError(f"{name} RMS must be finite")
    return float(rms)


@dataclass(frozen=True, slots=True)
class PowerLawFit:
    """A registered log-log least-squares fit over a declared time window."""

    slope: float
    intercept: float
    sample_count: int
    time_start: float
    time_stop: float

    def __post_init__(self) -> None:
        slope = _finite_real(self.slope, "slope")
        intercept = _finite_real(self.intercept, "intercept")
        if type(self.sample_count) is not int:
            raise TypeError("sample_count must be a built-in integer")
        if self.sample_count < 10:
            raise ValueError("sample_count must be at least ten")
        start = _finite_real(self.time_start, "time_start")
        stop = _finite_real(self.time_stop, "time_stop")
        if start <= 0.0 or stop <= start:
            raise ValueError("fit times must be positive and strictly ordered")
        object.__setattr__(self, "slope", slope)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "time_start", start)
        object.__setattr__(self, "time_stop", stop)


@dataclass(frozen=True, slots=True)
class BesselCollapseError:
    """Immutable absolute and away-from-zero Bessel collapse metrics."""

    max_absolute: float
    rms_absolute: float
    max_relative_away_from_zeros: float
    away_zero_sample_count: int

    def __post_init__(self) -> None:
        for name in (
            "max_absolute",
            "rms_absolute",
            "max_relative_away_from_zeros",
        ):
            value = _finite_real(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if type(self.away_zero_sample_count) is not int:
            raise TypeError("away_zero_sample_count must be a built-in integer")
        if self.away_zero_sample_count <= 0:
            raise ValueError("away_zero_sample_count must be positive")


def rms_envelope(signal: object, window_samples: int) -> np.ndarray:
    """Return a centered moving RMS envelope with reflection at both ends."""

    values = _metric_complex_vector(signal, "signal")
    if type(window_samples) is not int:
        raise TypeError("window_samples must be a built-in integer")
    if window_samples < 3:
        raise ValueError("RMS window must be at least three samples")
    if window_samples % 2 == 0:
        raise ValueError("RMS window must be odd")
    if values.size < window_samples:
        raise ValueError("signal length must be at least the RMS window")

    with np.errstate(over="ignore", invalid="ignore"):
        magnitude = np.abs(values)
    scale = float(np.max(magnitude))
    if not np.isfinite(scale):
        raise ValueError("signal magnitude must be finite")
    if scale == 0.0:
        return np.zeros(values.shape, dtype=np.float64)
    half_width = window_samples // 2
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized_power = np.square(np.divide(magnitude, scale))
        padded = np.pad(normalized_power, half_width, mode="reflect")
        moving_power = np.convolve(
            padded,
            np.full(window_samples, 1.0 / window_samples),
            mode="valid",
        )
        envelope = np.multiply(scale, np.sqrt(moving_power))
    if not np.isfinite(envelope).all():
        raise ValueError("RMS envelope must be finite")
    return np.array(envelope, dtype=np.float64, copy=True)


def fit_power_law(time: object, envelope: object, mask: object) -> PowerLawFit:
    """Fit a power law on exactly the samples selected by a boolean mask."""

    times = _metric_real_vector(time, "time")
    amplitudes = _metric_real_vector(envelope, "envelope")
    selected_mask = _metric_boolean_mask(mask)
    if times.shape != amplitudes.shape or times.shape != selected_mask.shape:
        raise ValueError("time, envelope, and mask must have the same shape")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("time must be strictly increasing")
    sample_count = int(np.count_nonzero(selected_mask))
    if sample_count < 10:
        raise ValueError("power-law fit requires at least ten selected samples")
    selected_time = times[selected_mask]
    selected_amplitude = amplitudes[selected_mask]
    if np.any(selected_time <= 0.0):
        raise ValueError("selected time samples must be positive")
    if np.any(selected_amplitude <= 0.0):
        raise ValueError("selected envelope samples must be positive")

    logarithmic_time = np.log(selected_time)
    logarithmic_amplitude = np.log(selected_amplitude)
    time_mean = float(np.mean(logarithmic_time))
    amplitude_mean = float(np.mean(logarithmic_amplitude))
    centered_time = logarithmic_time - time_mean
    centered_amplitude = logarithmic_amplitude - amplitude_mean
    scale = float(np.max(np.abs(centered_time)))
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("selected logarithmic times must span a finite nonzero interval")
    normalized_time = centered_time / scale
    denominator = float(np.dot(normalized_time, normalized_time))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        slope = np.dot(normalized_time, centered_amplitude) / (denominator * scale)
        intercept = amplitude_mean - slope * time_mean
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("power-law fit must be finite")
    return PowerLawFit(
        float(slope),
        float(intercept),
        sample_count,
        float(selected_time[0]),
        float(selected_time[-1]),
    )


def operational_crossover_time(epsilon: float, V4: float) -> float:
    """Return the first time at which the canonical Bessel kernel reaches 0.9."""

    perturbation = _finite_real(epsilon, "epsilon")
    coefficient = _finite_real(V4, "V4")
    with np.errstate(over="ignore", invalid="ignore"):
        rate = np.abs(np.multiply(np.float64(perturbation), np.float64(coefficient)))
    if not np.isfinite(rate):
        raise ValueError("abs(epsilon * V4) must be finite")
    if rate == 0.0:
        raise ValueError("epsilon and V4 must define a nonzero representable rate")
    transition = brentq(
        lambda value: float(j0(value) - 0.9),
        0.0,
        2.404825557695773,
        xtol=5.0e-15,
        rtol=8.0 * np.finfo(np.float64).eps,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        crossover = np.divide(np.float64(transition), rate)
    if not np.isfinite(crossover):
        raise ValueError("operational crossover time must be finite")
    return float(crossover)


def bessel_collapse_error(
    scaled_response: object,
    target: object,
    zero_threshold: float = 0.05,
) -> BesselCollapseError:
    """Measure complex collapse error, using relative error only away from zeros."""

    scaled = _metric_complex_vector(scaled_response, "scaled_response")
    reference = _metric_complex_vector(target, "target")
    if scaled.shape != reference.shape:
        raise ValueError("scaled_response and target must have the same shape")
    threshold = _finite_real(zero_threshold, "zero_threshold")
    if threshold <= 0.0:
        raise ValueError("zero_threshold must be positive")
    with np.errstate(over="ignore", invalid="ignore"):
        absolute = np.abs(np.subtract(scaled, reference))
        reference_magnitude = np.abs(reference)
    if not np.isfinite(absolute).all() or not np.isfinite(reference_magnitude).all():
        raise ValueError("Bessel collapse magnitudes must be finite")
    away = reference_magnitude >= threshold
    sample_count = int(np.count_nonzero(away))
    if sample_count == 0:
        raise ValueError("Bessel collapse requires samples away from zeros")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative = np.divide(absolute[away], reference_magnitude[away])
    if not np.isfinite(relative).all():
        raise ValueError("away-from-zero relative collapse error must be finite")
    return BesselCollapseError(
        float(np.max(absolute)),
        _stable_rms_nonnegative(absolute, "absolute collapse error"),
        float(np.max(relative)),
        sample_count,
    )
