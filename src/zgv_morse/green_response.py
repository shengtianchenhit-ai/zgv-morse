"""Phase-controlled quadrature of a tracked local plate-wave branch."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson

from .dispersion import RingAnchoredSpectralEvaluator, TrackedSpectralSample


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


COMPACT_TAPER_PLATEAU_SIGMA = 1.25
DIRECT_RADIAL_SUPPORT_SIGMA = 1.5


def _finite_real(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_complex(value: object, name: str) -> complex:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Complex):
        raise TypeError(f"{name} must be a finite numeric scalar")
    try:
        result = complex(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _real_array(
    values: ArrayLike,
    name: str,
    *,
    ndim: int | None = None,
    positive: bool = False,
    nonnegative: bool = False,
) -> FloatArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array") from error
    if (
        candidate.size == 0
        or np.iscomplexobj(candidate)
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
        or (ndim is not None and candidate.ndim != ndim)
    ):
        suffix = "" if ndim is None else f" with ndim={ndim}"
        raise ValueError(f"{name} must be a nonempty real numeric array{suffix}")
    try:
        result = np.array(candidate, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} entries must be finite")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} entries must be positive")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} entries must be nonnegative")
    return result


def _complex_array(values: ArrayLike, name: str, *, ndim: int | None = None) -> ComplexArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if (
        candidate.size == 0
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
        or (ndim is not None and candidate.ndim != ndim)
    ):
        suffix = "" if ndim is None else f" with ndim={ndim}"
        raise ValueError(f"{name} must be a nonempty numeric array{suffix}")
    try:
        result = np.array(candidate, dtype=np.complex128, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} entries must be finite")
    return result


def compact_radial_taper(q: ArrayLike, window_sigma: float) -> FloatArray:
    r"""Return a deterministic :math:`C^\infty` compact radial taper.

    The taper is exactly one for ``|q| <= 1.25 * window_sigma`` and exactly
    zero for ``|q| >= 1.5 * window_sigma``.  On the intervening strip it uses
    the standard flat smooth step built from ``exp(-1/x)``.  Consequently the
    taper and all of its derivatives match the constant pieces at both joins.
    """

    offsets = _real_array(q, "q")
    sigma = _finite_real(window_sigma, "window_sigma", positive=True)
    plateau = COMPACT_TAPER_PLATEAU_SIGMA * sigma
    support = DIRECT_RADIAL_SUPPORT_SIGMA * sigma
    absolute_offset = np.abs(offsets)
    taper = np.ones(offsets.shape, dtype=np.float64)
    taper[absolute_offset >= support] = 0.0
    transition = (absolute_offset > plateau) & (absolute_offset < support)
    if np.any(transition):
        coordinate = (absolute_offset[transition] - plateau) / (support - plateau)
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            rising_numerator = np.exp(-1.0 / coordinate)
            falling_numerator = np.exp(-1.0 / (1.0 - coordinate))
            taper[transition] = falling_numerator / (rising_numerator + falling_numerator)
    if not np.isfinite(taper).all():
        raise ValueError("compact radial taper must be finite")
    return taper


def _read_only_float(values: ArrayLike) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _read_only_complex(values: ArrayLike) -> ComplexArray:
    result = np.array(values, dtype=np.complex128, copy=True)
    result.setflags(write=False)
    return result


def _validate_q(values: ArrayLike) -> FloatArray:
    q = _real_array(values, "q", ndim=1)
    if q.size < 2 or not np.all(np.diff(q) > 0.0):
        raise ValueError("q must be strictly increasing and contain at least two nodes")
    return q


def _validate_theta(values: ArrayLike) -> FloatArray:
    theta = _real_array(values, "theta", ndim=1)
    if theta.size < 4:
        raise ValueError("theta must contain at least four endpoint-free nodes")
    expected = 2.0 * np.pi * np.arange(theta.size) / theta.size
    if not np.allclose(theta, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError("theta must be endpoint-free and uniform on [0, 2*pi)")
    return theta


def _broadcast_nonnegative_error(
    values: ArrayLike,
    shape: tuple[int, int],
    name: str,
) -> FloatArray:
    error = _real_array(values, name, nonnegative=True)
    try:
        broadcast = np.broadcast_to(error, shape)
    except ValueError as exception:
        raise ValueError(f"{name} must broadcast to surface shape {shape}") from exception
    return np.array(broadcast, dtype=np.float64, copy=True)


@dataclass(frozen=True, slots=True)
class BranchNodeSample:
    """One tracked full-wave node used to tabulate a polar branch surface."""

    omega: float
    amplitude: complex
    frequency_uncertainty: float
    amplitude_uncertainty: float = 0.0
    relative_eigengap: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "omega", _finite_real(self.omega, "omega", positive=True))
        object.__setattr__(self, "amplitude", _finite_complex(self.amplitude, "amplitude"))
        object.__setattr__(
            self,
            "frequency_uncertainty",
            _finite_real(
                self.frequency_uncertainty,
                "frequency_uncertainty",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "amplitude_uncertainty",
            _finite_real(
                self.amplitude_uncertainty,
                "amplitude_uncertainty",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "relative_eigengap",
            _finite_real(
                self.relative_eigengap,
                "relative_eigengap",
                nonnegative=True,
            ),
        )


@runtime_checkable
class RadialBranchEvaluator(Protocol):
    """Stateful evaluator continued outward from a ring anchor."""

    def __call__(self, kxy: FloatArray) -> BranchNodeSample: ...


@runtime_checkable
class RadialEvaluatorFactory(Protocol):
    """Create a fresh inward or outward tracker at one angular ring anchor."""

    def __call__(self, theta: float, radial_direction: int) -> RadialBranchEvaluator: ...


@dataclass(frozen=True, slots=True)
class PolarBranchSurface:
    """A periodic polar tabulation of frequency, amplitude, and local error."""

    q: FloatArray
    theta: FloatArray
    omega: FloatArray
    amplitude: ComplexArray
    frequency_error: FloatArray
    amplitude_error: FloatArray | float = 0.0
    relative_eigengap: FloatArray | float = 1.0

    def __post_init__(self) -> None:
        q = _validate_q(self.q)
        theta = _validate_theta(self.theta)
        shape = (q.size, theta.size)
        omega = _real_array(self.omega, "omega", ndim=2, positive=True)
        amplitude = _complex_array(self.amplitude, "amplitude", ndim=2)
        if omega.shape != shape or amplitude.shape != shape:
            raise ValueError(f"omega and amplitude must have shape {shape}")
        frequency_error = _broadcast_nonnegative_error(
            self.frequency_error,
            shape,
            "frequency_error",
        )
        amplitude_error = _broadcast_nonnegative_error(
            self.amplitude_error,
            shape,
            "amplitude_error",
        )
        relative_eigengap = _broadcast_nonnegative_error(
            self.relative_eigengap,
            shape,
            "relative_eigengap",
        )
        object.__setattr__(self, "q", _read_only_float(q))
        object.__setattr__(self, "theta", _read_only_float(theta))
        object.__setattr__(self, "omega", _read_only_float(omega))
        object.__setattr__(self, "amplitude", _read_only_complex(amplitude))
        object.__setattr__(self, "frequency_error", _read_only_float(frequency_error))
        object.__setattr__(self, "amplitude_error", _read_only_float(amplitude_error))
        object.__setattr__(self, "relative_eigengap", _read_only_float(relative_eigengap))

    def validate(self) -> None:
        """Compatibility no-op: construction already performs complete validation."""


@dataclass(frozen=True, slots=True)
class BranchResponse:
    """A carrier-demodulated complex branch response."""

    time: FloatArray
    carrier_frequency: float
    demodulated: ComplexArray

    def __post_init__(self) -> None:
        time = _real_array(self.time, "time", ndim=1, nonnegative=True)
        carrier = _finite_real(
            self.carrier_frequency,
            "carrier_frequency",
            positive=True,
        )
        demodulated = _complex_array(self.demodulated, "demodulated", ndim=1)
        if demodulated.shape != time.shape:
            raise ValueError("demodulated and time must have the same shape")
        object.__setattr__(self, "time", _read_only_float(time))
        object.__setattr__(self, "carrier_frequency", carrier)
        object.__setattr__(self, "demodulated", _read_only_complex(demodulated))

    def analytic_signal(self) -> ComplexArray:
        """Restore the carrier without exposing internal array storage."""

        with np.errstate(over="ignore", invalid="ignore"):
            phase = np.multiply(self.carrier_frequency, self.time)
            result = np.multiply(self.demodulated, np.exp(-1j * phase))
        if not np.isfinite(result).all():
            raise ValueError("analytic signal must remain finite")
        return np.array(result, dtype=np.complex128, copy=True)


def assert_phase_accuracy(
    t_max: float,
    frequency_error: float,
    phase_limit: float = 0.05,
) -> float:
    """Enforce a threshold on an accumulated frequency-discrepancy estimate."""

    maximum_time = _finite_real(t_max, "t_max", nonnegative=True)
    error = _finite_real(frequency_error, "frequency_error", nonnegative=True)
    limit = _finite_real(phase_limit, "phase_limit", positive=True)
    with np.errstate(over="ignore", invalid="ignore"):
        accumulated = np.multiply(np.float64(maximum_time), np.float64(error))
    if not np.isfinite(accumulated):
        raise ValueError("accumulated phase must be finite")
    value = float(accumulated)
    if value > limit:
        raise ValueError(f"accumulated phase {value:.6g} exceeds {limit:.6g}")
    return value


def integrate_branch_response(
    surface: PolarBranchSurface,
    time: ArrayLike,
    k0: float,
    omega_reference: float,
    source_radius: float,
    window_sigma: float,
    chunk_size: int = 64,
    fourier_normalization: float = (2.0 * np.pi) ** -2,
    phase_limit: float = 0.05,
) -> BranchResponse:
    """Integrate a tabulated branch without using an asymptotic Bessel result."""

    if not isinstance(surface, PolarBranchSurface):
        raise TypeError("surface must be a PolarBranchSurface")
    times = _real_array(time, "time", ndim=1, nonnegative=True)
    ring_radius = _finite_real(k0, "k0", positive=True)
    carrier = _finite_real(omega_reference, "omega_reference", positive=True)
    source = _finite_real(source_radius, "source_radius", nonnegative=True)
    sigma = _finite_real(window_sigma, "window_sigma", positive=True)
    normalization = _finite_real(
        fourier_normalization,
        "fourier_normalization",
        positive=True,
    )
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive built-in integer")
    limit = _finite_real(phase_limit, "phase_limit", positive=True)

    with np.errstate(over="ignore", invalid="ignore"):
        radial_k = np.add(ring_radius, surface.q)
    if not np.isfinite(radial_k).all() or np.any(radial_k <= 0.0):
        raise ValueError("the radial wave number k0 + q must remain positive")
    support_half_width = DIRECT_RADIAL_SUPPORT_SIGMA * sigma
    if (
        surface.q[0] > -support_half_width
        or surface.q[-1] < support_half_width
    ):
        raise ValueError(
            "surface radial domain must cover compact support "
            f"|q| <= {DIRECT_RADIAL_SUPPORT_SIGMA:g} * window_sigma"
        )
    assert_phase_accuracy(
        float(np.max(times)),
        float(np.max(surface.frequency_error)),
        limit,
    )

    with np.errstate(
        over="ignore",
        under="ignore",
        invalid="ignore",
        divide="ignore",
    ):
        source_argument = np.multiply(0.5 * source, radial_k)
        source_weight = np.exp(-np.multiply(source_argument, source_argument))
        window_argument = np.divide(surface.q, sigma)
        window_weight = np.exp(-np.power(window_argument, 8))
        window_weight *= compact_radial_taper(surface.q, sigma)
        radial_weight = radial_k * source_weight * window_weight
        static = surface.amplitude * radial_weight[:, np.newaxis]
        shifted_frequency = surface.omega - carrier
    if not np.isfinite(static).all() or not np.isfinite(shifted_frequency).all():
        raise ValueError("branch quadrature weights and shifted frequency must be finite")

    result = np.empty(times.size, dtype=np.complex128)
    angular_weight = 2.0 * np.pi / surface.theta.size
    for start in range(0, times.size, chunk_size):
        stop = min(start + chunk_size, times.size)
        with np.errstate(over="ignore", invalid="ignore"):
            phase_argument = np.multiply(
                times[start:stop, np.newaxis, np.newaxis],
                shifted_frequency[np.newaxis, :, :],
            )
            phase = np.exp(-1j * phase_argument)
            angular = angular_weight * np.sum(
                phase * static[np.newaxis, :, :],
                axis=2,
            )
            result[start:stop] = normalization * simpson(
                angular,
                x=surface.q,
                axis=1,
            )
    if not np.isfinite(result).all():
        raise ValueError("branch response quadrature must be finite")
    return BranchResponse(times, carrier, result)


def estimate_nested_frequency_error(coarse: ArrayLike, fine: ArrayLike) -> float:
    """Return the maximum registered-node frequency discrepancy."""

    coarse_array = _real_array(coarse, "coarse", ndim=2)
    fine_array = _real_array(fine, "fine", ndim=2)
    expected = (2 * (coarse_array.shape[0] - 1) + 1, 2 * coarse_array.shape[1])
    if fine_array.shape != expected:
        raise ValueError(f"expected doubled shape {expected}, got {fine_array.shape}")
    error = float(np.max(np.abs(coarse_array - fine_array[::2, ::2])))
    if not np.isfinite(error):
        raise ValueError("nested frequency error must be finite")
    return error


def normal_impulse_amplitude(omega: float, top_normal_component: complex) -> complex:
    """Return the phase-invariant normal impulse modal amplitude."""

    frequency = _finite_real(omega, "omega", positive=True)
    component = _finite_complex(top_normal_component, "top_normal_component")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        magnitude_squared = np.multiply(
            np.complex128(component),
            np.conj(np.complex128(component)),
        )
        result = 1j * magnitude_squared / (2.0 * frequency)
    if not np.isfinite(result):
        raise ValueError("normal impulse amplitude must be finite")
    return complex(result)


class FullWaveRadialEvaluatorFactory:
    """Adapt a nested-order spectral evaluator to physical Green-function nodes."""

    __slots__ = ("amplitude_atol", "amplitude_rtol", "spectral_evaluator")

    def __init__(
        self,
        spectral_evaluator: RingAnchoredSpectralEvaluator,
        *,
        amplitude_rtol: float = 1.0e-2,
        amplitude_atol: float = 1.0e-12,
    ) -> None:
        if not isinstance(spectral_evaluator, RingAnchoredSpectralEvaluator):
            raise TypeError("spectral_evaluator must be a RingAnchoredSpectralEvaluator")
        self.spectral_evaluator = spectral_evaluator
        self.amplitude_rtol = _finite_real(
            amplitude_rtol,
            "amplitude_rtol",
            nonnegative=True,
        )
        self.amplitude_atol = _finite_real(
            amplitude_atol,
            "amplitude_atol",
            nonnegative=True,
        )

    def __call__(
        self,
        theta: float,
        radial_direction: int,
    ) -> _FullWaveRadialEvaluator:
        tracker = self.spectral_evaluator.radial_tracker(theta, radial_direction)
        return _FullWaveRadialEvaluator(
            tracker,
            self.amplitude_rtol,
            self.amplitude_atol,
        )


class _FullWaveRadialEvaluator:
    """Convert every tracked eigenvector into its physical normal impulse weight."""

    __slots__ = ("_amplitude_atol", "_amplitude_rtol", "_tracker")

    def __init__(
        self,
        tracker: Callable[[FloatArray], TrackedSpectralSample],
        amplitude_rtol: float,
        amplitude_atol: float,
    ) -> None:
        self._tracker = tracker
        self._amplitude_rtol = amplitude_rtol
        self._amplitude_atol = amplitude_atol

    def __call__(self, kxy: FloatArray) -> BranchNodeSample:
        sample = self._tracker(kxy)
        if not isinstance(sample, TrackedSpectralSample):
            raise TypeError("spectral radial tracker returned an invalid sample")
        fine_amplitude = normal_impulse_amplitude(
            sample.frequency.omega,
            sample.top_normal_component,
        )
        coarse_amplitude = normal_impulse_amplitude(
            sample.coarse_omega,
            sample.coarse_top_normal_component,
        )
        amplitude_uncertainty = abs(fine_amplitude - coarse_amplitude)
        amplitude_scale = max(abs(fine_amplitude), abs(coarse_amplitude))
        with np.errstate(over="ignore", invalid="ignore"):
            amplitude_tolerance = np.add(
                self._amplitude_atol,
                np.multiply(self._amplitude_rtol, amplitude_scale),
            )
        if amplitude_uncertainty > amplitude_tolerance:
            raise RuntimeError(
                "nested spectral amplitude discrepancy "
                f"{amplitude_uncertainty:.6g} exceeds "
                f"{float(amplitude_tolerance):.6g}"
            )
        return BranchNodeSample(
            sample.frequency.omega,
            fine_amplitude,
            sample.frequency.frequency_uncertainty,
            amplitude_uncertainty,
            sample.relative_eigengap,
        )


def _sample_node(evaluator: RadialBranchEvaluator, point: FloatArray) -> BranchNodeSample:
    sample = evaluator(point)
    if not isinstance(sample, BranchNodeSample):
        raise TypeError("radial evaluator must return a BranchNodeSample")
    return sample


def _validate_anchor_samples(
    inward: BranchNodeSample,
    outward: BranchNodeSample,
) -> None:
    frequency_tolerance = inward.frequency_uncertainty + outward.frequency_uncertainty
    frequency_tolerance += (
        32.0
        * np.finfo(np.float64).eps
        * max(
            inward.omega,
            outward.omega,
            1.0,
        )
    )
    amplitude_scale = max(abs(inward.amplitude), abs(outward.amplitude), 1.0)
    amplitude_tolerance = inward.amplitude_uncertainty + outward.amplitude_uncertainty
    amplitude_tolerance += 32.0 * np.finfo(np.float64).eps * amplitude_scale
    if abs(inward.omega - outward.omega) > frequency_tolerance:
        raise ValueError("fresh-ray anchor omega values are inconsistent")
    if abs(inward.amplitude - outward.amplitude) > amplitude_tolerance:
        raise ValueError("fresh-ray anchor amplitude values are inconsistent")


def build_tracked_surface(
    evaluator_factory: RadialEvaluatorFactory,
    q: ArrayLike,
    theta: ArrayLike,
    k0: float,
) -> PolarBranchSurface:
    """Build a surface from independent inward/outward anchor continuations."""

    if not callable(evaluator_factory):
        raise TypeError("evaluator_factory must be callable")
    radial_offsets = _validate_q(q)
    angles = _validate_theta(theta)
    ring_radius = _finite_real(k0, "k0", positive=True)
    with np.errstate(over="ignore", invalid="ignore"):
        radial_k = np.add(ring_radius, radial_offsets)
    if not np.isfinite(radial_k).all() or np.any(radial_k <= 0.0):
        raise ValueError("the radial wave number k0 + q must remain positive")
    zero_indices = np.flatnonzero(radial_offsets == 0.0)
    if zero_indices.size != 1:
        raise ValueError("q must contain exactly one q=0 ring anchor")
    zero_index = int(zero_indices[0])

    shape = (radial_offsets.size, angles.size)
    omega = np.empty(shape, dtype=np.float64)
    amplitude = np.empty(shape, dtype=np.complex128)
    uncertainty = np.empty(shape, dtype=np.float64)
    amplitude_uncertainty = np.empty(shape, dtype=np.float64)
    relative_eigengap = np.empty(shape, dtype=np.float64)
    negative_indices = list(range(zero_index - 1, -1, -1))
    positive_indices = list(range(zero_index + 1, radial_offsets.size))

    for angular_index, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        inward = evaluator_factory(float(angle), -1)
        outward = evaluator_factory(float(angle), 1)
        if not callable(inward) or not callable(outward):
            raise TypeError("evaluator_factory must return callable radial evaluators")

        anchor_point = ring_radius * direction
        inward_anchor = _sample_node(inward, anchor_point)
        outward_anchor = _sample_node(outward, anchor_point)
        _validate_anchor_samples(inward_anchor, outward_anchor)
        omega[zero_index, angular_index] = 0.5 * inward_anchor.omega + 0.5 * outward_anchor.omega
        amplitude[zero_index, angular_index] = (
            0.5 * inward_anchor.amplitude + 0.5 * outward_anchor.amplitude
        )
        uncertainty[zero_index, angular_index] = max(
            inward_anchor.frequency_uncertainty,
            outward_anchor.frequency_uncertainty,
            0.5 * abs(inward_anchor.omega - outward_anchor.omega),
        )
        amplitude_uncertainty[zero_index, angular_index] = max(
            inward_anchor.amplitude_uncertainty,
            outward_anchor.amplitude_uncertainty,
            0.5 * abs(inward_anchor.amplitude - outward_anchor.amplitude),
        )
        relative_eigengap[zero_index, angular_index] = min(
            inward_anchor.relative_eigengap,
            outward_anchor.relative_eigengap,
        )

        for radial_index in negative_indices:
            sample = _sample_node(inward, radial_k[radial_index] * direction)
            omega[radial_index, angular_index] = sample.omega
            amplitude[radial_index, angular_index] = sample.amplitude
            uncertainty[radial_index, angular_index] = sample.frequency_uncertainty
            amplitude_uncertainty[radial_index, angular_index] = sample.amplitude_uncertainty
            relative_eigengap[radial_index, angular_index] = sample.relative_eigengap
        for radial_index in positive_indices:
            sample = _sample_node(outward, radial_k[radial_index] * direction)
            omega[radial_index, angular_index] = sample.omega
            amplitude[radial_index, angular_index] = sample.amplitude
            uncertainty[radial_index, angular_index] = sample.frequency_uncertainty
            amplitude_uncertainty[radial_index, angular_index] = sample.amplitude_uncertainty
            relative_eigengap[radial_index, angular_index] = sample.relative_eigengap

    return PolarBranchSurface(
        radial_offsets,
        angles,
        omega,
        amplitude,
        uncertainty,
        amplitude_uncertainty,
        relative_eigengap,
    )


@dataclass(frozen=True, slots=True)
class RegisteredGridConvergence:
    """Three-level complex-response and phase-discrepancy control record."""

    surfaces: tuple[PolarBranchSurface, PolarBranchSurface, PolarBranchSurface]
    responses: tuple[BranchResponse, BranchResponse, BranchResponse]
    complex_response_errors: FloatArray
    nested_frequency_errors: FloatArray
    accumulated_phase_errors: FloatArray
    grid_shapes: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    def __post_init__(self) -> None:
        surfaces = tuple(self.surfaces)
        responses = tuple(self.responses)
        if len(surfaces) != 3 or any(
            not isinstance(surface, PolarBranchSurface) for surface in surfaces
        ):
            raise ValueError("surfaces must contain three PolarBranchSurface records")
        if len(responses) != 3 or any(
            not isinstance(response, BranchResponse) for response in responses
        ):
            raise ValueError("responses must contain three BranchResponse records")
        response_errors = _real_array(
            self.complex_response_errors,
            "complex_response_errors",
            ndim=1,
            nonnegative=True,
        )
        frequency_errors = _real_array(
            self.nested_frequency_errors,
            "nested_frequency_errors",
            ndim=1,
            nonnegative=True,
        )
        phase_errors = _real_array(
            self.accumulated_phase_errors,
            "accumulated_phase_errors",
            ndim=1,
            nonnegative=True,
        )
        if any(array.shape != (2,) for array in (response_errors, frequency_errors, phase_errors)):
            raise ValueError("convergence error arrays must each have shape (2,)")
        expected_shapes = tuple(surface.omega.shape for surface in surfaces)
        if tuple(self.grid_shapes) != expected_shapes:
            raise ValueError("grid_shapes must agree with the registered surfaces")
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "responses", responses)
        object.__setattr__(self, "complex_response_errors", _read_only_float(response_errors))
        object.__setattr__(self, "nested_frequency_errors", _read_only_float(frequency_errors))
        object.__setattr__(self, "accumulated_phase_errors", _read_only_float(phase_errors))
        object.__setattr__(self, "grid_shapes", expected_shapes)

    @property
    def finest_response(self) -> BranchResponse:
        """Return the accepted finest-grid response by identity."""

        return self.responses[-1]


def _registered_axes_are_nested(
    coarse: PolarBranchSurface,
    fine: PolarBranchSurface,
) -> None:
    expected_shape = (2 * (coarse.q.size - 1) + 1, 2 * coarse.theta.size)
    if fine.omega.shape != expected_shape:
        raise ValueError(
            f"registered surface expected doubled shape {expected_shape}, got {fine.omega.shape}"
        )
    if not np.allclose(coarse.q, fine.q[::2], rtol=0.0, atol=1.0e-13):
        raise ValueError("registered grids must use nested q nodes")
    if not np.allclose(coarse.theta, fine.theta[::2], rtol=0.0, atol=1.0e-13):
        raise ValueError("registered grids must use nested theta nodes")


def _responses_share_reference(responses: Sequence[BranchResponse]) -> None:
    first = responses[0]
    for response in responses[1:]:
        if not np.array_equal(response.time, first.time):
            raise ValueError("registered responses must use identical time nodes")
        if response.carrier_frequency != first.carrier_frequency:
            raise ValueError("registered responses must use the same carrier frequency")


def _rms_complex(values: ComplexArray, name: str) -> float:
    """Evaluate an overflow-resistant complex root-mean-square norm."""

    with np.errstate(over="ignore", invalid="ignore"):
        magnitudes = np.abs(values)
    scale = float(np.max(magnitudes))
    if not np.isfinite(scale):
        raise ValueError(f"{name} must be finite")
    if scale == 0.0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = np.divide(magnitudes, scale)
        rms = np.multiply(scale, np.sqrt(np.mean(np.square(normalized))))
    if not np.isfinite(rms):
        raise ValueError(f"{name} must be finite")
    return float(rms)


def verify_registered_grid_convergence(
    surfaces: Sequence[PolarBranchSurface],
    responses: Sequence[BranchResponse],
    phase_limit: float = 0.05,
    response_rtol: float = 0.05,
    response_atol: float = 1.0e-8,
    maximum_error_ratio: float = 0.5,
) -> RegisteredGridConvergence:
    """Check declared refinement, response discrepancy, rate, and phase discrepancy."""

    surface_tuple = tuple(surfaces)
    response_tuple = tuple(responses)
    if len(surface_tuple) != 3 or any(
        not isinstance(surface, PolarBranchSurface) for surface in surface_tuple
    ):
        raise ValueError("surfaces must contain three PolarBranchSurface records")
    if len(response_tuple) != 3 or any(
        not isinstance(response, BranchResponse) for response in response_tuple
    ):
        raise ValueError("responses must contain three BranchResponse records")
    limit = _finite_real(phase_limit, "phase_limit", positive=True)
    relative_tolerance = _finite_real(
        response_rtol,
        "response_rtol",
        nonnegative=True,
    )
    absolute_tolerance = _finite_real(
        response_atol,
        "response_atol",
        nonnegative=True,
    )
    error_ratio_limit = _finite_real(
        maximum_error_ratio,
        "maximum_error_ratio",
        positive=True,
    )
    if error_ratio_limit >= 1.0:
        raise ValueError("maximum_error_ratio must be less than one")
    _registered_axes_are_nested(surface_tuple[0], surface_tuple[1])
    _registered_axes_are_nested(surface_tuple[1], surface_tuple[2])
    _responses_share_reference(response_tuple)

    response_error_values: list[float] = []
    for previous, current in zip(
        response_tuple[:-1],
        response_tuple[1:],
        strict=True,
    ):
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.subtract(current.demodulated, previous.demodulated)
        response_error_values.append(_rms_complex(difference, "complex response differences"))
    response_errors = np.array(response_error_values, dtype=np.float64)
    finest_scale = _rms_complex(
        response_tuple[-1].demodulated,
        "finest complex response",
    )
    roundoff_floor = 64.0 * np.finfo(np.float64).eps * max(finest_scale, 1.0)
    if response_errors[1] >= response_errors[0] and response_errors[1] > roundoff_floor:
        raise ValueError("complex response differences must decrease under refinement")
    with np.errstate(over="ignore", invalid="ignore"):
        accepted_error = np.add(
            absolute_tolerance,
            np.multiply(relative_tolerance, finest_scale),
        )
    if response_errors[1] > accepted_error:
        raise ValueError(
            "finest-grid complex response discrepancy "
            f"{response_errors[1]:.6g} exceeds "
            f"{float(accepted_error):.6g}"
        )
    rate_floor = max(absolute_tolerance, roundoff_floor)
    if response_errors[1] > rate_floor:
        error_ratio = response_errors[1] / response_errors[0]
        if error_ratio > error_ratio_limit:
            raise ValueError(
                "complex response convergence rate "
                f"{error_ratio:.6g} exceeds {error_ratio_limit:.6g}"
            )

    nested_errors = np.array(
        [
            estimate_nested_frequency_error(previous.omega, current.omega)
            for previous, current in zip(
                surface_tuple[:-1],
                surface_tuple[1:],
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    maximum_time = float(np.max(response_tuple[0].time))
    accumulated = np.empty(2, dtype=np.float64)
    for index, (previous, current, nested_error) in enumerate(
        zip(surface_tuple[:-1], surface_tuple[1:], nested_errors, strict=True)
    ):
        combined_error = max(
            float(nested_error),
            float(np.max(previous.frequency_error)),
            float(np.max(current.frequency_error)),
        )
        accumulated[index] = assert_phase_accuracy(maximum_time, combined_error, limit)

    return RegisteredGridConvergence(
        surface_tuple,  # type: ignore[arg-type]
        response_tuple,  # type: ignore[arg-type]
        response_errors,
        nested_errors,
        accumulated,
        tuple(surface.omega.shape for surface in surface_tuple),  # type: ignore[arg-type]
    )


def _refine_q(q: FloatArray) -> FloatArray:
    refined = np.empty(2 * q.size - 1, dtype=np.float64)
    refined[::2] = q
    refined[1::2] = 0.5 * q[:-1] + 0.5 * q[1:]
    return refined


def _refine_theta(theta: FloatArray) -> FloatArray:
    return 2.0 * np.pi * np.arange(2 * theta.size) / (2 * theta.size)


def integrate_registered_grid_convergence(
    evaluator_factory: RadialEvaluatorFactory,
    q: ArrayLike,
    theta: ArrayLike,
    time: ArrayLike,
    k0: float,
    omega_reference: float,
    source_radius: float,
    window_sigma: float,
    chunk_size: int = 64,
    fourier_normalization: float = (2.0 * np.pi) ** -2,
    phase_limit: float = 0.05,
    response_rtol: float = 0.05,
    response_atol: float = 1.0e-8,
    maximum_error_ratio: float = 0.5,
) -> RegisteredGridConvergence:
    """Build and integrate the three registered refinement levels."""

    base_q = _validate_q(q)
    base_theta = _validate_theta(theta)
    q_levels = (base_q, _refine_q(base_q), _refine_q(_refine_q(base_q)))
    theta_levels = (
        base_theta,
        _refine_theta(base_theta),
        _refine_theta(_refine_theta(base_theta)),
    )
    surfaces = tuple(
        build_tracked_surface(evaluator_factory, q_level, theta_level, k0)
        for q_level, theta_level in zip(q_levels, theta_levels, strict=True)
    )
    responses = tuple(
        integrate_branch_response(
            surface,
            time,
            k0,
            omega_reference,
            source_radius,
            window_sigma,
            chunk_size,
            fourier_normalization,
            phase_limit,
        )
        for surface in surfaces
    )
    return verify_registered_grid_convergence(
        surfaces,
        responses,
        phase_limit,
        response_rtol,
        response_atol,
        maximum_error_ratio,
    )
