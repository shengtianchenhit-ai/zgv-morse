"""Uniform response asymptotics for a weakly split ZGV critical ring."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Complex, Integral, Real
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import j0, jv

if TYPE_CHECKING:
    from .critical_points import CriticalPoint
    from .green_response import BranchNodeSample


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BooleanArray = NDArray[np.bool_]


def _finite_real(value: object, name: str, *, positive: bool = False) -> float:
    """Return a strictly typed finite real scalar."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar")
    try:
        scalar = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _finite_complex(value: object, name: str, *, nonzero: bool = False) -> complex:
    """Return a strictly typed finite complex scalar."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Complex):
        raise TypeError(f"{name} must be a finite numeric scalar")
    try:
        scalar = complex(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
        raise ValueError(f"{name} must be finite")
    if nonzero and scalar == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return scalar


def _real_array(
    values: ArrayLike,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> FloatArray:
    """Copy and validate a nonempty finite real scalar or array."""

    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array") from error
    if (
        candidate.size == 0
        or np.iscomplexobj(candidate)
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a nonempty real numeric array")
    try:
        array = np.array(candidate, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} entries must be finite")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} entries must be positive")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} entries must be nonnegative")
    return array


def _complex_array(values: ArrayLike, name: str) -> ComplexArray:
    """Copy and validate a nonempty finite numeric scalar or array."""

    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if (
        candidate.size == 0
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a nonempty numeric array")
    try:
        array = np.array(candidate, dtype=np.complex128, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} entries must be finite")
    return array


@dataclass(frozen=True, slots=True)
class UniformParameters:
    """Parameters in the leading uniform ZGV-ring response theorem.

    The canonical channel assumes a nonzero angular mean amplitude ``A0`` as
    required by the theorem.  Exceptional nodal channels with ``A0 == 0`` need
    a different leading asymptotic amplitude and are outside this model.
    """

    omega0: float
    k0: float
    curvature: float
    epsilon: float
    V0: float
    V4: float
    amplitude: complex
    fourier_normalization: float = (2.0 * np.pi) ** -2

    def __post_init__(self) -> None:
        object.__setattr__(self, "omega0", _finite_real(self.omega0, "omega0", positive=True))
        object.__setattr__(self, "k0", _finite_real(self.k0, "k0", positive=True))
        object.__setattr__(
            self,
            "curvature",
            _finite_real(self.curvature, "curvature", positive=True),
        )
        for name in ("epsilon", "V0", "V4"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        object.__setattr__(
            self,
            "amplitude",
            _finite_complex(self.amplitude, "amplitude", nonzero=True),
        )
        object.__setattr__(
            self,
            "fourier_normalization",
            _finite_real(
                self.fourier_normalization,
                "fourier_normalization",
                positive=True,
            ),
        )


def _validated_parameters(parameters: UniformParameters) -> UniformParameters:
    if not isinstance(parameters, UniformParameters):
        raise TypeError("parameters must be UniformParameters")
    return parameters


def uniform_prefactor(parameters: UniformParameters) -> complex:
    """Return the full Fourier-normalized radial stationary-phase prefactor."""

    parameters = _validated_parameters(parameters)
    if parameters.curvature <= 0.0:
        raise ValueError("the implemented theorem requires positive radial curvature")
    radial = np.exp(-0.25j * np.pi) * np.sqrt(2.0 * np.pi / parameters.curvature)
    prefactor = (
        parameters.fourier_normalization
        * parameters.amplitude
        * parameters.k0
        * radial
        * 2.0
        * np.pi
    )
    if not np.isfinite(prefactor):
        raise ValueError("the uniform prefactor must be finite")
    return complex(prefactor)


def _phase(frequency: float, time: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        phase = frequency * time
    if not np.isfinite(phase).all():
        raise ValueError("carrier phase must be finite")
    return phase


def uniform_bessel_response(
    time: ArrayLike,
    parameters: UniformParameters,
) -> ComplexArray:
    """Evaluate the leading response uniformly in ``epsilon * V4 * time``."""

    parameters = _validated_parameters(parameters)
    times = _real_array(time, "time", positive=True)
    shifted_frequency = parameters.omega0 + parameters.epsilon * parameters.V0
    if not np.isfinite(shifted_frequency):
        raise ValueError("the shifted carrier frequency must be finite")
    carrier = np.exp(-1j * _phase(shifted_frequency, times))
    with np.errstate(over="ignore", invalid="ignore"):
        tau = parameters.epsilon * parameters.V4 * times
    if not np.isfinite(tau).all():
        raise ValueError("the transition variable must be finite")
    response = uniform_prefactor(parameters) * carrier * times**-0.5 * j0(tau)
    if not np.isfinite(response).all():
        raise ValueError("the uniform response must be finite")
    return np.array(response, dtype=np.complex128, copy=True)


def scale_transition_response(
    time: ArrayLike,
    response: ArrayLike,
    parameters: UniformParameters,
) -> ComplexArray:
    """Remove the carrier, stationary-phase decay, and analytic prefactor."""

    parameters = _validated_parameters(parameters)
    times = _real_array(time, "time", positive=True)
    values = _complex_array(response, "response")
    if values.shape != times.shape:
        raise ValueError("response and time must have the same shape")
    shifted_frequency = parameters.omega0 + parameters.epsilon * parameters.V0
    if not np.isfinite(shifted_frequency):
        raise ValueError("the shifted carrier frequency must be finite")
    demodulation = np.exp(1j * _phase(shifted_frequency, times))
    scaled = times**0.5 * demodulation * values / uniform_prefactor(parameters)
    if not np.isfinite(scaled).all():
        raise ValueError("the scaled transition response must be finite")
    return np.array(scaled, dtype=np.complex128, copy=True)


def weighted_bessel_kernel(
    tau: ArrayLike,
    angular_coefficients: Mapping[int, complex],
) -> ComplexArray:
    """Integrate an angular Fourier weight against the fourfold phase kernel."""

    transition = _real_array(tau, "tau")
    if not isinstance(angular_coefficients, Mapping):
        raise TypeError("angular_coefficients must be a mapping")
    result = np.zeros(transition.shape, dtype=np.complex128)
    for raw_harmonic, raw_coefficient in angular_coefficients.items():
        if isinstance(raw_harmonic, (bool, np.bool_)) or not isinstance(
            raw_harmonic, Integral
        ):
            raise TypeError("angular harmonics must be integers")
        harmonic = int(raw_harmonic)
        coefficient = _finite_complex(
            raw_coefficient,
            f"angular coefficient {harmonic}",
        )
        if harmonic % 4 != 0:
            continue
        order = -harmonic // 4
        phase = (-1j) ** (order % 4)
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                weighted_coefficient = np.multiply(
                    np.complex128(coefficient),
                    np.complex128(phase),
                )
                term = np.multiply(weighted_coefficient, jv(order, transition))
                result = np.add(result, term)
        except (OverflowError, ValueError) as error:
            raise ValueError("weighted Bessel kernel must be finite") from error
        if not np.isfinite(result).all():
            raise ValueError("weighted Bessel kernel must be finite")
    return result


def _splitting_rate(epsilon: object, V4: object) -> float:
    epsilon_value = _finite_real(epsilon, "epsilon")
    coefficient = _finite_real(V4, "V4")
    with np.errstate(over="ignore", invalid="ignore"):
        rate = abs(epsilon_value * coefficient)
    if not np.isfinite(rate):
        raise ValueError("abs(epsilon * V4) must be finite")
    return float(rate)


def crossover_time(epsilon: float, V4: float) -> float:
    """Return the distinguished transition time ``1 / |epsilon V4|``."""

    rate = _splitting_rate(epsilon, V4)
    if rate == 0.0:
        raise ValueError("epsilon and V4 must define a nonzero first-order splitting")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        time = np.divide(np.float64(1.0), np.float64(rate))
    if not np.isfinite(time):
        raise ValueError("the crossover time must be finite")
    return float(time)


def critical_frequency_separation(epsilon: float, V4: float) -> float:
    """Return the minimum-to-saddle frequency separation ``2 |epsilon V4|``."""

    separation = 2.0 * _splitting_rate(epsilon, V4)
    if not np.isfinite(separation):
        raise ValueError("the critical-frequency separation must be finite")
    return separation


def signed_modulation_rate(epsilon: float, V4: float) -> float:
    """Return the positive Bessel modulation rate ``|epsilon V4|``."""

    return _splitting_rate(epsilon, V4)


def bessel_overlap_is_valid(
    time: ArrayLike,
    epsilon: float,
    V4: float,
    second_order_bound: ArrayLike,
) -> BooleanArray:
    """Return where the late-Bessel and first-order perturbation regimes overlap."""

    times = _real_array(time, "time", positive=True)
    bound = _real_array(second_order_bound, "second_order_bound", nonnegative=True)
    epsilon_value = _finite_real(epsilon, "epsilon")
    rate = _splitting_rate(epsilon_value, V4)
    try:
        times, bound = np.broadcast_arrays(times, bound)
    except ValueError as error:
        raise ValueError("time and second_order_bound must have broadcast-compatible shapes") from error
    with np.errstate(over="ignore", invalid="ignore"):
        epsilon_squared = np.multiply(np.float64(epsilon_value), np.float64(epsilon_value))
        tau = np.multiply(np.float64(rate), times)
        remainder = np.multiply(np.multiply(epsilon_squared, times), bound)
    if not np.isfinite(tau).all() or not np.isfinite(remainder).all():
        raise ValueError("overlap regime variables must be finite")
    return np.array((tau > 1.0) & (remainder < 0.1), dtype=np.bool_, copy=True)


def _nonnegative_scalar(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _symmetric_hessian(values: ArrayLike) -> FloatArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("hessian must be a finite real array with shape (2, 2)") from error
    if (
        candidate.shape != (2, 2)
        or np.iscomplexobj(candidate)
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError("hessian must be a finite real array with shape (2, 2)")
    try:
        hessian = np.array(candidate, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("hessian must be a finite real array with shape (2, 2)") from error
    if not np.isfinite(hessian).all():
        raise ValueError("hessian must be a finite real array with shape (2, 2)")
    scale = max(
        float(np.max(np.abs(hessian), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    if float(np.max(np.abs(hessian - hessian.T), initial=0.0)) > 1.0e-12 * scale:
        raise ValueError("hessian must be symmetric within relative tolerance 1e-12")
    result = 0.5 * hessian + 0.5 * hessian.T
    result.setflags(write=False)
    return result


def _read_only_real_vector(values: ArrayLike, name: str, *, positive: bool) -> FloatArray:
    result = _real_array(values, name, positive=positive)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    result.setflags(write=False)
    return result


def _read_only_complex_vector(values: ArrayLike, name: str) -> ComplexArray:
    result = _complex_array(values, name)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MorseContribution:
    """One exact Cartesian Morse point in the fixed-anisotropy sum."""

    omega: float
    hessian: FloatArray
    amplitude: complex
    frequency_uncertainty: float
    hessian_uncertainty: float
    amplitude_uncertainty: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "omega", _finite_real(self.omega, "omega", positive=True))
        object.__setattr__(self, "hessian", _symmetric_hessian(self.hessian))
        object.__setattr__(self, "amplitude", _finite_complex(self.amplitude, "amplitude"))
        for name in (
            "frequency_uncertainty",
            "hessian_uncertainty",
            "amplitude_uncertainty",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_scalar(getattr(self, name), name),
            )


@dataclass(frozen=True, slots=True)
class DemodulatedAsymptoticResponse:
    """A carrier-demodulated stationary-phase response and phase-discrepancy gate."""

    time: FloatArray
    carrier_frequency: float
    demodulated: ComplexArray
    maximum_accumulated_phase_error: float = 0.0
    fitted_amplitude: None = field(default=None, init=False)
    fitted_phase: None = field(default=None, init=False)
    fitted_frequency: None = field(default=None, init=False)
    fitted_time_shift: None = field(default=None, init=False)

    def __post_init__(self) -> None:
        time = _read_only_real_vector(self.time, "time", positive=True)
        carrier = _finite_real(
            self.carrier_frequency,
            "carrier_frequency",
            positive=True,
        )
        demodulated = _read_only_complex_vector(self.demodulated, "demodulated")
        if demodulated.shape != time.shape:
            raise ValueError("demodulated and time must have the same shape")
        accumulated = _nonnegative_scalar(
            self.maximum_accumulated_phase_error,
            "maximum_accumulated_phase_error",
        )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "carrier_frequency", carrier)
        object.__setattr__(self, "demodulated", demodulated)
        object.__setattr__(self, "maximum_accumulated_phase_error", accumulated)

    def analytic_signal(self) -> ComplexArray:
        """Restore the carrier and return independent writable storage."""

        carrier = np.exp(-1j * _phase(self.carrier_frequency, self.time))
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.multiply(self.demodulated, carrier)
        if not np.isfinite(result).all():
            raise ValueError("analytic asymptotic response must be finite")
        return np.array(result, dtype=np.complex128, copy=True)


def _morse_denominator_and_signature(
    contribution: MorseContribution,
) -> tuple[float, int]:
    eigenvalues = np.linalg.eigvalsh(contribution.hessian)
    with np.errstate(over="ignore", invalid="ignore"):
        separation_threshold = np.multiply(
            10.0,
            contribution.hessian_uncertainty,
        )
    if float(np.min(np.abs(eigenvalues))) <= separation_threshold:
        raise ValueError("Morse Hessian is not resolved")
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = np.multiply(
            np.sqrt(abs(eigenvalues[0])),
            np.sqrt(abs(eigenvalues[1])),
        )
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Morse Hessian determinant scale must be finite and positive")
    signature = int(
        np.count_nonzero(eigenvalues > 0.0)
        - np.count_nonzero(eigenvalues < 0.0)
    )
    return float(denominator), signature


def morse_stationary_phase_response(
    time: ArrayLike,
    contributions: Iterable[MorseContribution],
    omega_reference: float,
    fourier_normalization: float = (2.0 * np.pi) ** -2,
    phase_limit: float = 0.05,
) -> DemodulatedAsymptoticResponse:
    """Evaluate the exact-Morse response without fitted alignment parameters."""

    times = _real_array(time, "time", positive=True)
    if times.ndim != 1:
        raise ValueError("time must be a one-dimensional array")
    carrier = _finite_real(
        omega_reference,
        "omega_reference",
        positive=True,
    )
    normalization = _finite_real(
        fourier_normalization,
        "fourier_normalization",
        positive=True,
    )
    limit = _finite_real(phase_limit, "phase_limit", positive=True)
    try:
        contribution_tuple = tuple(contributions)
    except TypeError as error:
        raise TypeError("contributions must be an iterable of MorseContribution records") from error
    if not contribution_tuple:
        raise ValueError("contributions must not be empty")
    if any(not isinstance(item, MorseContribution) for item in contribution_tuple):
        raise TypeError("contributions must contain only MorseContribution records")

    result = np.zeros(times.shape, dtype=np.complex128)
    maximum_frequency_uncertainty = max(
        item.frequency_uncertainty for item in contribution_tuple
    )
    from .green_response import assert_phase_accuracy

    accumulated = assert_phase_accuracy(
        float(np.max(times)),
        maximum_frequency_uncertainty,
        limit,
    )
    for contribution in contribution_tuple:
        denominator, signature = _morse_denominator_and_signature(contribution)
        delta_frequency = contribution.omega - carrier
        oscillation = np.exp(-1j * _phase(delta_frequency, times))
        signature_phase = np.exp(-0.25j * np.pi * signature)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            coefficient = np.divide(
                np.multiply(contribution.amplitude, signature_phase),
                denominator,
            )
            result = np.add(result, np.multiply(coefficient, oscillation))
        if not np.isfinite(result).all():
            raise ValueError("Morse stationary-phase accumulation must be finite")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        prefactor = np.divide(2.0 * np.pi * normalization, times)
        result = np.multiply(result, prefactor)
    if not np.isfinite(result).all():
        raise ValueError("Morse stationary-phase response must be finite")
    return DemodulatedAsymptoticResponse(times, carrier, result, accumulated)


def build_morse_contribution(
    point: CriticalPoint,
    node_sample: BranchNodeSample,
    k0: float,
    source_radius: float,
    window_sigma: float,
) -> MorseContribution:
    """Combine an exact critical point with its full-wave Cartesian amplitude.

    The modal amplitude is multiplied by the Gaussian source and branch window.
    No polar ``k`` Jacobian appears because the contribution uses the Cartesian
    Hessian; the polar Jacobian cancels in that Hessian transformation.
    """

    from .critical_points import CriticalPoint
    from .green_response import BranchNodeSample

    if not isinstance(point, CriticalPoint):
        raise TypeError("point must be a CriticalPoint")
    if not isinstance(node_sample, BranchNodeSample):
        raise TypeError("node_sample must be a BranchNodeSample")
    ring_radius = _finite_real(k0, "k0", positive=True)
    source = _nonnegative_scalar(source_radius, "source_radius")
    sigma = _finite_real(window_sigma, "window_sigma", positive=True)
    frequency_tolerance = node_sample.frequency_uncertainty
    frequency_tolerance += 64.0 * np.finfo(np.float64).eps * max(
        point.omega,
        node_sample.omega,
        1.0,
    )
    if abs(point.omega - node_sample.omega) > frequency_tolerance:
        raise ValueError("critical-point and modal-node frequencies are inconsistent")

    q = point.radius - ring_radius
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        source_argument = np.multiply(0.5 * source, point.radius)
        source_weight = np.exp(-np.multiply(source_argument, source_argument))
        window_argument = np.divide(q, sigma)
        window_weight = np.exp(-np.power(window_argument, 8))
        weight = np.multiply(source_weight, window_weight)
        amplitude = np.multiply(node_sample.amplitude, weight)
        amplitude_uncertainty = np.multiply(
            node_sample.amplitude_uncertainty,
            abs(weight),
        )
    if (
        not np.isfinite(weight)
        or not np.isfinite(amplitude)
        or not np.isfinite(amplitude_uncertainty)
    ):
        raise ValueError("Morse source-window amplitude must be finite")
    return MorseContribution(
        node_sample.omega,
        point.hessian,
        complex(amplitude),
        node_sample.frequency_uncertainty,
        point.hessian_uncertainty,
        float(amplitude_uncertainty),
    )
