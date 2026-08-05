"""Normalization-invariant eigenmode sensitivities and reduced resolvents."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import LinAlgWarning, eigvalsh, solve

from .mode_tracking import TrackedMode
from .spectral_plate import PlateMatrices, WavevectorDerivatives


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]


def _finite_real(value: object, name: str, *, positive: bool = False) -> float:
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


def _numeric_vector(values: object, name: str) -> ComplexArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric vector") from error
    if (
        candidate.ndim != 1
        or candidate.size == 0
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a nonempty numeric vector")
    try:
        vector = np.array(candidate, dtype=np.complex128, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric vector") from error
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} entries must be finite")
    return vector


def _real_array(
    values: object,
    name: str,
    *,
    one_dimensional: bool = False,
) -> FloatArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array") from error
    if (
        candidate.ndim == 0
        or candidate.size == 0
        or (one_dimensional and candidate.ndim != 1)
        or np.iscomplexobj(candidate)
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        dimension = " one-dimensional" if one_dimensional else ""
        raise ValueError(f"{name} must be a nonempty{dimension} real array")
    try:
        array = np.array(candidate, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} entries must be finite")
    return array


def _harmonic_orders(values: object) -> IntegerArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("order must be a consecutive integer vector starting at zero") from error
    if (
        candidate.ndim != 1
        or candidate.size == 0
        or not np.issubdtype(candidate.dtype, np.integer)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError("order must be a consecutive integer vector starting at zero")
    order = np.array(candidate, dtype=np.int64, copy=True)
    if not np.array_equal(order, np.arange(order.size, dtype=np.int64)):
        raise ValueError("order must be a consecutive integer vector starting at zero")
    return order


def _read_only(array: NDArray[np.generic]) -> None:
    array.setflags(write=False)


def _hermitian_matrix(values: object, name: str, dimension: int | None = None) -> ComplexArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite Hermitian matrix") from error
    if (
        candidate.ndim != 2
        or candidate.shape[0] != candidate.shape[1]
        or candidate.shape[0] == 0
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a nonempty square numeric matrix")
    try:
        matrix = np.array(candidate, dtype=np.complex128, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite Hermitian matrix") from error
    if dimension is not None and matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} entries must be finite")
    scale = float(np.max(np.abs(matrix), initial=0.0))
    defect = float(np.max(np.abs(matrix - matrix.conj().T), initial=0.0))
    if defect > 1.0e-11 * max(scale, np.finfo(np.float64).tiny):
        raise ValueError(f"{name} must be Hermitian")
    return 0.5 * (matrix + matrix.conj().T)


def _validated_mass(mass: object, dimension: int) -> ComplexArray:
    matrix = _hermitian_matrix(mass, "mass", dimension)
    try:
        eigenvalues = eigvalsh(matrix, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("mass must be Hermitian positive definite") from error
    if eigenvalues[0] <= 0.0:
        raise ValueError("mass must be Hermitian positive definite")
    return matrix


def _mass_normalize(vector: object, mass: object) -> tuple[ComplexArray, ComplexArray]:
    mode = _numeric_vector(vector, "vector")
    mass_matrix = _validated_mass(mass, mode.size)
    norm_squared = float(np.real(np.vdot(mode, mass_matrix @ mode)))
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("mode must have a finite positive mass norm")
    return mode / np.sqrt(norm_squared), mass_matrix


def _real_form(vector: ComplexArray, matrix: ComplexArray, name: str) -> float:
    value = np.vdot(vector, matrix @ vector)
    tolerance = 1.0e-11 * max(abs(value), 1.0)
    if not np.isfinite(value) or abs(float(np.imag(value))) > tolerance:
        raise ValueError(f"{name} must define a real Hermitian quadratic form")
    return float(np.real(value))


def frequency_sensitivity(
    omega: float,
    vector: object,
    mass: object,
    k_epsilon: object,
) -> float:
    """Return the first frequency sensitivity per unit perturbation."""

    frequency = _finite_real(omega, "omega", positive=True)
    mode, _mass = _mass_normalize(vector, mass)
    perturbation = _hermitian_matrix(k_epsilon, "k_epsilon", mode.size)
    return _real_form(mode, perturbation, "k_epsilon") / (2.0 * frequency)


@dataclass(frozen=True, slots=True)
class RadialSensitivity:
    """First perturbation and mixed radial/perturbation sensitivities."""

    V: float
    B: float
    lambda_epsilon: float
    lambda_radial: float
    lambda_radial_epsilon: float
    mode_radial_derivative: ComplexArray
    differentiated_mode_residual: float

    def __post_init__(self) -> None:
        for name in (
            "V",
            "B",
            "lambda_epsilon",
            "lambda_radial",
            "lambda_radial_epsilon",
        ):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        residual = _finite_real(
            self.differentiated_mode_residual,
            "differentiated_mode_residual",
        )
        if residual < 0.0:
            raise ValueError("differentiated_mode_residual must be nonnegative")
        derivative = _numeric_vector(
            self.mode_radial_derivative,
            "mode_radial_derivative",
        )
        derivative.setflags(write=False)
        object.__setattr__(self, "mode_radial_derivative", derivative)
        object.__setattr__(self, "differentiated_mode_residual", residual)


@dataclass(frozen=True, slots=True)
class AngularHarmonics:
    """Real Fourier coefficients on an endpoint-free angular grid."""

    order: IntegerArray
    cosine: FloatArray
    sine: FloatArray

    def __post_init__(self) -> None:
        order = _harmonic_orders(self.order)
        cosine = _real_array(self.cosine, "cosine", one_dimensional=True)
        sine = _real_array(self.sine, "sine", one_dimensional=True)
        if cosine.shape != order.shape or sine.shape != order.shape:
            raise ValueError("order, cosine, and sine must have the same shape")
        _read_only(order)
        _read_only(cosine)
        _read_only(sine)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "cosine", cosine)
        object.__setattr__(self, "sine", sine)


@dataclass(frozen=True, slots=True)
class CubicHarmonicReport:
    """Fourfold sensitivity coefficients and symmetry diagnostics."""

    V0: float
    V4: float
    V8: float
    reconstruction: FloatArray
    periodicity_defect: float
    mirror_defect: float
    non_cubic_leakage: float

    def __post_init__(self) -> None:
        for name in ("V0", "V4", "V8"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        reconstruction = _real_array(
            self.reconstruction,
            "reconstruction",
            one_dimensional=True,
        )
        _read_only(reconstruction)
        object.__setattr__(self, "reconstruction", reconstruction)
        for name in ("periodicity_defect", "mirror_defect", "non_cubic_leakage"):
            value = _finite_real(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class PhysicalCubicShift:
    """Physical anisotropy and frequency shifts with epsilon applied once."""

    delta_c: FloatArray
    frequency_shift: FloatArray

    def __post_init__(self) -> None:
        delta_c = _real_array(self.delta_c, "delta_c")
        frequency_shift = _real_array(self.frequency_shift, "frequency_shift")
        if delta_c.shape != frequency_shift.shape:
            raise ValueError("delta_c and frequency_shift must have the same shape")
        _read_only(delta_c)
        _read_only(frequency_shift)
        object.__setattr__(self, "delta_c", delta_c)
        object.__setattr__(self, "frequency_shift", frequency_shift)


def _require_accurate_eigenpair(
    stiffness: ComplexArray,
    mass: ComplexArray,
    eigenvalue: float,
    mode: ComplexArray,
) -> None:
    left = stiffness @ mode
    right = eigenvalue * (mass @ mode)
    residual = np.linalg.norm(left - right) / max(
        np.linalg.norm(left) + np.linalg.norm(right),
        np.finfo(np.float64).tiny,
    )
    if not np.isfinite(residual) or residual > 1.0e-8:
        raise ValueError("vector and eigenvalue do not form a sufficiently accurate eigenpair")


def differentiated_mode(
    stiffness: object,
    mass: object,
    eigenvalue: float,
    vector: object,
    k_parameter: object,
) -> tuple[ComplexArray, float, float]:
    """Differentiate a simple mass-normalized generalized eigenmode."""

    mode, mass_matrix = _mass_normalize(vector, mass)
    dimension = mode.size
    stiffness_matrix = _hermitian_matrix(stiffness, "stiffness", dimension)
    parameter_matrix = _hermitian_matrix(k_parameter, "k_parameter", dimension)
    eigenvalue_value = _finite_real(eigenvalue, "eigenvalue")
    if eigenvalue_value < 0.0:
        raise ValueError("eigenvalue must be nonnegative")
    _require_accurate_eigenpair(stiffness_matrix, mass_matrix, eigenvalue_value, mode)

    lambda_parameter = _real_form(mode, parameter_matrix, "k_parameter")
    operator = stiffness_matrix - eigenvalue_value * mass_matrix
    mass_mode = mass_matrix @ mode
    operator_scale = max(
        float(np.linalg.norm(stiffness_matrix, ord=2)),
        abs(eigenvalue_value) * float(np.linalg.norm(mass_matrix, ord=2)),
        np.finfo(np.float64).tiny,
    )
    constraint_scale = max(
        float(np.linalg.norm(mass_mode)),
        np.finfo(np.float64).tiny,
    )
    augmented = np.block(
        [
            [
                operator / operator_scale,
                (mass_mode / constraint_scale)[:, np.newaxis],
            ],
            [
                ((mode.conj() @ mass_matrix) / constraint_scale)[np.newaxis, :],
                np.zeros((1, 1), complex),
            ],
        ]
    )
    right_hand_side = -(parameter_matrix - lambda_parameter * mass_matrix) @ mode
    augmented_rhs = np.concatenate(
        (
            right_hand_side / operator_scale,
            np.zeros(1, dtype=np.complex128),
        )
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            solution = solve(
                augmented,
                augmented_rhs,
                assume_a="gen",
                check_finite=False,
            )
    except (LinAlgWarning, np.linalg.LinAlgError, ValueError) as error:
        raise ValueError(
            "simple-eigenvalue reduced-resolvent solve is singular or ill-conditioned"
        ) from error
    derivative = np.asarray(solution[:-1], dtype=np.complex128)
    if not np.isfinite(derivative).all():
        raise ValueError("reduced-resolvent solve produced a non-finite mode derivative")

    equation_residual = (
        operator @ derivative + (parameter_matrix - lambda_parameter * mass_matrix) @ mode
    )
    gauge_residual = abs(np.vdot(mode, mass_matrix @ derivative))
    scale = np.linalg.norm(parameter_matrix @ mode) + abs(lambda_parameter) * np.linalg.norm(
        mass_matrix @ mode
    )
    relative_residual = max(
        float(np.linalg.norm(equation_residual) / max(scale, np.finfo(np.float64).tiny)),
        float(gauge_residual),
    )
    if not np.isfinite(relative_residual):
        raise ValueError("differentiated-mode residual is non-finite")
    if relative_residual > 1.0e-8:
        raise ValueError("differentiated-mode residual exceeds 1e-8")
    derivative = np.array(derivative, copy=True)
    derivative.setflags(write=False)
    return derivative, lambda_parameter, relative_residual


def radial_frequency_sensitivity(
    stiffness: object,
    mass: object,
    omega: float,
    vector: object,
    k_radial: object,
    k_epsilon: object,
    k_radial_epsilon: object,
) -> RadialSensitivity:
    """Return ``V`` and ``B = dV/dk`` for a simple generalized eigenmode."""

    frequency = _finite_real(omega, "omega", positive=True)
    mode, mass_matrix = _mass_normalize(vector, mass)
    dimension = mode.size
    stiffness_matrix = _hermitian_matrix(stiffness, "stiffness", dimension)
    radial_matrix = _hermitian_matrix(k_radial, "k_radial", dimension)
    perturbation_matrix = _hermitian_matrix(k_epsilon, "k_epsilon", dimension)
    mixed_matrix = _hermitian_matrix(
        k_radial_epsilon,
        "k_radial_epsilon",
        dimension,
    )
    mode_radial, lambda_radial, residual = differentiated_mode(
        stiffness_matrix,
        mass_matrix,
        frequency**2,
        mode,
        radial_matrix,
    )
    lambda_epsilon = _real_form(mode, perturbation_matrix, "k_epsilon")
    direct_mixed = _real_form(mode, mixed_matrix, "k_radial_epsilon")
    differentiated_form = np.vdot(mode, perturbation_matrix @ mode_radial)
    if not np.isfinite(differentiated_form):
        raise ValueError("differentiated perturbation form is non-finite")
    lambda_radial_epsilon = direct_mixed + 2.0 * float(np.real(differentiated_form))
    V = lambda_epsilon / (2.0 * frequency)
    B = lambda_radial_epsilon / (2.0 * frequency) - lambda_epsilon * lambda_radial / (
        4.0 * frequency**3
    )
    return RadialSensitivity(
        V=V,
        B=B,
        lambda_epsilon=lambda_epsilon,
        lambda_radial=lambda_radial,
        lambda_radial_epsilon=lambda_radial_epsilon,
        mode_radial_derivative=mode_radial,
        differentiated_mode_residual=residual,
    )


def sensitivity_from_plate(
    mode: TrackedMode,
    base_matrices: PlateMatrices,
    base_derivatives: WavevectorDerivatives,
    perturbation_matrices: PlateMatrices,
    perturbation_derivatives: WavevectorDerivatives,
    theta: float,
) -> RadialSensitivity:
    """Adapt compatible fixed-density plate records to the sensitivity calculation.

    The perturbation record carries only ``K_epsilon`` in its stiffness field;
    its mass must equal the base mass, enforcing the paper's fixed-density
    assumption ``M_epsilon = 0``.  All records must use the same node-major
    discretization.
    """

    if not isinstance(mode, TrackedMode):
        raise TypeError("mode must be a TrackedMode instance")
    if not isinstance(base_matrices, PlateMatrices):
        raise TypeError("base_matrices must be a PlateMatrices instance")
    if not isinstance(base_derivatives, WavevectorDerivatives):
        raise TypeError("base_derivatives must be a WavevectorDerivatives instance")
    if not isinstance(perturbation_matrices, PlateMatrices):
        raise TypeError("perturbation_matrices must be a PlateMatrices instance")
    if not isinstance(perturbation_derivatives, WavevectorDerivatives):
        raise TypeError("perturbation_derivatives must be a WavevectorDerivatives instance")
    angle = _finite_real(theta, "theta")
    base_nodes = np.asarray(base_matrices.nodes)
    perturbation_nodes = np.asarray(perturbation_matrices.nodes)
    if not np.array_equal(base_nodes, perturbation_nodes):
        raise ValueError("base and perturbation plate nodes must be identical")
    base_mass = np.asarray(base_matrices.mass)
    perturbation_mass = np.asarray(perturbation_matrices.mass)
    if base_mass.shape != perturbation_mass.shape or not np.allclose(
        base_mass,
        perturbation_mass,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise ValueError("fixed-density sensitivity requires identical base and perturbation mass")
    uncertainty = max(float(mode.residual), np.finfo(np.float64).eps)
    if not np.isfinite(mode.eigengap) or mode.eigengap <= 10.0 * uncertainty:
        raise ValueError("tracked-mode eigengap must exceed ten times its numerical uncertainty")
    cosine = np.cos(angle)
    sine = np.sin(angle)
    k_radial = cosine * base_derivatives.dkx + sine * base_derivatives.dky
    k_radial_epsilon = cosine * perturbation_derivatives.dkx + sine * perturbation_derivatives.dky
    return radial_frequency_sensitivity(
        base_matrices.stiffness,
        base_matrices.mass,
        mode.omega,
        mode.vector,
        k_radial,
        perturbation_matrices.stiffness,
        k_radial_epsilon,
    )


def extract_angular_harmonics(
    theta: object,
    values: object,
    max_order: int,
) -> AngularHarmonics:
    """Extract deterministic real Fourier coefficients through ``max_order``."""

    angles = _real_array(theta, "theta", one_dimensional=True)
    samples = _real_array(values, "values", one_dimensional=True)
    if angles.shape != samples.shape:
        raise ValueError("theta and values must have the same shape")
    if type(max_order) is not int:
        raise TypeError("max_order must be a nonnegative built-in integer")
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    if 2 * max_order >= angles.size:
        raise ValueError("angular grid does not resolve the requested harmonics")

    expected = angles[0] + 2.0 * np.pi * np.arange(angles.size) / angles.size
    if not np.allclose(angles, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError("theta must be a uniform endpoint-free full-period grid")

    order = np.arange(max_order + 1, dtype=np.int64)
    cosine = np.zeros(max_order + 1, dtype=np.float64)
    sine = np.zeros(max_order + 1, dtype=np.float64)
    cosine[0] = float(np.mean(samples))
    for harmonic in order[1:]:
        cosine[harmonic] = 2.0 * float(np.mean(samples * np.cos(harmonic * angles)))
        sine[harmonic] = 2.0 * float(np.mean(samples * np.sin(harmonic * angles)))
    return AngularHarmonics(order, cosine, sine)


def cubic_harmonic_report(
    theta: object,
    values: object,
    max_order: int = 16,
) -> CubicHarmonicReport:
    """Report the constant/fourfold cubic model and discrete symmetry defects."""

    spectrum = extract_angular_harmonics(theta, values, max_order)
    if max_order < 8:
        raise ValueError("max_order must include order 8")
    angles = _real_array(theta, "theta", one_dimensional=True)
    samples = _real_array(values, "values", one_dimensional=True)
    if angles.size % 4 != 0:
        raise ValueError("cubic report grid size must be divisible by four")

    V0 = float(spectrum.cosine[0])
    V4 = float(spectrum.cosine[4])
    reconstruction = V0 + V4 * np.cos(4.0 * angles)
    centered_scale = float(np.linalg.norm(samples - np.mean(samples)))
    residual_norm = float(np.linalg.norm(samples - reconstruction))
    sample_scale = max(float(np.linalg.norm(samples)), 1.0)
    noise_floor = 64.0 * np.finfo(np.float64).eps * sample_scale
    if centered_scale <= noise_floor and residual_norm <= noise_floor:
        non_cubic_leakage = 0.0
    else:
        non_cubic_leakage = residual_norm / max(centered_scale, sample_scale * np.finfo(float).eps)

    angular_step = 2.0 * np.pi / angles.size
    reflection_offset = 2.0 * angles[0] / angular_step
    reflection_integer = int(np.rint(reflection_offset))
    if not np.isclose(reflection_offset, reflection_integer, rtol=0.0, atol=1.0e-10):
        raise ValueError("cubic report grid must be closed under theta -> -theta reflection")
    reflected_indices = (-np.arange(angles.size) - reflection_integer) % angles.size
    return CubicHarmonicReport(
        V0=V0,
        V4=V4,
        V8=float(spectrum.cosine[8]),
        reconstruction=reconstruction,
        periodicity_defect=float(np.max(np.abs(samples - np.roll(samples, angles.size // 4)))),
        mirror_defect=float(np.max(np.abs(samples - samples[reflected_indices]))),
        non_cubic_leakage=non_cubic_leakage,
    )


def physical_cubic_shift(
    epsilon: object,
    delta: float,
    V4: float,
) -> PhysicalCubicShift:
    """Apply epsilon once to the cubic invariant and per-unit frequency coefficient."""

    epsilon_values = _real_array(epsilon, "epsilon")
    delta_value = _finite_real(delta, "delta")
    V4_value = _finite_real(V4, "V4")
    if delta_value == 0.0:
        raise ValueError("delta must be nonzero")
    with np.errstate(over="ignore", invalid="ignore"):
        delta_c = epsilon_values * delta_value
        frequency_shift = epsilon_values * V4_value
    return PhysicalCubicShift(delta_c=delta_c, frequency_shift=frequency_shift)
