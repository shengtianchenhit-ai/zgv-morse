"""Locate, classify, and locally exhaust critical points in an annulus."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from .dispersion import DispersionEvaluator, FrequencyGradient


FloatArray = NDArray[np.float64]


def _finite_real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _real_array(values: object, shape: tuple[int, ...], name: str) -> FloatArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array with shape {shape}") from error
    if (
        candidate.shape != shape
        or np.iscomplexobj(candidate)
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a finite real array with shape {shape}")
    try:
        result = np.array(candidate, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array with shape {shape}") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite real array with shape {shape}")
    return result


def _read_only_float(values: object) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _positive_integer(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class Annulus:
    """A positive-radius closed annulus centered on the origin."""

    k0: float
    half_width: float

    def __post_init__(self) -> None:
        k0 = _finite_real(self.k0, "k0", positive=True)
        half_width = _finite_real(self.half_width, "half_width", positive=True)
        if half_width >= k0:
            raise ValueError("annulus inner radius must be positive")
        object.__setattr__(self, "k0", k0)
        object.__setattr__(self, "half_width", half_width)

    @property
    def inner_radius(self) -> float:
        """Return the inner boundary radius."""

        return self.k0 - self.half_width

    @property
    def outer_radius(self) -> float:
        """Return the outer boundary radius."""

        return self.k0 + self.half_width


@dataclass(frozen=True, slots=True)
class CriticalPoint:
    """One independently residual-checked and Hessian-resolved critical point."""

    kx: float
    ky: float
    radius: float
    theta: float
    omega: float
    gradient_residual: float
    gradient_uncertainty: float
    hessian: FloatArray
    hessian_eigenvalues: FloatArray
    hessian_uncertainty: float
    kind: str
    morse_index: int

    def __post_init__(self) -> None:
        kx = _finite_real(self.kx, "kx")
        ky = _finite_real(self.ky, "ky")
        radius = _finite_real(self.radius, "radius", positive=True)
        theta = _finite_real(self.theta, "theta")
        if not 0.0 <= theta < 2.0 * np.pi:
            raise ValueError("theta must lie in [0, 2*pi)")
        omega = _finite_real(self.omega, "omega", positive=True)
        gradient_residual = _nonnegative_real(self.gradient_residual, "gradient_residual")
        gradient_uncertainty = _nonnegative_real(
            self.gradient_uncertainty,
            "gradient_uncertainty",
        )
        if gradient_residual > gradient_uncertainty:
            raise ValueError("gradient_residual must not exceed gradient_uncertainty")

        hessian = _real_array(self.hessian, (2, 2), "hessian")
        hessian_scale = max(
            float(np.max(np.abs(hessian), initial=0.0)),
            np.finfo(np.float64).tiny,
        )
        symmetry_defect = float(np.max(np.abs(hessian - hessian.T), initial=0.0))
        if symmetry_defect > 1.0e-12 * hessian_scale:
            raise ValueError("hessian must be symmetric within relative tolerance 1e-12")
        hessian = 0.5 * hessian + 0.5 * hessian.T
        eigenvalues = _real_array(self.hessian_eigenvalues, (2,), "hessian_eigenvalues")
        if np.any(np.diff(eigenvalues) < 0.0):
            raise ValueError("hessian_eigenvalues must be sorted in nondecreasing order")
        computed_eigenvalues = np.linalg.eigvalsh(hessian)
        eigenvalue_scale = max(
            float(np.max(np.abs(computed_eigenvalues), initial=0.0)),
            1.0,
        )
        if not np.allclose(
            eigenvalues,
            computed_eigenvalues,
            rtol=1.0e-11,
            atol=1.0e-12 * eigenvalue_scale,
        ):
            raise ValueError("hessian_eigenvalues must agree with hessian")
        hessian_uncertainty = _nonnegative_real(
            self.hessian_uncertainty,
            "hessian_uncertainty",
        )
        kind, morse_index = _classify(eigenvalues, hessian_uncertainty)
        if self.kind != kind:
            raise ValueError("kind must agree with the resolved Hessian eigenvalues")
        if type(self.morse_index) is not int or self.morse_index != morse_index:
            raise ValueError("morse_index must agree with kind and equal +1 or -1")

        coordinate_scale = max(radius, 1.0)
        if abs(float(np.hypot(kx, ky)) - radius) > 1.0e-10 * coordinate_scale:
            raise ValueError("radius must agree with kx and ky")
        direction = np.array([kx, ky]) / radius
        angular_direction = np.array([np.cos(theta), np.sin(theta)])
        if np.linalg.norm(direction - angular_direction) > 1.0e-9:
            raise ValueError("theta must agree with kx and ky")

        object.__setattr__(self, "kx", kx)
        object.__setattr__(self, "ky", ky)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "gradient_residual", gradient_residual)
        object.__setattr__(self, "gradient_uncertainty", gradient_uncertainty)
        object.__setattr__(self, "hessian", _read_only_float(hessian))
        object.__setattr__(self, "hessian_eigenvalues", _read_only_float(eigenvalues))
        object.__setattr__(self, "hessian_uncertainty", hessian_uncertainty)


@dataclass(frozen=True, slots=True)
class ExhaustionReport:
    """Finite-resolution boundary-separation and gradient-index consistency record."""

    boundary_is_noncritical: bool
    index_closes: bool
    minimum_boundary_gradient: float
    maximum_gradient_uncertainty: float

    def __post_init__(self) -> None:
        if type(self.boundary_is_noncritical) is not bool:
            raise TypeError("boundary_is_noncritical must be a bool")
        if type(self.index_closes) is not bool:
            raise TypeError("index_closes must be a bool")
        minimum = _nonnegative_real(
            self.minimum_boundary_gradient,
            "minimum_boundary_gradient",
        )
        maximum_uncertainty = _nonnegative_real(
            self.maximum_gradient_uncertainty,
            "maximum_gradient_uncertainty",
        )
        if self.boundary_is_noncritical and minimum <= 10.0 * maximum_uncertainty:
            raise ValueError(
                "a noncritical boundary must exceed ten times its gradient uncertainty"
            )
        object.__setattr__(self, "minimum_boundary_gradient", minimum)
        object.__setattr__(self, "maximum_gradient_uncertainty", maximum_uncertainty)


def _sample(evaluator: DispersionEvaluator, point: FloatArray) -> FrequencyGradient:
    result = evaluator(point)
    if not isinstance(result, FrequencyGradient):
        raise TypeError("evaluator must return a FrequencyGradient")
    return result


def _centered_hessian(
    evaluator: DispersionEvaluator,
    point: FloatArray,
    step: float,
) -> tuple[FloatArray, float]:
    columns: list[FloatArray] = []
    column_uncertainties: list[float] = []
    for axis in range(2):
        shift = np.zeros(2, dtype=np.float64)
        shift[axis] = step
        plus = _sample(evaluator, point + shift)
        minus = _sample(evaluator, point - shift)
        columns.append((plus.gradient - minus.gradient) / (2.0 * step))
        column_uncertainties.append(
            (plus.gradient_uncertainty + minus.gradient_uncertainty) / (2.0 * step)
        )
    raw = np.column_stack(columns)
    symmetric = 0.5 * raw + 0.5 * raw.T
    if not np.isfinite(symmetric).all():
        raise ValueError("centered Hessian must be finite")
    matrix_bound = float(np.linalg.norm(column_uncertainties))
    return symmetric, matrix_bound


def cartesian_hessian(
    evaluator: DispersionEvaluator,
    point: FloatArray,
    step: float,
) -> tuple[FloatArray, float]:
    """Return a Richardson Cartesian Hessian and a conservative scalar error.

    The error adds the observed fine/Richardson step difference to evaluator
    gradient uncertainties propagated through the two centered differences and
    the Richardson weights.
    """

    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    location = _real_array(point, (2,), "point")
    step_value = _finite_real(step, "step", positive=True)
    coarse, coarse_uncertainty = _centered_hessian(evaluator, location, step_value)
    fine, fine_uncertainty = _centered_hessian(evaluator, location, step_value / 2.0)
    richardson = (4.0 * fine - coarse) / 3.0
    step_uncertainty = float(np.linalg.norm(richardson - fine, ord=2))
    propagated_uncertainty = (4.0 * fine_uncertainty + coarse_uncertainty) / 3.0
    uncertainty = step_uncertainty + propagated_uncertainty
    if not np.isfinite(richardson).all() or not np.isfinite(uncertainty):
        raise ValueError("Richardson Hessian and uncertainty must be finite")
    return _read_only_float(richardson), float(uncertainty)


def _classify(eigenvalues: FloatArray, uncertainty: float) -> tuple[str, int]:
    """Classify a resolved planar Hessian and return its gradient index."""

    values = _real_array(eigenvalues, (2,), "eigenvalues")
    uncertainty_value = _nonnegative_real(uncertainty, "uncertainty")
    if float(np.min(np.abs(values))) <= 10.0 * uncertainty_value:
        raise ValueError("Hessian is not separated from degeneracy")
    if np.all(values > 0.0):
        return "minimum", 1
    if np.all(values < 0.0):
        return "maximum", 1
    return "saddle", -1


def _polar_candidates(
    evaluator: DispersionEvaluator,
    annulus: Annulus,
    n_radial: int,
    n_theta: int,
) -> list[FloatArray]:
    radii = np.linspace(annulus.inner_radius, annulus.outer_radius, n_radial)
    angular_step = 2.0 * np.pi / n_theta
    angles = angular_step * np.arange(n_theta)
    radial_gradient = np.empty((n_radial, n_theta), dtype=np.float64)
    tangential_gradient = np.empty_like(radial_gradient)

    for radial_index, radius in enumerate(radii):
        for angular_index, theta in enumerate(angles):
            radial = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
            tangent = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
            gradient = _sample(evaluator, radius * radial).gradient
            radial_gradient[radial_index, angular_index] = float(gradient @ radial)
            tangential_gradient[radial_index, angular_index] = float(gradient @ tangent)

    seeds: list[FloatArray] = []
    for radial_index in range(n_radial - 1):
        for angular_index in range(n_theta):
            next_angular_index = (angular_index + 1) % n_theta
            row_indices = np.array([radial_index, radial_index + 1])
            column_indices = np.array([angular_index, next_angular_index])
            radial_cell = radial_gradient[np.ix_(row_indices, column_indices)]
            tangential_cell = tangential_gradient[np.ix_(row_indices, column_indices)]
            if (
                float(np.min(radial_cell)) <= 0.0 <= float(np.max(radial_cell))
                and float(np.min(tangential_cell))
                <= 0.0
                <= float(np.max(tangential_cell))
            ):
                radius = 0.5 * (radii[radial_index] + radii[radial_index + 1])
                theta = (angles[angular_index] + 0.5 * angular_step) % (2.0 * np.pi)
                seeds.append(radius * np.array([np.cos(theta), np.sin(theta)]))
    return seeds


def _canonical_theta(point: FloatArray, angular_tolerance: float) -> float:
    theta = float(np.mod(np.arctan2(point[1], point[0]), 2.0 * np.pi))
    if min(theta, 2.0 * np.pi - theta) <= angular_tolerance:
        return 0.0
    return theta


def _critical_point_position_uncertainty(point: CriticalPoint) -> float:
    minimum_curvature = float(np.min(np.abs(point.hessian_eigenvalues)))
    return point.gradient_uncertainty / minimum_curvature


def _deduplication_tolerance(
    first: CriticalPoint,
    second: CriticalPoint,
    base_tolerance: float,
    maximum_tolerance: float,
) -> float:
    uncertainty_scale = (
        _critical_point_position_uncertainty(first)
        + _critical_point_position_uncertainty(second)
    )
    return min(maximum_tolerance, max(base_tolerance, uncertainty_scale))


def locate_critical_points(
    evaluator: DispersionEvaluator,
    annulus: Annulus,
    n_radial: int,
    n_theta: int,
    hessian_step: float,
) -> list[CriticalPoint]:
    """Find all residual- and Hessian-resolved roots seeded by a polar grid."""

    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not isinstance(annulus, Annulus):
        raise TypeError("annulus must be an Annulus")
    radial_count = _positive_integer(n_radial, "n_radial", minimum=2)
    angular_count = _positive_integer(n_theta, "n_theta", minimum=4)
    step = _finite_real(hessian_step, "hessian_step", positive=True)
    seeds = _polar_candidates(evaluator, annulus, radial_count, angular_count)

    distance_tolerance = 5.0e-9 * max(annulus.k0, 1.0)
    maximum_deduplication_tolerance = max(
        distance_tolerance,
        min(0.05 * annulus.half_width, 0.01 * annulus.k0),
    )
    radial_tolerance = 20.0 * np.finfo(np.float64).eps * max(annulus.outer_radius, 1.0)
    angular_tolerance = distance_tolerance / annulus.inner_radius
    accepted: list[CriticalPoint] = []

    def gradient(point: FloatArray) -> FloatArray:
        return np.asarray(_sample(evaluator, point).gradient)

    def jacobian(point: FloatArray) -> FloatArray:
        return np.asarray(cartesian_hessian(evaluator, point, step)[0])

    for seed in seeds:
        solution = root(gradient, seed, jac=jacobian, tol=1.0e-11)
        if not solution.success:
            continue
        point = _real_array(solution.x, (2,), "root point")
        radius = float(np.linalg.norm(point))
        if (
            radius < annulus.inner_radius - radial_tolerance
            or radius > annulus.outer_radius + radial_tolerance
        ):
            continue
        sample = _sample(evaluator, point)
        residual = float(np.linalg.norm(sample.gradient))
        if not np.isfinite(residual) or residual > sample.gradient_uncertainty:
            continue
        hessian, hessian_uncertainty = cartesian_hessian(evaluator, point, step)
        eigenvalues = np.linalg.eigvalsh(hessian)
        kind, morse_index = _classify(eigenvalues, hessian_uncertainty)
        theta = _canonical_theta(point, angular_tolerance)
        candidate = CriticalPoint(
            float(point[0]),
            float(point[1]),
            radius,
            theta,
            sample.omega,
            residual,
            sample.gradient_uncertainty,
            hessian,
            eigenvalues,
            hessian_uncertainty,
            kind,
            morse_index,
        )

        duplicate_index = next(
            (
                index
                for index, previous in enumerate(accepted)
                if np.hypot(candidate.kx - previous.kx, candidate.ky - previous.ky)
                <= _deduplication_tolerance(
                    candidate,
                    previous,
                    distance_tolerance,
                    maximum_deduplication_tolerance,
                )
            ),
            None,
        )
        if duplicate_index is None:
            accepted.append(candidate)
        elif candidate.gradient_residual < accepted[duplicate_index].gradient_residual:
            accepted[duplicate_index] = candidate

    return sorted(accepted, key=lambda item: item.theta)


def _boundary_winding(
    evaluator: DispersionEvaluator,
    radius: float,
    count: int,
) -> tuple[int, float, float, float, bool]:
    gradients = np.empty((count, 2), dtype=np.float64)
    uncertainties = np.empty(count, dtype=np.float64)
    for index, theta in enumerate(2.0 * np.pi * np.arange(count) / count):
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        sample = _sample(evaluator, radius * direction)
        gradients[index] = sample.gradient
        uncertainties[index] = sample.gradient_uncertainty

    norms = np.hypot(gradients[:, 0], gradients[:, 1])
    directions_resolved = bool(np.isfinite(norms).all() and np.all(norms > 0.0))
    if not directions_resolved:
        return 0, 0.0, float(np.max(uncertainties)), np.inf, False
    unit_gradient = gradients[:, 0] / norms + 1j * (gradients[:, 1] / norms)
    increments = np.angle(np.roll(unit_gradient, -1) * np.conj(unit_gradient))
    winding_value = float(np.sum(increments) / (2.0 * np.pi))
    winding = int(np.rint(winding_value))
    integer_resolved = abs(winding_value - winding) <= 1.0e-8
    maximum_increment = float(np.max(np.abs(increments), initial=0.0))
    return (
        winding,
        float(np.min(norms)),
        float(np.max(uncertainties)),
        maximum_increment,
        integer_resolved,
    )


def verify_annular_exhaustion(
    evaluator: DispersionEvaluator,
    annulus: Annulus,
    points: Iterable[CriticalPoint],
    n_boundary: int,
) -> ExhaustionReport:
    """Return a finite-resolution annular boundary/index consistency check.

    Both boundaries are sampled at ``n_boundary`` and ``2*n_boundary``.
    Acceptance requires ten-times gradient separation, matching coarse/fine
    direction windings, and fine-grid phase increments below ``pi/2``.  The
    point-index sum is compared with outer winding minus inner winding.  This
    finite-resolution check does not by itself exclude unresolved canceling
    critical-point pairs in the annulus interior.
    """

    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not isinstance(annulus, Annulus):
        raise TypeError("annulus must be an Annulus")
    boundary_count = _positive_integer(n_boundary, "n_boundary", minimum=4)
    try:
        critical_points = tuple(points)
    except TypeError as error:
        raise TypeError("points must be an iterable of CriticalPoint records") from error
    if any(not isinstance(point, CriticalPoint) for point in critical_points):
        raise TypeError("points must contain only CriticalPoint records")

    results: dict[tuple[str, str], tuple[int, float, float, float, bool]] = {}
    for boundary_name, radius in (
        ("inner", annulus.inner_radius),
        ("outer", annulus.outer_radius),
    ):
        results[(boundary_name, "coarse")] = _boundary_winding(
            evaluator,
            radius,
            boundary_count,
        )
        results[(boundary_name, "fine")] = _boundary_winding(
            evaluator,
            radius,
            2 * boundary_count,
        )

    minimum = min(result[1] for result in results.values())
    maximum_uncertainty = max(result[2] for result in results.values())
    separated = minimum > 10.0 * maximum_uncertainty
    winding_resolved = all(
        results[(name, "coarse")][0] == results[(name, "fine")][0]
        and results[(name, "coarse")][4]
        and results[(name, "fine")][4]
        and results[(name, "fine")][3] < 0.5 * np.pi
        for name in ("inner", "outer")
    )
    boundary_is_noncritical = separated and winding_resolved
    inner_winding = results[("inner", "fine")][0]
    outer_winding = results[("outer", "fine")][0]
    point_index = sum(point.morse_index for point in critical_points)
    index_closes = (
        boundary_is_noncritical and point_index == outer_winding - inner_winding
    )
    return ExhaustionReport(
        boundary_is_noncritical,
        index_closes,
        minimum,
        maximum_uncertainty,
    )
