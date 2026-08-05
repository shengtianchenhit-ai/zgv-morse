"""Tracked full-wave dispersion frequencies and analytic wavevector gradients.

The spectral evaluator keeps an independently tracked ring of angular anchors
for its working GLL order and for the nested order four degrees higher.  Each
query continues the nearest anchor to the requested Cartesian wavevector,
which avoids selecting a branch by eigenvalue order alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .mode_tracking import (
    TrackedMode,
    mass_mac,
    mass_subspace_singular_values,
    seed_tracked_mode,
    track_mode,
)
from .spectral_plate import (
    ModeSet,
    assemble_plate_matrices,
    assemble_wavevector_derivatives,
    solve_plate_modes,
)


ComplexArray = NDArray[np.complex128]
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


def _real_vector2(values: object, name: str) -> FloatArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real vector of length 2") from error
    if (
        candidate.shape != (2,)
        or np.iscomplexobj(candidate)
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a finite real vector of length 2")
    try:
        result = np.array(candidate, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real vector of length 2") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite real vector of length 2")
    return result


def _read_only_float(values: object) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FrequencyGradient:
    """A positive frequency, its Cartesian gradient, and absolute uncertainties."""

    omega: float
    gradient: FloatArray
    frequency_uncertainty: float
    gradient_uncertainty: float

    def __post_init__(self) -> None:
        omega = _finite_real(self.omega, "omega", positive=True)
        gradient = _real_vector2(self.gradient, "gradient")
        frequency_uncertainty = _nonnegative_real(
            self.frequency_uncertainty,
            "frequency_uncertainty",
        )
        gradient_uncertainty = _nonnegative_real(
            self.gradient_uncertainty,
            "gradient_uncertainty",
        )
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "gradient", _read_only_float(gradient))
        object.__setattr__(self, "frequency_uncertainty", frequency_uncertainty)
        object.__setattr__(self, "gradient_uncertainty", gradient_uncertainty)


@runtime_checkable
class DispersionEvaluator(Protocol):
    """Callable interface used by the critical-point search."""

    def __call__(self, kxy: FloatArray) -> FrequencyGradient:
        """Evaluate a frequency and its Cartesian wavevector gradient."""


def _complex_vector(values: object, name: str) -> ComplexArray:
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
        result = np.array(candidate, dtype=np.complex128, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric vector") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} entries must be finite")
    return result


def _hermitian_matrix(values: object, name: str, dimension: int) -> ComplexArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric matrix") from error
    if (
        candidate.ndim != 2
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a numeric matrix")
    try:
        matrix = np.array(candidate, dtype=np.complex128, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric matrix") from error
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"{name} shape must agree with vector length {dimension} "
            f"and equal ({dimension}, {dimension})"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} entries must be finite")
    scale = max(float(np.max(np.abs(matrix), initial=0.0)), np.finfo(np.float64).tiny)
    defect = float(np.max(np.abs(matrix - matrix.conj().T), initial=0.0))
    if defect > 1.0e-12 * scale:
        raise ValueError(f"{name} must be Hermitian within relative tolerance 1e-12")
    return 0.5 * matrix + 0.5 * matrix.conj().T


def hellmann_feynman_gradient(
    omega: float,
    vector: NDArray[np.complex128],
    mass: NDArray[np.complex128],
    dkx: NDArray[np.complex128],
    dky: NDArray[np.complex128],
) -> FloatArray:
    """Return the exact simple-mode gradient for ``K u = omega**2 M u``.

    The eigenvector need not be mass normalized.  All three matrices are
    validated as finite complex-Hermitian matrices before the real quadratic
    forms are divided by ``2 * omega * (uᴴ M u)``.
    """

    omega_value = _finite_real(omega, "omega", positive=True)
    displacement = _complex_vector(vector, "vector")
    dimension = displacement.size
    mass_matrix = _hermitian_matrix(mass, "mass", dimension)
    derivative_x = _hermitian_matrix(dkx, "dkx", dimension)
    derivative_y = _hermitian_matrix(dky, "dky", dimension)
    try:
        mass_factor = np.linalg.cholesky(mass_matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("mass must be Hermitian positive definite") from error
    if not np.isfinite(mass_factor).all():
        raise ValueError("mass must be Hermitian positive definite")

    with np.errstate(over="ignore", invalid="ignore"):
        norm = float(np.real(np.vdot(displacement, mass_matrix @ displacement)))
        numerators = np.array(
            [
                np.real(np.vdot(displacement, derivative_x @ displacement)),
                np.real(np.vdot(displacement, derivative_y @ displacement)),
            ],
            dtype=np.float64,
        )
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("vector generalized mass norm must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gradient = numerators / (2.0 * omega_value * norm)
    if not np.isfinite(gradient).all():
        raise ValueError("Hellmann-Feynman gradient must be finite")
    return gradient


@dataclass(frozen=True, slots=True)
class _AngularAnchors:
    order: int
    nodes: FloatArray
    modes: tuple[TrackedMode, ...]


@dataclass(frozen=True, slots=True)
class _OrderEvaluation:
    mode: TrackedMode
    gradient: FloatArray
    modes: ModeSet


@dataclass(frozen=True, slots=True)
class TrackedSpectralSample:
    """A nested-order tracked mode sample with its top-face normal component."""

    frequency: FrequencyGradient
    top_normal_component: complex
    relative_eigengap: float
    coarse_omega: float
    coarse_top_normal_component: complex

    def __post_init__(self) -> None:
        if not isinstance(self.frequency, FrequencyGradient):
            raise TypeError("frequency must be a FrequencyGradient")
        try:
            component = complex(self.top_normal_component)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("top_normal_component must be finite") from error
        if not np.isfinite(component.real) or not np.isfinite(component.imag):
            raise ValueError("top_normal_component must be finite")
        gap = _nonnegative_real(self.relative_eigengap, "relative_eigengap")
        coarse_omega = _finite_real(self.coarse_omega, "coarse_omega", positive=True)
        try:
            coarse_component = complex(self.coarse_top_normal_component)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("coarse_top_normal_component must be finite") from error
        if not np.isfinite(coarse_component.real) or not np.isfinite(
            coarse_component.imag
        ):
            raise ValueError("coarse_top_normal_component must be finite")
        object.__setattr__(self, "top_normal_component", component)
        object.__setattr__(self, "relative_eigengap", gap)
        object.__setattr__(self, "coarse_omega", coarse_omega)
        object.__setattr__(self, "coarse_top_normal_component", coarse_component)


def _positive_integer(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _elasticity_tensor(values: object) -> NDArray[np.float64]:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("tensor must be a finite real array with shape (3, 3, 3, 3)") from error
    if (
        candidate.shape != (3, 3, 3, 3)
        or np.iscomplexobj(candidate)
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError("tensor must be a finite real array with shape (3, 3, 3, 3)")
    result = np.array(candidate, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("tensor must be a finite real array with shape (3, 3, 3, 3)")
    result.setflags(write=False)
    return result


def _relative_eigenvalue_uncertainty(coarse: TrackedMode, fine: TrackedMode) -> float:
    scale = max(abs(coarse.eigenvalue), abs(fine.eigenvalue), 1.0)
    discretization = abs(fine.eigenvalue - coarse.eigenvalue) / scale
    result = max(discretization, coarse.residual, fine.residual)
    if not np.isfinite(result) or result < 0.0:
        raise RuntimeError("relative eigenvalue uncertainty is not finite")
    return float(result)


def _relative_eigenvalue_discrepancy(coarse: TrackedMode, fine: TrackedMode) -> float:
    scale = max(abs(coarse.eigenvalue), abs(fine.eigenvalue), 1.0)
    discrepancy = abs(fine.eigenvalue - coarse.eigenvalue) / scale
    if not np.isfinite(discrepancy) or discrepancy < 0.0:
        raise RuntimeError("relative coarse/fine eigenvalue discrepancy is not finite")
    return float(discrepancy)


def _require_untruncated(mode: TrackedMode, modes: ModeSet, context: str) -> None:
    last_computed_index = len(modes.eigenvalues) - 1
    if max(mode.cluster_indices) >= last_computed_index:
        raise RuntimeError(
            f"{context} cluster touches the top of the computed spectrum; "
            "increase num_modes"
        )


class RingAnchoredSpectralEvaluator:
    """Evaluate one full-wave plate branch from pretracked angular anchors.

    Parameters use the same nondimensional conventions as
    :func:`assemble_plate_matrices`: ``tensor`` is a stable fourth-order
    elasticity tensor, ``rho`` is density, and ``half_thickness`` is ``h``.
    ``k0`` and ``target_omega`` select the ring branch.  One anchor is tracked
    at each of ``angular_sectors`` equally spaced directions, independently at
    GLL orders ``order`` and ``order + 4``.
    """

    __slots__ = (
        "tensor",
        "rho",
        "half_thickness",
        "k0",
        "target_omega",
        "order",
        "num_modes",
        "angular_sectors",
        "_angles",
        "_coarse",
        "_fine",
    )

    def __init__(
        self,
        tensor: NDArray[np.float64],
        *,
        rho: float,
        half_thickness: float,
        k0: float,
        target_omega: float,
        order: int,
        num_modes: int,
        angular_sectors: int,
    ) -> None:
        self.tensor = _elasticity_tensor(tensor)
        self.rho = _finite_real(rho, "rho", positive=True)
        self.half_thickness = _finite_real(
            half_thickness,
            "half_thickness",
            positive=True,
        )
        self.k0 = _finite_real(k0, "k0", positive=True)
        self.target_omega = _finite_real(target_omega, "target_omega", positive=True)
        self.order = _positive_integer(order, "order", minimum=2)
        self.num_modes = _positive_integer(num_modes, "num_modes", minimum=2)
        self.angular_sectors = _positive_integer(
            angular_sectors,
            "angular_sectors",
            minimum=4,
        )
        if self.num_modes > 3 * (self.order + 1):
            raise ValueError("num_modes exceeds the node-major coarse matrix dimension")

        self._angles = _read_only_float(
            2.0 * np.pi * np.arange(self.angular_sectors) / self.angular_sectors
        )
        self._coarse = self._build_anchors(self.order)
        self._fine = self._build_anchors(self.order + 4)
        self._validate_anchor_order_consistency()

    def _assemble_and_solve(self, point: FloatArray, order: int) -> ModeSet:
        matrices = assemble_plate_matrices(
            float(point[0]),
            float(point[1]),
            self.tensor,
            self.rho,
            self.half_thickness,
            order=order,
        )
        modes = solve_plate_modes(matrices, self.num_modes)
        self._validate_node_major_layout(modes, order)
        return modes

    @staticmethod
    def _validate_node_major_layout(modes: ModeSet, order: int) -> None:
        nodes = np.asarray(modes.matrices.nodes)
        dimension = 3 * nodes.size
        if (
            nodes.shape != (order + 1,)
            or modes.matrices.stiffness.shape != (dimension, dimension)
            or modes.matrices.mass.shape != (dimension, dimension)
            or modes.vectors.shape[0] != dimension
        ):
            raise ValueError("spectral solve must use fixed node-major displacement triples")

    def _build_anchors(self, order: int) -> _AngularAnchors:
        tracked: list[TrackedMode] = []
        expected_nodes: FloatArray | None = None
        first_modes: ModeSet | None = None
        for angle in self._angles:
            point = self.k0 * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            modes = self._assemble_and_solve(point, order)
            nodes = np.asarray(modes.matrices.nodes)
            if expected_nodes is None:
                expected_nodes = np.array(nodes, dtype=np.float64, copy=True)
                seed_index = int(np.argmin(np.abs(modes.omega - self.target_omega)))
                seed = seed_tracked_mode(modes, seed_index)
                _require_untruncated(seed, modes, f"order-{order} seed anchor")
                tracked.append(seed)
                first_modes = modes
            else:
                if not np.array_equal(nodes, expected_nodes):
                    raise ValueError("spectral anchor nodes must stay fixed within each GLL order")
                current = track_mode(
                    tracked[-1],
                    modes,
                    predicted_eigenvalue=tracked[-1].eigenvalue,
                )
                _require_untruncated(current, modes, f"order-{order} angular anchor")
                tracked.append(current)
        if (
            expected_nodes is None or first_modes is None
        ):  # pragma: no cover - constructor validation prevents this
            raise RuntimeError("no angular anchors were constructed")
        closure = track_mode(
            tracked[-1],
            first_modes,
            predicted_eigenvalue=tracked[-1].eigenvalue,
        )
        _require_untruncated(closure, first_modes, f"order-{order} closure")
        closure_mac = mass_mac(
            tracked[0].vector,
            closure.vector,
            first_modes.matrices.mass,
        )
        closure_overlaps = mass_subspace_singular_values(
            tracked[0].cluster_basis,
            closure.cluster_basis,
            first_modes.matrices.mass,
        )
        closure_overlap = float(closure_overlaps[-1])
        closure_threshold = 1.0 - 1.0e-8
        if closure_mac < closure_threshold:
            raise RuntimeError(
                f"order-{order} angular anchor closure MAC is inconsistent: "
                f"MAC={closure_mac:.6e}"
            )
        if closure_overlap < closure_threshold:
            raise RuntimeError(
                f"order-{order} angular anchor closure cluster overlap is inconsistent: "
                f"overlap={closure_overlap:.6e}"
            )
        return _AngularAnchors(order, _read_only_float(expected_nodes), tuple(tracked))

    def _validate_anchor_order_consistency(self) -> None:
        for sector, (coarse, fine) in enumerate(
            zip(self._coarse.modes, self._fine.modes, strict=True)
        ):
            discrepancy = _relative_eigenvalue_discrepancy(coarse, fine)
            resolved_gap = min(coarse.eigengap, fine.eigengap)
            if discrepancy >= 0.1 * resolved_gap:
                raise RuntimeError(
                    "coarse/fine angular anchor branch mismatch at "
                    f"sector {sector}: discrepancy={discrepancy:.6e}, "
                    f"resolved_gap={resolved_gap:.6e}"
                )

    def _nearest_anchor_index(self, point: FloatArray) -> int:
        angle = float(np.mod(np.arctan2(point[1], point[0]), 2.0 * np.pi))
        spacing = 2.0 * np.pi / self.angular_sectors
        return int(np.floor(angle / spacing + 0.5)) % self.angular_sectors

    def _evaluate_order(
        self,
        point: FloatArray,
        anchors: _AngularAnchors,
        anchor_index: int,
    ) -> _OrderEvaluation:
        return self._continue_order(
            point,
            anchors.order,
            anchors.modes[anchor_index],
            context=f"order-{anchors.order} query",
        )

    def _continue_order(
        self,
        point: FloatArray,
        order: int,
        previous: TrackedMode,
        *,
        context: str,
    ) -> _OrderEvaluation:
        modes = self._assemble_and_solve(point, order)
        tracked = track_mode(
            previous,
            modes,
            predicted_eigenvalue=previous.eigenvalue,
        )
        _require_untruncated(tracked, modes, context)
        derivatives = assemble_wavevector_derivatives(
            float(point[0]),
            float(point[1]),
            self.tensor,
            self.rho,
            self.half_thickness,
            order=order,
        )
        gradient = hellmann_feynman_gradient(
            tracked.omega,
            tracked.vector,
            modes.matrices.mass,
            derivatives.dkx,
            derivatives.dky,
        )
        return _OrderEvaluation(tracked, _read_only_float(gradient), modes)

    @staticmethod
    def _combine_nested_orders(
        coarse: _OrderEvaluation,
        fine: _OrderEvaluation,
    ) -> tuple[FrequencyGradient, float]:
        relative_uncertainty = _relative_eigenvalue_uncertainty(coarse.mode, fine.mode)
        relative_gap = min(coarse.mode.eigengap, fine.mode.eigengap)
        if relative_gap <= 10.0 * relative_uncertainty:
            raise RuntimeError(
                "tracked relative eigengap is unresolved: "
                f"gap={relative_gap:.6e}, uncertainty={relative_uncertainty:.6e}"
            )
        frequency_uncertainty = abs(fine.mode.omega - coarse.mode.omega)
        gradient_uncertainty = float(np.linalg.norm(fine.gradient - coarse.gradient))
        return (
            FrequencyGradient(
                fine.mode.omega,
                fine.gradient,
                frequency_uncertainty,
                gradient_uncertainty,
            ),
            relative_gap,
        )

    def __call__(self, kxy: FloatArray) -> FrequencyGradient:
        point = _real_vector2(kxy, "kxy")
        anchor_index = self._nearest_anchor_index(point)
        coarse = self._evaluate_order(point, self._coarse, anchor_index)
        fine = self._evaluate_order(point, self._fine, anchor_index)

        frequency, _relative_gap = self._combine_nested_orders(coarse, fine)
        return frequency

    def radial_tracker(
        self,
        theta: float,
        radial_direction: int,
    ) -> _RadialSpectralTracker:
        """Create a fresh stateful tracker continued away from the ring anchor."""

        angle = _finite_real(theta, "theta")
        if not 0.0 <= angle < 2.0 * np.pi:
            raise ValueError("theta must lie in [0, 2*pi)")
        if type(radial_direction) is not int or radial_direction not in (-1, 1):
            raise ValueError("radial_direction must be the built-in integer -1 or +1")
        point = self.k0 * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        anchor_index = self._nearest_anchor_index(point)
        return _RadialSpectralTracker(
            self,
            angle,
            radial_direction,
            self._coarse.modes[anchor_index],
            self._fine.modes[anchor_index],
        )


class _RadialSpectralTracker:
    """Stateful two-order continuation along one signed radial ray."""

    __slots__ = (
        "_evaluator",
        "_theta",
        "_radial_direction",
        "_unit_direction",
        "_coarse_previous",
        "_fine_previous",
        "_last_distance",
    )

    def __init__(
        self,
        evaluator: RingAnchoredSpectralEvaluator,
        theta: float,
        radial_direction: int,
        coarse_previous: TrackedMode,
        fine_previous: TrackedMode,
    ) -> None:
        self._evaluator = evaluator
        self._theta = theta
        self._radial_direction = radial_direction
        self._unit_direction = _read_only_float(
            [np.cos(theta), np.sin(theta)]
        )
        self._coarse_previous = coarse_previous
        self._fine_previous = fine_previous
        self._last_distance = -np.inf

    def __call__(self, kxy: FloatArray) -> TrackedSpectralSample:
        point = _real_vector2(kxy, "kxy")
        radius = float(np.linalg.norm(point))
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radial tracker point must have positive finite radius")
        direction_defect = float(np.linalg.norm(point / radius - self._unit_direction))
        if direction_defect > 1.0e-10:
            raise ValueError("radial tracker point must stay on its angular ray")
        q = radius - self._evaluator.k0
        tolerance = 1.0e-11 * max(self._evaluator.k0, 1.0)
        if self._radial_direction * q < -tolerance:
            raise ValueError("radial tracker point lies on the wrong signed ray")
        distance = abs(q)
        if distance + tolerance < self._last_distance:
            raise ValueError("radial tracker points must move monotonically away from q=0")

        coarse = self._evaluator._continue_order(
            point,
            self._evaluator.order,
            self._coarse_previous,
            context=f"order-{self._evaluator.order} radial continuation",
        )
        fine_order = self._evaluator.order + 4
        fine = self._evaluator._continue_order(
            point,
            fine_order,
            self._fine_previous,
            context=f"order-{fine_order} radial continuation",
        )
        frequency, relative_gap = self._evaluator._combine_nested_orders(coarse, fine)
        coarse_top_normal_dof = 3 * (coarse.modes.matrices.nodes.size - 1) + 2
        top_normal_dof = 3 * (fine.modes.matrices.nodes.size - 1) + 2
        coarse_top_normal_component = complex(
            coarse.mode.vector[coarse_top_normal_dof]
        )
        top_normal_component = complex(fine.mode.vector[top_normal_dof])

        self._coarse_previous = coarse.mode
        self._fine_previous = fine.mode
        self._last_distance = distance
        return TrackedSpectralSample(
            frequency,
            top_normal_component,
            relative_gap,
            coarse.mode.omega,
            coarse_top_normal_component,
        )
