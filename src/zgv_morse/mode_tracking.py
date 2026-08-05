"""Mass-weighted mode, eigenspace, and branch tracking utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from .gll import GLLMesh
from .spectral_plate import ModeSet, PlateMatrices


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class ModeTrackingError(RuntimeError):
    """Raised when neither a mode nor its local eigenspace can be continued."""


@dataclass(frozen=True, slots=True)
class TrackedMode:
    """One continued eigenmode and its overlap diagnostics.

    ``mac`` is a squared scalar overlap, while ``subspace_overlap`` is an
    amplitude cosine in ``[0, 1]``.  ``cluster_basis`` preserves the full
    eigenspace state through degeneracies without changing the original
    ten-positional-argument constructor.
    """

    omega: float
    eigenvalue: float
    vector: ComplexArray
    index: int
    mac: float
    eigengap: float
    residual: float
    cluster_indices: tuple[int, ...]
    subspace_overlap: float
    cluster_basis: ComplexArray | None = None

    def __post_init__(self) -> None:
        vector = _numeric_vector(self.vector, "vector")
        omega = _finite_real(self.omega, "omega")
        eigenvalue = _finite_real(self.eigenvalue, "eigenvalue")
        if omega < 0.0 or eigenvalue < 0.0:
            raise ValueError("omega and eigenvalue must be nonnegative")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be a nonnegative built-in integer")
        mac = _unit_interval(self.mac, "mac")
        if isinstance(self.eigengap, (bool, np.bool_)) or not isinstance(self.eigengap, Real):
            raise ValueError("eigengap must be a nonnegative real scalar")
        eigengap = float(self.eigengap)
        if np.isnan(eigengap) or eigengap < 0.0:
            raise ValueError("eigengap must be a nonnegative real scalar")
        residual = _finite_real(self.residual, "residual")
        if residual < 0.0:
            raise ValueError("residual must be nonnegative")
        try:
            cluster_indices = tuple(self.cluster_indices)
        except TypeError as error:
            raise ValueError("cluster_indices must be a nonempty integer sequence") from error
        if (
            not cluster_indices
            or any(type(candidate) is not int or candidate < 0 for candidate in cluster_indices)
            or len(set(cluster_indices)) != len(cluster_indices)
            or self.index not in cluster_indices
        ):
            raise ValueError(
                "cluster_indices must be distinct nonnegative integers containing index"
            )
        subspace_overlap = _unit_interval(self.subspace_overlap, "subspace_overlap")
        if self.cluster_basis is None:
            basis = vector[:, np.newaxis]
        else:
            basis = _numeric_basis(self.cluster_basis, "cluster_basis")
            if basis.shape[0] != vector.size:
                raise ValueError("cluster_basis row count must match vector length")
            if basis.shape[1] != len(cluster_indices):
                raise ValueError("cluster_basis column count must match cluster_indices")

        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "eigenvalue", eigenvalue)
        object.__setattr__(self, "vector", _read_only_complex(vector))
        object.__setattr__(self, "mac", mac)
        object.__setattr__(self, "eigengap", eigengap)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "cluster_indices", cluster_indices)
        object.__setattr__(self, "subspace_overlap", subspace_overlap)
        object.__setattr__(self, "cluster_basis", _read_only_complex(basis))


@dataclass(frozen=True, slots=True)
class TrackedBranch:
    """A mode continued along a sequence of two-dimensional wavevectors."""

    wavevectors: FloatArray
    modes: tuple[TrackedMode, ...]

    def __post_init__(self) -> None:
        points = _validated_wavevectors(self.wavevectors)
        try:
            modes = tuple(self.modes)
        except TypeError as error:
            raise ValueError("modes must be a nonempty sequence of TrackedMode records") from error
        if not modes or any(not isinstance(mode, TrackedMode) for mode in modes):
            raise ValueError("modes must be a nonempty sequence of TrackedMode records")
        if points.shape[0] != len(modes):
            raise ValueError("wavevector and mode count must match")
        object.__setattr__(self, "wavevectors", _read_only_float(points))
        object.__setattr__(self, "modes", modes)

    @property
    def minimum_eigengap(self) -> float:
        """Return the smallest relative eigengap along the branch."""

        return min(mode.eigengap for mode in self.modes)


def _read_only_complex(values: object) -> ComplexArray:
    result = np.array(values, dtype=np.complex128, copy=True)
    result.setflags(write=False)
    return result


def _read_only_float(values: object) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _numeric_vector(values: object, name: str) -> ComplexArray:
    """Return a finite, nonempty complex vector without aliasing ``values``."""

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


def _numeric_basis(values: object, name: str) -> ComplexArray:
    """Return a finite, nonempty complex basis matrix."""

    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric matrix") from error
    if (
        candidate.ndim != 2
        or 0 in candidate.shape
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a nonempty numeric matrix")
    try:
        basis = np.array(candidate, dtype=np.complex128, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric matrix") from error
    if not np.isfinite(basis).all():
        raise ValueError(f"{name} entries must be finite")
    return basis


def _validated_mass(mass: object, dimension: int) -> tuple[ComplexArray, ComplexArray]:
    """Validate a Hermitian positive-definite mass and return it and its Cholesky factor."""

    try:
        candidate = np.asarray(mass)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("mass must be a finite numeric matrix") from error
    if (
        candidate.ndim != 2
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError("mass must be a numeric matrix")
    try:
        matrix = np.array(candidate, dtype=np.complex128, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("mass must be a finite numeric matrix") from error
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"mass must have shape ({dimension}, {dimension})")
    if not np.isfinite(matrix).all():
        raise ValueError("mass entries must be finite")

    scale = float(np.max(np.abs(matrix), initial=0.0))
    defect = float(np.max(np.abs(matrix - matrix.conj().T), initial=0.0))
    if defect > 1.0e-12 * scale:
        raise ValueError("mass must be Hermitian within relative tolerance 1e-12")
    matrix = 0.5 * matrix + 0.5 * matrix.conj().T
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("mass must be Hermitian positive definite") from error
    if not np.isfinite(factor).all():
        raise ValueError("mass must be Hermitian positive definite")
    return matrix, factor


def _validated_overlap_inputs(
    reference: object,
    candidate: object,
    mass: object,
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    reference_vector = _numeric_vector(reference, "reference")
    candidate_vector = _numeric_vector(candidate, "candidate")
    if candidate_vector.shape != reference_vector.shape:
        raise ValueError("reference and candidate must have the same length")
    mass_matrix, _factor = _validated_mass(mass, reference_vector.size)
    return reference_vector, candidate_vector, mass_matrix


def _mass_norm_squared(vector: ComplexArray, mass: ComplexArray, name: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        norm_squared = float(np.real(np.vdot(vector, mass @ vector)))
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError(f"{name} mass norm must be finite and positive")
    return norm_squared


def mass_mac(reference: object, candidate: object, mass: object) -> float:
    """Return the normalized squared mass overlap of two vectors.

    The value is invariant to nonzero complex scaling of either vector.  Both
    vectors must be finite, nonempty, and compatible with a finite Hermitian
    positive-definite mass matrix.
    """

    reference_vector, candidate_vector, mass_matrix = _validated_overlap_inputs(
        reference,
        candidate,
        mass,
    )
    reference_norm = _mass_norm_squared(reference_vector, mass_matrix, "reference")
    candidate_norm = _mass_norm_squared(candidate_vector, mass_matrix, "candidate")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        overlap = np.vdot(reference_vector, mass_matrix @ candidate_vector)
        value = float(abs(overlap) ** 2 / (reference_norm * candidate_norm))
    tolerance = 1.0e-12
    if not np.isfinite(value) or value < -tolerance or value > 1.0 + tolerance:
        raise ValueError("mass overlap is numerically outside [0, 1]")
    return float(np.clip(value, 0.0, 1.0))


def phase_align(reference: object, candidate: object, mass: object) -> ComplexArray:
    """Return an independent candidate copy with real nonnegative mass overlap."""

    reference_vector, candidate_vector, mass_matrix = _validated_overlap_inputs(
        reference,
        candidate,
        mass,
    )
    _mass_norm_squared(reference_vector, mass_matrix, "reference")
    _mass_norm_squared(candidate_vector, mass_matrix, "candidate")
    overlap = np.vdot(reference_vector, mass_matrix @ candidate_vector)
    if overlap == 0.0:
        return candidate_vector
    aligned = candidate_vector * np.exp(-1j * np.angle(overlap))
    if not np.isfinite(aligned).all():
        raise ValueError("phase alignment produced non-finite output")
    return aligned


def _orthonormalized_weighted_basis(
    basis: ComplexArray,
    mass_factor: ComplexArray,
    name: str,
) -> ComplexArray:
    with np.errstate(over="ignore", invalid="ignore"):
        weighted = mass_factor.conj().T @ basis
    if not np.isfinite(weighted).all():
        raise ValueError(f"{name} cannot be weighted finitely by mass")
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    rank_tolerance = np.finfo(np.float64).eps * max(weighted.shape) * float(singular_values[0])
    if basis.shape[1] > basis.shape[0] or singular_values[-1] <= rank_tolerance:
        raise ValueError(f"{name} columns must be linearly independent in the mass metric")
    orthonormal, _triangular = np.linalg.qr(weighted, mode="reduced")
    return orthonormal


def mass_subspace_singular_values(
    previous_basis: object,
    current_basis: object,
    mass: object,
) -> FloatArray:
    """Return principal-overlap singular values for two bases in the mass metric.

    Input columns need not already be mass-orthonormal; each full-rank basis is
    robustly orthonormalized after Cholesky weighting.
    """

    previous = _numeric_basis(previous_basis, "previous_basis")
    current = _numeric_basis(current_basis, "current_basis")
    if previous.shape[0] != current.shape[0]:
        raise ValueError("previous_basis and current_basis must have the same row count")
    _mass_matrix, factor = _validated_mass(mass, previous.shape[0])
    previous_orthonormal = _orthonormalized_weighted_basis(
        previous,
        factor,
        "previous_basis",
    )
    current_orthonormal = _orthonormalized_weighted_basis(
        current,
        factor,
        "current_basis",
    )
    overlaps = np.linalg.svd(
        previous_orthonormal.conj().T @ current_orthonormal,
        compute_uv=False,
    )
    tolerance = 1.0e-12
    if not np.isfinite(overlaps).all() or np.any(overlaps > 1.0 + tolerance):
        raise ValueError("subspace overlaps are numerically outside [0, 1]")
    return np.clip(np.asarray(overlaps, dtype=np.float64), 0.0, 1.0)


def _real_vector(values: object, name: str) -> FloatArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real vector") from error
    if (
        candidate.ndim != 1
        or candidate.size == 0
        or np.iscomplexobj(candidate)
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a nonempty real vector")
    try:
        vector = np.array(candidate, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real vector") from error
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} entries must be finite")
    return vector


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        scalar = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _unit_interval(value: object, name: str) -> float:
    scalar = _finite_real(value, name)
    if not 0.0 <= scalar <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return scalar


def _strict_index(index: object, size: int, name: str = "index") -> int:
    if type(index) is not int:
        raise TypeError(f"{name} must be a built-in integer")
    if not 0 <= index < size:
        raise ValueError(f"{name} must be between 0 and {size - 1}")
    return index


def relative_eigengap(eigenvalues: object, index: int) -> float:
    """Return the nearest symmetric relative gap, with an absolute unit floor.

    For values ``a`` and ``b`` the pairwise convention is
    ``abs(a-b) / max(abs(a), abs(b), 1)``.  A singleton spectrum has infinite
    eigengap because it has no competing reported mode.
    """

    values = _real_vector(eigenvalues, "eigenvalues")
    if np.any(values < 0.0):
        raise ValueError("eigenvalues must be nonnegative")
    selected_index = _strict_index(index, values.size)
    if values.size == 1:
        return float("inf")
    selected = float(values[selected_index])
    gaps = [
        abs(selected - float(value)) / max(abs(selected), abs(float(value)), 1.0)
        for candidate_index, value in enumerate(values)
        if candidate_index != selected_index
    ]
    return float(min(gaps))


_relative_gap = relative_eigengap


def _validated_modes(
    modes: object,
) -> tuple[ModeSet, FloatArray, FloatArray, ComplexArray, FloatArray, ComplexArray]:
    if not isinstance(modes, ModeSet):
        raise TypeError("modes must be a ModeSet instance")
    eigenvalues = _real_vector(modes.eigenvalues, "modes.eigenvalues")
    omega = _real_vector(modes.omega, "modes.omega")
    residuals = _real_vector(modes.residuals, "modes.residuals")
    vectors = _numeric_basis(modes.vectors, "modes.vectors")
    mode_count = eigenvalues.size
    if omega.size != mode_count or residuals.size != mode_count or vectors.shape[1] != mode_count:
        raise ValueError(
            "mode eigenvalues, omega, vectors, and residuals must have compatible sizes"
        )
    if np.any(eigenvalues < 0.0) or np.any(omega < 0.0):
        raise ValueError("mode eigenvalues and omega must be nonnegative")
    if np.any(residuals < 0.0):
        raise ValueError("mode residuals must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        squared_omega = omega * omega
    if not np.isfinite(squared_omega).all() or not np.allclose(
        squared_omega,
        eigenvalues,
        rtol=1.0e-10,
        atol=1.0e-13,
    ):
        raise ValueError("modes.omega squared must agree with modes.eigenvalues")
    defect = _finite_real(modes.mass_orthogonality_defect, "mass_orthogonality_defect")
    if defect < 0.0:
        raise ValueError("mass_orthogonality_defect must be nonnegative")
    if not isinstance(modes.matrices, PlateMatrices):
        raise TypeError("modes.matrices must be a PlateMatrices instance")
    mass, _factor = _validated_mass(modes.matrices.mass, vectors.shape[0])
    return modes, eigenvalues, omega, vectors, residuals, mass


def seed_tracked_mode(modes: ModeSet, index: int) -> TrackedMode:
    """Create an immutable starting record from one member of ``modes``."""

    _modes, eigenvalues, omega, vectors, residuals, mass = _validated_modes(modes)
    selected_index = _strict_index(index, eigenvalues.size)
    _mass_norm_squared(vectors[:, selected_index], mass, "selected mode")
    return TrackedMode(
        omega=float(omega[selected_index]),
        eigenvalue=float(eigenvalues[selected_index]),
        vector=_read_only_complex(vectors[:, selected_index]),
        index=selected_index,
        mac=1.0,
        eigengap=relative_eigengap(eigenvalues, selected_index),
        residual=float(residuals[selected_index]),
        cluster_indices=(selected_index,),
        subspace_overlap=1.0,
    )


def _validated_previous(
    previous: object,
    dimension: int,
    mass: ComplexArray,
) -> tuple[ComplexArray, ComplexArray]:
    if not isinstance(previous, TrackedMode):
        raise TypeError("previous must be a TrackedMode instance")
    vector = _numeric_vector(previous.vector, "previous.vector")
    if vector.size != dimension:
        raise ValueError("previous.vector length must match current mode vectors")
    _mass_norm_squared(vector, mass, "previous.vector")
    eigenvalue = _finite_real(previous.eigenvalue, "previous.eigenvalue")
    if eigenvalue < 0.0:
        raise ValueError("previous.eigenvalue must be nonnegative")
    basis = _numeric_basis(previous.cluster_basis, "previous.cluster_basis")
    if basis.shape[0] != dimension:
        raise ValueError("previous.cluster_basis row count must match current mode vectors")
    mass_subspace_singular_values(basis, basis, mass)
    return vector, basis


def _all_mass_macs(
    reference: ComplexArray,
    candidates: ComplexArray,
    mass: ComplexArray,
) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        weighted_candidates = mass @ candidates
        reference_norm = float(np.real(np.vdot(reference, mass @ reference)))
        candidate_norms = np.real(np.sum(candidates.conj() * weighted_candidates, axis=0))
        overlaps = reference.conj() @ weighted_candidates
        values = np.abs(overlaps) ** 2 / (reference_norm * candidate_norms)
    if (
        not np.isfinite(reference_norm)
        or reference_norm <= 0.0
        or not np.isfinite(candidate_norms).all()
        or np.any(candidate_norms <= 0.0)
    ):
        raise ValueError("current and previous mode mass norms must be finite and positive")
    tolerance = 1.0e-12
    if (
        not np.isfinite(values).all()
        or np.any(values < -tolerance)
        or np.any(values > 1.0 + tolerance)
    ):
        raise ValueError("mode MAC values are numerically outside [0, 1]")
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def _subspace_projection_macs(
    previous_basis: ComplexArray,
    candidates: ComplexArray,
    mass: ComplexArray,
) -> FloatArray:
    factor = np.linalg.cholesky(mass)
    previous_orthonormal = _orthonormalized_weighted_basis(
        previous_basis,
        factor,
        "previous.cluster_basis",
    )
    weighted_candidates = factor.conj().T @ candidates
    candidate_norms = np.sum(np.abs(weighted_candidates) ** 2, axis=0)
    overlaps = previous_orthonormal.conj().T @ weighted_candidates
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projections = np.sum(np.abs(overlaps) ** 2, axis=0) / candidate_norms
    tolerance = 1.0e-12
    if (
        not np.isfinite(projections).all()
        or np.any(projections < -tolerance)
        or np.any(projections > 1.0 + tolerance)
    ):
        raise ValueError("mode subspace projections are numerically outside [0, 1]")
    return np.clip(np.asarray(projections, dtype=np.float64), 0.0, 1.0)


def _connected_eigenvalue_cluster(
    eigenvalues: FloatArray,
    index: int,
    threshold: float,
) -> tuple[int, ...]:
    members = {index}
    frontier = [index]
    while frontier:
        source_index = frontier.pop()
        source = float(eigenvalues[source_index])
        for candidate_index, value in enumerate(eigenvalues):
            if candidate_index in members:
                continue
            candidate = float(value)
            relative_gap = abs(candidate - source) / max(abs(candidate), abs(source), 1.0)
            if relative_gap <= threshold:
                members.add(candidate_index)
                frontier.append(candidate_index)
    return tuple(sorted(members))


def track_mode(
    previous: TrackedMode,
    modes: ModeSet,
    *,
    cluster_rel_gap: float = 1.0e-3,
    min_mac: float = 0.2,
    predicted_eigenvalue: float | None = None,
) -> TrackedMode:
    """Continue ``previous`` into ``modes`` by scalar and clustered mass overlap."""

    cluster_threshold = _unit_interval(cluster_rel_gap, "cluster_rel_gap")
    mac_threshold = _unit_interval(min_mac, "min_mac")
    _modes, eigenvalues, omega, vectors, residuals, mass = _validated_modes(modes)
    previous_vector, previous_basis = _validated_previous(previous, vectors.shape[0], mass)
    if predicted_eigenvalue is None:
        predicted = _finite_real(previous.eigenvalue, "previous.eigenvalue")
    else:
        predicted = _finite_real(predicted_eigenvalue, "predicted_eigenvalue")

    mac_values = _all_mass_macs(previous_vector, vectors, mass)
    projection_values = _subspace_projection_macs(previous_basis, vectors, mass)
    penalties = np.array(
        [
            abs(float(value) - predicted) / max(abs(float(value)), abs(predicted), 1.0)
            for value in eigenvalues
        ],
        dtype=np.float64,
    )
    index = int(np.argmax(projection_values - 1.0e-6 * penalties))
    selected = float(eigenvalues[index])
    cluster = _connected_eigenvalue_cluster(eigenvalues, index, cluster_threshold)
    cluster_overlaps = mass_subspace_singular_values(
        previous_basis,
        vectors[:, cluster],
        mass,
    )
    # For equal dimensions this is the smallest principal cosine.  For
    # unequal dimensions it is the smaller-space containment cosine.
    subspace_overlap = float(cluster_overlaps[-1])
    if mac_values[index] < mac_threshold and subspace_overlap**2 < mac_threshold:
        raise ModeTrackingError(
            "scalar MAC and cluster overlap both failed "
            f"(best MAC={mac_values[index]:.6g}, cluster overlap={subspace_overlap:.6g}, "
            f"threshold={mac_threshold:.6g})"
        )

    aligned = phase_align(previous_vector, vectors[:, index], mass)
    return TrackedMode(
        omega=float(omega[index]),
        eigenvalue=selected,
        vector=_read_only_complex(aligned),
        index=index,
        mac=float(mac_values[index]),
        eigengap=relative_eigengap(eigenvalues, index),
        residual=float(residuals[index]),
        cluster_indices=cluster,
        subspace_overlap=subspace_overlap,
        cluster_basis=vectors[:, cluster],
    )


def _validated_wavevectors(wavevectors: object) -> FloatArray:
    try:
        candidate = np.asarray(wavevectors)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("wavevectors must be a finite real array with shape (N, 2)") from error
    if (
        candidate.ndim != 2
        or candidate.shape[0] == 0
        or candidate.shape[1:] != (2,)
        or np.iscomplexobj(candidate)
        or not np.issubdtype(candidate.dtype, np.number)
        or np.issubdtype(candidate.dtype, np.bool_)
    ):
        raise ValueError("wavevectors must be a nonempty real array with shape (N, 2)")
    try:
        points = np.array(candidate, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("wavevectors must be a finite real array with shape (N, 2)") from error
    if not np.isfinite(points).all():
        raise ValueError("wavevectors entries must be finite")
    return points


def _solver_modes(
    solver: Callable[[float, float], ModeSet],
    point: FloatArray,
) -> ModeSet:
    modes = solver(float(point[0]), float(point[1]))
    if not isinstance(modes, ModeSet):
        raise TypeError("solver must return a ModeSet instance")
    return modes


def _require_same_mesh_layout(reference: object, current: object) -> None:
    if not isinstance(reference, GLLMesh) or not isinstance(current, GLLMesh):
        raise ValueError("solver mesh metadata must contain GLLMesh instances")
    if not np.array_equal(reference.nodes, current.nodes):
        raise ValueError("solver mesh nodes must remain identical along a branch")
    if len(reference.elements) != len(current.elements):
        raise ValueError("solver mesh layout must remain identical along a branch")
    for reference_element, current_element in zip(
        reference.elements,
        current.elements,
        strict=True,
    ):
        for field in ("nodes", "weights", "derivative", "connectivity"):
            if not np.array_equal(
                getattr(reference_element, field),
                getattr(current_element, field),
            ):
                raise ValueError("solver mesh layout must remain identical along a branch")


def _require_branch_solver_consistency(reference: ModeSet, current: ModeSet) -> None:
    _reference, _eigenvalues, _omega, reference_vectors, _residuals, reference_mass = (
        _validated_modes(reference)
    )
    _current, _eigenvalues, _omega, current_vectors, _residuals, current_mass = _validated_modes(
        current
    )
    if reference_vectors.shape[0] != current_vectors.shape[0]:
        raise ValueError("solver mode vector dimension must remain constant along a branch")
    if not np.array_equal(reference.matrices.nodes, current.matrices.nodes):
        raise ValueError("solver node-major nodes must remain identical along a branch")
    _require_same_mesh_layout(reference.matrices.mesh, current.matrices.mesh)
    mass_scale = max(
        float(np.max(np.abs(reference_mass), initial=0.0)),
        float(np.max(np.abs(current_mass), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    mass_defect = float(np.max(np.abs(reference_mass - current_mass), initial=0.0))
    if mass_defect > 1.0e-12 * mass_scale:
        raise ValueError("solver mass matrix must remain constant along a branch")


def track_branch(
    wavevectors: Sequence[tuple[float, float]] | FloatArray,
    solver: Callable[[float, float], ModeSet],
    seed_index: int,
    *,
    min_mac: float = 0.2,
) -> TrackedBranch:
    """Predict and correct a mode branch along an ordered wavevector path."""

    points = _validated_wavevectors(wavevectors)
    if not callable(solver):
        raise TypeError("solver must be callable")
    if type(seed_index) is not int:
        raise TypeError("seed_index must be a built-in integer")
    mac_threshold = _unit_interval(min_mac, "min_mac")

    first_modes = _solver_modes(solver, points[0])
    tracked = [seed_tracked_mode(first_modes, seed_index)]
    for point in points[1:]:
        current = _solver_modes(solver, point)
        _require_branch_solver_consistency(first_modes, current)
        predicted = (
            2.0 * tracked[-1].eigenvalue - tracked[-2].eigenvalue
            if len(tracked) > 1
            else tracked[-1].eigenvalue
        )
        tracked.append(
            track_mode(
                tracked[-1],
                current,
                min_mac=mac_threshold,
                predicted_eigenvalue=predicted,
            )
        )
    return TrackedBranch(
        wavevectors=_read_only_float(points),
        modes=tuple(tracked),
    )


def symmetric_lamb_parity_score(vector: object, nodes: object, mass: object) -> float:
    """Score symmetric Lamb parity in node-major order from ``-1`` to ``+1``.

    The symmetric mirror keeps ``ux`` and ``uy`` even and makes ``uz`` odd.
    Nodes must be strictly increasing and mirror-symmetric, and the mass must
    itself be invariant under that node/component parity operation.
    """

    node_values = _real_vector(nodes, "nodes")
    if node_values.size > 1 and not np.all(np.diff(node_values) > 0.0):
        raise ValueError("nodes must be strictly increasing")
    mirror_tolerance = 1.0e-12 * max(float(np.max(np.abs(node_values))), 1.0)
    if not np.allclose(
        node_values,
        -node_values[::-1],
        rtol=0.0,
        atol=mirror_tolerance,
    ):
        raise ValueError("nodes must be mirror-symmetric about zero")

    displacement = _numeric_vector(vector, "vector")
    dimension = 3 * node_values.size
    if displacement.size != dimension:
        raise ValueError("vector must contain exactly 3N node-major displacement entries")
    mass_matrix, _factor = _validated_mass(mass, dimension)
    norm_squared = _mass_norm_squared(displacement, mass_matrix, "vector")

    source_indices = np.arange(dimension, dtype=np.int64).reshape(-1, 3)[::-1].reshape(-1)
    signs = np.ones((node_values.size, 3), dtype=np.float64)
    signs[:, 2] = -1.0
    signs = signs.reshape(-1)
    parity = np.zeros((dimension, dimension), dtype=np.complex128)
    parity[np.arange(dimension), source_indices] = signs
    reflected_mass = parity.conj().T @ mass_matrix @ parity
    mass_scale = float(np.max(np.abs(mass_matrix), initial=0.0))
    parity_defect = float(np.max(np.abs(reflected_mass - mass_matrix), initial=0.0))
    if parity_defect > 1.0e-12 * mass_scale:
        raise ValueError("mass must be compatible with node-major mirror parity")

    mirrored = signs * displacement[source_indices]
    overlap = np.vdot(displacement, mass_matrix @ mirrored)
    imaginary_tolerance = 1.0e-12 * max(abs(overlap), norm_squared, 1.0)
    if abs(float(np.imag(overlap))) > imaginary_tolerance:
        raise ValueError("parity overlap must be real for a compatible mass")
    score = float(np.real(overlap) / norm_squared)
    tolerance = 1.0e-12
    if not np.isfinite(score) or score < -1.0 - tolerance or score > 1.0 + tolerance:
        raise ValueError("parity score is numerically outside [-1, 1]")
    return float(np.clip(score, -1.0, 1.0))
