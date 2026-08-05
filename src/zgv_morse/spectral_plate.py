"""GLL weak-form discretization of an infinite elastic plate."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, eigvalsh

from zgv_morse.elasticity import tensor_to_mandel
from zgv_morse.gll import GLLMesh, build_gll_mesh


ComplexArray: TypeAlias = NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PlateMatrices:
    """Assembled weak-form stiffness and mass matrices."""

    stiffness: ComplexArray
    mass: ComplexArray
    nodes: FloatArray
    mesh: GLLMesh
    hermitian_defect: float


@dataclass(frozen=True, slots=True)
class WavevectorDerivatives:
    """Analytic first and second stiffness derivatives in wavevector space."""

    dkx: ComplexArray
    dky: ComplexArray
    dkx2: ComplexArray
    dkx_dky: ComplexArray
    dky2: ComplexArray


@dataclass(frozen=True, slots=True)
class ModeSet:
    """Lowest generalized plate modes and numerical diagnostics."""

    eigenvalues: FloatArray
    omega: FloatArray
    vectors: ComplexArray
    residuals: FloatArray
    mass_orthogonality_defect: float
    matrices: PlateMatrices


def _interpolation(shape: FloatArray) -> ComplexArray:
    """Return the vector interpolation matrix in node-major DOF order."""

    values = np.asarray(shape, dtype=np.float64)
    interpolation = np.zeros((3, 3 * values.size), dtype=np.complex128)
    for node, value in enumerate(values):
        interpolation[:, 3 * node : 3 * node + 3] = value * np.eye(3)
    return interpolation


def _global_dofs(connectivity: NDArray[np.int64]) -> NDArray[np.int64]:
    """Expand global node indices into node-major displacement triples."""

    nodes = np.asarray(connectivity, dtype=np.int64)
    return (3 * nodes[:, np.newaxis] + np.arange(3, dtype=np.int64)).reshape(-1)


def _mandel_b_parts(
    shape: FloatArray,
    derivative: FloatArray,
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    """Return ``B0, Bx, By`` for ``B = B0 + kx*Bx + ky*By``."""

    values = np.asarray(shape, dtype=np.float64)
    gradients = np.asarray(derivative, dtype=np.float64)
    width = 3 * values.size
    b0 = np.zeros((6, width), dtype=np.complex128)
    bx = np.zeros_like(b0)
    by = np.zeros_like(b0)
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)

    for node, (value, gradient) in enumerate(zip(values, gradients, strict=True)):
        ux, uy, uz = 3 * node, 3 * node + 1, 3 * node + 2
        b0[2, uz] = gradient
        b0[3, uy] = gradient * inverse_sqrt_two
        b0[4, ux] = gradient * inverse_sqrt_two

        bx[0, ux] = 1j * value
        bx[4, uz] = 1j * value * inverse_sqrt_two
        bx[5, uy] = 1j * value * inverse_sqrt_two

        by[1, uy] = 1j * value
        by[3, uz] = 1j * value * inverse_sqrt_two
        by[5, ux] = 1j * value * inverse_sqrt_two

    return b0, bx, by


def _read_only_complex(values: object) -> ComplexArray:
    result = np.array(values, dtype=np.complex128, copy=True)
    result.setflags(write=False)
    return result


def _read_only_float(values: object) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _finite_real(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        scalar = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _validated_constitutive(C: object) -> FloatArray:
    try:
        candidate = np.asarray(C)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("C must be a finite real elasticity tensor") from error
    if np.iscomplexobj(candidate):
        raise ValueError("C must be a real elasticity tensor")
    try:
        tensor = np.array(candidate, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("C must be a finite real elasticity tensor") from error
    if tensor.shape != (3, 3, 3, 3):
        raise ValueError("C must have shape (3, 3, 3, 3)")
    if not np.isfinite(tensor).all():
        raise ValueError("C entries must be finite")

    scale = float(np.max(np.abs(tensor), initial=0.0))
    tolerance = 1.0e-12 * scale
    symmetry_images = (
        tensor.swapaxes(0, 1),
        tensor.swapaxes(2, 3),
        tensor.transpose(2, 3, 0, 1),
    )
    if any(
        float(np.max(np.abs(tensor - image), initial=0.0)) > tolerance for image in symmetry_images
    ):
        raise ValueError("C must satisfy the minor and major elasticity symmetries")

    constitutive = tensor_to_mandel(tensor)
    constitutive = 0.5 * constitutive + 0.5 * constitutive.T
    if not np.isfinite(constitutive).all():
        raise ValueError("C must produce a finite Mandel matrix")
    try:
        constitutive_eigenvalues = eigvalsh(constitutive, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("C must produce a positive-definite Mandel matrix") from error
    if constitutive_eigenvalues[0] <= 0.0:
        raise ValueError("C must produce a positive-definite Mandel matrix")
    return constitutive


def _validated_problem(
    kx: object,
    ky: object,
    C: object,
    rho: object,
    half_thickness: object,
    order: object,
    element_bounds: object,
) -> tuple[float, float, float, FloatArray, GLLMesh]:
    kx_value = _finite_real(kx, "kx")
    ky_value = _finite_real(ky, "ky")
    rho_value = _finite_real(rho, "rho", positive=True)
    thickness_value = _finite_real(half_thickness, "half_thickness", positive=True)
    bounds = (-thickness_value, thickness_value) if element_bounds is None else element_bounds
    mesh = build_gll_mesh(order, bounds)  # type: ignore[arg-type]
    if element_bounds is not None and (
        mesh.nodes[0] != -thickness_value or mesh.nodes[-1] != thickness_value
    ):
        raise ValueError("element_bounds must exactly span [-half_thickness, half_thickness]")
    constitutive = _validated_constitutive(C)
    return kx_value, ky_value, rho_value, constitutive, mesh


def _require_finite_outputs(*arrays: NDArray[np.generic], context: str) -> None:
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{context} produced non-finite output")


def assemble_plate_matrices(
    kx: float,
    ky: float,
    C: NDArray[np.float64],
    rho: float,
    half_thickness: float,
    order: int = 24,
    element_bounds: tuple[float, ...] | list[float] | None = None,
) -> PlateMatrices:
    """Assemble the unconstrained GLL weak form with natural free faces."""

    kx_value, ky_value, rho_value, constitutive, mesh = _validated_problem(
        kx,
        ky,
        C,
        rho,
        half_thickness,
        order,
        element_bounds,
    )
    number_of_dofs = 3 * mesh.nodes.size
    stiffness = np.zeros((number_of_dofs, number_of_dofs), dtype=np.complex128)
    mass = np.zeros_like(stiffness)

    for element in mesh.elements:
        global_dofs = _global_dofs(element.connectivity)
        local_size = element.connectivity.size
        local_stiffness = np.zeros((3 * local_size, 3 * local_size), dtype=np.complex128)
        local_mass = np.zeros_like(local_stiffness)
        shapes = np.eye(local_size, dtype=np.float64)
        for quadrature_index, weight in enumerate(element.weights):
            shape = shapes[quadrature_index]
            b0, bx, by = _mandel_b_parts(shape, element.derivative[quadrature_index])
            strain = b0 + kx_value * bx + ky_value * by
            interpolation = _interpolation(shape)
            with np.errstate(over="ignore", invalid="ignore"):
                local_stiffness += weight * (strain.conj().T @ constitutive @ strain)
                local_mass += weight * rho_value * (interpolation.conj().T @ interpolation)

        block = np.ix_(global_dofs, global_dofs)
        stiffness[block] += local_stiffness
        mass[block] += local_mass

    _require_finite_outputs(stiffness, mass, context="plate assembly")
    stiffness_scale = float(np.linalg.norm(stiffness, ord="fro"))
    antihermitian_scale = float(np.linalg.norm(stiffness - stiffness.conj().T, ord="fro"))
    hermitian_defect = antihermitian_scale / max(
        stiffness_scale,
        np.finfo(np.float64).tiny,
    )
    return PlateMatrices(
        stiffness=_read_only_complex(stiffness),
        mass=_read_only_complex(mass),
        nodes=_read_only_float(mesh.nodes),
        mesh=mesh,
        hermitian_defect=hermitian_defect,
    )


def assemble_wavevector_derivatives(
    kx: float,
    ky: float,
    C: NDArray[np.float64],
    rho: float,
    half_thickness: float,
    order: int = 24,
    element_bounds: tuple[float, ...] | list[float] | None = None,
) -> WavevectorDerivatives:
    """Assemble analytic first and second wavevector stiffness derivatives."""

    kx_value, ky_value, _rho_value, constitutive, mesh = _validated_problem(
        kx,
        ky,
        C,
        rho,
        half_thickness,
        order,
        element_bounds,
    )
    number_of_dofs = 3 * mesh.nodes.size
    assembled = [np.zeros((number_of_dofs, number_of_dofs), dtype=np.complex128) for _ in range(5)]

    for element in mesh.elements:
        global_dofs = _global_dofs(element.connectivity)
        local_size = element.connectivity.size
        local_derivatives = [
            np.zeros((3 * local_size, 3 * local_size), dtype=np.complex128) for _ in range(5)
        ]
        shapes = np.eye(local_size, dtype=np.float64)
        for quadrature_index, weight in enumerate(element.weights):
            b0, bx, by = _mandel_b_parts(
                shapes[quadrature_index],
                element.derivative[quadrature_index],
            )
            strain = b0 + kx_value * bx + ky_value * by
            with np.errstate(over="ignore", invalid="ignore"):
                first_x_term = bx.conj().T @ constitutive @ strain
                first_y_term = by.conj().T @ constitutive @ strain
                second_x = 2.0 * (bx.conj().T @ constitutive @ bx)
                mixed_term = bx.conj().T @ constitutive @ by
                second_y = 2.0 * (by.conj().T @ constitutive @ by)
            contributions = (
                first_x_term + first_x_term.conj().T,
                first_y_term + first_y_term.conj().T,
                second_x,
                mixed_term + mixed_term.conj().T,
                second_y,
            )
            for local, contribution in zip(
                local_derivatives,
                contributions,
                strict=True,
            ):
                local += weight * contribution

        block = np.ix_(global_dofs, global_dofs)
        for global_matrix, local_matrix in zip(assembled, local_derivatives, strict=True):
            global_matrix[block] += local_matrix

    _require_finite_outputs(*assembled, context="wavevector derivative assembly")
    for matrix in assembled:
        matrix[...] = 0.5 * matrix + 0.5 * matrix.conj().T
    return WavevectorDerivatives(*(_read_only_complex(matrix) for matrix in assembled))


def _numeric_matrix(values: object, name: str) -> ComplexArray:
    try:
        candidate = np.asarray(values)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric matrix") from error
    if candidate.ndim != 2:
        raise ValueError(f"{name} must have a two-dimensional square shape")
    if not np.issubdtype(candidate.dtype, np.number) or np.issubdtype(candidate.dtype, np.bool_):
        raise ValueError(f"{name} must be a numeric matrix")
    try:
        matrix = np.array(candidate, dtype=np.complex128, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric matrix") from error
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} entries must be finite")
    return matrix


def _require_hermitian(matrix: ComplexArray, name: str) -> None:
    scale = float(np.max(np.abs(matrix), initial=0.0))
    defect = float(np.max(np.abs(matrix - matrix.conj().T), initial=0.0))
    if defect > 1.0e-12 * scale:
        raise ValueError(f"{name} must be Hermitian within relative tolerance 1e-12")


def _validated_eigenproblem(
    matrices: object,
    num_modes: object,
) -> tuple[PlateMatrices, ComplexArray, ComplexArray, FloatArray, int]:
    if not isinstance(matrices, PlateMatrices):
        raise TypeError("matrices must be a PlateMatrices instance")
    if type(num_modes) is not int:
        raise TypeError("num_modes must be a built-in integer")

    stiffness = _numeric_matrix(matrices.stiffness, "stiffness")
    mass = _numeric_matrix(matrices.mass, "mass")
    if (
        stiffness.shape[0] != stiffness.shape[1]
        or mass.shape != stiffness.shape
        or stiffness.shape[0] == 0
        or stiffness.shape[0] % 3 != 0
    ):
        raise ValueError("stiffness and mass must have the same nonempty square shape")
    if not 1 <= num_modes <= stiffness.shape[0]:
        raise ValueError("num_modes must be between 1 and the matrix dimension")
    _require_hermitian(stiffness, "stiffness")
    _require_hermitian(mass, "mass")

    try:
        node_candidate = np.asarray(matrices.nodes)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("nodes must be a finite real vector") from error
    if np.iscomplexobj(node_candidate):
        raise ValueError("nodes must be a finite real vector")
    try:
        nodes = np.array(node_candidate, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("nodes must be a finite real vector") from error
    if nodes.shape != (stiffness.shape[0] // 3,) or not np.isfinite(nodes).all():
        raise ValueError("nodes shape must agree with node-major matrix DOFs and be finite")
    if not isinstance(matrices.mesh, GLLMesh):
        raise ValueError("mesh must be a GLLMesh instance")
    if not np.array_equal(nodes, matrices.mesh.nodes):
        raise ValueError("nodes must agree with mesh.nodes")
    defect = _finite_real(matrices.hermitian_defect, "hermitian_defect")
    if defect < 0.0:
        raise ValueError("hermitian_defect must be nonnegative")

    try:
        mass_eigenvalues = eigvalsh(mass, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("mass must be Hermitian positive definite") from error
    if not np.isfinite(mass_eigenvalues).all() or mass_eigenvalues[0] <= 0.0:
        raise ValueError("mass must be Hermitian positive definite")
    return matrices, stiffness, mass, mass_eigenvalues, num_modes


def solve_plate_modes(matrices: PlateMatrices, num_modes: int) -> ModeSet:
    """Solve for the lowest mass-normalized generalized Hermitian modes."""

    matrices, stiffness, mass, mass_eigenvalues, mode_count = _validated_eigenproblem(
        matrices,
        num_modes,
    )
    try:
        eigenvalues, vectors = eigh(
            stiffness,
            mass,
            subset_by_index=(0, mode_count - 1),
            driver="gvx",
            check_finite=False,
        )
    except np.linalg.LinAlgError as error:
        raise RuntimeError("generalized Hermitian eigensolver failed") from error
    _require_finite_outputs(eigenvalues, vectors, context="generalized eigensolver")

    stiffness_norm = float(np.linalg.norm(stiffness, ord=2))
    mass_norm = float(np.linalg.norm(mass, ord=2))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eigenvalue_scale = stiffness_norm / float(mass_eigenvalues[0])
    if not all(np.isfinite(value) for value in (stiffness_norm, mass_norm, eigenvalue_scale)):
        raise ValueError("generalized eigenproblem must have a finite numerical scale")
    negative_tolerance = (
        1.0e3
        * np.finfo(np.float64).eps
        * max(
            eigenvalue_scale,
            np.finfo(np.float64).tiny,
        )
    )
    if eigenvalues[0] < -negative_tolerance:
        raise RuntimeError("generalized eigensolver returned a materially negative eigenvalue")
    eigenvalues = np.maximum(eigenvalues, 0.0)

    for column in range(vectors.shape[1]):
        vector = vectors[:, column]
        mass_norm_squared = float(np.real(vector.conj() @ mass @ vector))
        if not np.isfinite(mass_norm_squared) or mass_norm_squared <= 0.0:
            raise RuntimeError("generalized eigensolver returned an invalid mass norm")
        vectors[:, column] /= np.sqrt(mass_norm_squared)

    residuals = np.empty(eigenvalues.size, dtype=np.float64)
    for column, eigenvalue in enumerate(eigenvalues):
        vector = vectors[:, column]
        left = stiffness @ vector
        right = eigenvalue * (mass @ vector)
        residuals[column] = np.linalg.norm(left - right) / max(
            np.linalg.norm(left) + np.linalg.norm(right),
            np.finfo(np.float64).tiny,
        )
    gram = vectors.conj().T @ mass @ vectors
    mass_orthogonality_defect = float(np.linalg.norm(gram - np.eye(mode_count), ord="fro"))
    _require_finite_outputs(
        eigenvalues,
        vectors,
        residuals,
        gram,
        context="mode solution",
    )
    if not np.isfinite(mass_orthogonality_defect):
        raise RuntimeError("mode diagnostics are non-finite")

    return ModeSet(
        eigenvalues=_read_only_float(eigenvalues),
        omega=_read_only_float(np.sqrt(eigenvalues)),
        vectors=_read_only_complex(vectors),
        residuals=_read_only_float(residuals),
        mass_orthogonality_defect=mass_orthogonality_defect,
        matrices=matrices,
    )
