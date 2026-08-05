"""Fourth-order isotropic and cubic elasticity tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


Tensor4: TypeAlias = NDArray[np.float64]


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _finite_sum(name: str, *values: float) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    return _finite_scalar(name, result)


def _isotropic_moduli(lam: object, mu: object) -> tuple[float, float]:
    lam_value = _finite_scalar("lam", lam)
    mu_value = _finite_scalar("mu", mu)
    if mu_value <= 0.0:
        raise ValueError("mu must be positive")
    if lam_value + (2.0 / 3.0) * mu_value <= 0.0:
        raise ValueError("bulk modulus lam + 2*mu/3 must be positive")
    return lam_value, mu_value


def _real_array(name: str, value: object) -> NDArray[np.float64]:
    try:
        unconverted = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real array") from error
    if np.iscomplexobj(unconverted):
        raise ValueError(f"{name} must be a finite real array")
    try:
        array = np.asarray(unconverted, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real array") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} entries must all be finite")
    return array


def _finite_result(name: str, result: NDArray[np.float64]) -> NDArray[np.float64]:
    if not np.isfinite(result).all():
        raise ValueError(f"{name} result entries must all be finite")
    return result


def _tensor4(value: object) -> Tensor4:
    tensor = _real_array("tensor", value)
    if tensor.shape != (3, 3, 3, 3):
        raise ValueError("tensor must have shape (3, 3, 3, 3)")
    return tensor


def _rotation_matrix(value: object) -> NDArray[np.float64]:
    rotation = _real_array("rotation", value)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    identity = np.eye(3)
    with np.errstate(over="ignore", invalid="ignore"):
        gram_matrix = rotation.T @ rotation
    if not np.allclose(gram_matrix, identity, rtol=0.0, atol=1.0e-12):
        raise ValueError("rotation must be orthogonal within tolerance 1e-12")
    return rotation


def _cubic_components(c11: float, c12: float, c44: float) -> Tensor4:
    tensor = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for first_axis in range(3):
        tensor[first_axis, first_axis, first_axis, first_axis] = c11
        for second_axis in range(3):
            if first_axis == second_axis:
                continue
            tensor[first_axis, first_axis, second_axis, second_axis] = c12
            tensor[first_axis, second_axis, first_axis, second_axis] = c44
            tensor[first_axis, second_axis, second_axis, first_axis] = c44
    return tensor


@dataclass(frozen=True, slots=True)
class CubicConstants:
    """Independent stiffness constants of a cubic elastic material."""

    c11: float
    c12: float
    c44: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "c11", _finite_scalar("c11", self.c11))
        object.__setattr__(self, "c12", _finite_scalar("c12", self.c12))
        object.__setattr__(self, "c44", _finite_scalar("c44", self.c44))


def assert_cubic_stability(constants: CubicConstants) -> None:
    """Raise when any positive-definiteness invariant is violated."""

    if not isinstance(constants, CubicConstants):
        raise TypeError("constants must be a CubicConstants instance")

    failed_invariants: list[str] = []
    if constants.c44 <= 0.0:
        failed_invariants.append("C44 > 0")
    if constants.c11 - constants.c12 <= 0.0:
        failed_invariants.append("C11 - C12 > 0")
    if constants.c11 + 2.0 * constants.c12 <= 0.0:
        failed_invariants.append("C11 + 2*C12 > 0")
    if failed_invariants:
        failures = ", ".join(failed_invariants)
        raise ValueError(f"unstable cubic constants; failed invariants: {failures}")


def isotropic_tensor(lam: float, mu: float) -> Tensor4:
    """Return the isotropic stiffness tensor for Lamé moduli ``lam`` and ``mu``."""

    lam_value, mu_value = _isotropic_moduli(lam, mu)
    identity = np.eye(3)
    with np.errstate(over="ignore", invalid="ignore"):
        tensor = (
            lam_value * np.einsum("ij,kl->ijkl", identity, identity)
            + mu_value * np.einsum("ik,jl->ijkl", identity, identity)
            + mu_value * np.einsum("il,jk->ijkl", identity, identity)
        )
    return _finite_result("isotropic tensor", tensor)


def cubic_tensor(c11: float, c12: float, c44: float) -> Tensor4:
    """Return a stable cubic stiffness tensor in its crystallographic frame."""

    constants = CubicConstants(c11=c11, c12=c12, c44=c44)
    assert_cubic_stability(constants)
    tensor = _cubic_components(constants.c11, constants.c12, constants.c44)
    return _finite_result("cubic tensor", tensor)


def cubic_family(
    lam: float,
    mu: float,
    delta: float,
    epsilon: float,
) -> tuple[Tensor4, CubicConstants]:
    """Return the volume-preserving cubic perturbation of isotropy."""

    lam_value, mu_value = _isotropic_moduli(lam, mu)
    delta_value = _finite_scalar("delta", delta)
    epsilon_value = _finite_scalar("epsilon", epsilon)
    if delta_value == 0.0:
        raise ValueError("delta must be nonzero")

    c11_shift = (2.0 / 3.0) * epsilon_value * delta_value
    c12_shift = (1.0 / 3.0) * epsilon_value * delta_value
    constants = CubicConstants(
        c11=_finite_sum("c11", lam_value, mu_value, mu_value, c11_shift),
        c12=_finite_sum("c12", lam_value, -c12_shift),
        c44=mu_value,
    )
    assert_cubic_stability(constants)
    tensor = _cubic_components(constants.c11, constants.c12, constants.c44)
    return _finite_result("cubic family tensor", tensor), constants


def cubic_perturbation_tensor(delta: float) -> Tensor4:
    """Return the exact derivative of :func:`cubic_family` with respect to epsilon."""

    delta_value = _finite_scalar("delta", delta)
    if delta_value == 0.0:
        raise ValueError("delta must be nonzero")
    tensor = _cubic_components(
        (2.0 / 3.0) * delta_value,
        -(1.0 / 3.0) * delta_value,
        0.0,
    )
    return _finite_result("cubic perturbation tensor", tensor)


def rotate_tensor(c: Tensor4, rotation: NDArray[np.float64]) -> Tensor4:
    """Rotate all four indices of an elasticity tensor into a new frame."""

    tensor = _tensor4(c)
    matrix = _rotation_matrix(rotation)
    with np.errstate(over="ignore", invalid="ignore"):
        rotated = np.einsum(
            "ia,jb,kc,ld,abcd->ijkl",
            matrix,
            matrix,
            matrix,
            matrix,
            tensor,
            optimize=True,
        )
    return _finite_result("rotated tensor", rotated)


def tensor_to_mandel(c: Tensor4) -> NDArray[np.float64]:
    """Project a fourth-order tensor onto the orthonormal Mandel basis."""

    tensor = _tensor4(c)
    basis = np.zeros((6, 3, 3), dtype=np.float64)
    for mandel_index, (first_axis, second_axis) in enumerate(
        ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    ):
        if first_axis == second_axis:
            basis[mandel_index, first_axis, second_axis] = 1.0
        else:
            basis[mandel_index, first_axis, second_axis] = np.sqrt(0.5)
            basis[mandel_index, second_axis, first_axis] = np.sqrt(0.5)
    with np.errstate(over="ignore", invalid="ignore"):
        mandel = np.einsum("aij,ijkl,bkl->ab", basis, tensor, basis, optimize=True)
    return _finite_result("Mandel matrix", mandel)
