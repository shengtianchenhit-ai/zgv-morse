"""Legendre--Gauss--Lobatto thickness discretization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_legendre, roots_jacobi


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_MAX_GLL_ORDER = 512


@dataclass(frozen=True, slots=True)
class GLLElement:
    """Read-only numerical data for one physical GLL element."""

    nodes: FloatArray
    weights: FloatArray
    derivative: FloatArray
    connectivity: IntArray


@dataclass(frozen=True, slots=True)
class GLLMesh:
    """Global shared-interface nodes and their local GLL elements."""

    nodes: FloatArray
    elements: tuple[GLLElement, ...]


def _read_only_float64(values: object) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _read_only_int64(values: object) -> IntArray:
    array = np.array(values, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


def _validate_order(order: object) -> int:
    if type(order) is not int:
        raise TypeError("order must be a built-in integer")
    if not 2 <= order <= _MAX_GLL_ORDER:
        raise ValueError(f"order must be between 2 and {_MAX_GLL_ORDER}")
    return order


def _float_vector(
    values: object,
    name: str,
    *,
    minimum_size: int = 0,
    distinct: bool = False,
) -> FloatArray:
    try:
        candidate = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real floating array") from error
    if candidate.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(candidate.dtype, np.floating):
        raise TypeError(f"{name} must contain real floating values")
    array = np.array(candidate, dtype=np.float64, copy=True)
    if array.size < minimum_size:
        raise ValueError(f"{name} must contain at least two values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if distinct and np.unique(array).size != array.size:
        raise ValueError(f"{name} must contain distinct values")
    return array


def _float_matrix(values: object, name: str) -> FloatArray:
    try:
        candidate = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real floating array") from error
    if candidate.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.issubdtype(candidate.dtype, np.floating):
        raise TypeError(f"{name} must contain real floating values")
    array = np.array(candidate, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _validated_element_bounds(element_bounds: object) -> FloatArray:
    if not isinstance(element_bounds, (tuple, list)):
        raise TypeError("element_bounds must be a tuple or list")
    if len(element_bounds) < 2:
        raise ValueError("element_bounds must contain at least two values")

    converted: list[float] = []
    for index, value in enumerate(element_bounds):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"element_bounds[{index}] must be a real scalar")
        try:
            bound = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"element_bounds[{index}] must be finite") from error
        if not np.isfinite(bound):
            raise ValueError(f"element_bounds[{index}] must be finite")
        converted.append(bound)

    bounds = np.array(converted, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        increasing = np.all(np.diff(bounds) > 0.0)
    if not increasing:
        raise ValueError("element_bounds must be strictly increasing after float64 conversion")
    return bounds


def gll_nodes_weights(order: int) -> tuple[FloatArray, FloatArray]:
    """Return an ``order + 1`` point Legendre--Gauss--Lobatto rule.

    ``order`` is the polynomial degree and must be a built-in integer between
    2 and 512.  The finite upper bound keeps accidental or impractical root
    computations from consuming unbounded resources.
    """

    degree = _validate_order(order)
    try:
        interior, _ = roots_jacobi(degree - 1, 1.0, 1.0)
    except (FloatingPointError, MemoryError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"failed to compute GLL roots for order {degree}") from error

    interior = np.asarray(interior, dtype=np.float64)
    if (
        interior.shape != (degree - 1,)
        or not np.isfinite(interior).all()
        or not np.all(np.diff(interior) > 0.0)
        or np.any(np.abs(interior) >= 1.0)
    ):
        raise RuntimeError(f"invalid GLL roots returned for order {degree}")

    # Jacobi roots are symmetric analytically.  Enforcing that symmetry removes
    # harmless platform-level asymmetry and makes odd moments cancel exactly.
    interior = 0.5 * (interior - interior[::-1])
    nodes = np.concatenate(([-1.0], interior, [1.0])).astype(np.float64, copy=False)
    legendre_values = np.asarray(eval_legendre(degree, nodes), dtype=np.float64)
    weights = 2.0 / (degree * (degree + 1) * legendre_values**2)
    endpoint_weight = 2.0 / (degree * (degree + 1))
    weights[0] = endpoint_weight
    weights[-1] = endpoint_weight
    weights = 0.5 * (weights + weights[::-1])

    if not np.isfinite(weights).all() or not np.all(weights > 0.0):
        raise RuntimeError(f"invalid GLL weights returned for order {degree}")
    return _read_only_float64(nodes), _read_only_float64(weights)


def differentiation_matrix(nodes: FloatArray) -> FloatArray:
    """Return the barycentric collocation derivative for distinct ``nodes``.

    The node order is preserved.  Inputs must be a one-dimensional real
    floating array-like object; integer, Boolean, complex, and object dtypes
    are rejected instead of being silently coerced.
    """

    x = _float_vector(nodes, "nodes", minimum_size=2, distinct=True)
    center = 0.5 * np.min(x) + 0.5 * np.max(x)
    scale = float(np.max(np.abs(x - center)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("nodes must have a finite, nonzero scale")
    scaled = (x - center) / scale

    differences = scaled[:, np.newaxis] - scaled[np.newaxis, :]
    np.fill_diagonal(differences, 1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        signs = np.prod(np.sign(differences), axis=1)
        logarithms = -np.sum(np.log(np.abs(differences)), axis=1)
        logarithms -= np.max(logarithms)
        barycentric_weights = signs * np.exp(logarithms)
        ratios = barycentric_weights[np.newaxis, :] / barycentric_weights[:, np.newaxis]
        derivative = ratios / differences
    np.fill_diagonal(derivative, 0.0)
    derivative[np.diag_indices_from(derivative)] = -np.sum(derivative, axis=1)
    derivative /= scale
    if not np.isfinite(derivative).all():
        raise ValueError("nodes do not define a finite differentiation matrix")
    return _read_only_float64(derivative)


def map_to_thickness(
    nodes: FloatArray,
    weights: FloatArray,
    derivative: FloatArray,
    h: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Map reference data from ``[-1, 1]`` to the symmetric interval ``[-h, h]``.

    Array inputs must have real floating dtypes.  Their sizes must agree, and
    ``h`` may be any finite positive real scalar other than a Boolean.
    """

    reference_nodes = _float_vector(nodes, "nodes", minimum_size=2, distinct=True)
    reference_weights = _float_vector(weights, "weights")
    reference_derivative = _float_matrix(derivative, "derivative")
    if reference_weights.shape != reference_nodes.shape:
        raise ValueError("weights must have the same length as nodes")
    expected_shape = (reference_nodes.size, reference_nodes.size)
    if reference_derivative.shape != expected_shape:
        raise ValueError(f"derivative must have shape {expected_shape}")
    half_thickness = _positive_real(h, "h")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mapped_nodes = reference_nodes * half_thickness
        mapped_weights = reference_weights * half_thickness
        mapped_derivative = reference_derivative / half_thickness
    if not all(
        np.isfinite(array).all()
        for array in (mapped_nodes, mapped_weights, mapped_derivative)
    ):
        raise ValueError("mapping must produce only finite values")
    return (
        _read_only_float64(mapped_nodes),
        _read_only_float64(mapped_weights),
        _read_only_float64(mapped_derivative),
    )


def build_gll_mesh(order: int, element_bounds: tuple[float, ...] | list[float]) -> GLLMesh:
    """Build a one-dimensional GLL mesh over strictly increasing element bounds.

    Bounds must be supplied as a tuple or list of at least two finite real
    scalars.  Integer bounds are accepted, but all bounds must remain strictly
    increasing after conversion to float64.  Adjacent elements share exactly
    one global interface node.
    """

    degree = _validate_order(order)
    bounds = _validated_element_bounds(element_bounds)
    reference_nodes, reference_weights = gll_nodes_weights(degree)
    reference_derivative = differentiation_matrix(reference_nodes)

    elements: list[GLLElement] = []
    global_pieces: list[FloatArray] = []
    for element_index, (left, right) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        center = 0.5 * left + 0.5 * right
        half_width = 0.5 * right - 0.5 * left
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            local_nodes = center + half_width * reference_nodes
            local_weights = half_width * reference_weights
            local_derivative = reference_derivative / half_width
        local_nodes = np.array(local_nodes, dtype=np.float64, copy=True)
        local_nodes[0] = left
        local_nodes[-1] = right
        local_arrays = (local_nodes, local_weights, local_derivative)
        if not all(np.isfinite(array).all() for array in local_arrays):
            raise ValueError(f"element {element_index} mapping must produce only finite values")
        if not np.all(np.diff(local_nodes) > 0.0):
            raise ValueError(f"element {element_index} mapped nodes must be strictly increasing")

        start = element_index * degree
        connectivity = _read_only_int64(np.arange(start, start + degree + 1, dtype=np.int64))
        element = GLLElement(
            nodes=_read_only_float64(local_nodes),
            weights=_read_only_float64(local_weights),
            derivative=_read_only_float64(local_derivative),
            connectivity=connectivity,
        )
        elements.append(element)
        global_pieces.append(element.nodes if element_index == 0 else element.nodes[1:])

    global_nodes = _read_only_float64(np.concatenate(global_pieces))
    return GLLMesh(nodes=global_nodes, elements=tuple(elements))
