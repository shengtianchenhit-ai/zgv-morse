from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import zgv_morse.gll as gll_module
from zgv_morse.gll import (
    GLLElement,
    GLLMesh,
    build_gll_mesh,
    differentiation_matrix,
    gll_nodes_weights,
    map_to_thickness,
)


def test_order_two_has_known_nodes_and_weights() -> None:
    nodes, weights = gll_nodes_weights(2)

    np.testing.assert_array_equal(nodes, np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_allclose(weights, np.array([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]))


def test_order_eight_integrates_degree_fourteen() -> None:
    nodes, weights = gll_nodes_weights(8)

    assert np.sum(weights) == pytest.approx(2.0, rel=0.0, abs=2.0e-14)
    assert np.dot(weights, nodes**14) == pytest.approx(2.0 / 15.0, rel=0.0, abs=2.0e-14)


@pytest.mark.parametrize("order", [2, 3, 5, 8, 12])
def test_rules_are_sorted_symmetric_positive_and_finite(order: int) -> None:
    nodes, weights = gll_nodes_weights(order)

    assert nodes.shape == weights.shape == (order + 1,)
    assert nodes[0] == -1.0
    assert nodes[-1] == 1.0
    assert np.all(np.diff(nodes) > 0.0)
    assert np.all(weights > 0.0)
    assert np.isfinite(nodes).all()
    assert np.isfinite(weights).all()
    np.testing.assert_allclose(nodes, -nodes[::-1], rtol=0.0, atol=4.0e-15)
    np.testing.assert_allclose(weights, weights[::-1], rtol=0.0, atol=4.0e-15)


@pytest.mark.parametrize("order", [2, 3, 5, 8, 12])
def test_rules_integrate_every_monomial_through_degree_two_order_minus_one(
    order: int,
) -> None:
    nodes, weights = gll_nodes_weights(order)

    for degree in range(2 * order):
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        assert np.dot(weights, nodes**degree) == pytest.approx(
            expected,
            rel=0.0,
            abs=5.0e-14,
        )


@pytest.mark.parametrize(
    "order",
    [True, False, 2.0, np.int64(2), "2", None],
    ids=["true", "false", "float", "numpy-int", "string", "none"],
)
def test_rule_requires_a_builtin_integer_order(order: object) -> None:
    with pytest.raises(TypeError, match="order.*integer"):
        gll_nodes_weights(order)  # type: ignore[arg-type]


@pytest.mark.parametrize("order", [-1, 0, 1, 513, 100_000])
def test_rule_rejects_unsupported_orders_cleanly(order: int) -> None:
    with pytest.raises(ValueError, match="order.*2.*512"):
        gll_nodes_weights(order)


def test_rule_arrays_are_independent_read_only_float64_values() -> None:
    nodes, weights = gll_nodes_weights(5)

    assert nodes.dtype == np.float64
    assert weights.dtype == np.float64
    assert not np.shares_memory(nodes, weights)
    assert not nodes.flags.writeable
    assert not weights.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        nodes[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        weights[0] = 0.0


def test_rule_translates_root_backend_failures_to_a_contextual_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_roots(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("backend-specific failure")

    monkeypatch.setattr(gll_module, "roots_jacobi", fail_roots)

    with pytest.raises(RuntimeError, match="failed to compute GLL roots for order 4"):
        gll_nodes_weights(4)


def test_rule_rejects_malformed_root_backend_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = np.array([-0.5, np.nan, 0.5])
    monkeypatch.setattr(
        gll_module,
        "roots_jacobi",
        lambda *_args, **_kwargs: (malformed, np.ones_like(malformed)),
    )

    with pytest.raises(RuntimeError, match="invalid GLL roots returned for order 4"):
        gll_nodes_weights(4)


def test_order_ten_differentiates_degree_ten_polynomial() -> None:
    nodes, _ = gll_nodes_weights(10)
    derivative = differentiation_matrix(nodes)
    values = nodes**10 - 0.3 * nodes**5 + 0.7
    expected = 10.0 * nodes**9 - 1.5 * nodes**4

    np.testing.assert_allclose(derivative @ values, expected, rtol=0.0, atol=2.0e-12)


@pytest.mark.parametrize("order", [2, 5, 10])
def test_derivative_rows_sum_to_zero_and_differentiate_monomials_through_order(
    order: int,
) -> None:
    nodes, _ = gll_nodes_weights(order)
    derivative = differentiation_matrix(nodes)

    np.testing.assert_allclose(np.sum(derivative, axis=1), 0.0, rtol=0.0, atol=3.0e-13)
    for degree in range(order + 1):
        expected = (
            np.zeros_like(nodes) if degree == 0 else degree * nodes ** (degree - 1)
        )
        np.testing.assert_allclose(
            derivative @ nodes**degree,
            expected,
            rtol=0.0,
            atol=5.0e-12,
        )


def test_differentiation_preserves_the_order_of_distinct_unsorted_nodes() -> None:
    nodes = np.array([1.0, -1.0, 0.25, -0.5], dtype=np.float64)
    derivative = differentiation_matrix(nodes)

    np.testing.assert_allclose(derivative @ nodes**3, 3.0 * nodes**2, atol=2.0e-14)


@pytest.mark.parametrize(
    "nodes",
    [
        pytest.param(np.array([[-1.0, 0.0], [0.5, 1.0]]), id="matrix"),
        pytest.param(np.array(0.0), id="scalar"),
    ],
)
def test_differentiation_requires_a_one_dimensional_node_array(nodes: np.ndarray) -> None:
    with pytest.raises(ValueError, match="nodes.*one-dimensional"):
        differentiation_matrix(nodes)


@pytest.mark.parametrize(
    "nodes",
    [
        pytest.param(np.array([-1, 0, 1]), id="integer"),
        pytest.param(np.array([False, True]), id="boolean"),
        pytest.param(np.array([-1.0 + 0.0j, 1.0 + 0.0j]), id="complex"),
        pytest.param(np.array(["-1.0", "1.0"]), id="string"),
        pytest.param(np.array([-1.0, 1.0], dtype=object), id="object"),
    ],
)
def test_differentiation_rejects_nonfloating_node_arrays(nodes: np.ndarray) -> None:
    with pytest.raises(TypeError, match="nodes.*floating"):
        differentiation_matrix(nodes)


@pytest.mark.parametrize(
    "nodes",
    [
        pytest.param(np.array([-1.0, np.nan, 1.0]), id="nan"),
        pytest.param(np.array([-1.0, np.inf, 1.0]), id="infinity"),
    ],
)
def test_differentiation_rejects_nonfinite_nodes(nodes: np.ndarray) -> None:
    with pytest.raises(ValueError, match="nodes.*finite"):
        differentiation_matrix(nodes)


def test_differentiation_requires_at_least_two_distinct_nodes() -> None:
    with pytest.raises(ValueError, match="nodes.*at least two"):
        differentiation_matrix(np.array([0.0]))
    with pytest.raises(ValueError, match="nodes.*distinct"):
        differentiation_matrix(np.array([-1.0, 0.0, 0.0, 1.0]))


def test_differentiation_returns_an_independent_read_only_float64_matrix() -> None:
    nodes = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    derivative = differentiation_matrix(nodes)
    expected = np.array(
        [
            [-1.5, 2.0, -0.5],
            [-0.5, 0.0, 0.5],
            [0.5, -2.0, 1.5],
        ]
    )

    np.testing.assert_allclose(derivative, expected, rtol=0.0, atol=2.0e-15)
    assert derivative.dtype == np.float64
    assert not np.shares_memory(derivative, nodes)
    assert not derivative.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        derivative[0, 0] = 0.0


def test_map_to_thickness_scales_quadrature_and_differentiates_a_cubic() -> None:
    nodes, weights = gll_nodes_weights(6)
    derivative = differentiation_matrix(nodes)

    z, mapped_weights, mapped_derivative = map_to_thickness(
        nodes,
        weights,
        derivative,
        h=2.5,
    )

    assert z[0] == -2.5
    assert z[-1] == 2.5
    assert np.sum(mapped_weights) == pytest.approx(5.0, rel=0.0, abs=2.0e-14)
    np.testing.assert_allclose(
        mapped_derivative @ z**3,
        3.0 * z**2,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_map_to_thickness_applies_the_documented_scaling() -> None:
    nodes = np.array([-1.0, 0.0, 1.0])
    weights = np.array([0.25, 1.5, 0.25])
    derivative = differentiation_matrix(nodes)

    mapped_nodes, mapped_weights, mapped_derivative = map_to_thickness(
        nodes,
        weights,
        derivative,
        h=2,
    )

    np.testing.assert_array_equal(mapped_nodes, 2.0 * nodes)
    np.testing.assert_array_equal(mapped_weights, 2.0 * weights)
    np.testing.assert_array_equal(mapped_derivative, derivative / 2.0)


@pytest.mark.parametrize(
    ("nodes", "weights", "derivative", "error", "message"),
    [
        pytest.param(
            np.array([[-1.0, 0.0, 1.0]]),
            np.ones(3),
            np.eye(3),
            ValueError,
            "nodes.*one-dimensional",
            id="node-shape",
        ),
        pytest.param(
            np.array([-1, 0, 1]),
            np.ones(3),
            np.eye(3),
            TypeError,
            "nodes.*floating",
            id="node-integer",
        ),
        pytest.param(
            np.array([-1.0, np.nan, 1.0]),
            np.ones(3),
            np.eye(3),
            ValueError,
            "nodes.*finite",
            id="node-nonfinite",
        ),
        pytest.param(
            np.array([-1.0, 0.0, 0.0]),
            np.ones(3),
            np.eye(3),
            ValueError,
            "nodes.*distinct",
            id="node-duplicate",
        ),
        pytest.param(
            np.array([-1.0, 0.0, 1.0]),
            np.ones((1, 3)),
            np.eye(3),
            ValueError,
            "weights.*one-dimensional",
            id="weight-shape",
        ),
        pytest.param(
            np.array([-1.0, 0.0, 1.0]),
            np.ones(3, dtype=np.int64),
            np.eye(3),
            TypeError,
            "weights.*floating",
            id="weight-integer",
        ),
        pytest.param(
            np.array([-1.0, 0.0, 1.0]),
            np.array([1.0, np.inf, 1.0]),
            np.eye(3),
            ValueError,
            "weights.*finite",
            id="weight-nonfinite",
        ),
        pytest.param(
            np.array([-1.0, 0.0, 1.0]),
            np.ones(2),
            np.eye(3),
            ValueError,
            "weights.*same length",
            id="weight-size",
        ),
        pytest.param(
            np.array([-1.0, 0.0, 1.0]),
            np.ones(3),
            np.ones(3),
            ValueError,
            "derivative.*two-dimensional",
            id="derivative-shape-rank",
        ),
        pytest.param(
            np.array([-1.0, 0.0, 1.0]),
            np.ones(3),
            np.eye(3, dtype=np.int64),
            TypeError,
            "derivative.*floating",
            id="derivative-integer",
        ),
        pytest.param(
            np.array([-1.0, 0.0, 1.0]),
            np.ones(3),
            np.eye(3, dtype=np.complex128),
            TypeError,
            "derivative.*floating",
            id="derivative-complex",
        ),
        pytest.param(
            np.array([-1.0, 0.0, 1.0]),
            np.ones(3),
            np.array([[1.0, np.nan, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            ValueError,
            "derivative.*finite",
            id="derivative-nonfinite",
        ),
        pytest.param(
            np.array([-1.0, 0.0, 1.0]),
            np.ones(3),
            np.eye(2),
            ValueError,
            "derivative.*shape",
            id="derivative-size",
        ),
    ],
)
def test_map_to_thickness_validates_array_inputs(
    nodes: object,
    weights: object,
    derivative: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        map_to_thickness(nodes, weights, derivative, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "h",
    [True, np.bool_(False), "1.0", 1.0 + 0.0j, None],
    ids=["bool", "numpy-bool", "string", "complex", "none"],
)
def test_map_to_thickness_requires_a_real_scalar_h(h: object) -> None:
    nodes, weights = gll_nodes_weights(2)
    with pytest.raises(TypeError, match="h.*real"):
        map_to_thickness(nodes, weights, differentiation_matrix(nodes), h)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "h",
    [0.0, -1.0, np.nan, np.inf, 10**1000],
    ids=["zero", "negative", "nan", "infinity", "out-of-range-integer"],
)
def test_map_to_thickness_requires_a_finite_positive_h(h: object) -> None:
    nodes, weights = gll_nodes_weights(2)
    with pytest.raises(ValueError, match="h.*finite and positive"):
        map_to_thickness(nodes, weights, differentiation_matrix(nodes), h)  # type: ignore[arg-type]


def test_map_to_thickness_returns_independent_read_only_float64_arrays() -> None:
    nodes = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    weights = np.array([0.25, 1.5, 0.25], dtype=np.float32)
    derivative = np.array(differentiation_matrix(nodes), copy=True)

    mapped = map_to_thickness(nodes, weights, derivative, 1.5)
    snapshots = tuple(np.array(array, copy=True) for array in mapped)
    nodes[:] = 7.0
    weights[:] = 7.0
    derivative[:] = 7.0

    for result, snapshot in zip(mapped, snapshots, strict=True):
        np.testing.assert_array_equal(result, snapshot)
        assert result.dtype == np.float64
        assert np.isfinite(result).all()
        assert not result.flags.writeable
        assert not np.shares_memory(result, nodes)
        assert not np.shares_memory(result, weights)
        assert not np.shares_memory(result, derivative)


def test_two_element_order_five_mesh_shares_one_interface_node() -> None:
    mesh = build_gll_mesh(5, (-1, 0, 1))

    assert isinstance(mesh, GLLMesh)
    assert len(mesh.nodes) == 11
    assert len(mesh.elements) == 2
    np.testing.assert_array_equal(mesh.elements[0].connectivity, np.arange(0, 6))
    np.testing.assert_array_equal(mesh.elements[1].connectivity, np.arange(5, 11))
    assert mesh.elements[0].connectivity[-1] == mesh.elements[1].connectivity[0]
    assert mesh.elements[0].nodes[-1] == 0.0
    assert mesh.elements[1].nodes[0] == 0.0
    assert sum(np.sum(element.weights) for element in mesh.elements) == pytest.approx(
        2.0,
        rel=0.0,
        abs=2.0e-14,
    )


def test_single_element_mesh_uses_all_local_nodes_and_the_physical_span() -> None:
    mesh = build_gll_mesh(3, [-2, 3])
    element = mesh.elements[0]

    assert len(mesh.nodes) == 4
    np.testing.assert_array_equal(element.connectivity, np.arange(4))
    np.testing.assert_array_equal(mesh.nodes, element.nodes)
    assert element.nodes[0] == -2.0
    assert element.nodes[-1] == 3.0
    assert np.sum(element.weights) == pytest.approx(5.0, rel=0.0, abs=2.0e-14)


def test_three_nonuniform_elements_have_contiguous_shared_connectivity() -> None:
    bounds = [-2.0, -1.75, 0.5, 4.0]
    order = 4
    mesh = build_gll_mesh(order, bounds)

    assert len(mesh.nodes) == 3 * order + 1
    assert np.all(np.diff(mesh.nodes) > 0.0)
    for index, element in enumerate(mesh.elements):
        expected_connectivity = np.arange(index * order, index * order + order + 1)
        np.testing.assert_array_equal(element.connectivity, expected_connectivity)
        np.testing.assert_array_equal(mesh.nodes[element.connectivity], element.nodes)
        assert element.nodes[0] == bounds[index]
        assert element.nodes[-1] == bounds[index + 1]
        np.testing.assert_allclose(
            element.derivative @ element.nodes**3,
            3.0 * element.nodes**2,
            rtol=0.0,
            atol=3.0e-11,
        )
    assert sum(np.sum(element.weights) for element in mesh.elements) == pytest.approx(
        bounds[-1] - bounds[0],
        rel=0.0,
        abs=3.0e-14,
    )


@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param(np.array([-1.0, 1.0]), id="numpy-array"),
        pytest.param(iter([-1.0, 1.0]), id="iterator"),
        pytest.param("-1,1", id="string"),
        pytest.param(None, id="none"),
    ],
)
def test_mesh_requires_bounds_as_a_tuple_or_list(bounds: object) -> None:
    with pytest.raises(TypeError, match="element_bounds.*tuple or list"):
        build_gll_mesh(3, bounds)  # type: ignore[arg-type]


@pytest.mark.parametrize("bounds", [[], [0.0], ()])
def test_mesh_requires_at_least_two_bounds(bounds: list[float] | tuple[()]) -> None:
    with pytest.raises(ValueError, match="element_bounds.*at least two"):
        build_gll_mesh(3, bounds)


@pytest.mark.parametrize(
    "invalid_bound",
    [True, False, "0.0", 0.0 + 0.0j, None],
    ids=["true", "false", "string", "complex", "none"],
)
def test_mesh_rejects_nonreal_bound_values(invalid_bound: object) -> None:
    with pytest.raises(TypeError, match="element_bounds.*real"):
        build_gll_mesh(3, [-1.0, invalid_bound, 1.0])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "invalid_bound",
    [np.nan, np.inf, -np.inf, 10**1000],
    ids=["nan", "positive-infinity", "negative-infinity", "out-of-range-integer"],
)
def test_mesh_rejects_nonfinite_or_out_of_range_bounds(invalid_bound: object) -> None:
    with pytest.raises(ValueError, match="element_bounds.*finite"):
        build_gll_mesh(3, [-1.0, invalid_bound, 1.0])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param([-1.0, 0.0, 0.0, 1.0], id="duplicate"),
        pytest.param([-1.0, 0.5, 0.25, 1.0], id="decreasing"),
        pytest.param([2**53, 2**53 + 1], id="collapses-when-converted"),
    ],
)
def test_mesh_requires_bounds_strictly_increasing_after_float64_conversion(
    bounds: list[float],
) -> None:
    with pytest.raises(ValueError, match="element_bounds.*strictly increasing"):
        build_gll_mesh(3, bounds)


@pytest.mark.parametrize("order", [True, 2.0, np.int64(2)])
def test_mesh_uses_the_same_strict_order_contract(order: object) -> None:
    with pytest.raises(TypeError, match="order.*integer"):
        build_gll_mesh(order, [-1.0, 1.0])  # type: ignore[arg-type]


def test_mesh_records_are_frozen_and_all_returned_buffers_are_read_only() -> None:
    mesh = build_gll_mesh(3, [-1.0, 0.25, 2.0])
    element = mesh.elements[0]

    assert isinstance(element, GLLElement)
    assert isinstance(mesh.elements, tuple)
    with pytest.raises(FrozenInstanceError):
        mesh.nodes = np.array([0.0])  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        element.connectivity = np.array([0], dtype=np.int64)  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        mesh.nodes[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        element.connectivity[0] = 7

    arrays = [mesh.nodes]
    for local_element in mesh.elements:
        arrays.extend(
            [
                local_element.nodes,
                local_element.weights,
                local_element.derivative,
                local_element.connectivity,
            ]
        )
    for array in arrays:
        assert not array.flags.writeable
        assert np.isfinite(array).all()
    assert mesh.nodes.dtype == np.float64
    for local_element in mesh.elements:
        assert local_element.nodes.dtype == np.float64
        assert local_element.weights.dtype == np.float64
        assert local_element.derivative.dtype == np.float64
        assert local_element.connectivity.dtype == np.int64


def test_mesh_buffers_do_not_alias_global_or_neighbor_buffers() -> None:
    mesh = build_gll_mesh(3, [-1.0, 0.0, 2.0])
    first, second = mesh.elements

    assert not np.shares_memory(mesh.nodes, first.nodes)
    assert not np.shares_memory(mesh.nodes, second.nodes)
    assert not np.shares_memory(first.nodes, second.nodes)
    assert not np.shares_memory(first.weights, second.weights)
    assert not np.shares_memory(first.derivative, second.derivative)
    assert not np.shares_memory(first.connectivity, second.connectivity)
