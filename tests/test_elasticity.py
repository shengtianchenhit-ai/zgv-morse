from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.typing import NDArray

from zgv_morse.elasticity import (
    CubicConstants,
    Tensor4,
    assert_cubic_stability,
    cubic_family,
    cubic_perturbation_tensor,
    cubic_tensor,
    isotropic_tensor,
    rotate_tensor,
    tensor_to_mandel,
)


def _in_plane_rotation(angle: float) -> NDArray[np.float64]:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _strain_energy(tensor: Tensor4, strain: NDArray[np.float64]) -> float:
    return float(0.5 * np.einsum("ij,ijkl,kl->", strain, tensor, strain))


def test_isotropic_energy_is_invariant_under_rotation() -> None:
    tensor = isotropic_tensor(lam=2.0, mu=1.0)
    strain = np.array(
        [
            [0.25, -0.15, 0.07],
            [-0.15, -0.10, 0.05],
            [0.07, 0.05, 0.30],
        ]
    )
    rotation = _in_plane_rotation(0.37)

    rotated_tensor = rotate_tensor(tensor, rotation)
    rotated_strain = rotation @ strain @ rotation.T

    assert _strain_energy(rotated_tensor, rotated_strain) == pytest.approx(
        _strain_energy(tensor, strain), abs=1e-13
    )


def test_cubic_family_preserves_bulk_combination_and_sets_anisotropy() -> None:
    epsilon = 0.04

    tensor, constants = cubic_family(lam=2.0, mu=1.0, delta=1.0, epsilon=epsilon)

    assert constants.c11 + 2.0 * constants.c12 == pytest.approx(8.0)
    assert constants.c11 - constants.c12 - 2.0 * constants.c44 == pytest.approx(epsilon)
    np.testing.assert_allclose(
        tensor,
        cubic_tensor(constants.c11, constants.c12, constants.c44),
    )
    assert_cubic_stability(constants)


def test_cubic_tensor_has_correct_shear_values_and_all_symmetries() -> None:
    tensor = cubic_tensor(c11=5.0, c12=2.0, c44=1.25)

    assert tensor[0, 0, 0, 0] == 5.0
    assert tensor[0, 0, 1, 1] == 2.0
    assert tensor[0, 1, 0, 1] == 1.25
    assert tensor[1, 0, 0, 1] == 1.25
    assert tensor[0, 1, 1, 0] == 1.25
    assert tensor[1, 0, 1, 0] == 1.25
    np.testing.assert_array_equal(tensor, tensor.swapaxes(0, 1))
    np.testing.assert_array_equal(tensor, tensor.swapaxes(2, 3))
    np.testing.assert_array_equal(tensor, tensor.transpose(2, 3, 0, 1))


def test_isotropic_tensor_has_orthonormal_mandel_representation() -> None:
    mandel = tensor_to_mandel(isotropic_tensor(lam=2.0, mu=1.0))

    np.testing.assert_allclose(np.diag(mandel)[:3], 4.0)
    np.testing.assert_allclose(np.diag(mandel)[3:], 2.0)
    np.testing.assert_allclose(mandel[np.triu_indices(3, k=1)], 2.0)
    np.testing.assert_allclose(mandel[:3, 3:], 0.0, atol=0.0)
    np.testing.assert_allclose(mandel[3:, :3], 0.0, atol=0.0)


def test_cubic_perturbation_matches_centered_finite_difference() -> None:
    lam = 2.0
    mu = 1.0
    delta = 1.7
    step = 1.0e-5
    tensor_plus, _ = cubic_family(lam, mu, delta, step)
    tensor_minus, _ = cubic_family(lam, mu, delta, -step)

    finite_difference = (tensor_plus - tensor_minus) / (2.0 * step)

    np.testing.assert_allclose(
        cubic_perturbation_tensor(delta),
        finite_difference,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_quarter_turn_about_001_leaves_cubic_tensor_unchanged() -> None:
    tensor = cubic_tensor(c11=5.0, c12=2.0, c44=1.25)

    rotated = rotate_tensor(tensor, _in_plane_rotation(np.pi / 2.0))

    np.testing.assert_allclose(rotated, tensor, rtol=0.0, atol=1.0e-14)


def test_rotation_uses_forward_matrix_on_every_tensor_index() -> None:
    tensor = cubic_tensor(c11=5.0, c12=2.0, c44=1.25)
    rotation = _in_plane_rotation(0.31)
    expected = np.zeros_like(tensor)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for ell in range(3):
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                for d in range(3):
                                    expected[i, j, k, ell] += (
                                        rotation[i, a]
                                        * rotation[j, b]
                                        * rotation[k, c]
                                        * rotation[ell, d]
                                        * tensor[a, b, c, d]
                                    )

    np.testing.assert_allclose(rotate_tensor(tensor, rotation), expected, atol=1.0e-14)


@pytest.mark.parametrize(
    "rotation",
    [
        pytest.param(np.eye(2), id="wrong-shape"),
        pytest.param(np.diag([1.0, 1.0, 2.0]), id="non-orthogonal"),
        pytest.param(np.diag([1.0, 1.0, np.nan]), id="nonfinite"),
    ],
)
def test_rotate_tensor_rejects_invalid_rotations(rotation: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="rotation"):
        rotate_tensor(isotropic_tensor(2.0, 1.0), rotation)


@pytest.mark.parametrize(
    ("constants", "invariant"),
    [
        pytest.param(CubicConstants(3.0, 1.0, 0.0), "C44 > 0", id="shear"),
        pytest.param(CubicConstants(1.0, 1.0, 1.0), "C11 - C12 > 0", id="tetragonal"),
        pytest.param(CubicConstants(1.0, -1.0, 1.0), "C11 + 2*C12 > 0", id="bulk"),
    ],
)
def test_cubic_stability_reports_failed_invariant(
    constants: CubicConstants, invariant: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(invariant)):
        assert_cubic_stability(constants)


def test_cubic_tensor_rejects_unstable_constants() -> None:
    with pytest.raises(ValueError, match="C44 > 0"):
        cubic_tensor(c11=3.0, c12=1.0, c44=0.0)


@pytest.mark.parametrize(
    ("lam", "mu", "message"),
    [
        pytest.param(np.nan, 1.0, "lam", id="nonfinite-lambda"),
        pytest.param(2.0, np.inf, "mu", id="nonfinite-mu"),
        pytest.param(2.0, 0.0, "mu", id="zero-shear"),
        pytest.param(-2.0 / 3.0, 1.0, "bulk", id="zero-bulk"),
        pytest.param(-1.0e308, 1.0e308, "bulk", id="overflow-safe-negative-bulk"),
    ],
)
def test_isotropic_tensor_rejects_invalid_moduli(lam: float, mu: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        isotropic_tensor(lam, mu)


def test_negative_lambda_with_positive_bulk_is_allowed() -> None:
    tensor = isotropic_tensor(lam=-0.5, mu=1.0)

    assert np.isfinite(tensor).all()


@pytest.mark.parametrize(
    ("delta", "epsilon", "message"),
    [
        pytest.param(0.0, 0.01, "delta", id="zero-delta"),
        pytest.param(np.inf, 0.01, "delta", id="nonfinite-delta"),
        pytest.param(1.0, np.nan, "epsilon", id="nonfinite-epsilon"),
    ],
)
def test_cubic_family_rejects_invalid_scale_or_perturbation(
    delta: float, epsilon: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cubic_family(lam=2.0, mu=1.0, delta=delta, epsilon=epsilon)


def test_cubic_family_allows_negative_nonzero_delta() -> None:
    tensor, constants = cubic_family(lam=2.0, mu=1.0, delta=-1.0, epsilon=0.04)

    assert np.isfinite(tensor).all()
    assert constants.c11 - constants.c12 - 2.0 * constants.c44 == pytest.approx(-0.04)


@pytest.mark.parametrize("delta", [0.0, np.inf, np.nan])
def test_cubic_perturbation_rejects_invalid_delta(delta: float) -> None:
    with pytest.raises(ValueError, match="delta"):
        cubic_perturbation_tensor(delta)


def test_cubic_constants_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="c12"):
        CubicConstants(c11=4.0, c12=np.nan, c44=1.0)


def test_cubic_constants_are_immutable() -> None:
    constants = CubicConstants(c11=4.0, c12=2.0, c44=1.0)

    with pytest.raises(FrozenInstanceError):
        constants.c11 = 5.0  # type: ignore[misc]


def test_mandel_conversion_uses_xx_yy_zz_yz_xz_xy_order() -> None:
    tensor = np.zeros((3, 3, 3, 3))
    for axis, value in enumerate((11.0, 22.0, 33.0)):
        tensor[axis, axis, axis, axis] = value
    for (first_axis, second_axis), value in zip(
        ((1, 2), (0, 2), (0, 1)),
        (4.0, 5.0, 6.0),
        strict=True,
    ):
        tensor[first_axis, second_axis, first_axis, second_axis] = value
        tensor[first_axis, second_axis, second_axis, first_axis] = value
        tensor[second_axis, first_axis, first_axis, second_axis] = value
        tensor[second_axis, first_axis, second_axis, first_axis] = value

    np.testing.assert_allclose(
        np.diag(tensor_to_mandel(tensor)),
        (11.0, 22.0, 33.0, 8.0, 10.0, 12.0),
    )


@pytest.mark.parametrize("function", [rotate_tensor, tensor_to_mandel])
def test_tensor_operations_reject_nonfinite_tensors(function: object) -> None:
    tensor = isotropic_tensor(2.0, 1.0)
    tensor[0, 0, 0, 0] = np.nan

    if function is rotate_tensor:
        with pytest.raises(ValueError, match="tensor"):
            rotate_tensor(tensor, np.eye(3))
    else:
        with pytest.raises(ValueError, match="tensor"):
            tensor_to_mandel(tensor)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda tensor: rotate_tensor(tensor, np.eye(3)), id="rotation"),
        pytest.param(tensor_to_mandel, id="mandel"),
    ],
)
@pytest.mark.parametrize("imaginary_part", [0.0, 1.0], ids=["zero", "nonzero"])
def test_tensor_operations_reject_complex_tensors_without_truncation(
    operation: Callable[[object], object], imaginary_part: float
) -> None:
    tensor = isotropic_tensor(2.0, 1.0).astype(np.complex128)
    tensor[0, 0, 0, 0] += imaginary_part * 1j

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="tensor.*real"):
            operation(tensor)


@pytest.mark.parametrize("imaginary_part", [0.0, 1.0], ids=["zero", "nonzero"])
def test_rotate_tensor_rejects_complex_rotations_without_truncation(
    imaginary_part: float,
) -> None:
    rotation = np.eye(3, dtype=np.complex128)
    rotation[0, 0] += imaginary_part * 1j

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="rotation.*real"):
            rotate_tensor(isotropic_tensor(2.0, 1.0), rotation)


@pytest.mark.parametrize(
    ("operation", "field"),
    [
        pytest.param(lambda value: CubicConstants(value, 0.0, 1.0), "c11", id="constants"),
        pytest.param(lambda value: isotropic_tensor(value, 1.0), "lam", id="isotropic"),
        pytest.param(lambda value: cubic_tensor(value, 0.0, 1.0), "c11", id="cubic"),
        pytest.param(lambda value: cubic_family(2.0, 1.0, value, 0.1), "delta", id="family"),
        pytest.param(cubic_perturbation_tensor, "delta", id="perturbation"),
    ],
)
def test_huge_integer_scalars_raise_value_error(
    operation: Callable[[object], object], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        operation(10**1000)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda tensor: rotate_tensor(tensor, np.eye(3)), id="rotation"),
        pytest.param(tensor_to_mandel, id="mandel"),
    ],
)
def test_huge_integer_tensors_raise_value_error(
    operation: Callable[[object], object],
) -> None:
    tensor = np.full((3, 3, 3, 3), 10**1000, dtype=object)

    with pytest.raises(ValueError, match="tensor"):
        operation(tensor)


def test_huge_integer_rotation_raises_value_error() -> None:
    rotation = np.full((3, 3), 10**1000, dtype=object)

    with pytest.raises(ValueError, match="rotation"):
        rotate_tensor(isotropic_tensor(2.0, 1.0), rotation)


def test_cubic_perturbation_avoids_representable_intermediate_overflow() -> None:
    delta = 1.0e308

    tensor = cubic_perturbation_tensor(delta)

    assert np.isfinite(tensor).all()
    assert tensor[0, 0, 0, 0] == pytest.approx((2.0 / 3.0) * delta)
    assert tensor[0, 0, 1, 1] == pytest.approx(-(1.0 / 3.0) * delta)


def test_cubic_family_avoids_representable_intermediate_overflow() -> None:
    lam = 1.0e307
    mu = 1.0e307
    delta = 0.8
    epsilon = 1.0e308

    tensor, constants = cubic_family(lam, mu, delta, epsilon)

    expected_c11 = lam + 2.0 * mu + (2.0 / 3.0) * epsilon * delta
    expected_c12 = lam - (1.0 / 3.0) * epsilon * delta
    assert constants.c11 == pytest.approx(expected_c11)
    assert constants.c12 == pytest.approx(expected_c12)
    assert constants.c44 == mu
    assert np.isfinite(tensor).all()
    assert_cubic_stability(constants)


def test_cubic_family_zero_epsilon_avoids_baseline_accumulation_overflow() -> None:
    lam = -5.0e307
    mu = 1.0e308

    tensor, constants = cubic_family(lam, mu, delta=1.0, epsilon=0.0)

    assert constants.c11 == pytest.approx(1.5e308)
    assert constants.c12 == lam
    assert constants.c44 == mu
    assert np.isfinite((constants.c11, constants.c12, constants.c44)).all()
    assert np.isfinite(tensor).all()
    np.testing.assert_allclose(tensor, isotropic_tensor(lam, mu), rtol=1.0e-15)


def test_isotropic_tensor_rejects_nonfinite_output_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="finite"):
            isotropic_tensor(1.0e308, 1.0e308)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda tensor: rotate_tensor(tensor, _in_plane_rotation(np.pi / 4.0)),
            id="rotation",
        ),
        pytest.param(tensor_to_mandel, id="mandel"),
    ],
)
def test_tensor_operations_reject_overflowing_results_without_warning(
    operation: Callable[[object], object],
) -> None:
    tensor = np.full((3, 3, 3, 3), np.finfo(np.float64).max)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="finite"):
            operation(tensor)
