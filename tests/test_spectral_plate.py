from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from itertools import pairwise

import numpy as np
import pytest

from zgv_morse.elasticity import cubic_tensor, isotropic_tensor, rotate_tensor
from zgv_morse.spectral_plate import (
    ModeSet,
    PlateMatrices,
    WavevectorDerivatives,
    _global_dofs,
    _interpolation,
    _mandel_b_parts,
    assemble_plate_matrices,
    assemble_wavevector_derivatives,
    solve_plate_modes,
)


def test_interpolation_and_global_dofs_use_node_major_triples() -> None:
    shape = np.array([0.25, 0.75])

    interpolation = _interpolation(shape)

    expected = np.zeros((3, 6))
    expected[:, :3] = 0.25 * np.eye(3)
    expected[:, 3:] = 0.75 * np.eye(3)
    np.testing.assert_array_equal(interpolation, expected)
    assert interpolation.dtype == np.complex128
    np.testing.assert_array_equal(
        _global_dofs(np.array([4, 1], dtype=np.int64)),
        np.array([12, 13, 14, 3, 4, 5]),
    )


def test_mandel_b_parts_reconstruct_the_fourier_strain() -> None:
    shape = np.array([0.3, 0.7])
    derivative = np.array([-1.2, 1.2])
    displacement = np.array([0.4, -0.2, 0.9, -0.1, 0.8, 0.3], dtype=np.complex128)
    kx = 0.7
    ky = -0.2

    b0, bx, by = _mandel_b_parts(shape, derivative)
    strain = (b0 + kx * bx + ky * by) @ displacement

    ux = np.dot(shape, displacement[0::3])
    uy = np.dot(shape, displacement[1::3])
    uz = np.dot(shape, displacement[2::3])
    dux = np.dot(derivative, displacement[0::3])
    duy = np.dot(derivative, displacement[1::3])
    duz = np.dot(derivative, displacement[2::3])
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    expected = np.array(
        [
            1j * kx * ux,
            1j * ky * uy,
            duz,
            (duy + 1j * ky * uz) * inverse_sqrt_two,
            (dux + 1j * kx * uz) * inverse_sqrt_two,
            (1j * ky * ux + 1j * kx * uy) * inverse_sqrt_two,
        ]
    )
    np.testing.assert_allclose(strain, expected, rtol=0.0, atol=2.0e-16)


@pytest.mark.parametrize("record", [PlateMatrices, WavevectorDerivatives, ModeSet])
def test_public_records_are_frozen(record: type[object]) -> None:
    assert record.__dataclass_params__.frozen
    assert record.__slots__

    fields = record.__dataclass_fields__
    assert fields
    with pytest.raises(FrozenInstanceError):
        instance = object.__new__(record)
        setattr(instance, next(iter(fields)), None)


def test_plate_assembly_is_hermitian_with_positive_mass() -> None:
    matrices = assemble_plate_matrices(
        0.73,
        0.21,
        isotropic_tensor(lam=2.0, mu=1.0),
        rho=1.0,
        half_thickness=1.0,
        order=12,
    )

    assert matrices.stiffness.shape == matrices.mass.shape == (39, 39)
    assert matrices.nodes.shape == (13,)
    assert matrices.hermitian_defect < 1.0e-14
    np.testing.assert_allclose(
        matrices.stiffness,
        matrices.stiffness.conj().T,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert np.linalg.eigvalsh(matrices.mass)[0] > 0.0


def test_hermitian_defect_is_the_direct_relative_frobenius_norm() -> None:
    matrices = assemble_plate_matrices(
        0.73,
        0.21,
        isotropic_tensor(lam=2.0, mu=1.0),
        rho=1.0,
        half_thickness=1.0,
        order=12,
    )
    expected = np.linalg.norm(
        matrices.stiffness - matrices.stiffness.conj().T,
        ord="fro",
    ) / max(
        np.linalg.norm(matrices.stiffness, ord="fro"),
        np.finfo(np.float64).tiny,
    )

    assert matrices.hermitian_defect == expected


def test_mass_is_node_major_diagonal_and_integrates_each_translation() -> None:
    rho = 2.4
    half_thickness = 1.3
    matrices = assemble_plate_matrices(
        0.0,
        0.0,
        isotropic_tensor(lam=2.0, mu=1.0),
        rho=rho,
        half_thickness=half_thickness,
        order=6,
    )

    np.testing.assert_allclose(
        matrices.mass,
        np.diag(np.diag(matrices.mass)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.diag(matrices.mass).reshape(-1, 3),
        np.repeat(np.diag(matrices.mass).reshape(-1, 3)[:, :1], 3, axis=1),
        rtol=0.0,
        atol=0.0,
    )
    for component in range(3):
        translation = np.zeros(3 * matrices.nodes.size)
        translation[component::3] = 1.0
        integrated_mass = translation @ matrices.mass @ translation
        assert integrated_mass.real == pytest.approx(2.0 * half_thickness * rho, abs=2e-14)
        assert integrated_mass.imag == 0.0


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def test_all_wavevector_derivatives_match_centered_differences_and_are_hermitian() -> None:
    kx = 0.71
    ky = 0.23
    first_step = 2.0e-6
    second_step = 2.0e-4
    tensor = isotropic_tensor(lam=2.0, mu=1.0)
    common = dict(C=tensor, rho=1.0, half_thickness=1.0, order=9)

    derivatives = assemble_wavevector_derivatives(kx, ky, **common)

    def stiffness(x: float, y: float) -> np.ndarray:
        return assemble_plate_matrices(x, y, **common).stiffness

    center = stiffness(kx, ky)
    first_x = (stiffness(kx + first_step, ky) - stiffness(kx - first_step, ky)) / (2.0 * first_step)
    first_y = (stiffness(kx, ky + first_step) - stiffness(kx, ky - first_step)) / (2.0 * first_step)
    second_x = (
        stiffness(kx + second_step, ky) - 2.0 * center + stiffness(kx - second_step, ky)
    ) / second_step**2
    second_y = (
        stiffness(kx, ky + second_step) - 2.0 * center + stiffness(kx, ky - second_step)
    ) / second_step**2
    mixed = (
        stiffness(kx + second_step, ky + second_step)
        - stiffness(kx + second_step, ky - second_step)
        - stiffness(kx - second_step, ky + second_step)
        + stiffness(kx - second_step, ky - second_step)
    ) / (4.0 * second_step**2)

    assert _relative_error(derivatives.dkx, first_x) < 2.0e-9
    assert _relative_error(derivatives.dky, first_y) < 2.0e-9
    assert _relative_error(derivatives.dkx2, second_x) < 2.0e-7
    assert _relative_error(derivatives.dkx_dky, mixed) < 2.0e-7
    assert _relative_error(derivatives.dky2, second_y) < 2.0e-7
    blocks = (
        derivatives.dkx,
        derivatives.dky,
        derivatives.dkx2,
        derivatives.dkx_dky,
        derivatives.dky2,
    )
    for block in blocks:
        np.testing.assert_allclose(block, block.conj().T, rtol=0.0, atol=2.0e-14)
        assert np.isfinite(block).all()
    for first, second in pairwise(blocks):
        assert not np.shares_memory(first, second)


def test_uniform_sh0_mode_has_the_bulk_shear_speed_and_solver_diagnostics_are_small() -> None:
    matrices = assemble_plate_matrices(
        0.7,
        0.0,
        isotropic_tensor(lam=2.0, mu=1.0),
        rho=1.0,
        half_thickness=1.0,
        order=10,
    )

    modes = solve_plate_modes(matrices, num_modes=12)

    assert np.min(np.abs(modes.omega - 0.7)) < 2.0e-11
    assert np.max(modes.residuals) < 1.0e-10
    assert modes.mass_orthogonality_defect < 2.0e-12
    expected_orthogonality_defect = np.linalg.norm(
        modes.vectors.conj().T @ matrices.mass @ modes.vectors - np.eye(12),
        ord="fro",
    )
    assert modes.mass_orthogonality_defect == expected_orthogonality_defect
    np.testing.assert_allclose(
        modes.vectors.conj().T @ matrices.mass @ modes.vectors,
        np.eye(12),
        rtol=0.0,
        atol=2.0e-12,
    )


@pytest.mark.parametrize(
    ("wavenumber", "order", "num_modes", "target_frequency"),
    [
        pytest.param(0.7, 10, 12, None, id="low-flexural-mode"),
        pytest.param(
            0.8042173193715181,
            18,
            18,
            2.85175877496009,
            id="reference-zgv-mode",
        ),
    ],
)
def test_each_reported_residual_uses_the_per_mode_force_balance(
    wavenumber: float,
    order: int,
    num_modes: int,
    target_frequency: float | None,
) -> None:
    matrices = assemble_plate_matrices(
        wavenumber,
        0.0,
        isotropic_tensor(lam=2.0, mu=1.0),
        rho=1.0,
        half_thickness=1.0,
        order=order,
    )
    modes = solve_plate_modes(matrices, num_modes)
    tiny = np.finfo(np.float64).tiny
    expected = np.empty(num_modes)
    for index, (eigenvalue, vector) in enumerate(
        zip(modes.eigenvalues, modes.vectors.T, strict=True)
    ):
        left = matrices.stiffness @ vector
        right = eigenvalue * (matrices.mass @ vector)
        expected[index] = np.linalg.norm(left - right) / max(
            np.linalg.norm(left) + np.linalg.norm(right),
            tiny,
        )

    if target_frequency is None:
        assert modes.omega[0] < 0.5
    else:
        assert np.min(np.abs(modes.omega - target_frequency)) < 2.0e-10
    np.testing.assert_allclose(modes.residuals, expected, rtol=2.0e-14, atol=0.0)


def test_zero_wavevector_retains_three_unconstrained_rigid_translations() -> None:
    matrices = assemble_plate_matrices(
        0.0,
        0.0,
        isotropic_tensor(lam=2.0, mu=1.0),
        rho=1.0,
        half_thickness=1.0,
        order=10,
    )

    modes = solve_plate_modes(matrices, num_modes=8)

    assert np.count_nonzero(modes.omega < 2.0e-6) == 3
    assert modes.omega[3] > 0.5


@pytest.mark.parametrize(
    ("updates", "error", "message"),
    [
        pytest.param({"kx": True}, TypeError, "kx", id="boolean-kx"),
        pytest.param({"ky": np.inf}, ValueError, "ky", id="nonfinite-ky"),
        pytest.param({"rho": 0.0}, ValueError, "rho", id="zero-density"),
        pytest.param(
            {"half_thickness": np.nan},
            ValueError,
            "half_thickness",
            id="nonfinite-thickness",
        ),
        pytest.param(
            {"C": np.zeros((3, 3, 3))},
            ValueError,
            "shape",
            id="tensor-shape",
        ),
        pytest.param(
            {"C": isotropic_tensor(2.0, 1.0).astype(np.complex128)},
            ValueError,
            "real",
            id="complex-tensor",
        ),
        pytest.param({"element_bounds": []}, ValueError, "at least two", id="empty-bounds"),
        pytest.param(
            {"element_bounds": [-0.9, 1.0]},
            ValueError,
            "span",
            id="incomplete-span",
        ),
    ],
)
def test_plate_assembly_rejects_invalid_problem_data(
    updates: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "kx": 0.4,
        "ky": 0.2,
        "C": isotropic_tensor(2.0, 1.0),
        "rho": 1.0,
        "half_thickness": 1.0,
        "order": 5,
    }
    arguments.update(updates)

    with pytest.raises(error, match=message):
        assemble_plate_matrices(**arguments)  # type: ignore[arg-type]


def test_plate_assembly_rejects_tensor_without_elastic_symmetries() -> None:
    tensor = isotropic_tensor(2.0, 1.0)
    tensor[0, 0, 0, 1] += 0.1

    with pytest.raises(ValueError, match="symmetr"):
        assemble_plate_matrices(0.4, 0.2, tensor, 1.0, 1.0, order=5)


@pytest.mark.parametrize(
    "tensor",
    [
        pytest.param(np.zeros((3, 3, 3, 3)), id="zero"),
        pytest.param(-isotropic_tensor(2.0, 1.0), id="negative-definite"),
    ],
)
def test_plate_assembly_rejects_nonpositive_constitutive_tensors(
    tensor: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="positive-definite"):
        assemble_plate_matrices(0.4, 0.2, tensor, 1.0, 1.0, order=5)


def test_wavevector_derivatives_validate_density_even_though_it_is_unused() -> None:
    with pytest.raises(ValueError, match="rho"):
        assemble_wavevector_derivatives(
            0.4,
            0.2,
            isotropic_tensor(2.0, 1.0),
            rho=0.0,
            half_thickness=1.0,
            order=5,
        )


def test_solver_rejects_invalid_mode_counts() -> None:
    matrices = assemble_plate_matrices(0.4, 0.2, isotropic_tensor(2.0, 1.0), 1.0, 1.0, order=4)

    for invalid in (True, np.int64(2), 2.0):
        with pytest.raises(TypeError, match="num_modes.*integer"):
            solve_plate_modes(matrices, invalid)  # type: ignore[arg-type]
    for invalid in (0, matrices.stiffness.shape[0] + 1):
        with pytest.raises(ValueError, match="num_modes"):
            solve_plate_modes(matrices, invalid)


@pytest.mark.parametrize(
    ("field", "mutation", "message"),
    [
        pytest.param(
            "stiffness",
            lambda value: value[:-1, :-1],
            "shape",
            id="wrong-stiffness-shape",
        ),
        pytest.param(
            "stiffness",
            lambda value: np.full_like(value, np.nan),
            "finite",
            id="nonfinite-stiffness",
        ),
        pytest.param(
            "stiffness",
            lambda value: value + np.triu(np.ones_like(value), 1),
            "Hermitian",
            id="nonhermitian-stiffness",
        ),
        pytest.param(
            "mass",
            lambda value: np.zeros_like(value),
            "positive definite",
            id="nonpositive-mass",
        ),
        pytest.param(
            "mass",
            lambda value: value + 1j * np.eye(value.shape[0]),
            "Hermitian",
            id="nonhermitian-mass",
        ),
    ],
)
def test_solver_rejects_malformed_matrix_records(
    field: str,
    mutation: object,
    message: str,
) -> None:
    matrices = assemble_plate_matrices(0.4, 0.2, isotropic_tensor(2.0, 1.0), 1.0, 1.0, order=4)
    changed = replace(matrices, **{field: mutation(getattr(matrices, field))})  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        solve_plate_modes(changed, 4)


def test_solver_rejects_materially_negative_eigenvalues() -> None:
    matrices = assemble_plate_matrices(0.4, 0.2, isotropic_tensor(2.0, 1.0), 1.0, 1.0, order=4)
    indefinite = replace(matrices, stiffness=-matrices.mass)

    with pytest.raises(RuntimeError, match="negative"):
        solve_plate_modes(indefinite, 4)


def _in_plane_rotation(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_isotropic_spectrum_is_rotationally_invariant() -> None:
    wavenumber = 0.8
    angle = 0.37
    tensor = isotropic_tensor(lam=2.0, mu=1.0)
    axial = solve_plate_modes(
        assemble_plate_matrices(wavenumber, 0.0, tensor, 1.0, 1.0, order=14),
        14,
    )
    oblique = solve_plate_modes(
        assemble_plate_matrices(
            wavenumber * np.cos(angle),
            wavenumber * np.sin(angle),
            tensor,
            1.0,
            1.0,
            order=14,
        ),
        14,
    )

    np.testing.assert_allclose(oblique.omega, axial.omega, rtol=0.0, atol=2.0e-10)


def test_isotropic_stiffness_transforms_covariantly_with_direction() -> None:
    wavenumber = 0.76
    angle = 0.41
    rotation = _in_plane_rotation(angle)
    tensor = isotropic_tensor(lam=2.0, mu=1.0)
    axial = assemble_plate_matrices(wavenumber, 0.0, tensor, 1.0, 1.0, order=8)
    oblique = assemble_plate_matrices(
        wavenumber * np.cos(angle),
        wavenumber * np.sin(angle),
        tensor,
        1.0,
        1.0,
        order=8,
    )
    dof_rotation = np.kron(np.eye(axial.nodes.size), rotation)

    np.testing.assert_allclose(
        oblique.stiffness,
        dof_rotation @ axial.stiffness @ dof_rotation.T,
        rtol=0.0,
        atol=3.0e-14,
    )


def test_frequencies_scale_as_inverse_square_root_density() -> None:
    density_ratio = 3.7
    tensor = isotropic_tensor(lam=2.0, mu=1.0)
    reference = solve_plate_modes(
        assemble_plate_matrices(0.61, 0.19, tensor, 1.0, 1.0, order=9),
        10,
    )
    heavier = solve_plate_modes(
        assemble_plate_matrices(0.61, 0.19, tensor, density_ratio, 1.0, order=9),
        10,
    )

    np.testing.assert_allclose(
        heavier.omega * np.sqrt(density_ratio),
        reference.omega,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_two_element_low_modes_converge_to_single_element_spectral_result() -> None:
    common = dict(
        kx=0.65,
        ky=0.17,
        C=isotropic_tensor(lam=2.0, mu=1.0),
        rho=1.0,
        half_thickness=1.0,
    )
    reference = solve_plate_modes(
        assemble_plate_matrices(**common, order=18),
        6,
    ).omega
    coarse = solve_plate_modes(
        assemble_plate_matrices(**common, order=4, element_bounds=[-1.0, 0.0, 1.0]),
        6,
    ).omega
    fine = solve_plate_modes(
        assemble_plate_matrices(**common, order=8, element_bounds=[-1.0, 0.0, 1.0]),
        6,
    ).omega

    coarse_error = np.linalg.norm(coarse - reference)
    fine_error = np.linalg.norm(fine - reference)
    assert fine_error < coarse_error
    assert np.max(np.abs(fine - reference)) < 2.0e-10


def test_rotated_cubic_tensor_still_assembles_a_hermitian_stiffness() -> None:
    tensor = rotate_tensor(
        cubic_tensor(c11=5.0, c12=2.0, c44=1.25),
        _in_plane_rotation(0.29),
    )

    matrices = assemble_plate_matrices(0.63, -0.24, tensor, 1.2, 0.9, order=9)

    assert matrices.hermitian_defect < 1.0e-14
    np.testing.assert_allclose(
        matrices.stiffness,
        matrices.stiffness.conj().T,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_result_arrays_are_independent_read_only_snapshots() -> None:
    matrices = assemble_plate_matrices(0.4, 0.2, isotropic_tensor(2.0, 1.0), 1.0, 1.0, order=5)
    derivatives = assemble_wavevector_derivatives(
        0.4, 0.2, isotropic_tensor(2.0, 1.0), 1.0, 1.0, order=5
    )
    modes = solve_plate_modes(matrices, 5)
    arrays = (
        matrices.stiffness,
        matrices.mass,
        matrices.nodes,
        derivatives.dkx,
        derivatives.dky,
        derivatives.dkx2,
        derivatives.dkx_dky,
        derivatives.dky2,
        modes.eigenvalues,
        modes.omega,
        modes.vectors,
        modes.residuals,
    )

    assert all(not array.flags.writeable for array in arrays)
    assert not np.shares_memory(matrices.nodes, matrices.mesh.nodes)
    for index, first in enumerate(arrays):
        for second in arrays[index + 1 :]:
            assert not np.shares_memory(first, second)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        pytest.param({"nodes": np.array([0.0])}, "nodes shape", id="nodes"),
        pytest.param({"mesh": object()}, "mesh", id="mesh"),
        pytest.param({"hermitian_defect": np.nan}, "hermitian_defect", id="defect"),
    ],
)
def test_solver_rejects_malformed_numerical_metadata(
    updates: dict[str, object],
    message: str,
) -> None:
    matrices = assemble_plate_matrices(0.4, 0.2, isotropic_tensor(2.0, 1.0), 1.0, 1.0, order=4)

    with pytest.raises(ValueError, match=message):
        solve_plate_modes(replace(matrices, **updates), 4)
