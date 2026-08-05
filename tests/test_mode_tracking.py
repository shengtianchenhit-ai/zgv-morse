from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from zgv_morse.elasticity import isotropic_tensor
from zgv_morse.gll import GLLMesh
from zgv_morse.mode_tracking import (
    ModeTrackingError,
    TrackedBranch,
    TrackedMode,
    mass_mac,
    mass_subspace_singular_values,
    phase_align,
    relative_eigengap,
    seed_tracked_mode,
    symmetric_lamb_parity_score,
    track_branch,
    track_mode,
)
from zgv_morse.spectral_plate import (
    ModeSet,
    PlateMatrices,
    assemble_plate_matrices,
    solve_plate_modes,
)


def test_mode_tracking_error_is_a_runtime_error() -> None:
    assert issubclass(ModeTrackingError, RuntimeError)


def test_mass_mac_and_phase_alignment_are_phase_and_scale_invariant() -> None:
    mass = np.diag([1.0, 2.0, 3.0])
    reference = np.array([1.0, 2.0j, -0.5])
    candidate = -3.4 * np.exp(0.73j) * reference
    untouched = candidate.copy()

    assert mass_mac(reference, candidate, mass) == pytest.approx(1.0, abs=2.0e-15)
    aligned = phase_align(reference, candidate, mass)

    overlap = reference.conj() @ mass @ aligned
    assert overlap.imag == pytest.approx(0.0, abs=2.0e-14)
    assert overlap.real > 0.0
    np.testing.assert_array_equal(candidate, untouched)
    assert not np.shares_memory(aligned, candidate)
    assert aligned.conj() @ mass @ aligned == pytest.approx(
        candidate.conj() @ mass @ candidate,
        rel=2.0e-15,
    )


def test_phase_align_returns_an_independent_copy_for_zero_overlap() -> None:
    reference = np.array([1.0, 0.0])
    candidate = np.array([0.0, 2.0j])

    aligned = phase_align(reference, candidate, np.eye(2))

    np.testing.assert_array_equal(aligned, candidate)
    assert not np.shares_memory(aligned, candidate)


def test_principal_overlaps_detect_identical_rotated_subspaces() -> None:
    angle = 0.41
    previous = np.eye(3)[:, :2]
    current = np.array(
        [
            [2.0 * np.cos(angle), -0.3 * np.sin(angle)],
            [2.0 * np.sin(angle), 0.3 * np.cos(angle)],
            [0.0, 0.0],
        ]
    )

    singular_values = mass_subspace_singular_values(previous, current, np.eye(3))

    np.testing.assert_allclose(singular_values, 1.0, rtol=0.0, atol=2.0e-14)
    assert np.all((0.0 <= singular_values) & (singular_values <= 1.0))


@pytest.mark.parametrize(
    ("reference", "candidate", "mass", "message"),
    [
        (np.ones((2, 1)), np.ones(2), np.eye(2), "reference"),
        (np.ones(2), np.ones(3), np.eye(2), "same length"),
        (np.array([1.0, np.nan]), np.ones(2), np.eye(2), "reference"),
        (np.ones(2), np.ones(2), np.ones(2), "mass"),
        (np.ones(2), np.ones(2), np.eye(3), "mass"),
        (np.ones(2), np.ones(2), np.array([[1.0, 1.0j], [1.0j, 1.0]]), "Hermitian"),
        (np.ones(2), np.ones(2), np.diag([1.0, 0.0]), "positive definite"),
        (np.zeros(2), np.ones(2), np.eye(2), "norm"),
    ],
)
def test_mass_mac_rejects_invalid_vectors_and_mass(
    reference: object,
    candidate: object,
    mass: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mass_mac(reference, candidate, mass)


@pytest.mark.parametrize(
    ("previous", "current", "mass", "message"),
    [
        (np.ones(2), np.eye(2), np.eye(2), "previous_basis"),
        (np.eye(2), np.ones((3, 1)), np.eye(2), "row"),
        (np.ones((2, 2)), np.eye(2), np.eye(2), "linearly independent"),
        (np.eye(2), np.eye(2), np.array([[1.0, 2.0], [0.0, 1.0]]), "Hermitian"),
        (np.eye(2), np.eye(2), np.diag([1.0, -1.0]), "positive definite"),
    ],
)
def test_subspace_overlap_rejects_invalid_bases_and_mass(
    previous: object,
    current: object,
    mass: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mass_subspace_singular_values(previous, current, mass)


def _mode_set(
    eigenvalues: object,
    vectors: object,
    *,
    mass: object | None = None,
    residuals: object | None = None,
) -> ModeSet:
    eigenvalue_array = np.asarray(eigenvalues, dtype=np.float64)
    vector_array = np.asarray(vectors, dtype=np.complex128)
    dimension = vector_array.shape[0]
    mass_array = (
        np.eye(dimension, dtype=np.complex128)
        if mass is None
        else np.asarray(mass, dtype=np.complex128)
    )
    residual_array = (
        np.zeros(eigenvalue_array.size)
        if residuals is None
        else np.asarray(residuals, dtype=np.float64)
    )
    nodes = np.arange(dimension, dtype=np.float64)
    matrices = PlateMatrices(
        stiffness=np.zeros_like(mass_array),
        mass=mass_array,
        nodes=nodes,
        mesh=GLLMesh(nodes=nodes, elements=()),
        hermitian_defect=0.0,
    )
    return ModeSet(
        eigenvalues=eigenvalue_array,
        omega=np.sqrt(eigenvalue_array),
        vectors=vector_array,
        residuals=residual_array,
        mass_orthogonality_defect=0.0,
        matrices=matrices,
    )


def test_tracking_records_are_frozen_slotted_and_report_minimum_gap() -> None:
    first = TrackedMode(1.0, 1.0, np.array([1.0]), 0, 1.0, 0.4, 0.0, (0,), 1.0)
    second = TrackedMode(2.0, 4.0, np.array([1.0]), 0, 0.8, 0.1, 1.0e-12, (0,), 0.9)
    branch = TrackedBranch(np.zeros((2, 2)), (first, second))

    assert TrackedMode.__dataclass_params__.frozen
    assert TrackedBranch.__dataclass_params__.frozen
    assert TrackedMode.__slots__
    assert TrackedBranch.__slots__
    assert branch.minimum_eigengap == 0.1
    with pytest.raises(FrozenInstanceError):
        first.index = 2  # type: ignore[misc]


def test_tracked_branch_defensively_copies_and_validates_nonempty_cardinality() -> None:
    mode = TrackedMode(1.0, 1.0, [1.0], 0, 1.0, np.inf, 0.0, (0,), 1.0)
    points = np.array([[0.0, 0.0]])

    branch = TrackedBranch(points, (mode,))
    points[:] = 9.0

    np.testing.assert_array_equal(branch.wavevectors, [[0.0, 0.0]])
    assert not branch.wavevectors.flags.writeable
    with pytest.raises(ValueError, match="nonempty"):
        TrackedBranch(np.empty((0, 2)), ())
    with pytest.raises(ValueError, match="count"):
        TrackedBranch(np.zeros((2, 2)), (mode,))


def test_relative_eigengap_uses_symmetric_unit_floored_scaling() -> None:
    assert relative_eigengap(np.array([0.0, 2.0, 5.0]), 1) == pytest.approx(3.0 / 5.0)
    assert relative_eigengap(np.array([7.0]), 0) == np.inf


def test_seed_mode_copies_diagnostics_and_returns_read_only_vector() -> None:
    source_vectors = np.eye(2, dtype=np.complex128)
    modes = _mode_set([1.0, 4.0], source_vectors, residuals=[1.0e-9, 2.0e-9])

    tracked = seed_tracked_mode(modes, 1)
    source_vectors[:, 1] = 7.0

    assert tracked.omega == 2.0
    assert tracked.eigenvalue == 4.0
    assert tracked.index == 1
    assert tracked.mac == tracked.subspace_overlap == 1.0
    assert tracked.eigengap == pytest.approx(3.0 / 4.0)
    assert tracked.residual == 2.0e-9
    assert tracked.cluster_indices == (1,)
    np.testing.assert_array_equal(tracked.vector, [0.0, 1.0])
    assert not tracked.vector.flags.writeable


def test_tracking_follows_shape_when_eigenvalue_order_changes() -> None:
    modes = _mode_set(
        [2.1, 1.1],
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    )
    previous = TrackedMode(
        1.0,
        1.0,
        np.array([1.0, 0.0], dtype=np.complex128),
        0,
        1.0,
        0.5,
        0.0,
        (0,),
        1.0,
    )

    tracked = track_mode(previous, modes)

    assert tracked.index == 1
    assert tracked.eigenvalue == 1.1
    assert tracked.mac == pytest.approx(1.0)
    assert tracked.cluster_indices == (1,)
    assert tracked.subspace_overlap == pytest.approx(1.0)
    assert not tracked.vector.flags.writeable


def test_cluster_rotation_succeeds_when_every_scalar_mac_is_too_low() -> None:
    inverse_sqrt_three = 1.0 / np.sqrt(3.0)
    rotated_basis = np.array(
        [
            [inverse_sqrt_three, inverse_sqrt_three, inverse_sqrt_three],
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0],
            [1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
        ]
    )
    previous = TrackedMode(
        1.0,
        1.0,
        np.array([1.0, 0.0, 0.0]),
        0,
        1.0,
        0.0,
        0.0,
        (0, 1, 2),
        1.0,
    )
    modes = _mode_set([0.9999, 1.0, 1.0001], rotated_basis)

    tracked = track_mode(previous, modes, min_mac=0.5, predicted_eigenvalue=1.0)

    assert tracked.index == 1
    assert tracked.mac == pytest.approx(1.0 / 3.0)
    assert tracked.cluster_indices == (0, 1, 2)
    assert tracked.subspace_overlap == pytest.approx(1.0, abs=2.0e-15)


def test_degenerate_cluster_state_makes_split_selection_basis_invariant() -> None:
    seed = seed_tracked_mode(_mode_set([0.8], [[1.0], [0.0]]), 0)
    split = _mode_set([0.9, 1.1], np.eye(2))

    outgoing_indices = []
    for angle in (0.17, 1.19):
        rotated = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        at_degeneracy = track_mode(seed, _mode_set([1.0, 1.0], rotated))
        outgoing = track_mode(at_degeneracy, split, predicted_eigenvalue=1.1)
        outgoing_indices.append(outgoing.index)
        assert at_degeneracy.cluster_basis is not None
        assert at_degeneracy.cluster_basis.shape == (2, 2)

    assert outgoing_indices == [1, 1]


def test_cluster_membership_is_transitive_under_relative_gap_connections() -> None:
    previous = seed_tracked_mode(_mode_set([1.0], [[1.0], [0.0], [0.0]]), 0)
    modes = _mode_set([1.0, 1.0009, 1.0018], np.eye(3))

    tracked = track_mode(previous, modes, cluster_rel_gap=1.0e-3)

    assert tracked.cluster_indices == (0, 1, 2)
    assert tracked.cluster_basis is not None
    assert tracked.cluster_basis.shape == (3, 3)


def test_tracked_mode_defensively_copies_vector_and_optional_cluster_basis() -> None:
    vector = np.array([1.0, 0.0])
    basis = np.eye(2)

    tracked = TrackedMode(1.0, 1.0, vector, 0, 1.0, 0.0, 0.0, (0, 1), 1.0, basis)
    vector[:] = 9.0
    basis[:] = 9.0
    default_basis = TrackedMode(1.0, 1.0, [1.0, 0.0], 0, 1.0, 1.0, 0.0, (0,), 1.0)

    np.testing.assert_array_equal(tracked.vector, [1.0, 0.0])
    np.testing.assert_array_equal(tracked.cluster_basis, np.eye(2))
    np.testing.assert_array_equal(default_basis.cluster_basis, [[1.0], [0.0]])
    assert not tracked.vector.flags.writeable
    assert tracked.cluster_basis is not None
    assert not tracked.cluster_basis.flags.writeable


def test_tracking_rejects_an_unrelated_mode_and_subspace() -> None:
    previous = TrackedMode(
        1.0,
        1.0,
        np.array([1.0, 0.0, 0.0]),
        0,
        1.0,
        0.5,
        0.0,
        (0,),
        1.0,
    )
    modes = _mode_set([1.2, 2.0], np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

    with pytest.raises(ModeTrackingError, match="scalar MAC and cluster overlap"):
        track_mode(previous, modes)


def test_subspace_threshold_uses_squared_overlap_like_scalar_mac() -> None:
    previous = seed_tracked_mode(_mode_set([1.0], [[1.0], [0.0]]), 0)
    modes = _mode_set([1.1], [[0.6], [0.8]])

    with pytest.raises(ModeTrackingError, match="scalar MAC and cluster overlap"):
        track_mode(previous, modes, min_mac=0.5)


def test_branch_predictor_continues_through_a_crossing_and_order_swap() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    def solver(kx: float, _ky: float) -> ModeSet:
        if kx == 0.0:
            return _mode_set([1.0, 1.4], np.eye(2))
        if kx == 1.0:
            return _mode_set(
                [1.2, 1.2],
                np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0),
            )
        return _mode_set([1.0, 1.4], np.array([[0.0, 1.0], [1.0, 0.0]]))

    branch = track_branch(points, solver, seed_index=0)
    points[:] = 99.0

    np.testing.assert_array_equal(branch.wavevectors[:, 0], [0.0, 1.0, 2.0])
    assert [mode.index for mode in branch.modes] == [0, 0, 1]
    np.testing.assert_allclose([mode.eigenvalue for mode in branch.modes], [1.0, 1.2, 1.4])
    assert not branch.wavevectors.flags.writeable
    assert all(not mode.vector.flags.writeable for mode in branch.modes)


@pytest.mark.parametrize(
    "wavevectors",
    [
        pytest.param([], id="empty"),
        pytest.param([0.0, 1.0], id="one-dimensional"),
        pytest.param([[0.0, 1.0, 2.0]], id="three-columns"),
        pytest.param([[0.0, np.nan]], id="nonfinite"),
        pytest.param([[0.0 + 0.0j, 1.0]], id="complex"),
    ],
)
def test_track_branch_rejects_invalid_wavevectors(wavevectors: object) -> None:
    with pytest.raises(ValueError, match="wavevectors"):
        track_branch(wavevectors, lambda _kx, _ky: _mode_set([1.0], [[1.0]]), 0)


def test_track_branch_rejects_a_noncallable_or_invalid_solver_output() -> None:
    with pytest.raises(TypeError, match="solver"):
        track_branch([[0.0, 0.0]], object(), 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="solver.*ModeSet"):
        track_branch([[0.0, 0.0]], lambda _kx, _ky: object(), 0)  # type: ignore[arg-type]


def test_track_branch_rejects_callback_node_or_mesh_permutation() -> None:
    first = _mode_set([1.0, 2.0], np.eye(2))
    second = _mode_set([1.1, 2.1], np.eye(2))
    permuted_nodes = second.matrices.nodes[::-1].copy()
    second = replace(
        second,
        matrices=replace(
            second.matrices,
            nodes=permuted_nodes,
            mesh=GLLMesh(nodes=permuted_nodes, elements=()),
        ),
    )

    with pytest.raises(ValueError, match="nodes|mesh"):
        track_branch([[0.0, 0.0], [1.0, 0.0]], lambda kx, _ky: first if kx == 0 else second, 0)


def test_track_branch_rejects_callback_mass_changes() -> None:
    first = _mode_set([1.0, 2.0], np.eye(2))
    second = _mode_set([1.1, 2.1], np.eye(2), mass=np.diag([1.0, 1.01]))

    with pytest.raises(ValueError, match="mass"):
        track_branch([[0.0, 0.0], [1.0, 0.0]], lambda kx, _ky: first if kx == 0 else second, 0)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("cluster_rel_gap", -1.0e-3),
        ("cluster_rel_gap", np.inf),
        ("min_mac", -0.1),
        ("min_mac", 1.1),
        ("predicted_eigenvalue", np.nan),
    ],
)
def test_track_mode_rejects_invalid_thresholds_and_prediction(keyword: str, value: float) -> None:
    previous = seed_tracked_mode(_mode_set([1.0], [[1.0]]), 0)
    with pytest.raises(ValueError, match=keyword):
        track_mode(previous, _mode_set([1.0], [[1.0]]), **{keyword: value})


@pytest.mark.parametrize("index", [-1, 2, True, 0.0])
def test_seed_and_relative_gap_reject_invalid_indices(index: object) -> None:
    modes = _mode_set([1.0, 2.0], np.eye(2))
    error = TypeError if type(index) is not int else ValueError
    with pytest.raises(error, match="index"):
        seed_tracked_mode(modes, index)  # type: ignore[arg-type]
    with pytest.raises(error, match="index"):
        relative_eigengap([1.0, 2.0], index)  # type: ignore[arg-type]


def test_symmetric_and_antisymmetric_synthetic_lamb_parity() -> None:
    nodes = np.array([-1.0, 0.0, 1.0])
    symmetric = np.array(
        [1.0, 2.0, -3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0],
        dtype=np.complex128,
    )
    antisymmetric = np.array(
        [1.0, 2.0, 3.0, 0.0, 0.0, 6.0, -1.0, -2.0, 3.0],
        dtype=np.complex128,
    )

    assert symmetric_lamb_parity_score(symmetric, nodes, np.eye(9)) == pytest.approx(1.0)
    assert symmetric_lamb_parity_score(antisymmetric, nodes, np.eye(9)) == pytest.approx(-1.0)


def test_isotropic_spectral_mode_near_reference_zgv_is_symmetric_and_mass_normalized() -> None:
    matrices = assemble_plate_matrices(
        0.8042173193715181,
        0.0,
        isotropic_tensor(lam=2.0, mu=1.0),
        rho=1.0,
        half_thickness=1.0,
        order=12,
    )
    modes = solve_plate_modes(matrices, num_modes=10)
    index = int(np.argmin(np.abs(modes.omega - 2.8517587749600901)))
    vector = modes.vectors[:, index]

    assert modes.omega[index] == pytest.approx(2.8517587749600901, rel=2.0e-6)
    assert np.real(vector.conj() @ matrices.mass @ vector) == pytest.approx(1.0, abs=2.0e-14)
    assert symmetric_lamb_parity_score(vector, matrices.nodes, matrices.mass) > 1.0 - 2.0e-12


@pytest.mark.parametrize(
    ("vector", "nodes", "mass", "message"),
    [
        (np.ones(6), np.array([-1.0, 2.0]), np.eye(6), "mirror-symmetric"),
        (np.ones(5), np.array([-1.0, 1.0]), np.eye(6), "3N"),
        (np.ones(6), np.array([-1.0, 1.0]), np.eye(5), "mass"),
        (np.ones(6), np.array([-1.0, 1.0]), np.diag([1.0, 1.0, 1.0, 2.0, 1.0, 1.0]), "parity"),
    ],
)
def test_parity_rejects_incompatible_nodes_vectors_and_mass(
    vector: object,
    nodes: object,
    mass: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        symmetric_lamb_parity_score(vector, nodes, mass)
