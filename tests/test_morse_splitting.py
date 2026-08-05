from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from zgv_morse.critical_points import (
    Annulus,
    CriticalPoint,
    ExhaustionReport,
    _classify,
    cartesian_hessian,
    locate_critical_points,
    verify_annular_exhaustion,
)
from zgv_morse.dispersion import (
    FrequencyGradient,
    RingAnchoredSpectralEvaluator,
    hellmann_feynman_gradient,
)
from zgv_morse.elasticity import cubic_family, isotropic_tensor


def test_frequency_gradient_is_frozen_slotted_and_defensively_copies_gradient() -> None:
    source = np.array([0.3, -0.2])
    sample = FrequencyGradient(2.0, source, 1.0e-8, 2.0e-7)
    source[:] = 9.0

    assert FrequencyGradient.__dataclass_params__.frozen
    assert FrequencyGradient.__slots__
    assert_allclose(sample.gradient, [0.3, -0.2])
    assert not sample.gradient.flags.writeable
    with pytest.raises(FrozenInstanceError):
        sample.omega = 3.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        sample.gradient[0] = 0.0


@pytest.mark.parametrize(
    ("omega", "gradient", "frequency_uncertainty", "gradient_uncertainty", "message"),
    [
        (0.0, [0.0, 0.0], 0.0, 0.0, "omega"),
        (np.inf, [0.0, 0.0], 0.0, 0.0, "omega"),
        (1.0, [0.0], 0.0, 0.0, "gradient"),
        (1.0, [0.0, np.nan], 0.0, 0.0, "gradient"),
        (1.0, [0.0, 0.0], -1.0, 0.0, "frequency_uncertainty"),
        (1.0, [0.0, 0.0], 0.0, np.inf, "gradient_uncertainty"),
    ],
)
def test_frequency_gradient_rejects_invalid_values(
    omega: object,
    gradient: object,
    frequency_uncertainty: object,
    gradient_uncertainty: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        FrequencyGradient(
            omega,  # type: ignore[arg-type]
            gradient,  # type: ignore[arg-type]
            frequency_uncertainty,  # type: ignore[arg-type]
            gradient_uncertainty,  # type: ignore[arg-type]
        )


def test_hellmann_feynman_gradient_matches_diagonal_problem() -> None:
    gradient = hellmann_feynman_gradient(
        2.0,
        np.array([1.0, 0.0], dtype=np.complex128),
        np.eye(2),
        np.diag([1.2, -0.4]),
        np.diag([-0.8, 0.7]),
    )

    assert_allclose(gradient, [0.3, -0.2], rtol=0.0, atol=0.0)


def test_hellmann_feynman_gradient_uses_generalized_normalization() -> None:
    vector = 3.0j * np.array([1.0, 2.0])
    mass = np.diag([2.0, 0.5])
    dkx = np.diag([4.0, 1.0])
    dky = np.array([[2.0, 1.0j], [-1.0j, 3.0]])

    gradient = hellmann_feynman_gradient(5.0, vector, mass, dkx, dky)
    expected = np.array(
        [
            np.real(np.vdot(vector, dkx @ vector)),
            np.real(np.vdot(vector, dky @ vector)),
        ]
    ) / (2.0 * 5.0 * np.real(np.vdot(vector, mass @ vector)))

    assert_allclose(gradient, expected, rtol=2.0e-15, atol=0.0)
    assert np.isfinite(gradient).all()


@pytest.mark.parametrize(
    ("omega", "vector", "mass", "dkx", "dky", "message"),
    [
        (0.0, [1.0, 0.0], np.eye(2), np.eye(2), np.eye(2), "omega"),
        (1.0, [1.0], np.eye(2), np.eye(2), np.eye(2), "length"),
        (1.0, [1.0, 0.0], [[1.0, 1.0], [0.0, 1.0]], np.eye(2), np.eye(2), "Hermitian"),
        (
            1.0,
            [1.0, 0.0],
            np.diag([1.0, -1.0]),
            np.eye(2),
            np.eye(2),
            "positive definite",
        ),
        (1.0, [1.0, 0.0], np.eye(2), [[1.0, 1.0j], [1.0j, 1.0]], np.eye(2), "Hermitian"),
        (1.0, [1.0, 0.0], np.eye(2), np.eye(2), [[np.nan, 0.0], [0.0, 1.0]], "finite"),
    ],
)
def test_hellmann_feynman_gradient_rejects_invalid_eigenproblems(
    omega: object,
    vector: object,
    mass: object,
    dkx: object,
    dky: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        hellmann_feynman_gradient(omega, vector, mass, dkx, dky)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def isotropic_evaluator() -> RingAnchoredSpectralEvaluator:
    return RingAnchoredSpectralEvaluator(
        isotropic_tensor(2.0, 1.0),
        rho=1.0,
        half_thickness=1.0,
        k0=0.8042173193715181,
        target_omega=2.8517587749600901,
        order=6,
        num_modes=12,
        angular_sectors=8,
    )


@pytest.fixture(scope="module")
def weak_cubic_evaluator() -> RingAnchoredSpectralEvaluator:
    tensor = cubic_family(2.0, 1.0, 1.0, 0.02)[0]
    return RingAnchoredSpectralEvaluator(
        tensor,
        rho=1.0,
        half_thickness=1.0,
        k0=0.8042173193715181,
        target_omega=2.8517587749600901,
        order=6,
        num_modes=12,
        angular_sectors=8,
    )


@pytest.mark.parametrize(
    ("fixture_name", "point"),
    [
        ("isotropic_evaluator", np.array([0.77, 0.16])),
        ("isotropic_evaluator", np.array([0.58, -0.54])),
        ("weak_cubic_evaluator", np.array([0.74, 0.24])),
    ],
)
def test_ring_anchored_gradient_matches_tracked_frequency_differences(
    fixture_name: str,
    point: np.ndarray,
    request: pytest.FixtureRequest,
) -> None:
    evaluator = request.getfixturevalue(fixture_name)
    sample = evaluator(point)
    step = 2.0e-5
    finite = np.empty(2)
    for axis in range(2):
        shift = np.zeros(2)
        shift[axis] = step
        finite[axis] = (evaluator(point + shift).omega - evaluator(point - shift).omega) / (
            2.0 * step
        )

    assert_allclose(sample.gradient, finite, rtol=3.0e-6, atol=2.0e-8)
    assert sample.frequency_uncertainty >= 0.0
    assert sample.gradient_uncertainty >= 0.0


def test_isotropic_evaluator_is_rotationally_covariant(
    isotropic_evaluator: RingAnchoredSpectralEvaluator,
) -> None:
    point = np.array([0.73, 0.21])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])

    original = isotropic_evaluator(point)
    rotated = isotropic_evaluator(rotation @ point)

    assert rotated.omega == pytest.approx(original.omega, abs=2.0e-12)
    assert_allclose(rotated.gradient, rotation @ original.gradient, rtol=0.0, atol=2.0e-11)


def test_weak_cubic_evaluator_has_fourfold_covariance(
    weak_cubic_evaluator: RingAnchoredSpectralEvaluator,
) -> None:
    point = np.array([0.71, 0.29])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])

    original = weak_cubic_evaluator(point)
    rotated = weak_cubic_evaluator(rotation @ point)

    assert rotated.omega == pytest.approx(original.omega, abs=2.0e-12)
    assert_allclose(rotated.gradient, rotation @ original.gradient, rtol=0.0, atol=3.0e-11)


def test_ring_anchored_evaluator_reports_nested_order_uncertainty(
    isotropic_evaluator: RingAnchoredSpectralEvaluator,
) -> None:
    sample = isotropic_evaluator(np.array([0.76, 0.18]))

    assert 0.0 <= sample.frequency_uncertainty < 1.0e-3
    assert 0.0 <= sample.gradient_uncertainty < 1.0e-2


def test_ring_anchored_evaluator_rejects_forced_unresolved_gap(
    isotropic_evaluator: RingAnchoredSpectralEvaluator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "zgv_morse.dispersion._relative_eigenvalue_uncertainty",
        lambda *_args: np.inf,
    )

    with pytest.raises(RuntimeError, match="eigengap"):
        isotropic_evaluator(np.array([0.76, 0.18]))


def test_ring_anchors_reject_failed_mass_mac_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "zgv_morse.dispersion.mass_mac",
        lambda *_args: 0.0,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="closure.*MAC"):
        RingAnchoredSpectralEvaluator(
            isotropic_tensor(2.0, 1.0),
            rho=1.0,
            half_thickness=1.0,
            k0=0.8042173193715181,
            target_omega=2.8517587749600901,
            order=6,
            num_modes=12,
            angular_sectors=8,
        )


def test_ring_anchors_reject_forced_coarse_fine_branch_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "zgv_morse.dispersion._relative_eigenvalue_discrepancy",
        lambda *_args: np.inf,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="coarse/fine.*branch"):
        RingAnchoredSpectralEvaluator(
            isotropic_tensor(2.0, 1.0),
            rho=1.0,
            half_thickness=1.0,
            k0=0.8042173193715181,
            target_omega=2.8517587749600901,
            order=6,
            num_modes=12,
            angular_sectors=8,
        )


def test_ring_anchors_reject_a_seed_at_the_computed_spectrum_edge() -> None:
    with pytest.raises(RuntimeError, match="top of the computed spectrum"):
        RingAnchoredSpectralEvaluator(
            isotropic_tensor(2.0, 1.0),
            rho=1.0,
            half_thickness=1.0,
            k0=0.8042173193715181,
            target_omega=100.0,
            order=6,
            num_modes=12,
            angular_sectors=8,
        )


def test_ring_query_rejects_a_forced_top_edge_cluster(
    isotropic_evaluator: RingAnchoredSpectralEvaluator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zgv_morse.dispersion as dispersion_module

    original_track_mode = dispersion_module.track_mode

    def force_top_cluster(previous, modes, **kwargs):
        tracked = original_track_mode(previous, modes, **kwargs)
        last = len(modes.eigenvalues) - 1
        return replace(
            tracked,
            index=last,
            cluster_indices=(last,),
            cluster_basis=tracked.vector[:, np.newaxis],
        )

    monkeypatch.setattr(dispersion_module, "track_mode", force_top_cluster)

    with pytest.raises(RuntimeError, match="top of the computed spectrum"):
        isotropic_evaluator(np.array([0.76, 0.18]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rho": 0.0}, "rho"),
        ({"half_thickness": -1.0}, "half_thickness"),
        ({"k0": 0.0}, "k0"),
        ({"target_omega": np.nan}, "target_omega"),
        ({"order": 1}, "order"),
        ({"num_modes": 0}, "num_modes"),
        ({"num_modes": 1}, "num_modes"),
        ({"angular_sectors": 3}, "angular_sectors"),
    ],
)
def test_ring_anchored_evaluator_rejects_invalid_constructor_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    valid: dict[str, object] = {
        "rho": 1.0,
        "half_thickness": 1.0,
        "k0": 0.8,
        "target_omega": 2.85,
        "order": 6,
        "num_modes": 12,
        "angular_sectors": 8,
    }
    valid.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=message):
        RingAnchoredSpectralEvaluator(isotropic_tensor(2.0, 1.0), **valid)  # type: ignore[arg-type]


class NormalFormEvaluator:
    """Exact fourfold normal form used independently of the full-wave solver."""

    def __init__(self, epsilon: float, *, gradient_uncertainty: float = 1.0e-10) -> None:
        self.epsilon = epsilon
        self.k0 = 2.0
        self.a = 3.0
        self.V4 = 0.5
        self.B0 = 0.3
        self.gradient_uncertainty = gradient_uncertainty

    def __call__(self, kxy: np.ndarray) -> FrequencyGradient:
        x, y = kxy
        radius = float(np.hypot(x, y))
        theta = float(np.arctan2(y, x))
        q = radius - self.k0
        omega = 5.0 + 0.5 * self.a * q**2 + self.epsilon * (
            0.1 + self.V4 * np.cos(4.0 * theta) + self.B0 * q
        )
        radial = np.array([np.cos(theta), np.sin(theta)])
        tangent = np.array([-np.sin(theta), np.cos(theta)])
        gradient = (
            (self.a * q + self.epsilon * self.B0) * radial
            - 4.0
            * self.epsilon
            * self.V4
            * np.sin(4.0 * theta)
            * tangent
            / radius
        )
        return FrequencyGradient(omega, gradient, 1.0e-12, self.gradient_uncertainty)


def _normal_form_points(epsilon: float, n_radial: int = 17, n_theta: int = 128):
    return locate_critical_points(
        NormalFormEvaluator(epsilon),
        Annulus(k0=2.0, half_width=0.10),
        n_radial,
        n_theta,
        1.0e-4,
    )


def test_critical_point_records_are_frozen_slotted_and_arrays_are_read_only() -> None:
    annulus = Annulus(2.0, 0.1)
    hessian = np.diag([2.0, -1.0])
    eigenvalues = np.array([-1.0, 2.0])
    point = CriticalPoint(
        2.0,
        0.0,
        2.0,
        0.0,
        5.0,
        1.0e-12,
        1.0e-10,
        hessian,
        eigenvalues,
        1.0e-8,
        "saddle",
        -1,
    )
    report = ExhaustionReport(True, True, 0.2, 1.0e-10)
    hessian[:] = 7.0
    eigenvalues[:] = 7.0

    for record in (Annulus, CriticalPoint, ExhaustionReport):
        assert record.__dataclass_params__.frozen
        assert record.__slots__
    assert annulus.k0 == 2.0
    assert_allclose(point.hessian, np.diag([2.0, -1.0]))
    assert_allclose(point.hessian_eigenvalues, [-1.0, 2.0])
    assert not point.hessian.flags.writeable
    assert not point.hessian_eigenvalues.flags.writeable
    assert report.boundary_is_noncritical and report.index_closes
    with pytest.raises(FrozenInstanceError):
        point.kind = "minimum"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("k0", "half_width", "message"),
    [
        (0.0, 0.1, "k0"),
        (2.0, 0.0, "half_width"),
        (2.0, 2.0, "inner radius"),
        (np.inf, 0.1, "k0"),
    ],
)
def test_annulus_rejects_invalid_geometry(
    k0: object,
    half_width: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Annulus(k0, half_width)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("eigenvalues", "uncertainty", "expected"),
    [
        ([1.0, 2.0], 1.0e-3, ("minimum", 1)),
        ([-2.0, -1.0], 1.0e-3, ("maximum", 1)),
        ([-2.0, 1.0], 1.0e-3, ("saddle", -1)),
    ],
)
def test_classify_assigns_planar_gradient_indices(
    eigenvalues: object,
    uncertainty: float,
    expected: tuple[str, int],
) -> None:
    assert _classify(np.asarray(eigenvalues), uncertainty) == expected


@pytest.mark.parametrize(
    ("eigenvalues", "uncertainty"),
    [
        ([1.0, 0.0], 0.0),
        ([1.0, 1.0e-9], 2.0e-10),
    ],
)
def test_classify_rejects_unresolved_or_exact_degeneracy(
    eigenvalues: object,
    uncertainty: float,
) -> None:
    with pytest.raises(ValueError, match="degeneracy"):
        _classify(np.asarray(eigenvalues), uncertainty)


class QuadraticEvaluator:
    def __init__(self, hessian: np.ndarray, uncertainty: float) -> None:
        self.hessian = hessian
        self.uncertainty = uncertainty

    def __call__(self, point: np.ndarray) -> FrequencyGradient:
        gradient = self.hessian @ np.asarray(point)
        omega = 4.0 + 0.5 * float(np.asarray(point) @ gradient)
        return FrequencyGradient(omega, gradient, 0.0, self.uncertainty)


def test_cartesian_hessian_richardson_result_and_propagated_uncertainty() -> None:
    exact = np.array([[2.0, 0.4], [0.4, 3.0]])
    evaluator = QuadraticEvaluator(exact, uncertainty=2.0e-8)

    hessian, uncertainty = cartesian_hessian(evaluator, np.array([0.2, -0.1]), 1.0e-3)

    assert_allclose(hessian, exact, rtol=0.0, atol=3.0e-13)
    # Central differences contribute u/h at the coarse step and 2u/h at
    # the fine step, propagated through (4*fine-coarse)/3 by absolute weights.
    assert uncertainty >= np.sqrt(2.0) * (4.0 * 4.0e-5 + 2.0e-5) / 3.0


class CoupledGradientErrorEvaluator:
    def __init__(self, center: np.ndarray, uncertainty: float) -> None:
        self.center = center
        self.uncertainty = uncertainty
        self.exact_hessian = np.array([[2.0, 0.3], [0.3, 1.5]])
        self.error_direction = np.ones(2) / np.sqrt(2.0)

    def __call__(self, point: np.ndarray) -> FrequencyGradient:
        point = np.asarray(point)
        displacement = point - self.center
        axis = int(np.argmax(np.abs(displacement)))
        sign = np.sign(displacement[axis])
        error = sign * self.uncertainty * self.error_direction
        gradient = self.exact_hessian @ point + error
        return FrequencyGradient(10.0, gradient, 0.0, self.uncertainty)


def test_cartesian_hessian_bounds_coupled_errors_from_both_columns() -> None:
    center = np.array([0.2, -0.1])
    evaluator = CoupledGradientErrorEvaluator(center, uncertainty=2.0e-8)

    hessian, uncertainty = cartesian_hessian(evaluator, center, 1.0e-3)
    actual_operator_error = np.linalg.norm(hessian - evaluator.exact_hessian, ord=2)

    assert actual_operator_error <= uncertainty


class NonlinearGradientEvaluator:
    def __call__(self, point: np.ndarray) -> FrequencyGradient:
        x, y = np.asarray(point)
        omega = 5.0 + x**6 / 6.0 + y**2 / 2.0
        return FrequencyGradient(omega, np.array([x**5, y]), 0.0, 0.0)


def test_cartesian_hessian_uncertainty_includes_step_difference() -> None:
    evaluator = NonlinearGradientEvaluator()
    point = np.array([0.7, -0.2])
    step = 0.08

    coarse_xx = ((point[0] + step) ** 5 - (point[0] - step) ** 5) / (2.0 * step)
    fine_step = step / 2.0
    fine_xx = ((point[0] + fine_step) ** 5 - (point[0] - fine_step) ** 5) / (
        2.0 * fine_step
    )
    expected_xx = (4.0 * fine_xx - coarse_xx) / 3.0
    hessian, uncertainty = cartesian_hessian(evaluator, point, step)

    assert hessian[0, 0] == pytest.approx(expected_xx, rel=2.0e-15)
    assert uncertainty >= abs(expected_xx - fine_xx)


def test_cubic_normal_form_has_four_minima_and_four_saddles() -> None:
    evaluator = NormalFormEvaluator(0.02)
    annulus = Annulus(k0=2.0, half_width=0.10)

    points = locate_critical_points(evaluator, annulus, 17, 128, 1.0e-4)
    report = verify_annular_exhaustion(evaluator, annulus, points, 256)

    assert len(points) == 8
    assert sum(point.kind == "minimum" for point in points) == 4
    assert sum(point.kind == "saddle" for point in points) == 4
    assert sum(point.morse_index for point in points) == 0
    assert report.boundary_is_noncritical and report.index_closes
    assert report.minimum_boundary_gradient > 10.0 * report.maximum_gradient_uncertainty
    assert_allclose(
        [point.radius for point in points],
        2.0 - 0.02 * 0.3 / 3.0,
        rtol=0.0,
        atol=2.0e-8,
    )
    assert max(point.gradient_residual for point in points) <= max(
        point.gradient_uncertainty for point in points
    )


def test_periodic_polar_cells_find_theta_zero_once_and_sort_angles() -> None:
    points = _normal_form_points(0.02)
    angles = np.asarray([point.theta for point in points])

    assert np.all(np.diff(angles) > 0.0)
    assert angles[0] == pytest.approx(0.0, abs=1.0e-9)
    assert np.count_nonzero(np.minimum(angles, 2.0 * np.pi - angles) < 1.0e-8) == 1


def test_sign_reversal_preserves_angles_and_swaps_minimum_saddle_roles() -> None:
    positive = _normal_form_points(+0.02)
    negative = _normal_form_points(-0.02)

    assert_allclose(
        [point.theta for point in positive],
        [point.theta for point in negative],
        rtol=0.0,
        atol=1.0e-8,
    )
    for plus, minus in zip(positive, negative, strict=True):
        assert plus.kind == ("saddle" if minus.kind == "minimum" else "minimum")
    assert_allclose(
        [point.radius for point in negative],
        2.0 + 0.02 * 0.3 / 3.0,
        rtol=0.0,
        atol=2.0e-8,
    )


def test_candidate_resolution_doubling_is_stable_and_deduplicated() -> None:
    coarse = _normal_form_points(0.02, 9, 64)
    fine = _normal_form_points(0.02, 17, 128)

    assert len(coarse) == len(fine) == 8
    assert_allclose(
        [[point.kx, point.ky] for point in coarse],
        [[point.kx, point.ky] for point in fine],
        rtol=0.0,
        atol=2.0e-9,
    )
    coordinates = np.array([[point.kx, point.ky] for point in fine])
    pairwise_distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    np.fill_diagonal(pairwise_distances, np.inf)
    assert np.min(pairwise_distances) > 1.0


def test_locator_skips_failed_refinements_before_reading_invalid_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def failed_root(*args: object, **_kwargs: object) -> SimpleNamespace:
        calls.append(args)
        return SimpleNamespace(success=False, x=np.array([np.nan, np.nan]))

    monkeypatch.setattr("zgv_morse.critical_points.root", failed_root)

    points = _normal_form_points(0.02)

    assert calls
    assert points == []


@pytest.mark.parametrize(
    ("separation", "expected_count"),
    [(1.0e-6, 1), (2.0e-2, 2)],
)
def test_dedup_uses_capped_gradient_to_hessian_position_uncertainty(
    separation: float,
    expected_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    center = np.array([2.0, 0.0])
    solutions = [
        center + np.array([0.0, -0.5 * separation]),
        center + np.array([0.0, +0.5 * separation]),
    ]

    class UncertainRootEvaluator:
        def __call__(self, point: np.ndarray) -> FrequencyGradient:
            gradient = np.asarray(point) - center
            return FrequencyGradient(5.0, gradient, 0.0, 1.0)

    monkeypatch.setattr(
        "zgv_morse.critical_points._polar_candidates",
        lambda *_args: [center.copy(), center.copy()],
    )
    monkeypatch.setattr(
        "zgv_morse.critical_points.root",
        lambda *_args, **_kwargs: SimpleNamespace(success=True, x=solutions.pop(0)),
    )
    monkeypatch.setattr(
        "zgv_morse.critical_points.cartesian_hessian",
        lambda *_args: (np.eye(2), 0.0),
    )

    points = locate_critical_points(
        UncertainRootEvaluator(),
        Annulus(2.0, 0.1),
        3,
        8,
        1.0e-4,
    )

    assert len(points) == expected_count


class OffsetMinimumEvaluator:
    def __call__(self, point: np.ndarray) -> FrequencyGradient:
        displacement = np.asarray(point) - np.array([2.0, 0.0])
        omega = 5.0 + 0.5 * float(displacement @ displacement)
        return FrequencyGradient(omega, displacement, 0.0, 1.0e-12)


def _offset_minimum_point() -> CriticalPoint:
    return CriticalPoint(
        2.0,
        0.0,
        2.0,
        0.0,
        5.0,
        0.0,
        1.0e-12,
        np.eye(2),
        np.ones(2),
        0.0,
        "minimum",
        1,
    )


def test_exhaustion_compares_point_index_with_outer_minus_inner_winding() -> None:
    evaluator = OffsetMinimumEvaluator()
    annulus = Annulus(2.0, 0.2)

    missing = verify_annular_exhaustion(evaluator, annulus, [], 64)
    complete = verify_annular_exhaustion(evaluator, annulus, [_offset_minimum_point()], 64)

    assert missing.boundary_is_noncritical
    assert not missing.index_closes
    assert complete.boundary_is_noncritical
    assert complete.index_closes


class RapidBoundaryRotationEvaluator:
    def __call__(self, point: np.ndarray) -> FrequencyGradient:
        theta = np.arctan2(point[1], point[0])
        gradient = np.array([np.cos(12.0 * theta), np.sin(12.0 * theta)])
        return FrequencyGradient(5.0, gradient, 0.0, 1.0e-12)


def test_exhaustion_rejects_aliased_boundary_winding() -> None:
    report = verify_annular_exhaustion(
        RapidBoundaryRotationEvaluator(),
        Annulus(2.0, 0.1),
        [],
        16,
    )

    assert not report.boundary_is_noncritical
    assert not report.index_closes


@pytest.mark.parametrize("scale", [1.0e-300, 1.0e300])
def test_boundary_winding_is_invariant_to_extreme_finite_gradient_scale(scale: float) -> None:
    class ScaledRadialEvaluator:
        def __call__(self, point: np.ndarray) -> FrequencyGradient:
            direction = np.asarray(point) / np.hypot(point[0], point[1])
            return FrequencyGradient(5.0, scale * direction, 0.0, 0.0)

    report = verify_annular_exhaustion(
        ScaledRadialEvaluator(),
        Annulus(2.0, 0.1),
        [],
        16,
    )

    assert report.boundary_is_noncritical
    assert report.index_closes


def test_exhaustion_report_detects_unresolved_boundary_uncertainty() -> None:
    evaluator = NormalFormEvaluator(0.02, gradient_uncertainty=1.0)
    report = verify_annular_exhaustion(
        evaluator,
        Annulus(2.0, 0.1),
        _normal_form_points(0.02),
        64,
    )

    assert not report.boundary_is_noncritical
    assert not report.index_closes


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: cartesian_hessian(NormalFormEvaluator(0.02), [2.0], 1.0e-4), "point"),
        (
            lambda: cartesian_hessian(NormalFormEvaluator(0.02), [2.0, 0.0], 0.0),
            "step",
        ),
        (
            lambda: locate_critical_points(
                NormalFormEvaluator(0.02), Annulus(2.0, 0.1), 1, 128, 1.0e-4
            ),
            "n_radial",
        ),
        (
            lambda: locate_critical_points(
                NormalFormEvaluator(0.02), Annulus(2.0, 0.1), 17, 3, 1.0e-4
            ),
            "n_theta",
        ),
        (
            lambda: verify_annular_exhaustion(
                NormalFormEvaluator(0.02), Annulus(2.0, 0.1), [], 3
            ),
            "n_boundary",
        ),
    ],
)
def test_critical_point_algorithms_reject_invalid_inputs(call: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        call()  # type: ignore[operator]
