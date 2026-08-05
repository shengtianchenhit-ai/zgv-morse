from __future__ import annotations

from dataclasses import FrozenInstanceError
from inspect import signature

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import simpson
import scipy.special as special

import zgv_morse.green_response as green_response
from zgv_morse.workflows import green as green_workflow

from zgv_morse.asymptotics import (
    UniformParameters,
    build_morse_contribution,
    morse_stationary_phase_response,
    scale_transition_response,
)
from zgv_morse.critical_points import (
    Annulus,
    locate_critical_points,
    verify_annular_exhaustion,
)
from zgv_morse.dispersion import RingAnchoredSpectralEvaluator
from zgv_morse.elasticity import cubic_family, isotropic_tensor
from zgv_morse.green_response import (
    BranchNodeSample,
    BranchResponse,
    FullWaveRadialEvaluatorFactory,
    PolarBranchSurface,
    assert_phase_accuracy,
    build_tracked_surface,
    estimate_nested_frequency_error,
    integrate_branch_response,
    integrate_registered_grid_convergence,
    normal_impulse_amplitude,
    verify_registered_grid_convergence,
)


def _normal_form_surface(
    *,
    number_of_q: int = 41,
    number_of_theta: int = 32,
    frequency_error: float | np.ndarray = 0.0,
) -> PolarBranchSurface:
    q = np.linspace(-0.6, 0.6, number_of_q)
    theta = 2.0 * np.pi * np.arange(number_of_theta) / number_of_theta
    radial, angular = np.meshgrid(q, theta, indexing="ij")
    omega = 5.0 + 1.5 * radial**2 + 0.02 * np.cos(4.0 * angular)
    amplitude = (1.0 + 0.1 * radial) * np.exp(0.2j * np.sin(angular))
    return PolarBranchSurface(q, theta, omega, amplitude, frequency_error)


def _nested_surfaces() -> tuple[PolarBranchSurface, PolarBranchSurface, PolarBranchSurface]:
    surfaces: list[PolarBranchSurface] = []
    for number_of_q, number_of_theta in ((3, 4), (5, 8), (9, 16)):
        q = np.linspace(-0.2, 0.2, number_of_q)
        theta = 2.0 * np.pi * np.arange(number_of_theta) / number_of_theta
        radial, angular = np.meshgrid(q, theta, indexing="ij")
        surfaces.append(
            PolarBranchSurface(
                q,
                theta,
                5.0 + radial**2 + 0.01 * np.cos(4.0 * angular),
                np.ones_like(radial, dtype=np.complex128),
                np.zeros_like(radial),
            )
        )
    return tuple(surfaces)  # type: ignore[return-value]


def test_direct_normal_form_quadrature_converges_to_bessel_kernel() -> None:
    k0, curvature, epsilon, V0, V4, sigma = 2.0, 3.0, 0.04, 0.1, 0.5, 0.30
    q = np.linspace(-3.0 * sigma, 3.0 * sigma, 1201)
    theta = 2.0 * np.pi * np.arange(256) / 256
    radial, angular = np.meshgrid(q, theta, indexing="ij")
    surface = PolarBranchSurface(
        q,
        theta,
        5.0 + 0.5 * curvature * radial**2 + epsilon * (V0 + V4 * np.cos(4 * angular)),
        np.ones_like(radial, dtype=np.complex128),
        0.0,
    )
    time = np.array([100.0, 140.0, 180.0])

    response = integrate_branch_response(
        surface,
        time,
        k0,
        5.0 + epsilon * V0,
        0.0,
        sigma,
    )
    parameters = UniformParameters(5.0, k0, curvature, epsilon, V0, V4, 1.0)

    assert_allclose(
        scale_transition_response(time, response.analytic_signal(), parameters),
        special.j0(epsilon * V4 * time),
        rtol=1.5e-2,
        atol=1.5e-2,
    )


def test_direct_integral_uses_full_measure_source_and_window() -> None:
    k0, source_radius, window_sigma = 1.2, 0.4, 0.25
    q = np.linspace(-1.5 * window_sigma, 1.5 * window_sigma, 7)
    theta = 2.0 * np.pi * np.arange(8) / 8
    radial, angular = np.meshgrid(q, theta, indexing="ij")
    omega_reference = 5.0
    amplitude = (1.0 + 0.2 * radial) * (1.0 + 0.3 * np.exp(1j * angular))
    surface = PolarBranchSurface(
        q,
        theta,
        np.full_like(radial, omega_reference),
        amplitude,
        0.0,
    )
    response = integrate_branch_response(
        surface,
        np.array([1.0]),
        k0,
        omega_reference,
        source_radius,
        window_sigma,
    )
    radial_k = k0 + q
    static = (
        amplitude
        * radial_k[:, None]
        * np.exp(-0.25 * source_radius**2 * radial_k[:, None] ** 2)
        * np.exp(-((q[:, None] / window_sigma) ** 8))
        * green_response.compact_radial_taper(q[:, None], window_sigma)
    )
    angular_integral = (2.0 * np.pi / theta.size) * np.sum(static, axis=1)
    expected = (2.0 * np.pi) ** -2 * simpson(angular_integral, x=q)

    assert response.demodulated[0] == pytest.approx(expected, rel=2.0e-14, abs=2.0e-14)


def test_compact_radial_taper_has_an_exact_plateau_and_flat_support_boundary() -> None:
    sigma = 0.2
    q = sigma * np.array([-1.6, -1.5, -1.4, -1.25, -1.0, 0.0, 1.0, 1.25, 1.4, 1.5, 1.6])

    taper = green_response.compact_radial_taper(q, sigma)

    assert_allclose(taper, taper[::-1], rtol=0.0, atol=0.0)
    assert_allclose(taper[np.abs(q) <= 1.25 * sigma], 1.0, rtol=0.0, atol=0.0)
    assert_allclose(taper[np.abs(q) >= 1.5 * sigma], 0.0, rtol=0.0, atol=0.0)
    assert np.all((taper[(np.abs(q) > 1.25 * sigma) & (np.abs(q) < 1.5 * sigma)] > 0.0))
    assert np.all((taper[(np.abs(q) > 1.25 * sigma) & (np.abs(q) < 1.5 * sigma)] < 1.0))


def test_direct_quadrature_annihilates_endpoint_data_at_compact_support() -> None:
    sigma = 0.2
    q = np.array([-1.5 * sigma, 0.0, 1.5 * sigma])
    theta = 2.0 * np.pi * np.arange(4) / 4
    radial, _angular = np.meshgrid(q, theta, indexing="ij")
    amplitude = np.zeros_like(radial, dtype=np.complex128)
    amplitude[[0, -1], :] = 1.0e12
    surface = PolarBranchSurface(
        q,
        theta,
        np.full_like(radial, 5.0),
        amplitude,
        0.0,
    )

    response = integrate_branch_response(surface, [1.0], 2.0, 5.0, 0.0, sigma)

    assert_allclose(response.demodulated, 0.0, rtol=0.0, atol=0.0)


def test_direct_quadrature_rejects_a_domain_that_cuts_compact_support() -> None:
    with pytest.raises(ValueError, match="radial domain must cover compact support"):
        integrate_branch_response(
            _normal_form_surface(),
            [1.0],
            2.0,
            5.0,
            0.0,
            0.5,
        )


@pytest.mark.parametrize("side", ("inner", "outer"))
def test_direct_quadrature_rejects_one_ulp_of_compact_support_undercoverage(
    side: str,
) -> None:
    sigma = 0.2
    support = 1.5 * sigma
    left = -support
    right = support
    if side == "inner":
        left = np.nextafter(left, 0.0)
    else:
        right = np.nextafter(right, 0.0)
    q = np.array([left, 0.0, right])
    theta = 2.0 * np.pi * np.arange(4) / 4
    radial, _angular = np.meshgrid(q, theta, indexing="ij")
    surface = PolarBranchSurface(
        q,
        theta,
        np.full_like(radial, 5.0),
        np.ones_like(radial, dtype=np.complex128),
        0.0,
    )

    with pytest.raises(ValueError, match="radial domain must cover compact support"):
        integrate_branch_response(surface, [1.0], 2.0, 5.0, 0.0, sigma)


def test_green_response_domain_covers_direct_support_with_search_margin() -> None:
    control = green_workflow._response_domain_control(
        k0=4.0,
        window_sigma=0.6,
        registered_annulus_half_width=0.6,
    )

    assert control.taper_plateau_half_width == pytest.approx(0.75)
    assert control.direct_support_half_width == pytest.approx(0.9)
    assert control.morse_search_margin == pytest.approx(0.03)
    assert control.morse_search_half_width == pytest.approx(0.93)
    assert control.morse_search_half_width > control.direct_support_half_width
    assert control.morse_radial_nodes == 15
    metadata = control.metadata(4.0)
    assert metadata["direct_support_abs_q_over_sigma"] == pytest.approx(1.5)
    assert metadata["direct_support_abs_q_over_k0"] == pytest.approx(0.225)
    assert metadata["morse_search_abs_q_over_sigma"] == pytest.approx(1.55)
    assert metadata["morse_search_abs_q_over_k0"] == pytest.approx(0.2325)


@pytest.mark.parametrize(
    ("k0", "window_sigma", "registered_half_width"),
    (
        (0.0, 0.6, 0.6),
        (4.0, 0.0, 0.6),
        (4.0, 0.6, 0.0),
        (4.0, np.nan, 0.6),
    ),
)
def test_green_response_domain_rejects_invalid_scales(
    k0: float,
    window_sigma: float,
    registered_half_width: float,
) -> None:
    with pytest.raises(ValueError, match="radial-domain scales"):
        green_workflow._response_domain_control(
            k0=k0,
            window_sigma=window_sigma,
            registered_annulus_half_width=registered_half_width,
        )


def test_green_response_domain_rejects_search_without_positive_inner_radius() -> None:
    with pytest.raises(ValueError, match="retain a positive inner radius"):
        green_workflow._response_domain_control(
            k0=0.9,
            window_sigma=0.6,
            registered_annulus_half_width=0.6,
        )


def test_morse_node_uses_the_same_compact_taper_as_direct_quadrature() -> None:
    sigma = 0.2
    radial_offset = 1.4 * sigma
    node = BranchNodeSample(5.0, 2.0 + 3.0j, 1.0e-6, 4.0e-5, 0.2)

    tapered = green_workflow._compactly_tapered_node(node, radial_offset, sigma)
    weight = float(green_response.compact_radial_taper([radial_offset], sigma)[0])

    assert tapered.omega == node.omega
    assert tapered.amplitude == pytest.approx(weight * node.amplitude)
    assert tapered.frequency_uncertainty == node.frequency_uncertainty
    assert tapered.amplitude_uncertainty == pytest.approx(weight * node.amplitude_uncertainty)
    assert tapered.relative_eigengap == node.relative_eigengap


def test_direct_quadrature_is_invariant_to_time_chunking() -> None:
    surface = _normal_form_surface()
    time = np.linspace(20.0, 80.0, 17)

    one_at_a_time = integrate_branch_response(surface, time, 2.0, 5.0, 0.1, 0.3, chunk_size=1)
    one_chunk = integrate_branch_response(surface, time, 2.0, 5.0, 0.1, 0.3, chunk_size=100)

    assert_allclose(one_at_a_time.demodulated, one_chunk.demodulated, rtol=2.0e-15, atol=2.0e-15)


def test_direct_quadrature_never_calls_bessel_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("direct quadrature called a Bessel function")

    monkeypatch.setattr(special, "j0", forbidden)
    monkeypatch.setattr(special, "jv", forbidden)

    response = integrate_branch_response(
        _normal_form_surface(number_of_q=21, number_of_theta=16),
        np.array([10.0, 20.0]),
        2.0,
        5.0,
        0.0,
        0.3,
    )
    assert np.isfinite(response.demodulated).all()


def test_surface_and_response_are_frozen_slotted_defensive_records() -> None:
    q = np.linspace(-0.2, 0.2, 5)
    theta = 2.0 * np.pi * np.arange(8) / 8
    omega = np.full((5, 8), 5.0)
    amplitude = np.ones((5, 8), dtype=np.complex128)
    uncertainty = np.full((5, 8), 1.0e-5)
    amplitude_uncertainty = np.full((5, 8), 2.0e-5)
    relative_eigengap = np.full((5, 8), 0.2)
    surface = PolarBranchSurface(
        q,
        theta,
        omega,
        amplitude,
        uncertainty,
        amplitude_uncertainty,
        relative_eigengap,
    )
    response = BranchResponse(np.array([1.0, 2.0]), 5.0, np.array([1.0j, 2.0j]))

    q[:] = 99.0
    theta[:] = 99.0
    omega[:] = 99.0
    amplitude[:] = 99.0
    uncertainty[:] = 99.0
    amplitude_uncertainty[:] = 99.0
    relative_eigengap[:] = 99.0
    assert_allclose(surface.q, np.linspace(-0.2, 0.2, 5))
    assert_allclose(surface.frequency_error, 1.0e-5)
    assert_allclose(surface.amplitude_error, 2.0e-5)
    assert not any(
        array.flags.writeable
        for array in (
            surface.q,
            surface.theta,
            surface.omega,
            surface.amplitude,
            surface.frequency_error,
            surface.amplitude_error,
            surface.relative_eigengap,
            response.time,
            response.demodulated,
        )
    )
    assert not hasattr(surface, "__dict__")
    assert not hasattr(response, "__dict__")
    with pytest.raises(FrozenInstanceError):
        surface.q = np.array([0.0])  # type: ignore[misc]

    analytic = response.analytic_signal()
    assert not np.shares_memory(analytic, response.demodulated)
    analytic[:] = 0.0
    assert_allclose(response.demodulated, [1.0j, 2.0j])


@pytest.mark.parametrize(
    "surface",
    [
        lambda: PolarBranchSurface(
            [0.0, -0.1, 0.1],
            2.0 * np.pi * np.arange(4) / 4,
            np.ones((3, 4)),
            np.ones((3, 4)),
            0.0,
        ),
        lambda: PolarBranchSurface(
            [-0.1, 0.0, 0.1],
            np.linspace(0.0, 2.0 * np.pi, 4),
            np.ones((3, 4)),
            np.ones((3, 4)),
            0.0,
        ),
        lambda: PolarBranchSurface(
            [-0.1, 0.0, 0.1],
            2.0 * np.pi * np.arange(4) / 4,
            np.ones((4, 3)),
            np.ones((3, 4)),
            0.0,
        ),
        lambda: PolarBranchSurface(
            [-0.1, 0.0, 0.1],
            2.0 * np.pi * np.arange(4) / 4,
            np.full((3, 4), np.nan),
            np.ones((3, 4)),
            0.0,
        ),
        lambda: PolarBranchSurface(
            [-0.1, 0.0, 0.1],
            2.0 * np.pi * np.arange(4) / 4,
            np.ones((3, 4)),
            np.ones((3, 4)),
            -1.0,
        ),
        lambda: PolarBranchSurface(
            [-0.1, 0.0, 0.1],
            2.0 * np.pi * np.arange(4) / 4,
            np.ones((3, 4)),
            np.ones((3, 4)),
            0.0,
            -1.0,
        ),
    ],
)
def test_surface_rejects_malformed_grids_and_values(surface) -> None:
    with pytest.raises((TypeError, ValueError)):
        surface()


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (([-1.0, 1.0], 2.0, 5.0, 0.0, 0.3), "nonnegative"),
        (([1.0, 2.0], 0.0, 5.0, 0.0, 0.3), "positive"),
        (([1.0, 2.0], 0.5, 5.0, 0.0, 0.3), "radial"),
        (([1.0, 2.0], 2.0, 0.0, 0.0, 0.3), "positive"),
        (([1.0, 2.0], 2.0, 5.0, -0.1, 0.3), "nonnegative"),
        (([1.0, 2.0], 2.0, 5.0, 0.0, 0.0), "positive"),
    ],
)
def test_direct_integral_rejects_malformed_geometry(arguments, match: str) -> None:
    time, k0, omega_reference, source_radius, window_sigma = arguments
    with pytest.raises((TypeError, ValueError), match=match):
        integrate_branch_response(
            _normal_form_surface(),
            time,
            k0,
            omega_reference,
            source_radius,
            window_sigma,
        )


def test_direct_integral_rejects_bad_chunk_normalization_and_phase_budget() -> None:
    per_node_error = np.zeros((41, 32))
    per_node_error[-1, -1] = 6.0e-4
    surface = _normal_form_surface(frequency_error=per_node_error)
    with pytest.raises((TypeError, ValueError), match="chunk_size"):
        integrate_branch_response(surface, [1.0], 2.0, 5.0, 0.0, 0.3, chunk_size=True)
    with pytest.raises(ValueError, match="normalization"):
        integrate_branch_response(
            surface,
            [1.0],
            2.0,
            5.0,
            0.0,
            0.3,
            fourier_normalization=0.0,
        )
    with pytest.raises(ValueError, match="accumulated phase"):
        integrate_branch_response(surface, [100.0], 2.0, 5.0, 0.0, 0.3)


def test_nested_frequency_error_requires_registered_doubling() -> None:
    coarse = np.zeros((3, 5))
    fine = np.zeros((5, 10))
    fine[2, 4] = 0.3

    assert estimate_nested_frequency_error(coarse, fine) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="doubled shape"):
        estimate_nested_frequency_error(coarse, np.zeros((6, 10)))
    with pytest.raises(ValueError, match="finite"):
        estimate_nested_frequency_error(coarse, np.full((5, 10), np.nan))


def test_phase_budget_accepts_point_zero_four_and_rejects_point_zero_six() -> None:
    assert assert_phase_accuracy(100.0, 4.0e-4) == pytest.approx(0.04)
    with pytest.raises(ValueError, match="accumulated phase"):
        assert_phase_accuracy(100.0, 6.0e-4)
    with pytest.raises(ValueError, match="nonnegative"):
        assert_phase_accuracy(-1.0, 1.0e-4)


def test_normal_impulse_amplitude_is_invariant_to_modal_phase() -> None:
    component = 0.3 - 0.7j
    expected = normal_impulse_amplitude(4.0, component)

    for phase in np.linspace(-np.pi, np.pi, 17):
        assert normal_impulse_amplitude(4.0, component * np.exp(1j * phase)) == pytest.approx(
            expected,
            rel=2.0e-15,
            abs=2.0e-15,
        )
    assert expected == pytest.approx(1j * abs(component) ** 2 / 8.0)
    with pytest.raises(ValueError, match="positive"):
        normal_impulse_amplitude(0.0, component)
    with pytest.raises(ValueError, match="finite"):
        normal_impulse_amplitude(4.0, complex(np.nan, 0.0))


class _RecordingFactory:
    def __init__(self, k0: float) -> None:
        self.k0 = k0
        self.factory_calls: list[tuple[float, int]] = []
        self.sample_calls: list[tuple[float, int, float]] = []

    def __call__(self, theta: float, radial_direction: int):
        self.factory_calls.append((theta, radial_direction))

        def evaluate(kxy: np.ndarray) -> BranchNodeSample:
            q_value = float(np.linalg.norm(kxy) - self.k0)
            self.sample_calls.append((theta, radial_direction, q_value))
            return BranchNodeSample(
                omega=6.0 + theta + q_value,
                amplitude=complex(1.0 + q_value, theta - q_value),
                frequency_uncertainty=1.0e-6 + abs(q_value) * 1.0e-4,
                amplitude_uncertainty=2.0e-6 + abs(q_value) * 2.0e-4,
            )

        return evaluate


def test_tracked_surface_uses_fresh_anchor_outward_rays_and_places_node_data() -> None:
    k0 = 2.0
    q = np.array([-0.3, -0.1, 0.0, 0.2, 0.4])
    theta = 2.0 * np.pi * np.arange(4) / 4
    factory = _RecordingFactory(k0)

    surface = build_tracked_surface(factory, q, theta, k0)

    assert factory.factory_calls == [
        item for angle in theta for item in ((float(angle), -1), (float(angle), 1))
    ]
    for angle in theta:
        calls = [
            (direction, radial)
            for called_angle, direction, radial in factory.sample_calls
            if called_angle == angle
        ]
        assert_allclose(
            [radial for direction, radial in calls if direction == -1], [0.0, -0.1, -0.3]
        )
        assert_allclose([radial for direction, radial in calls if direction == 1], [0.0, 0.2, 0.4])

    radial, angular = np.meshgrid(q, theta, indexing="ij")
    assert_allclose(surface.omega, 6.0 + angular + radial)
    assert_allclose(surface.amplitude.real, 1.0 + radial)
    assert_allclose(surface.amplitude.imag, angular - radial)
    assert_allclose(surface.frequency_error, 1.0e-6 + np.abs(radial) * 1.0e-4)
    assert_allclose(surface.amplitude_error, 2.0e-6 + np.abs(radial) * 2.0e-4)


def test_tracked_surface_rejects_missing_anchor_and_invalid_samples() -> None:
    theta = 2.0 * np.pi * np.arange(4) / 4
    with pytest.raises(ValueError, match="q=0"):
        build_tracked_surface(_RecordingFactory(2.0), [-0.2, 0.1, 0.2], theta, 2.0)

    def bad_factory(_theta: float, _direction: int):
        return lambda _kxy: BranchNodeSample(np.nan, 1.0, 0.0)

    with pytest.raises(ValueError, match="omega"):
        build_tracked_surface(bad_factory, [-0.2, 0.0, 0.2], theta, 2.0)


@pytest.mark.parametrize("mismatch", ["omega", "amplitude"])
def test_tracked_surface_rejects_inconsistent_fresh_anchors(mismatch: str) -> None:
    k0 = 2.0
    theta = 2.0 * np.pi * np.arange(4) / 4

    def inconsistent_factory(_theta: float, radial_direction: int):
        def evaluate(kxy: np.ndarray) -> BranchNodeSample:
            q_value = float(np.linalg.norm(kxy) - k0)
            omega_offset = 1.0e-2 * radial_direction if mismatch == "omega" else 0.0
            amplitude_offset = 0.1 * radial_direction if mismatch == "amplitude" else 0.0
            return BranchNodeSample(
                omega=6.0 + q_value + omega_offset,
                amplitude=1.0 + amplitude_offset,
                frequency_uncertainty=1.0e-4,
            )

        return evaluate

    with pytest.raises(ValueError, match=f"anchor {mismatch}"):
        build_tracked_surface(inconsistent_factory, [-0.2, 0.0, 0.2], theta, k0)


@pytest.fixture(scope="module")
def full_wave_spectral_evaluator() -> RingAnchoredSpectralEvaluator:
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


def test_concrete_full_wave_factory_tracks_modes_and_uses_node_amplitudes(
    full_wave_spectral_evaluator: RingAnchoredSpectralEvaluator,
) -> None:
    evaluator = full_wave_spectral_evaluator
    factory = FullWaveRadialEvaluatorFactory(evaluator)
    q = np.array([-0.02, 0.0, 0.02])
    theta = 2.0 * np.pi * np.arange(4) / 4

    surface = build_tracked_surface(factory, q, theta, evaluator.k0)

    assert surface.omega.shape == (3, 4)
    assert np.isfinite(surface.omega).all()
    assert np.isfinite(surface.amplitude).all()
    assert np.max(np.abs(surface.amplitude)) > 0.0
    assert_allclose(surface.amplitude.real, 0.0, atol=2.0e-15)
    assert np.all(surface.amplitude.imag >= 0.0)
    assert 0.0 <= np.max(surface.frequency_error) < 1.0e-3
    assert 0.0 <= np.max(surface.amplitude_error)
    relative_amplitude_error = surface.amplitude_error / np.maximum(
        np.abs(surface.amplitude),
        1.0e-12,
    )
    assert np.max(relative_amplitude_error) < 1.0e-2
    assert not surface.amplitude_error.flags.writeable
    assert np.min(surface.relative_eigengap) > 0.0
    assert not surface.relative_eigengap.flags.writeable
    assert np.ptp(surface.omega[:, 0]) > 0.0

    spectral_sample = evaluator.radial_tracker(0.0, 1)(np.array([evaluator.k0, 0.0]))
    physical_sample = factory(0.0, 1)(np.array([evaluator.k0, 0.0]))
    assert physical_sample.omega == pytest.approx(spectral_sample.frequency.omega)
    assert physical_sample.amplitude == pytest.approx(
        normal_impulse_amplitude(
            spectral_sample.frequency.omega,
            spectral_sample.top_normal_component,
        )
    )
    assert physical_sample.frequency_uncertainty == pytest.approx(
        spectral_sample.frequency.frequency_uncertainty
    )
    assert physical_sample.amplitude_uncertainty == pytest.approx(
        abs(
            normal_impulse_amplitude(
                spectral_sample.frequency.omega,
                spectral_sample.top_normal_component,
            )
            - normal_impulse_amplitude(
                spectral_sample.coarse_omega,
                spectral_sample.coarse_top_normal_component,
            )
        )
    )
    assert physical_sample.relative_eigengap == pytest.approx(spectral_sample.relative_eigengap)


def test_full_wave_factory_rejects_unresolved_spectral_amplitude(
    full_wave_spectral_evaluator: RingAnchoredSpectralEvaluator,
) -> None:
    evaluator = full_wave_spectral_evaluator
    factory = FullWaveRadialEvaluatorFactory(
        evaluator,
        amplitude_rtol=1.0e-8,
        amplitude_atol=0.0,
    )

    with pytest.raises(RuntimeError, match="spectral amplitude"):
        factory(0.0, 1)(np.array([evaluator.k0, 0.0]))


@pytest.mark.parametrize(
    "keywords",
    [
        {"amplitude_rtol": -1.0},
        {"amplitude_atol": -1.0},
    ],
)
def test_full_wave_factory_rejects_invalid_amplitude_tolerances(
    full_wave_spectral_evaluator: RingAnchoredSpectralEvaluator,
    keywords: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="amplitude"):
        FullWaveRadialEvaluatorFactory(full_wave_spectral_evaluator, **keywords)


def _responses(values: tuple[complex, complex, complex]) -> tuple[BranchResponse, ...]:
    time = np.array([1.0, 2.0])
    return tuple(
        BranchResponse(time, 5.0, np.full(time.shape, value, dtype=np.complex128))
        for value in values
    )


def test_registered_grid_verifier_uses_nested_complex_response_differences() -> None:
    result = verify_registered_grid_convergence(
        _nested_surfaces(),
        _responses((1.0 + 0.0j, 1.1 + 0.0j, 1.12 + 0.0j)),
    )

    assert result.grid_shapes == ((3, 4), (5, 8), (9, 16))
    assert_allclose(result.complex_response_errors, [0.1, 0.02], atol=2.0e-15)
    assert_allclose(result.nested_frequency_errors, 0.0, atol=1.0e-15)
    assert_allclose(result.accumulated_phase_errors, 0.0, atol=1.0e-15)
    assert result.finest_response is result.responses[-1]


def test_registered_grid_verifier_rejects_equal_magnitudes_with_diverging_phase() -> None:
    responses = _responses((np.exp(0.0j), np.exp(0.2j), np.exp(0.5j)))
    assert_allclose(
        [np.abs(response.demodulated) for response in responses],
        1.0,
        atol=2.0e-15,
    )

    with pytest.raises(ValueError, match="complex response differences must decrease"):
        verify_registered_grid_convergence(_nested_surfaces(), responses)


def test_registered_grid_verifier_rejects_large_but_slightly_decreasing_errors() -> None:
    with pytest.raises(ValueError, match="finest-grid complex response discrepancy"):
        verify_registered_grid_convergence(
            _nested_surfaces(),
            _responses((0.0, 100.0, 199.0)),
        )


def test_registered_grid_verifier_rejects_slow_error_reduction() -> None:
    with pytest.raises(ValueError, match="convergence rate"):
        verify_registered_grid_convergence(
            _nested_surfaces(),
            _responses((1.0, 1.04, 1.07)),
        )


def test_registered_grid_verifier_accepts_near_zero_absolute_floor() -> None:
    result = verify_registered_grid_convergence(
        _nested_surfaces(),
        _responses((0.0, 5.0e-9, 9.0e-9)),
    )

    assert_allclose(result.complex_response_errors, [5.0e-9, 4.0e-9])


@pytest.mark.parametrize(
    ("keywords", "value"),
    [
        ({"response_rtol": -1.0}, "response_rtol"),
        ({"response_atol": -1.0}, "response_atol"),
        ({"maximum_error_ratio": 1.0}, "maximum_error_ratio"),
    ],
)
def test_registered_grid_verifier_rejects_invalid_tolerances(
    keywords: dict[str, float],
    value: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=value):
        verify_registered_grid_convergence(
            _nested_surfaces(),
            _responses((1.0, 1.1, 1.12)),
            **keywords,
        )


def test_registered_grid_verifier_rejects_phase_budget_and_non_nested_nodes() -> None:
    coarse, fine, finest = _nested_surfaces()
    shifted_omega = np.array(fine.omega, copy=True)
    shifted_omega[::2, ::2] += 0.03
    bad_frequency = PolarBranchSurface(
        fine.q,
        fine.theta,
        shifted_omega,
        fine.amplitude,
        fine.frequency_error,
    )
    with pytest.raises(ValueError, match="accumulated phase"):
        verify_registered_grid_convergence(
            (coarse, bad_frequency, finest),
            _responses((1.0, 1.1, 1.12)),
        )

    shifted_q = np.array(fine.q, copy=True)
    shifted_q[2] += 1.0e-3
    non_nested = PolarBranchSurface(
        shifted_q,
        fine.theta,
        fine.omega,
        fine.amplitude,
        fine.frequency_error,
    )
    with pytest.raises(ValueError, match="nested q"):
        verify_registered_grid_convergence(
            (coarse, non_nested, finest),
            _responses((1.0, 1.1, 1.12)),
        )


@pytest.mark.parametrize("mismatch", ["time", "carrier"])
def test_registered_grid_verifier_rejects_response_coordinate_mismatch(mismatch: str) -> None:
    responses = list(_responses((1.0, 1.1, 1.12)))
    if mismatch == "time":
        responses[1] = BranchResponse([1.0, 2.1], 5.0, [1.1, 1.1])
    else:
        responses[1] = BranchResponse([1.0, 2.0], 5.1, [1.1, 1.1])

    with pytest.raises(ValueError, match=mismatch):
        verify_registered_grid_convergence(_nested_surfaces(), tuple(responses))


class _AnalyticFactory:
    def __init__(self, k0: float) -> None:
        self.k0 = k0

    def __call__(self, theta: float, _radial_direction: int):
        def evaluate(kxy: np.ndarray) -> BranchNodeSample:
            q_value = float(np.linalg.norm(kxy) - self.k0)
            return BranchNodeSample(
                omega=5.0 + q_value**2 + 0.01 * np.cos(4.0 * theta),
                amplitude=1.0 + 0.05j * q_value,
                frequency_uncertainty=0.0,
            )

        return evaluate


def test_registered_grid_orchestrator_builds_three_declared_levels() -> None:
    window_sigma = 0.2
    support = green_response.DIRECT_RADIAL_SUPPORT_SIGMA * window_sigma
    result = integrate_registered_grid_convergence(
        _AnalyticFactory(2.0),
        np.linspace(-support, support, 9),
        2.0 * np.pi * np.arange(4) / 4,
        np.array([1.0, 2.0]),
        2.0,
        5.0,
        0.0,
        window_sigma,
    )

    assert result.grid_shapes == ((9, 4), (17, 8), (33, 16))
    assert result.complex_response_errors[1] < result.complex_response_errors[0]


class _CachedCubicSymmetryFactory:
    """Reuse exact C4v-equivalent full-wave rays in the expensive benchmark."""

    def __init__(self, factory: FullWaveRadialEvaluatorFactory) -> None:
        self.factory = factory
        self.samples: dict[tuple[float, int, float], BranchNodeSample] = {}

    def __call__(self, theta: float, radial_direction: int):
        quadrant_angle = float(np.mod(theta, 0.5 * np.pi))
        canonical_angle = min(quadrant_angle, 0.5 * np.pi - quadrant_angle)
        canonical_angle = float(np.round(canonical_angle, 14))
        evaluator = self.factory(canonical_angle, radial_direction)
        direction = np.array(
            [np.cos(canonical_angle), np.sin(canonical_angle)],
            dtype=np.float64,
        )

        def evaluate(kxy: np.ndarray) -> BranchNodeSample:
            radius = float(np.linalg.norm(kxy))
            key = (
                canonical_angle,
                radial_direction,
                float(np.round(radius, 14)),
            )
            if key not in self.samples:
                self.samples[key] = evaluator(radius * direction)
            return self.samples[key]

        return evaluate


# Half-open indices on the fixed 1500:25:10200 time grid.  These three
# cancellation neighborhoods were preregistered from the exact-Morse phasor
# coherence before the direct full-wave residual was evaluated.
_FIXED_EPSILON_CANCELLATION_RANGES = (
    (15, 37),
    (128, 150),
    (242, 264),
)


@pytest.mark.slow
def test_fixed_epsilon_full_wave_response_matches_exact_morse_sum_without_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare certified direct quadrature with all eight exact full-wave points."""

    epsilon = 0.08
    k0 = 0.8042173193715181
    target_omega = 2.8517587749600901
    source_radius = 0.5
    window_sigma = 0.15 * k0
    evaluator = RingAnchoredSpectralEvaluator(
        cubic_family(2.0, 1.0, 1.0, epsilon)[0],
        rho=1.0,
        half_thickness=1.0,
        k0=k0,
        target_omega=target_omega,
        order=9,
        num_modes=12,
        angular_sectors=8,
    )
    domain_control = green_workflow._response_domain_control(
        k0=k0,
        window_sigma=window_sigma,
        registered_annulus_half_width=0.15 * k0,
    )
    annulus = Annulus(k0, domain_control.morse_search_half_width)
    coarse_points = locate_critical_points(
        evaluator,
        annulus,
        n_radial=7,
        n_theta=16,
        hessian_step=2.0e-3,
    )
    points = locate_critical_points(
        evaluator,
        annulus,
        n_radial=domain_control.morse_radial_nodes,
        n_theta=32,
        hessian_step=2.0e-3,
    )
    exhaustion = verify_annular_exhaustion(evaluator, annulus, points, 16)

    assert len(coarse_points) == len(points) == 8
    assert sum(point.kind == "minimum" for point in points) == 4
    assert sum(point.kind == "saddle" for point in points) == 4
    assert exhaustion.boundary_is_noncritical and exhaustion.index_closes
    for coarse, fine in zip(coarse_points, points, strict=True):
        assert coarse.kind == fine.kind
        coarse_position = np.array([coarse.kx, coarse.ky])
        fine_position = np.array([fine.kx, fine.ky])
        position_difference = float(np.linalg.norm(coarse_position - fine_position))
        position_uncertainty = coarse.gradient_uncertainty / np.min(
            np.abs(coarse.hessian_eigenvalues)
        ) + fine.gradient_uncertainty / np.min(np.abs(fine.hessian_eigenvalues))
        assert position_difference <= max(5.0e-9 * k0, position_uncertainty)
        coarse_sample = evaluator(coarse_position)
        fine_sample = evaluator(fine_position)
        positional_frequency_error = (
            0.5
            * max(
                np.max(np.abs(coarse.hessian_eigenvalues)),
                np.max(np.abs(fine.hessian_eigenvalues)),
            )
            * position_difference**2
        )
        assert abs(coarse.omega - fine.omega) <= (
            coarse_sample.frequency_uncertainty
            + fine_sample.frequency_uncertainty
            + positional_frequency_error
        )
        assert_allclose(
            coarse.hessian_eigenvalues,
            fine.hessian_eigenvalues,
            rtol=0.0,
            atol=coarse.hessian_uncertainty + fine.hessian_uncertainty,
        )

    minimum_frequency = min(point.omega for point in points)
    maximum_frequency = max(point.omega for point in points)
    omega_reference = 0.5 * (minimum_frequency + maximum_frequency)
    time = np.linspace(1500.0, 10200.0, 349)

    # These registered grids were fixed before evaluating the response.  C4v
    # covariance only avoids recomputing symmetry-equivalent modal rays; every
    # surface node still carries its full p/(p+4) frequency and modal amplitude.
    q = np.linspace(-1.5 * window_sigma, 1.5 * window_sigma, 129)
    theta = 2.0 * np.pi * np.arange(32) / 32
    physical_factory = FullWaveRadialEvaluatorFactory(evaluator)
    probe_angle = 0.125 * np.pi
    probe_angles = (
        probe_angle,
        0.5 * np.pi - probe_angle,
        0.5 * np.pi + probe_angle,
    )
    for radial_offset in (-0.5 * window_sigma, 0.0, 0.5 * window_sigma):
        radial_direction = -1 if radial_offset < 0.0 else 1
        probe_radius = k0 + radial_offset
        probe_samples = [
            physical_factory(angle, radial_direction)(
                probe_radius * np.array([np.cos(angle), np.sin(angle)])
            )
            for angle in probe_angles
        ]
        for equivalent in probe_samples[1:]:
            assert abs(equivalent.omega - probe_samples[0].omega) <= (
                equivalent.frequency_uncertainty + probe_samples[0].frequency_uncertainty + 1.0e-12
            )
            assert abs(equivalent.amplitude - probe_samples[0].amplitude) <= (
                equivalent.amplitude_uncertainty + probe_samples[0].amplitude_uncertainty + 1.0e-12
            )
    cached_factory = _CachedCubicSymmetryFactory(physical_factory)
    convergence = integrate_registered_grid_convergence(
        cached_factory,
        q,
        theta,
        time,
        k0,
        omega_reference,
        source_radius,
        window_sigma,
        chunk_size=2,
    )
    finest_response_scale = float(
        np.sqrt(np.mean(np.abs(convergence.finest_response.demodulated) ** 2))
    )
    assert convergence.complex_response_errors[1] <= (1.0e-8 + 0.05 * finest_response_scale)
    assert convergence.complex_response_errors[1] / convergence.complex_response_errors[0] <= 0.5
    assert np.max(convergence.accumulated_phase_errors) <= 0.05

    contributions = []
    for point in points:
        direction = np.array([np.cos(point.theta), np.sin(point.theta)])
        radial_direction = -1 if point.radius < k0 else 1
        node = physical_factory(point.theta, radial_direction)(point.radius * direction)
        taper = float(
            green_response.compact_radial_taper(
                [point.radius - k0],
                window_sigma,
            )[0]
        )
        node = BranchNodeSample(
            node.omega,
            taper * node.amplitude,
            node.frequency_uncertainty,
            taper * node.amplitude_uncertainty,
            node.relative_eigengap,
        )
        contributions.append(
            build_morse_contribution(
                point,
                node,
                k0,
                source_radius,
                window_sigma,
            )
        )
    assert all(abs(contribution.amplitude) > 0.0 for contribution in contributions)

    def forbidden_bessel(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fixed-epsilon Morse response called a Bessel kernel")

    monkeypatch.setattr("zgv_morse.asymptotics.j0", forbidden_bessel)
    monkeypatch.setattr("zgv_morse.asymptotics.jv", forbidden_bessel)
    morse = morse_stationary_phase_response(
        time,
        contributions,
        omega_reference,
    )
    direct = convergence.finest_response.demodulated

    # The cancellation mask is derived only from the precomputed Morse phasor
    # coherence and a predeclared threshold, never from the direct residual.
    stationary_coefficients = []
    for contribution in contributions:
        eigenvalues = np.linalg.eigvalsh(contribution.hessian)
        signature_value = int(
            np.count_nonzero(eigenvalues > 0.0) - np.count_nonzero(eigenvalues < 0.0)
        )
        stationary_coefficients.append(
            contribution.amplitude
            * np.exp(-0.25j * np.pi * signature_value)
            / np.sqrt(abs(np.prod(eigenvalues)))
        )
    phasor_sum = sum(
        coefficient * np.exp(-1j * (contribution.omega - omega_reference) * time)
        for coefficient, contribution in zip(
            stationary_coefficients,
            contributions,
            strict=True,
        )
    )
    coherence = np.abs(phasor_sum) / sum(map(abs, stationary_coefficients))
    cancellation_mask = np.zeros(time.shape, dtype=np.bool_)
    for start, stop in _FIXED_EPSILON_CANCELLATION_RANGES:
        cancellation_mask[start:stop] = True
    assert_allclose(time, 1500.0 + 25.0 * np.arange(time.size), atol=0.0)
    assert np.array_equal(cancellation_mask, coherence < 0.30)
    comparison_mask = ~cancellation_mask
    assert 0.0 < np.mean(cancellation_mask) < 0.25

    complex_error = direct - morse.demodulated
    comparison_scale = np.sqrt(np.mean(np.abs(morse.demodulated[comparison_mask]) ** 2))
    relative_rms_error = (
        np.sqrt(np.mean(np.abs(complex_error[comparison_mask]) ** 2)) / comparison_scale
    )
    relative_max_error = np.max(np.abs(complex_error[comparison_mask])) / comparison_scale
    cancellation_absolute_rms = (
        np.sqrt(np.mean(np.abs(complex_error[cancellation_mask]) ** 2)) / comparison_scale
    )
    assert relative_rms_error <= 0.055
    assert relative_max_error <= 0.17
    assert cancellation_absolute_rms <= 0.09

    # A beat-period RMS removes the signed minimum/saddle modulation without
    # fitting the stationary-phase amplitude, phase, frequency, or time origin.
    modulation_period = 2.0 * np.pi / (maximum_frequency - minimum_frequency)
    rms_centers: list[float] = []
    direct_rms: list[float] = []
    left = float(time[0])
    while left + modulation_period <= float(time[-1]):
        in_window = (time >= left) & (time < left + modulation_period)
        rms_centers.append(float(np.exp(np.mean(np.log(time[in_window])))))
        direct_rms.append(float(np.sqrt(np.mean(np.abs(direct[in_window]) ** 2))))
        left += modulation_period
    assert len(rms_centers) == 3
    fixed_epsilon_rms_slope = float(np.polyfit(np.log(rms_centers), np.log(direct_rms), 1)[0])
    assert abs(fixed_epsilon_rms_slope + 1.0) <= 0.05
    compensated_rms = np.multiply(rms_centers, direct_rms)
    assert np.ptp(compensated_rms) / np.mean(compensated_rms) <= 0.05

    forbidden_fit_parameters = {
        "fitted_amplitude",
        "fitted_phase",
        "fitted_frequency",
        "fitted_time_shift",
    }
    assert all(getattr(morse, name) is None for name in forbidden_fit_parameters)
    public_parameters = signature(morse_stationary_phase_response).parameters
    assert not ({"epsilon", "V4"} | forbidden_fit_parameters).intersection(public_parameters)
