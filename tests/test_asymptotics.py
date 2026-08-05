from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import j0, j1, jv

from zgv_morse.asymptotics import (
    DemodulatedAsymptoticResponse,
    MorseContribution,
    UniformParameters,
    bessel_overlap_is_valid,
    build_morse_contribution,
    critical_frequency_separation,
    crossover_time,
    morse_stationary_phase_response,
    scale_transition_response,
    signed_modulation_rate,
    uniform_bessel_response,
    uniform_prefactor,
    weighted_bessel_kernel,
)
from zgv_morse.critical_points import CriticalPoint
from zgv_morse.green_response import BranchNodeSample


def _parameters(**updates: object) -> UniformParameters:
    values: dict[str, object] = {
        "omega0": 5.0,
        "k0": 2.0,
        "curvature": 3.0,
        "epsilon": 0.02,
        "V0": 0.1,
        "V4": -0.5,
        "amplitude": 0.7 + 0.2j,
    }
    values.update(updates)
    return UniformParameters(**values)  # type: ignore[arg-type]


def test_uniform_response_scales_exactly_to_j0() -> None:
    parameters = _parameters()
    time = np.linspace(20.0, 500.0, 101)

    assert_allclose(
        scale_transition_response(time, uniform_bessel_response(time, parameters), parameters),
        j0(parameters.epsilon * parameters.V4 * time),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_uniform_prefactor_retains_full_fourier_normalization() -> None:
    parameters = _parameters()
    expected = (
        (2.0 * np.pi) ** -2
        * parameters.amplitude
        * parameters.k0
        * np.exp(-0.25j * np.pi)
        * np.sqrt(2.0 * np.pi / parameters.curvature)
        * 2.0
        * np.pi
    )

    assert uniform_prefactor(parameters) == pytest.approx(expected, rel=2.0e-15)


def test_only_fourfold_compatible_weights_survive() -> None:
    tau = np.linspace(-3.0, 3.0, 31)

    assert_allclose(weighted_bessel_kernel(tau, {0: 1.0}), j0(tau), atol=1.0e-14)
    assert_allclose(
        weighted_bessel_kernel(tau, {4: 0.5, -4: 0.5}),
        -1j * j1(tau),
        atol=1.0e-14,
    )
    assert_allclose(weighted_bessel_kernel(tau, {2: 0.5, -2: 0.5}), 0.0, atol=1.0e-14)


def test_weighted_kernel_preserves_complex_fourfold_coefficients() -> None:
    tau = np.linspace(-2.0, 2.0, 17)
    coefficients = {0: 0.3 - 0.4j, 4: 0.2 + 0.7j, -8: -0.1 + 0.5j}
    expected = (
        coefficients[0] * j0(tau)
        + coefficients[4] * (-1j) ** -1 * jv(-1, tau)
        + coefficients[-8] * (-1j) ** 2 * jv(2, tau)
    )
    assert_allclose(weighted_bessel_kernel(tau, coefficients), expected, atol=2.0e-14)


def test_small_tau_expansion_has_the_predicted_quadratic_coefficient() -> None:
    tau = np.array([-2.0e-3, -1.0e-3, 0.0, 1.0e-3, 2.0e-3])
    quadratic = 1.0 - 0.25 * tau**2

    error = weighted_bessel_kernel(tau, {0: 1.0}).real - quadratic
    assert_allclose(error, tau**4 / 64.0, rtol=2.0e-6, atol=2.0e-16)


def test_large_tau_leading_asymptotic_error_decreases_away_from_zeros() -> None:
    tau = 2.0 * np.pi * np.array([20.0, 40.0, 80.0]) + np.pi / 4.0
    leading = np.sqrt(2.0 / (np.pi * tau)) * np.cos(tau - np.pi / 4.0)
    relative_error = np.abs(j0(tau) - leading) / np.abs(j0(tau))

    assert np.all(relative_error[1:] < 0.51 * relative_error[:-1])


def test_frequency_separation_is_twice_the_modulation_rate() -> None:
    epsilon, V4 = -0.03, 0.7
    rate = signed_modulation_rate(epsilon, V4)

    assert rate == pytest.approx(abs(epsilon * V4))
    assert critical_frequency_separation(epsilon, V4) == pytest.approx(2.0 * rate)
    assert crossover_time(epsilon, V4) == pytest.approx(1.0 / rate)


@pytest.mark.parametrize(("epsilon", "V4"), [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
def test_crossover_rejects_an_absent_first_order_splitting(epsilon: float, V4: float) -> None:
    with pytest.raises(ValueError, match="nonzero"):
        crossover_time(epsilon, V4)


def test_crossover_rejects_a_nonrepresentable_finite_time() -> None:
    smallest_positive = np.nextafter(0.0, 1.0)

    with pytest.raises(ValueError, match="finite"):
        crossover_time(smallest_positive, 1.0)


def test_overlap_mask_has_input_shape_and_strict_regime_bounds() -> None:
    time = np.array([10.0, 10.0 + 1.0e-12, 20.0, 25.0])
    mask = bessel_overlap_is_valid(
        time,
        epsilon=0.1,
        V4=1.0,
        second_order_bound=np.array([0.0, 0.0, 0.5, 0.4]),
    )

    assert mask.shape == time.shape
    assert mask.dtype == np.bool_
    assert_allclose(mask, [False, True, False, False])


def test_overlap_converts_finite_intermediate_overflow_to_value_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        bessel_overlap_is_valid(1.0, 1.0e308, 1.0e-308, 0.0)


def test_weighted_kernel_rejects_nonfinite_accumulation() -> None:
    coefficients = {0: 1.7e308, 4: 1.7e308j}

    with pytest.raises(ValueError, match="finite"):
        weighted_bessel_kernel(1.0, coefficients)


def test_uniform_parameters_are_frozen_slotted_and_strictly_validated() -> None:
    parameters = _parameters()
    assert not hasattr(parameters, "__dict__")
    with pytest.raises(FrozenInstanceError):
        parameters.V4 = 1.0  # type: ignore[misc]

    invalid_updates = (
        {"omega0": 0.0},
        {"k0": -1.0},
        {"curvature": np.inf},
        {"epsilon": np.nan},
        {"V0": "0.1"},
        {"V4": True},
        {"amplitude": complex(np.inf, 0.0)},
        {"amplitude": 0.0j},
        {"fourier_normalization": 0.0},
    )
    for updates in invalid_updates:
        with pytest.raises((TypeError, ValueError)):
            _parameters(**updates)


def test_prefactor_requires_positive_radial_curvature() -> None:
    parameters = object.__new__(UniformParameters)
    for field, value in {
        "omega0": 5.0,
        "k0": 2.0,
        "curvature": -3.0,
        "epsilon": 0.02,
        "V0": 0.1,
        "V4": -0.5,
        "amplitude": 0.7 + 0.2j,
        "fourier_normalization": (2.0 * np.pi) ** -2,
    }.items():
        object.__setattr__(parameters, field, value)

    with pytest.raises(ValueError, match="positive radial curvature"):
        uniform_prefactor(parameters)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda p: uniform_bessel_response([1.0, 0.0], p), "positive"),
        (lambda p: uniform_bessel_response([1.0, np.inf], p), "finite"),
        (lambda p: uniform_bessel_response([1.0, 2.0j], p), "real"),
        (lambda p: scale_transition_response([1.0, 2.0], [1.0], p), "shape"),
        (lambda _p: weighted_bessel_kernel([0.0, np.nan], {0: 1.0}), "finite"),
        (lambda _p: weighted_bessel_kernel([0.0], {4.5: 1.0}), "integer"),
        (lambda _p: weighted_bessel_kernel([0.0], {4: np.inf}), "finite"),
        (lambda _p: crossover_time(True, 1.0), "real"),
        (
            lambda _p: bessel_overlap_is_valid([1.0], 0.1, 1.0, -0.1),
            "nonnegative",
        ),
    ],
)
def test_public_functions_reject_malformed_inputs(call, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        call(_parameters())


def test_array_results_do_not_alias_mutable_inputs() -> None:
    time = np.linspace(1.0, 5.0, 7)
    response = uniform_bessel_response(time, _parameters())
    kernel = weighted_bessel_kernel(time, {0: 1.0})
    overlap = bessel_overlap_is_valid(time, 0.2, 1.0, 0.01)

    assert not np.shares_memory(response, time)
    assert not np.shares_memory(kernel, time)
    assert not np.shares_memory(overlap, time)
    before = (response.copy(), kernel.copy(), overlap.copy())
    time[:] = 99.0
    assert_allclose(response, before[0])
    assert_allclose(kernel, before[1])
    assert_allclose(overlap, before[2])


def _morse_contribution(
    hessian: np.ndarray,
    *,
    omega: float = 5.0,
    amplitude: complex = 1.0,
    frequency_uncertainty: float = 0.0,
    hessian_uncertainty: float = 0.0,
) -> MorseContribution:
    return MorseContribution(
        omega,
        hessian,
        amplitude,
        frequency_uncertainty,
        hessian_uncertainty,
    )


def test_morse_signature_phases_cover_minimum_saddle_and_maximum() -> None:
    time = np.array([100.0, 200.0])
    cases = (
        (np.diag([2.0, 3.0]), -1j),
        (np.diag([2.0, -3.0]), 1.0),
        (np.diag([-2.0, -3.0]), 1j),
    )

    for hessian, signature_phase in cases:
        response = morse_stationary_phase_response(
            time,
            [_morse_contribution(hessian)],
            5.0,
        )
        assert_allclose(
            response.demodulated * time,
            signature_phase / (2.0 * np.pi * np.sqrt(6.0)),
            rtol=0.0,
            atol=2.0e-16,
        )


def test_morse_response_respects_normalization_and_rotated_hessian() -> None:
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    diagonal = np.diag([1.7, -2.4])
    time = np.array([50.0, 80.0])
    normalization = 0.031

    reference = morse_stationary_phase_response(
        time,
        [_morse_contribution(diagonal, amplitude=0.4 - 0.2j)],
        5.0,
        fourier_normalization=normalization,
    )
    rotated = morse_stationary_phase_response(
        time,
        [_morse_contribution(rotation @ diagonal @ rotation.T, amplitude=0.4 - 0.2j)],
        5.0,
        fourier_normalization=normalization,
    )

    assert_allclose(rotated.demodulated, reference.demodulated, atol=3.0e-17)
    expected = normalization * 2.0 * np.pi * (0.4 - 0.2j) / (
        time * np.sqrt(abs(np.linalg.det(diagonal)))
    )
    assert_allclose(reference.demodulated, expected, rtol=2.0e-15)


@pytest.mark.parametrize("eigenvalues", [(2.0, 3.0), (2.0, -3.0)])
def test_morse_asymptotic_converges_to_exact_gaussian_quadratic_integral(
    eigenvalues: tuple[float, float],
) -> None:
    alpha = 0.7
    time = np.array([50.0, 100.0, 200.0, 400.0])
    hessian = np.diag(eigenvalues)
    response = morse_stationary_phase_response(
        time,
        [_morse_contribution(hessian)],
        5.0,
    )
    factors = np.stack(
        [
            np.sqrt(alpha + 0.5j * time * eigenvalue)
            for eigenvalue in eigenvalues
        ]
    )
    exact = (2.0 * np.pi) ** -2 * np.pi / np.prod(factors, axis=0)
    relative_error = np.abs(response.demodulated - exact) / np.abs(exact)

    assert np.all(relative_error[1:] < 0.51 * relative_error[:-1])


def test_morse_response_enforces_hessian_resolution_and_phase_budget() -> None:
    unresolved = _morse_contribution(
        np.diag([1.0, 2.0]),
        hessian_uncertainty=0.1,
    )
    with pytest.raises(ValueError, match="Morse Hessian is not resolved"):
        morse_stationary_phase_response([100.0], [unresolved], 5.0)

    accepted = _morse_contribution(
        np.eye(2),
        frequency_uncertainty=4.0e-4,
    )
    rejected = _morse_contribution(
        np.eye(2),
        frequency_uncertainty=6.0e-4,
    )
    accepted_response = morse_stationary_phase_response([100.0], [accepted], 5.0)
    assert accepted_response.maximum_accumulated_phase_error == pytest.approx(0.04)
    with pytest.raises(ValueError, match="accumulated phase"):
        morse_stationary_phase_response([100.0], [rejected], 5.0)


def test_morse_contribution_builder_uses_cartesian_amplitude_without_jacobian() -> None:
    point = CriticalPoint(
        2.0,
        0.0,
        2.0,
        0.0,
        5.0,
        0.0,
        1.0e-10,
        np.diag([-1.0, 2.0]),
        np.array([-1.0, 2.0]),
        1.0e-6,
        "saddle",
        -1,
    )
    node = BranchNodeSample(5.0 + 1.0e-8, 2.0j, 2.0e-8, 3.0e-5)
    source_radius = 0.5
    k0 = 2.1
    window_sigma = 0.3

    contribution = build_morse_contribution(
        point,
        node,
        k0,
        source_radius,
        window_sigma,
    )
    weight = np.exp(-0.25 * source_radius**2 * point.radius**2) * np.exp(
        -((point.radius - k0) / window_sigma) ** 8
    )

    assert contribution.omega == node.omega
    assert contribution.amplitude == pytest.approx(node.amplitude * weight)
    assert contribution.amplitude != pytest.approx(
        node.amplitude * weight * point.radius
    )
    assert contribution.frequency_uncertainty == node.frequency_uncertainty
    assert contribution.hessian_uncertainty == point.hessian_uncertainty
    assert contribution.amplitude_uncertainty == pytest.approx(
        node.amplitude_uncertainty * weight
    )

    inconsistent = BranchNodeSample(5.1, 2.0j, 2.0e-8, 3.0e-5)
    with pytest.raises(ValueError, match="frequencies are inconsistent"):
        build_morse_contribution(
            point,
            inconsistent,
            k0,
            source_radius,
            window_sigma,
        )


def test_morse_records_are_defensive_and_public_inputs_are_strict() -> None:
    hessian = np.diag([2.0, -3.0])
    contribution = _morse_contribution(hessian)
    response = morse_stationary_phase_response([10.0, 20.0], [contribution], 5.0)
    hessian[:] = 99.0

    assert_allclose(contribution.hessian, np.diag([2.0, -3.0]))
    assert not contribution.hessian.flags.writeable
    assert not response.time.flags.writeable
    assert not response.demodulated.flags.writeable
    assert not hasattr(contribution, "__dict__")
    assert not hasattr(response, "__dict__")
    assert isinstance(response, DemodulatedAsymptoticResponse)
    analytic = response.analytic_signal()
    assert not np.shares_memory(analytic, response.demodulated)
    with pytest.raises(FrozenInstanceError):
        contribution.omega = 6.0  # type: ignore[misc]

    invalid_calls = (
        lambda: morse_stationary_phase_response([0.0], [contribution], 5.0),
        lambda: morse_stationary_phase_response([1.0], [], 5.0),
        lambda: morse_stationary_phase_response([1.0], [object()], 5.0),
        lambda: morse_stationary_phase_response(
            [1.0], [contribution], 5.0, fourier_normalization=0.0
        ),
        lambda: _morse_contribution(np.array([[1.0, 1.0], [0.0, 1.0]])),
        lambda: _morse_contribution(np.eye(2), frequency_uncertainty=-1.0),
    )
    for call in invalid_calls:
        with pytest.raises((TypeError, ValueError)):
            call()
