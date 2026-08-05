from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.linalg import eigh

from zgv_morse.mode_tracking import (
    mass_mac,
    seed_tracked_mode,
    symmetric_lamb_parity_score,
    track_mode,
)
from zgv_morse.perturbation import (
    AngularHarmonics,
    CubicHarmonicReport,
    PhysicalCubicShift,
    RadialSensitivity,
    cubic_harmonic_report,
    differentiated_mode,
    extract_angular_harmonics,
    frequency_sensitivity,
    physical_cubic_shift,
    radial_frequency_sensitivity,
    sensitivity_from_plate,
)
from zgv_morse.spectral_plate import (
    WavevectorDerivatives,
    assemble_plate_matrices,
    assemble_wavevector_derivatives,
    solve_plate_modes,
)
from zgv_morse.elasticity import cubic_family, isotropic_tensor


def test_frequency_sensitivity_is_scale_and_phase_invariant() -> None:
    mode = 3.0 * np.exp(0.37j) * np.array([1.0, 0.0])
    value = frequency_sensitivity(
        omega=2.0,
        vector=mode,
        mass=np.eye(2),
        k_epsilon=np.array([[0.7, -0.3], [-0.3, 0.2]]),
    )
    assert_allclose(value, 0.7 / 4.0, rtol=1.0e-14, atol=1.0e-14)


def test_radial_sensitivity_includes_differentiated_mode() -> None:
    omega, other, beta, h12 = 2.0, 9.0, 0.4, -0.3
    stiffness = np.diag([omega**2, other]).astype(complex)
    k_radial = beta * np.array(
        [[0.0, omega**2 - other], [omega**2 - other, 0.0]],
        complex,
    )
    result = radial_frequency_sensitivity(
        stiffness=stiffness,
        mass=np.eye(2, dtype=complex),
        omega=omega,
        vector=np.array([1.0, 0.0], complex),
        k_radial=k_radial,
        k_epsilon=np.array([[0.7, h12], [h12, 0.2]], complex),
        k_radial_epsilon=np.zeros((2, 2), complex),
    )
    assert_allclose(result.mode_radial_derivative, [0.0, beta], atol=1.0e-13)
    assert_allclose(result.B, beta * h12 / omega, atol=1.0e-13)
    assert result.differentiated_mode_residual < 1.0e-13


def test_nonzero_radial_eigenvalue_derivative_enters_frequency_denominator() -> None:
    result = radial_frequency_sensitivity(
        stiffness=np.diag([4.0, 9.0]),
        mass=np.eye(2),
        omega=2.0,
        vector=np.array([1.0, 0.0]),
        k_radial=np.diag([0.8, 0.1]),
        k_epsilon=np.diag([0.6, -0.2]),
        k_radial_epsilon=np.diag([-0.2, 0.3]),
    )

    expected = -0.2 / 4.0 - 0.6 * 0.8 / (4.0 * 2.0**3)
    assert result.lambda_radial == pytest.approx(0.8)
    assert result.lambda_epsilon == pytest.approx(0.6)
    assert result.lambda_radial_epsilon == pytest.approx(-0.2)
    assert result.B == pytest.approx(expected)


def test_B_is_invariant_to_mode_scale_and_phase() -> None:
    stiffness = np.diag([4.0, 9.0]).astype(complex)
    radial = np.array([[0.3, -0.7], [-0.7, 0.1]], complex)
    perturbation = np.array([[0.6, 0.25], [0.25, -0.2]], complex)
    mixed = np.array([[0.11, -0.04], [-0.04, 0.07]], complex)
    reference = radial_frequency_sensitivity(
        stiffness,
        np.eye(2),
        2.0,
        np.array([1.0, 0.0]),
        radial,
        perturbation,
        mixed,
    )
    scaled = radial_frequency_sensitivity(
        stiffness,
        np.eye(2),
        2.0,
        7.0 * np.exp(0.83j) * np.array([1.0, 0.0]),
        radial,
        perturbation,
        mixed,
    )

    assert scaled.V == pytest.approx(reference.V, rel=1.0e-14, abs=1.0e-14)
    assert scaled.B == pytest.approx(reference.B, rel=1.0e-14, abs=1.0e-14)
    assert_allclose(
        abs(scaled.mode_radial_derivative),
        abs(reference.mode_radial_derivative),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_V_and_B_match_independent_generalized_eigenvalue_finite_differences() -> None:
    mass = np.diag([1.2, 0.8, 1.5])
    stiffness = mass @ np.diag([4.0, 9.0, 16.0])
    radial = np.array([[0.45, -0.55, 0.12], [-0.55, 0.2, -0.18], [0.12, -0.18, -0.1]])
    perturbation = np.array([[0.72, 0.31, -0.22], [0.31, -0.15, 0.27], [-0.22, 0.27, 0.08]])
    mixed = np.array([[-0.16, 0.09, 0.03], [0.09, 0.12, -0.04], [0.03, -0.04, 0.05]])
    vector = np.array([1.0 / np.sqrt(1.2), 0.0, 0.0])
    analytic = radial_frequency_sensitivity(
        stiffness,
        mass,
        2.0,
        vector,
        radial,
        perturbation,
        mixed,
    )

    def frequency(kappa: float, epsilon: float) -> float:
        matrix = stiffness + kappa * radial + epsilon * perturbation + kappa * epsilon * mixed
        eigenvalues = eigh(matrix, mass, eigvals_only=True, check_finite=True)
        return float(np.sqrt(eigenvalues[0]))

    epsilon_step = 2.0e-5
    finite_V = (frequency(0.0, epsilon_step) - frequency(0.0, -epsilon_step)) / (2.0 * epsilon_step)

    def mixed_difference(step: float) -> float:
        return (
            frequency(step, step)
            - frequency(step, -step)
            - frequency(-step, step)
            + frequency(-step, -step)
        ) / (4.0 * step**2)

    coarse = mixed_difference(1.0e-3)
    fine = mixed_difference(5.0e-4)
    assert abs(fine - analytic.B) < abs(coarse - analytic.B)
    assert finite_V == pytest.approx(analytic.V, rel=2.0e-8, abs=2.0e-9)
    assert fine == pytest.approx(analytic.B, rel=2.0e-6, abs=2.0e-7)


def test_differentiated_mode_obeys_gauge_and_equation() -> None:
    stiffness = np.diag([4.0, 9.0, 16.0])
    parameter = np.array([[0.3, -0.4, 0.2], [-0.4, 0.1, -0.3], [0.2, -0.3, -0.2]])
    derivative, lambda_parameter, residual = differentiated_mode(
        stiffness,
        np.eye(3),
        4.0,
        np.array([1.0, 0.0, 0.0]),
        parameter,
    )

    assert lambda_parameter == pytest.approx(0.3)
    assert abs(derivative[0]) < 1.0e-14
    assert residual < 1.0e-13
    assert not derivative.flags.writeable


def test_simple_low_unit_eigenvalues_are_not_misclassified_as_degenerate() -> None:
    stiffness = np.diag([4.0e-12, 5.0e-12])
    parameter = np.array([[0.3e-12, -0.4e-12], [-0.4e-12, 0.1e-12]])

    derivative, lambda_parameter, residual = differentiated_mode(
        stiffness,
        np.eye(2),
        4.0e-12,
        [1.0, 0.0],
        parameter,
    )

    assert_allclose(derivative, [0.0, 0.4], rtol=1.0e-13, atol=1.0e-13)
    assert lambda_parameter == pytest.approx(0.3e-12)
    assert residual < 1.0e-13


def test_common_pencil_scaling_preserves_V_and_B() -> None:
    base = None
    for scale in (1.0e-12, 1.0, 1.0e12):
        result = radial_frequency_sensitivity(
            scale * np.diag([4.0, 9.0]),
            scale * np.eye(2),
            2.0,
            [1.0, 0.0],
            scale * np.array([[0.3, -0.7], [-0.7, 0.1]]),
            scale * np.array([[0.6, 0.25], [0.25, -0.2]]),
            scale * np.array([[0.11, -0.04], [-0.04, 0.07]]),
        )
        if base is None:
            base = result
        else:
            assert result.V == pytest.approx(base.V, rel=2.0e-14, abs=2.0e-14)
            assert result.B == pytest.approx(base.B, rel=2.0e-14, abs=2.0e-14)


def test_complex_HPD_pencil_matches_mixed_finite_difference() -> None:
    mass = np.array([[2.0, 0.3 + 0.2j], [0.3 - 0.2j, 1.4]], complex)
    cholesky = np.linalg.cholesky(mass)
    basis = np.linalg.solve(cholesky.conj().T, np.eye(2))
    stiffness = mass @ basis @ np.diag([4.0, 9.0]) @ basis.conj().T @ mass
    vector = basis[:, 0]
    radial = np.array([[0.4, -0.2 + 0.3j], [-0.2 - 0.3j, -0.1]], complex)
    perturbation = np.array([[0.6, 0.25 - 0.15j], [0.25 + 0.15j, 0.2]], complex)
    mixed = np.array([[-0.12, 0.07 + 0.04j], [0.07 - 0.04j, 0.08]], complex)
    analytic = radial_frequency_sensitivity(
        stiffness,
        mass,
        2.0,
        vector,
        radial,
        perturbation,
        mixed,
    )

    def frequency(kappa: float, epsilon: float) -> float:
        matrix = stiffness + kappa * radial + epsilon * perturbation + kappa * epsilon * mixed
        return float(np.sqrt(eigh(matrix, mass, eigvals_only=True)[0]))

    step = 8.0e-4
    finite_B = (
        frequency(step, step)
        - frequency(step, -step)
        - frequency(-step, step)
        + frequency(-step, -step)
    ) / (4.0 * step**2)
    assert finite_B == pytest.approx(analytic.B, rel=2.0e-6, abs=2.0e-7)
    assert analytic.differentiated_mode_residual < 1.0e-13


def test_plate_adapter_matches_direct_matrix_call() -> None:
    stiffness_tensor = isotropic_tensor(2.0, 1.0)
    kappa = 0.8042173193715181
    base = assemble_plate_matrices(kappa, 0.0, stiffness_tensor, 1.0, 1.0, order=8)
    derivatives = assemble_wavevector_derivatives(
        kappa,
        0.0,
        stiffness_tensor,
        1.0,
        1.0,
        order=8,
    )
    modes = solve_plate_modes(base, 12)
    tracked = seed_tracked_mode(modes, int(np.argmin(abs(modes.omega - 2.85175877496009))))
    scale = 0.025
    perturbation = replace(base, stiffness=scale * base.stiffness)
    perturbation_derivatives = WavevectorDerivatives(
        dkx=scale * derivatives.dkx,
        dky=scale * derivatives.dky,
        dkx2=scale * derivatives.dkx2,
        dkx_dky=scale * derivatives.dkx_dky,
        dky2=scale * derivatives.dky2,
    )
    theta = 0.31
    adapted = sensitivity_from_plate(
        tracked,
        base,
        derivatives,
        perturbation,
        perturbation_derivatives,
        theta,
    )
    radial = np.cos(theta) * derivatives.dkx + np.sin(theta) * derivatives.dky
    radial_mixed = (
        np.cos(theta) * perturbation_derivatives.dkx + np.sin(theta) * perturbation_derivatives.dky
    )
    direct = radial_frequency_sensitivity(
        base.stiffness,
        base.mass,
        tracked.omega,
        tracked.vector,
        radial,
        perturbation.stiffness,
        radial_mixed,
    )

    for field in (
        "V",
        "B",
        "lambda_epsilon",
        "lambda_radial",
        "lambda_radial_epsilon",
        "differentiated_mode_residual",
    ):
        assert getattr(adapted, field) == pytest.approx(getattr(direct, field))
    assert_allclose(adapted.mode_radial_derivative, direct.mode_radial_derivative)

    with pytest.raises(ValueError, match="fixed-density"):
        sensitivity_from_plate(
            tracked,
            base,
            derivatives,
            replace(perturbation, mass=1.01 * base.mass),
            perturbation_derivatives,
            theta,
        )
    with pytest.raises(ValueError, match="nodes"):
        sensitivity_from_plate(
            tracked,
            base,
            derivatives,
            replace(perturbation, nodes=perturbation.nodes + 1.0e-3),
            perturbation_derivatives,
            theta,
        )
    with pytest.raises(ValueError, match="theta"):
        sensitivity_from_plate(
            tracked,
            base,
            derivatives,
            perturbation,
            perturbation_derivatives,
            np.nan,
        )


def test_radial_sensitivity_record_is_frozen_and_defensive() -> None:
    derivative = np.array([0.0, 0.2j])
    record = RadialSensitivity(0.1, -0.2, 0.4, 0.0, -0.8, derivative, 1.0e-12)
    derivative[:] = 9.0

    assert_allclose(record.mode_radial_derivative, [0.0, 0.2j])
    assert not record.mode_radial_derivative.flags.writeable
    with pytest.raises(Exception):
        record.B = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mass", "message"),
    [
        (np.zeros((2, 2)), "positive definite"),
        (np.array([[1.0, 0.2], [0.0, 1.0]]), "Hermitian"),
    ],
)
def test_sensitivity_rejects_invalid_mass(mass: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        frequency_sensitivity(2.0, [1.0, 0.0], mass, np.eye(2))


def test_sensitivity_rejects_invalid_frequency_vector_and_matrix() -> None:
    with pytest.raises(ValueError, match="positive"):
        frequency_sensitivity(0.0, [1.0, 0.0], np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="mass norm"):
        frequency_sensitivity(2.0, [0.0, 0.0], np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="Hermitian"):
        frequency_sensitivity(2.0, [1.0, 0.0], np.eye(2), [[1.0, 1.0], [0.0, 1.0]])


def test_differentiated_mode_rejects_inaccurate_or_degenerate_eigenpairs() -> None:
    with pytest.raises(ValueError, match="eigenpair"):
        differentiated_mode(np.diag([4.0, 9.0]), np.eye(2), 5.0, [1.0, 0.0], np.eye(2))
    with pytest.raises(ValueError, match="simple"):
        differentiated_mode(np.diag([4.0, 4.0]), np.eye(2), 4.0, [1.0, 0.0], np.eye(2))


def test_plate_adapter_rejects_invalid_types_and_angle() -> None:
    with pytest.raises(TypeError, match="mode"):
        sensitivity_from_plate(object(), object(), object(), object(), object(), 0.0)  # type: ignore[arg-type]


def test_fourier_extraction_recovers_constant_cos4_and_cos8() -> None:
    theta = 2.0 * np.pi * np.arange(64) / 64
    values = 1.25 - 0.30 * np.cos(4.0 * theta) + 0.02 * np.cos(8.0 * theta)

    spectrum = extract_angular_harmonics(theta, values, max_order=12)

    assert_allclose(spectrum.order, np.arange(13))
    assert_allclose(spectrum.cosine[[0, 4, 8]], [1.25, -0.30, 0.02], atol=1.0e-14)
    omitted = np.delete(spectrum.cosine, [0, 4, 8])
    assert_allclose(omitted, 0.0, atol=1.0e-14)
    assert_allclose(spectrum.sine, 0.0, atol=1.0e-14)


def test_cubic_report_has_only_the_proved_first_order_fourfold_term() -> None:
    theta = 2.0 * np.pi * np.arange(128) / 128
    values = 0.17 + 0.043 * np.cos(4.0 * theta)

    report = cubic_harmonic_report(theta, values)

    assert_allclose([report.V0, report.V4, report.V8], [0.17, 0.043, 0.0], atol=1.0e-14)
    assert_allclose(report.reconstruction, values, atol=1.0e-14)
    assert report.periodicity_defect < 1.0e-14
    assert report.mirror_defect < 1.0e-14
    assert report.non_cubic_leakage < 1.0e-13


def test_cubic_mirror_defect_is_correct_on_a_half_step_grid() -> None:
    count = 64
    theta = 2.0 * np.pi * (np.arange(count) + 0.5) / count
    values = 0.17 + 0.043 * np.cos(4.0 * theta)

    report = cubic_harmonic_report(theta, values)

    assert report.mirror_defect < 1.0e-14
    assert report.periodicity_defect < 1.0e-14
    assert_allclose([report.V0, report.V4], [0.17, 0.043], atol=1.0e-14)


def test_constant_cubic_signal_has_stable_zero_leakage() -> None:
    theta = 2.0 * np.pi * np.arange(64) / 64
    report = cubic_harmonic_report(theta, np.full(64, 0.17))

    assert report.non_cubic_leakage == 0.0
    assert report.mirror_defect == 0.0
    assert report.periodicity_defect == 0.0


def test_physical_shift_counts_epsilon_once() -> None:
    epsilon = np.array([-0.02, 0.01, 0.04])

    shift = physical_cubic_shift(epsilon, delta=1.7, V4=-0.12 * 1.7)

    assert_allclose(shift.delta_c, epsilon * 1.7)
    assert_allclose(shift.frequency_shift / shift.delta_c, -0.12)


def test_angular_records_are_frozen_slotted_and_defensive() -> None:
    theta = 2.0 * np.pi * np.arange(64) / 64
    values = 0.3 + 0.2 * np.cos(4.0 * theta)
    epsilon = np.array([-0.02, 0.01])

    spectrum = extract_angular_harmonics(theta, values, 12)
    report = cubic_harmonic_report(theta, values, 12)
    shift = physical_cubic_shift(epsilon, 1.5, -0.08)
    values[:] = 9.0
    epsilon[:] = 9.0

    for record in (spectrum, report, shift):
        assert not hasattr(record, "__dict__")
        with pytest.raises(Exception):
            record.extra = 1.0  # type: ignore[attr-defined]
    for array in (
        spectrum.order,
        spectrum.cosine,
        spectrum.sine,
        report.reconstruction,
        shift.delta_c,
        shift.frequency_shift,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0.0
    assert_allclose(report.reconstruction, 0.3 + 0.2 * np.cos(4.0 * theta), atol=1.0e-14)
    assert_allclose(shift.delta_c, [-0.03, 0.015])


@pytest.mark.parametrize(
    ("theta", "values", "max_order", "message"),
    [
        pytest.param(
            np.linspace(0.0, 2.0 * np.pi, 64),
            np.ones(64),
            12,
            "endpoint-free",
            id="duplicated-endpoint",
        ),
        pytest.param(
            np.r_[2.0 * np.pi * np.arange(63) / 64, 1.99 * np.pi],
            np.ones(64),
            12,
            "uniform",
            id="nonuniform",
        ),
        pytest.param(
            2.0 * np.pi * np.arange(24) / 24,
            np.ones(24),
            12,
            "resolve",
            id="underresolved",
        ),
        pytest.param(
            np.r_[np.nan, 2.0 * np.pi * np.arange(1, 64) / 64],
            np.ones(64),
            12,
            "finite",
            id="nonfinite-theta",
        ),
        pytest.param(
            2.0 * np.pi * np.arange(64) / 64,
            np.r_[np.inf, np.ones(63)],
            12,
            "finite",
            id="nonfinite-values",
        ),
    ],
)
def test_fourier_extraction_rejects_invalid_grid_and_values(
    theta: np.ndarray,
    values: np.ndarray,
    max_order: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        extract_angular_harmonics(theta, values, max_order)


@pytest.mark.parametrize(
    ("theta", "values", "message"),
    [
        pytest.param(
            2.0 * np.pi * np.arange(18) / 18, np.ones(18), "divisible by four", id="quarter-turn"
        ),
        pytest.param(2.0 * np.pi * np.arange(64) / 64, np.ones(63), "same shape", id="shape"),
    ],
)
def test_cubic_report_rejects_incompatible_samples(
    theta: np.ndarray,
    values: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        cubic_harmonic_report(theta, values, max_order=8)
    with pytest.raises(ValueError, match="order 8"):
        cubic_harmonic_report(2.0 * np.pi * np.arange(64) / 64, np.ones(64), max_order=7)


@pytest.mark.parametrize(
    ("epsilon", "delta", "V4", "message"),
    [
        pytest.param([0.0, np.nan], 1.0, 0.2, "epsilon", id="epsilon-nonfinite"),
        pytest.param([0.0, 0.1j], 1.0, 0.2, "epsilon", id="epsilon-complex"),
        pytest.param([0.0], 0.0, 0.2, "delta", id="delta-zero"),
        pytest.param([0.0], np.inf, 0.2, "delta", id="delta-nonfinite"),
        pytest.param([0.0], 1.0, np.nan, "V4", id="V4-nonfinite"),
    ],
)
def test_physical_cubic_shift_rejects_invalid_inputs(
    epsilon: object,
    delta: float,
    V4: float,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        physical_cubic_shift(epsilon, delta, V4)


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(
            lambda: AngularHarmonics(np.arange(2), np.array([0.0, np.nan]), np.zeros(2)),
            id="harmonics",
        ),
        pytest.param(
            lambda: CubicHarmonicReport(0.0, 0.0, 0.0, [np.inf], 0.0, 0.0, 0.0),
            id="report",
        ),
        pytest.param(
            lambda: PhysicalCubicShift(np.zeros(2), np.zeros(3)),
            id="shift",
        ),
    ],
)
def test_angular_record_constructors_validate_arrays(record: object) -> None:
    with pytest.raises(ValueError):
        record()  # type: ignore[operator]


@pytest.mark.slow
def test_cubic_V_and_B_match_full_spectral_finite_differences() -> None:
    k0 = 0.8042173193715181
    omega0 = 2.8517587749600901
    order = 10
    num_modes = 12
    theta = 2.0 * np.pi * np.arange(64) / 64
    base_tensor = isotropic_tensor(2.0, 1.0)
    auxiliary_eta = 0.25
    plus_auxiliary = cubic_family(2.0, 1.0, 1.0, auxiliary_eta)[0]
    minus_auxiliary = cubic_family(2.0, 1.0, 1.0, -auxiliary_eta)[0]
    derivative_names = ("dkx", "dky", "dkx2", "dkx_dky", "dky2")

    def locked(values: np.ndarray) -> np.ndarray:
        result = np.array(values, dtype=np.complex128, copy=True)
        result.setflags(write=False)
        return result

    analytic_V = np.empty(theta.size)
    analytic_B = np.empty(theta.size)
    references = []
    relative_gaps: list[float] = []
    selected_uncertainties: list[float] = []

    for index, angle in enumerate(theta):
        cosine, sine = np.cos(angle), np.sin(angle)
        kx, ky = k0 * cosine, k0 * sine
        base = assemble_plate_matrices(kx, ky, base_tensor, 1.0, 1.0, order=order)
        base_derivatives = assemble_wavevector_derivatives(
            kx,
            ky,
            base_tensor,
            1.0,
            1.0,
            order=order,
        )
        modes = solve_plate_modes(base, num_modes)
        reference = seed_tracked_mode(modes, int(np.argmin(np.abs(modes.omega - omega0))))
        reference_parity = symmetric_lamb_parity_score(
            reference.vector,
            base.nodes,
            base.mass,
        )
        assert reference_parity > 1.0 - 1.0e-10

        plus = assemble_plate_matrices(
            kx,
            ky,
            plus_auxiliary,
            1.0,
            1.0,
            order=order,
        )
        minus = assemble_plate_matrices(
            kx,
            ky,
            minus_auxiliary,
            1.0,
            1.0,
            order=order,
        )
        plus_derivatives = assemble_wavevector_derivatives(
            kx,
            ky,
            plus_auxiliary,
            1.0,
            1.0,
            order=order,
        )
        minus_derivatives = assemble_wavevector_derivatives(
            kx,
            ky,
            minus_auxiliary,
            1.0,
            1.0,
            order=order,
        )

        # Assembly is linear in C.  The moderate auxiliary step keeps C(+/-eta)
        # stable while obtaining K_epsilon and K_k_epsilon without a fitted model.
        perturbation = replace(
            base,
            stiffness=locked((plus.stiffness - minus.stiffness) / (2.0 * auxiliary_eta)),
            mass=base.mass,
            nodes=base.nodes,
        )
        perturbation_derivatives = WavevectorDerivatives(
            *(
                locked(
                    (getattr(plus_derivatives, name) - getattr(minus_derivatives, name))
                    / (2.0 * auxiliary_eta)
                )
                for name in derivative_names
            )
        )
        sensitivity = sensitivity_from_plate(
            reference,
            base,
            base_derivatives,
            perturbation,
            perturbation_derivatives,
            float(angle),
        )
        analytic_V[index] = sensitivity.V
        analytic_B[index] = sensitivity.B
        references.append((reference, reference_parity, cosine, sine))
        relative_gaps.append(reference.eigengap)
        selected_uncertainties.append(max(reference.residual, np.finfo(np.float64).eps))

    def tracked_frequency(
        radius: float,
        epsilon: float,
        reference_data: tuple[object, ...],
    ) -> float:
        reference, reference_parity, cosine, sine = reference_data
        tensor = cubic_family(2.0, 1.0, 1.0, epsilon)[0]
        matrices = assemble_plate_matrices(
            radius * cosine,
            radius * sine,
            tensor,
            1.0,
            1.0,
            order=order,
        )
        tracked = track_mode(
            reference,
            solve_plate_modes(matrices, num_modes),
            min_mac=0.8,
            predicted_eigenvalue=reference.eigenvalue,
        )
        overlap = mass_mac(reference.vector, tracked.vector, matrices.mass)
        parity = symmetric_lamb_parity_score(
            tracked.vector,
            matrices.nodes,
            matrices.mass,
        )
        assert overlap > 0.95
        assert parity * reference_parity > 1.0 - 1.0e-10
        relative_gaps.append(tracked.eigengap)
        # The normalized selected eigenpair residual is the relative eigenvalue
        # uncertainty proxy used in the gap audit.
        selected_uncertainties.append(max(tracked.residual, np.finfo(np.float64).eps))
        return tracked.omega

    epsilon_steps = np.array([0.04, 0.02, 0.01, 0.005])
    analytic_report = cubic_harmonic_report(theta, analytic_V, max_order=16)
    V4_errors: list[float] = []
    B_errors: list[float] = []
    finite_V_values: list[np.ndarray] = []
    finite_B_values: list[np.ndarray] = []

    for epsilon_step in epsilon_steps:
        finite_V = np.empty(theta.size)
        finite_B = np.empty(theta.size)
        radial_step = float(epsilon_step)
        for index, reference_data in enumerate(references):
            omega_plus = tracked_frequency(k0, float(epsilon_step), reference_data)
            omega_minus = tracked_frequency(k0, -float(epsilon_step), reference_data)
            finite_V[index] = (omega_plus - omega_minus) / (2.0 * epsilon_step)

            plus_plus = tracked_frequency(
                k0 + radial_step,
                float(epsilon_step),
                reference_data,
            )
            plus_minus = tracked_frequency(
                k0 + radial_step,
                -float(epsilon_step),
                reference_data,
            )
            minus_plus = tracked_frequency(
                k0 - radial_step,
                float(epsilon_step),
                reference_data,
            )
            minus_minus = tracked_frequency(
                k0 - radial_step,
                -float(epsilon_step),
                reference_data,
            )
            finite_B[index] = (plus_plus - plus_minus - minus_plus + minus_minus) / (
                4.0 * radial_step * epsilon_step
            )

        finite_report = cubic_harmonic_report(theta, finite_V, max_order=16)
        V4_errors.append(abs(finite_report.V4 - analytic_report.V4))
        B_errors.append(float(np.linalg.norm(finite_B - analytic_B) / np.linalg.norm(analytic_B)))
        finite_V_values.append(finite_V)
        finite_B_values.append(finite_B)

    V4_errors_array = np.asarray(V4_errors)
    B_errors_array = np.asarray(B_errors)
    V4_ratios = V4_errors_array[:-1] / V4_errors_array[1:]
    B_ratios = B_errors_array[:-1] / B_errors_array[1:]
    assert np.count_nonzero((2.5 < V4_ratios) & (V4_ratios < 6.0)) >= 2
    assert np.count_nonzero((2.5 < B_ratios) & (B_ratios < 6.0)) >= 2

    resolved_report = cubic_harmonic_report(theta, finite_V_values[-1], max_order=16)
    resolved_relative_error_V4 = abs(resolved_report.V4 - analytic_report.V4) / abs(
        analytic_report.V4
    )
    resolved_relative_error_B = B_errors[-1]
    resolved_full_V_error = float(
        np.linalg.norm(finite_V_values[-1] - analytic_V) / np.linalg.norm(analytic_V)
    )
    minimum_relative_eigengap = min(relative_gaps)
    eigenvalue_uncertainty = max(selected_uncertainties)

    assert resolved_relative_error_V4 < 1.0e-4
    assert resolved_relative_error_B < 1.0e-4
    assert resolved_full_V_error < 1.0e-4
    assert abs(analytic_report.V8) / abs(analytic_report.V4) < 1.0e-8
    assert minimum_relative_eigengap > 10.0 * eigenvalue_uncertainty

    print(
        "cubic spectral diagnostics: "
        f"V0={analytic_report.V0:.16g}, V4={analytic_report.V4:.16g}, "
        f"V8={analytic_report.V8:.3e}, "
        f"B_error={resolved_relative_error_B:.3e}, "
        f"V4_error={resolved_relative_error_V4:.3e}, "
        f"gap={minimum_relative_eigengap:.3e}, "
        f"uncertainty={eigenvalue_uncertainty:.3e}, "
        f"V4_ratios={V4_ratios}, B_ratios={B_ratios}"
    )
