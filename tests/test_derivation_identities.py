from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_analytic_identities import (  # noqa: E402
    angular_bessel_identity_errors,
    bessel_morse_overlap_error,
    check_reference_substitution,
    closed_form_cubic_coefficients,
    cubic_strain_contraction_residual,
    cubic_tensor_decomposition_error,
    exceptional_second_order_splitting,
    frequency_modulation_errors,
    growing_tau_overlap_errors,
    independent_traction_determinant_mp,
    implicit_derivative_formulas,
    large_bessel_phase_errors,
    morse_splitting_data,
    normal_form_hessian_identity,
    oscillator_positive_frequency_residual,
    polar_jacobian_cancellation_error,
    production_eight_point_overlap_error,
    quadratic_signature_phase_errors,
    radial_hessian_diagonalization,
    regularized_fresnel_identity_error,
    shear_cutoff_limit_residual,
    sine_ratio_series,
    synthetic_branch_projection_errors,
    synthetic_sensitivity_formula_errors,
    traction_determinant_residual,
    weighted_bessel_identity_errors,
)
from zgv_morse.rayleigh_lamb import det_symmetric_mp  # noqa: E402


DERIVATION = ROOT / "docs" / "derivations" / "01_isotropic_rayleigh_lamb.tex"
ANISOTROPIC_DERIVATION = ROOT / "docs" / "derivations" / "02_anisotropic_morse_unfolding.tex"
GREEN_DERIVATION = ROOT / "docs" / "derivations" / "03_green_function_asymptotics.tex"


def test_sine_ratio_series_is_exact_through_sixth_order() -> None:
    s, h = sp.symbols("s h")
    expected = h - s**2 * h**3 / 6 + s**4 * h**5 / 120 - s**6 * h**7 / 5040

    assert sp.expand(sine_ratio_series() - expected) == 0


def test_scaled_traction_determinant_is_the_desingularized_determinant() -> None:
    assert traction_determinant_residual() == 0


@pytest.mark.parametrize(
    ("k", "omega"),
    ((0.6, 2.5), (0.8, 2.85), (0.7, 0.7)),
)
def test_independent_traction_determinant_matches_production_at_regular_and_cutoff_points(
    k: float,
    omega: float,
) -> None:
    independent = independent_traction_determinant_mp(k, omega, 2.0, 1.0, 1.0)
    production = det_symmetric_mp(k, omega, 2.0, 1.0, 1.0)

    assert complex(independent) == pytest.approx(complex(production), rel=1.0e-14, abs=1.0e-14)


def test_desingularized_determinant_has_the_documented_shear_cutoff_limit() -> None:
    assert shear_cutoff_limit_residual() == 0


def test_implicit_first_and_second_derivative_formulas() -> None:
    d_k, d_omega, d_kk, d_komega, d_omegaomega = sp.symbols(
        "D_k D_omega D_kk D_komega D_omegaomega"
    )
    formulas = implicit_derivative_formulas()
    expected_first = -d_k / d_omega
    expected_second = (
        -(d_kk + 2 * d_komega * expected_first + d_omegaomega * expected_first**2) / d_omega
    )

    assert sp.simplify(formulas["omega_k"] - expected_first) == 0
    assert sp.simplify(formulas["omega_kk"] - expected_second) == 0
    assert sp.simplify(formulas["zgv_curvature"] + d_kk / d_omega) == 0


def test_cartesian_radial_hessian_diagonalizes_into_normal_and_tangent() -> None:
    r, omega_r, omega_rr, a = sp.symbols("r omega_r omega_rr a")
    general, at_zgv = radial_hessian_diagonalization()

    assert general == sp.diag(omega_rr, omega_r / r)
    assert at_zgv == sp.diag(a, 0)


def test_high_precision_substitution_matches_task4_and_checked_artifact() -> None:
    report = check_reference_substitution(ROOT, dps=80)

    assert report.determinant_residual < 1.0e-70
    assert report.production_determinant_residual < 1.0e-70
    assert abs(report.group_velocity) < 1.0e-70
    assert report.d_omega == pytest.approx(-9.856471487573888, rel=0.0, abs=5.0e-14)
    assert report.kappa0 == pytest.approx(0.8042173193715181, rel=0.0, abs=2.0e-15)
    assert report.omega0 == pytest.approx(2.8517587749600901, rel=0.0, abs=2.0e-15)
    assert report.curvature_a == pytest.approx(1.1968627250739301, rel=0.0, abs=2.0e-15)
    assert report.task4_kappa_error < 5.0e-16
    assert report.task4_omega_error < 5.0e-16
    assert report.task4_curvature_error < 5.0e-15
    assert report.artifact_kappa_error < 5.0e-16
    assert report.artifact_omega_error < 5.0e-16
    assert report.artifact_curvature_error < 5.0e-15


def test_isotropic_derivation_has_fixed_labels_and_complete_route() -> None:
    text = DERIVATION.read_text(encoding="utf-8")

    for label in (
        r"\label{eq:desingularized-determinant}",
        r"\label{eq:zgv-curvature}",
        r"\label{thm:morse-bott-ring}",
    ):
        assert text.count(label) == 1

    for subsection in (
        "Helmholtz potentials and symmetric parity",
        "Face tractions and the original determinant",
        "Desingularization and the shear cutoff",
        "Implicit dispersion derivatives",
        "Cartesian Hessian and the Morse--Bott ring",
    ):
        assert rf"\subsection{{{subsection}}}" in text

    assert "TODO" not in text
    assert "omitted" not in text.lower()


def test_cubic_tensor_and_rotated_strain_have_only_the_documented_fourfold_term() -> None:
    assert cubic_tensor_decomposition_error(delta=1.7) < 1.0e-15
    assert cubic_strain_contraction_residual() == 0


def test_closed_form_cubic_coefficients_match_full_spectral_artifact() -> None:
    coefficients = closed_form_cubic_coefficients(ROOT, order=10)
    with np.load(ROOT / "data/generated/angular_sensitivity.npz", allow_pickle=False) as data:
        artifact_v0 = float(data["V0"])
        artifact_v4 = float(data["V4"])

    assert coefficients["V0"] == pytest.approx(artifact_v0, rel=0.0, abs=2.0e-13)
    assert coefficients["V4"] == pytest.approx(artifact_v4, rel=0.0, abs=2.0e-13)
    assert coefficients["Q4"] > 0.0
    assert coefficients["V4"] > 0.0
    assert coefficients["mass_norm"] == pytest.approx(1.0, rel=0.0, abs=2.0e-13)


def test_documented_generalized_eigenvalue_V_and_B_match_code_interfaces() -> None:
    errors = synthetic_sensitivity_formula_errors()

    assert set(errors) == {
        "V",
        "B",
        "arbitrary_B",
        "lambda_epsilon",
        "lambda_radial_epsilon",
        "mode",
    }
    assert max(errors.values()) < 2.0e-14


@pytest.mark.parametrize("harmonic", [2, 4, 6])
def test_general_even_harmonic_has_alternating_morse_points_and_zero_index(
    harmonic: int,
) -> None:
    data = morse_splitting_data(harmonic, epsilon=0.03, V_m=0.4)

    assert len(data["theta"]) == 2 * harmonic
    assert np.count_nonzero(data["kind"] == "minimum") == harmonic
    assert np.count_nonzero(data["kind"] == "saddle") == harmonic
    assert int(np.sum(data["index"])) == 0
    assert np.allclose(data["radial_shift"], -0.03 * 0.3 / 3.0)
    assert data["frequency_separation"] == pytest.approx(2.0 * 0.03 * 0.4)

    reversed_data = morse_splitting_data(harmonic, epsilon=-0.03, V_m=0.4)
    assert np.allclose(data["theta"], reversed_data["theta"])
    assert np.all(data["kind"] != reversed_data["kind"])


@pytest.mark.parametrize(
    "updates",
    (
        {"epsilon": 0.0},
        {"V_m": 0.0},
        {"k0": 0.0},
        {"curvature": -1.0},
    ),
)
def test_morse_splitting_rejects_degenerate_or_invalid_parameters(
    updates: dict[str, float],
) -> None:
    parameters = {"epsilon": 0.03, "V_m": 0.4, "k0": 2.0, "curvature": 3.0}
    parameters.update(updates)

    with pytest.raises(ValueError):
        morse_splitting_data(4, **parameters)


def test_morse_hessian_is_adaptive_and_rejects_unresolved_nonzero_splitting() -> None:
    resolved = morse_splitting_data(4, epsilon=1.0e-10, V_m=0.4)

    assert np.count_nonzero(resolved["kind"] == "minimum") == 4
    assert np.count_nonzero(resolved["kind"] == "saddle") == 4
    assert np.all(np.abs(resolved["tangential_curvature"]) > 0.0)

    with pytest.raises(ValueError, match="numerically resolved"):
        morse_splitting_data(4, epsilon=1.0e-14, V_m=0.4)


def test_normal_form_cartesian_hessian_and_schur_inertia_identity() -> None:
    matrix_residual, schur_residual = normal_form_hessian_identity()

    assert matrix_residual == sp.zeros(2)
    assert schur_residual == 0


def test_vfour_zero_can_split_at_second_order_instead_of_remaining_a_ring() -> None:
    first = exceptional_second_order_splitting(0.04, harmonic=4, amplitude=0.7)
    half = exceptional_second_order_splitting(0.02, harmonic=4, amplitude=0.7)

    assert first["point_count"] == 8
    assert first["minimum_count"] == first["saddle_count"] == 4
    assert first["curvature_scale"] > 0.0
    assert half["curvature_scale"] / first["curvature_scale"] == pytest.approx(0.25)

    with pytest.raises(ValueError, match="nonzero"):
        exceptional_second_order_splitting(0.0, harmonic=4, amplitude=0.7)
    with pytest.raises(ValueError, match="nonzero"):
        exceptional_second_order_splitting(0.04, harmonic=4, amplitude=0.0)
    with pytest.raises(ValueError, match="resolved"):
        exceptional_second_order_splitting(np.nextafter(0.0, 1.0), harmonic=4, amplitude=0.7)


def test_anisotropic_derivation_has_fixed_labels_and_explicit_boundaries() -> None:
    text = ANISOTROPIC_DERIVATION.read_text(encoding="utf-8")

    for label in (
        r"\label{thm:anisotropic-normal-form}",
        r"\label{thm:cubic-morse-splitting}",
        r"\label{eq:radial-shift}",
        r"\label{eq:cubic-vfour}",
    ):
        assert text.count(label) == 1
    for required in (
        "arbitrary normalization",
        "reduced resolvent",
        "fixed-density",
        "higher harmonic",
        "second order",
        "eigengap",
    ):
        assert required.lower() in text.lower()
    assert "TODO" not in text
    assert "omitted" not in text.lower()
    assert not any(ord(character) < 32 and character not in "\n\t" for character in text)


def test_positive_and_negative_angular_integrals_equal_two_pi_jzero() -> None:
    errors = angular_bessel_identity_errors((-11.25, -2.0, 0.0, 3.75, 14.0))

    assert set(errors) == {-11.25, -2.0, 0.0, 3.75, 14.0}
    assert max(errors.values()) < 1.0e-45


def test_gaussian_regularized_fresnel_factor_has_documented_phase() -> None:
    assert regularized_fresnel_identity_error() < 1.0e-45


def test_large_jzero_phase_error_decreases_away_from_cancellation_zeros() -> None:
    errors = large_bessel_phase_errors()

    assert errors.shape == (3,)
    assert errors[0] < 4.5e-6
    assert np.all(errors[1:] < 0.27 * errors[:-1])


def test_quadratic_stationary_phase_recovers_all_morse_signature_factors() -> None:
    errors = quadratic_signature_phase_errors()

    assert set(errors) == {"minimum", "saddle", "maximum"}
    assert max(errors.values()) < 1.0e-8


def test_branch_projection_is_scale_invariant_and_uses_undoubled_positive_frequency() -> None:
    errors = synthetic_branch_projection_errors()

    assert set(errors) == {"normalization", "normal_impulse", "physical_reconstruction"}
    assert max(errors.values()) < 2.0e-14
    assert oscillator_positive_frequency_residual() == 0


def test_cartesian_morse_hessian_cancels_the_polar_jacobian() -> None:
    assert polar_jacobian_cancellation_error() < 2.0e-14


def test_eight_point_morse_sum_matches_large_bessel_with_all_constants() -> None:
    assert bessel_morse_overlap_error() < 2.0e-14
    assert production_eight_point_overlap_error() < 2.0e-14


def test_growing_tau_overlap_controls_nontrivial_effective_second_order_phase() -> None:
    errors = growing_tau_overlap_errors()

    assert errors.shape == (3,)
    assert errors[0] < 1.9e-2
    assert np.all(errors[1:] < 0.51 * errors[:-1])


def test_frequency_separation_is_twice_the_nonnegative_modulation_magnitude() -> None:
    errors = frequency_modulation_errors()

    assert set(errors) == {"separation", "modulation", "factor_two"}
    assert max(errors.values()) < 2.0e-15


def test_general_angular_weight_matches_the_fourier_bessel_selection_rule() -> None:
    errors = weighted_bessel_identity_errors()

    assert set(errors) == {-5.5, -1.25, 0.0, 2.75, 9.0}
    assert max(errors.values()) < 1.0e-40


def test_green_derivation_has_fixed_labels_constants_and_limit_boundaries() -> None:
    text = GREEN_DERIVATION.read_text(encoding="utf-8")

    for label in (
        r"\label{thm:uniform-bessel-crossover}",
        r"\label{thm:fixed-anisotropy-decay}",
        r"\label{thm:growing-bessel-morse-overlap}",
        r"\label{eq:uniform-response}",
        r"\label{eq:morse-stationary-sum}",
        r"\label{eq:branch-projected-response}",
        r"\label{eq:normal-impulse-overlap}",
        r"\label{eq:radial-fresnel-factor}",
        r"\label{eq:weighted-bessel-series}",
        r"\label{eq:bessel-morse-overlap}",
        r"\label{eq:critical-frequency-separation}",
        r"\label{eq:signed-modulation-rate}",
        r"\label{rem:nodal-channels}",
        r"\label{rem:noncommuting-limits}",
    ):
        assert text.count(label) == 1
    for required in (
        "positive-frequency complex response",
        "source--mode--detector",
        "W_2^{\\mathrm{eff}}",
        "noncommuting limits",
        "nodal",
        "Cartesian Hessian",
        "signed modulation",
        "physical real response",
        "normalized angular kernel",
        "compact-\\(\\tau\\) theorem does not",
    ):
        assert required.lower() in text.lower()
    assert "TODO" not in text
    assert "omitted" not in text.lower()
    math_spacing_removed = text.replace(r"\qquad", "").replace(r"\quad", "")
    assert "quad" not in math_spacing_removed
    assert not any(ord(character) < 32 and character not in "\n\t" for character in text)
