from __future__ import annotations

import json
from pathlib import Path

from zgv_morse.artifact_schema import SCHEMAS


ROOT = Path(__file__).resolve().parents[1]
DERIVATION = ROOT / "docs" / "derivations" / "04_spectral_numerics.tex"
CONFIG = ROOT / "config" / "reference.yaml"


def _normalized_text() -> str:
    return DERIVATION.read_text(encoding="utf-8").replace(r"\_", "_")


def test_numerical_derivation_has_required_labels_and_acceptance_boundaries() -> None:
    text = _normalized_text()

    for label in (
        "eq:gll-element-map",
        "eq:spectral-weak-form",
        "eq:natural-traction-boundary",
        "eq:mandel-convention",
        "eq:wavevector-matrix-derivatives",
        "eq:normalized-eigen-residual",
        "eq:mass-orthogonality-defect",
        "eq:mass-mac",
        "alg:mode-tracking",
        "eq:eigengap-rejection",
        "eq:differentiated-mode-solve",
        "alg:critical-point-search",
        "eq:critical-point-uncertainty",
        "eq:annular-index-certificate",
        "eq:polar-green-quadrature",
        "eq:fit-masks",
        "eq:phase-error-certificate",
    ):
        assert text.count(rf"\label{{{label}}}") == 1

    for phrase in (
        "Legendre--Gauss--Lobatto",
        "natural traction-free",
        "Mandel",
        "normalized residual",
        "mass orthogonality",
        "subspace",
        "eigengap",
        "gradient uncertainty",
        "boundary winding",
        "index closes",
        "three-grid",
        "fit_window_early",
        "fit_window_late",
        "no fitted alignment",
        "not upper bounds",
    ):
        assert phrase.lower() in text.lower()


def test_every_live_reference_configuration_key_is_mapped_in_the_derivation() -> None:
    keys = []
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(" ") and ":" in line:
            keys.append(line.split(":", maxsplit=1)[0])
    text = _normalized_text()

    assert keys
    for key in keys:
        assert f"config/reference.yaml::{key}" in text, (
            f"missing qualified config/reference.yaml key {key}"
        )


def test_all_registered_artifact_and_tolerance_keys_are_qualified() -> None:
    text = _normalized_text()

    for artifact, schema in SCHEMAS.items():
        for key in schema:
            assert f"{artifact}.{key}" in text, f"missing qualified artifact key {artifact}.{key}"
        metadata = json.loads(
            (ROOT / "data" / "generated" / f"{artifact}.json").read_text(encoding="utf-8")
        )
        for key in metadata["tolerances"]:
            assert f"{artifact}.tolerances.{key}" in text, (
                f"missing sidecar tolerance key {artifact}.tolerances.{key}"
            )


def test_documentation_maps_each_numerical_stage_to_the_real_code_interface() -> None:
    text = _normalized_text()
    interfaces = (
        "gll_nodes_weights",
        "differentiation_matrix",
        "build_gll_mesh",
        "assemble_plate_matrices",
        "assemble_wavevector_derivatives",
        "solve_plate_modes",
        "mass_mac",
        "RingAnchoredSpectralEvaluator",
        "radial_frequency_sensitivity",
        "locate_critical_points",
        "verify_annular_exhaustion",
        "integrate_registered_grid_convergence",
        "assert_phase_accuracy",
        "fit_power_law",
        "_EARLY_TAU_WINDOW",
        "_FIXED_EPSILON",
        "_MORSE_COMPARISON_START",
        "_MORSE_COMPARISON_STOP",
        "_MORSE_COHERENCE_THRESHOLD",
        "_UNIFORM_TAU_MAXIMUM",
    )

    for interface in interfaces:
        assert interface in text, f"missing code interface {interface}"


def test_phase_and_resolution_guards_are_numerically_explicit() -> None:
    text = _normalized_text()
    compact = "".join(text.split())

    assert "phase_error_tolerance" in text
    assert r"t_{\max}" in text
    assert r"\max" in text
    assert r"|\delta\omega|" in compact
    assert r"\le0.05" in compact
    assert "ten times" in text.lower()
    assert "10^{-8}" in text
    assert "10^{-12}" in text
    assert "at least ten" in text.lower()


def test_documentation_distinguishes_the_registered_grids_indices_and_fit_masks() -> None:
    text = _normalized_text()
    compact = "".join(text.split())

    assert "poincar" in text.lower()
    assert "not" in text.lower().split("morse_index", maxsplit=1)[1][:500].lower()
    for shape in (
        r"129\times32",
        r"257\times64",
        r"513\times128",
        r"129\times64",
        r"257\times128",
        r"513\times256",
    ):
        assert shape in compact
    for phrase in (
        "independent full-wave",
        "nested subsampling",
        "gradient_uncertainty",
        "not stored",
        "half-open",
        "log--log intercept",
        "finite-resolution",
    ):
        assert phrase.lower() in text.lower()
    for value in ("1500", "0.10", "0.30", "0.08"):
        assert value in text


def test_numerical_derivation_contains_no_placeholders_or_control_characters() -> None:
    text = DERIVATION.read_text(encoding="utf-8")

    assert "TODO" not in text
    assert "omitted" not in text.lower()
    assert "TBD" not in text
    assert not any(character == "�" for character in text)
    assert not any(ord(character) < 32 and character not in "\n\t" for character in text)
