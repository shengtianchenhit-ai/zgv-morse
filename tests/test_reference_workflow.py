from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from zgv_morse.artifacts import validate_npz_sidecar
from zgv_morse.config import load_reference_config
from zgv_morse.provenance import validate_manifest
from zgv_morse.workflows import OUTPUTS, STAGES


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_all_smoke_workflows_close_the_registered_evidence_chain(tmp_path: Path) -> None:
    cfg = load_reference_config(ROOT / "config/reference.yaml")
    expected_order = (
        "isotropic",
        "sensitivity",
        "critical_points",
        "scaling",
        "green",
        "convergence",
        "silicon",
    )

    assert tuple(STAGES) == expected_order
    assert set(OUTPUTS) == set(STAGES)
    reports = {}
    for stage, run in STAGES.items():
        path = run(cfg, tmp_path, "smoke")
        assert path == tmp_path / OUTPUTS[stage]
        reports[stage] = validate_npz_sidecar(path)

    expected_npz = set(OUTPUTS.values())
    expected_sidecars = {Path(filename).with_suffix(".json").name for filename in OUTPUTS.values()}
    assert {path.name for path in tmp_path.glob("*.npz")} == expected_npz
    assert {
        path.name for path in tmp_path.glob("*.json") if path.name != "provenance_manifest.json"
    } == expected_sidecars

    manifest = validate_manifest(tmp_path / "provenance_manifest.json")
    assert set(manifest["artifacts"]) == {Path(filename).stem for filename in OUTPUTS.values()}

    with np.load(tmp_path / OUTPUTS["isotropic"], allow_pickle=False) as data:
        assert_allclose(data["kappa0"], 0.8042173193715181, rtol=2.0e-7)
        assert_allclose(data["omega0"], 2.8517587749600901, rtol=2.0e-7)
        assert float(data["curvature_a"]) > 0.0
    isotropic_metadata = reports["isotropic"]["metadata"]
    assert isotropic_metadata["tolerances"]["minimum_relative_eigengap"] > 0.0

    with np.load(tmp_path / OUTPUTS["sensitivity"], allow_pickle=False) as data:
        relative_V = np.linalg.norm(data["V_fd"] - data["V"]) / np.linalg.norm(data["V"])
        relative_B = np.linalg.norm(data["B_fd"] - data["B"]) / np.linalg.norm(data["B"])
        assert relative_V < cfg.sensitivity_match_tolerance
        assert relative_B < cfg.sensitivity_match_tolerance
        assert abs(float(data["V4"])) > 0.0

    with np.load(tmp_path / OUTPUTS["critical_points"], allow_pickle=False) as data:
        assert data["kind"].size == 8
        assert np.count_nonzero(data["kind"] == "minimum") == 4
        assert np.count_nonzero(data["kind"] == "saddle") == 4
        assert np.sum(data["morse_index"]) == 0
    reversal = reports["critical_points"]["metadata"]["sign_reversal_certificate"]
    assert reversal["positive_count"] == reversal["negative_count"] == 8
    assert reversal["roles_exchanged_at_same_angles"] is True
    assert all(
        positive != negative
        for positive, negative in zip(
            reversal["positive_kinds"],
            reversal["negative_kinds"],
            strict=True,
        )
    )

    with np.load(tmp_path / OUTPUTS["scaling"], allow_pickle=False) as data:
        assert abs(float(data["slope_splitting"]) - 1.0) <= 0.1
        assert abs(float(data["slope_radial"]) - 1.0) <= 0.1
        assert abs(float(data["slope_remainder"]) - 2.0) <= 0.2
        assert_allclose(
            data["q_min_full"],
            data["q_min_pred"],
            rtol=0.12,
            atol=2.0e-7,
        )
        assert_allclose(
            data["q_saddle_full"],
            data["q_saddle_pred"],
            rtol=0.12,
            atol=2.0e-7,
        )
        assert data["role_reversal_theta_min"].size == 4
        assert data["role_reversal_theta_saddle"].size == 4

    green_metadata = reports["green"]["metadata"]
    with np.load(tmp_path / OUTPUTS["green"], allow_pickle=False) as data:
        assert data["epsilon"].size >= 3
        assert np.max(data["phase_error"]) <= cfg.phase_error_tolerance
        uniform = (
            (data["epsilon"][:, None] <= 0.04)
            & (data["time"][None, :] >= 1500.0)
            & (np.abs(data["tau"]) <= 2.0)
        )
        assert np.max(np.abs(data["scaled_response"] - data["J0"])[uniform]) <= 0.08
        assert np.all(np.abs(data["slope_early"] + 0.5) <= 0.05)
        assert np.all(np.abs(data["slope_late"] + 1.0) <= 0.05)
        crossover_slope = np.polyfit(
            np.log(data["epsilon"]),
            np.log(data["crossover_time"]),
            1,
        )[0]
        assert abs(crossover_slope + 1.0) <= 0.1
        assert np.any(np.abs(data["G_morse"]) > 0.0)
        assert data["spectrum_omega"][0] < np.min(data["omega_min"])
        assert data["spectrum_omega"][-1] > np.max(data["omega_saddle"])
        late = green_metadata["late_fit_evidence"]
        assert len(late["geometric_time_centers"]) >= 4
        assert len(late["direct_response_rms"]) == len(late["geometric_time_centers"])
        reproduced_late_slope = np.polyfit(
            np.log(late["geometric_time_centers"]),
            np.log(late["direct_response_rms"]),
            1,
        )[0]
        assert_allclose(reproduced_late_slope, data["slope_late"][0], atol=1.0e-12)
        cancellation = green_metadata["fixed_morse_cancellation_evidence"]
        assert len(cancellation["coherence"]) == data["time"].size
        assert np.array_equal(
            np.flatnonzero(
                (data["time"] >= cancellation["comparison_start_time"])
                & (data["time"] <= cancellation["comparison_stop_time"])
                & (np.asarray(cancellation["coherence"]) < cancellation["coherence_threshold"])
            ),
            cancellation["cancellation_time_indices"],
        )
    green_tolerances = green_metadata["tolerances"]
    radial_control = green_metadata["radial_domain_control"]
    assert radial_control["compact_taper_plateau_abs_q_over_sigma"] == 1.25
    assert radial_control["direct_support_abs_q_over_sigma"] == 1.5
    assert radial_control["direct_support_abs_q_over_k0"] == pytest.approx(0.225)
    assert radial_control["endpoint_taper_value"] == 0.0
    assert radial_control["endpoint_is_flat_to_all_derivative_orders"] is True
    assert radial_control["morse_search_margin_abs_q_over_sigma"] == pytest.approx(0.05)
    assert radial_control["morse_search_abs_q_over_sigma"] == pytest.approx(1.55)
    assert radial_control["morse_search_abs_q_over_k0"] == pytest.approx(0.2325)
    assert radial_control["morse_search_strictly_contains_direct_support"] is True
    assert radial_control["direct_and_morse_responses_use_identical_compact_taper"] is True
    assert all(radial_control["boundary_is_noncritical_by_epsilon"].values())
    assert all(radial_control["gradient_index_closes_by_epsilon"].values())
    for key, minimum in radial_control["minimum_boundary_gradient_by_epsilon"].items():
        uncertainty = radial_control["maximum_boundary_gradient_uncertainty_by_epsilon"][key]
        assert minimum > 10.0 * uncertainty
    fixed_key = f"{green_metadata['fixed_morse_epsilon']:.8g}"
    assert (
        radial_control["stationary_point_count_by_epsilon"][fixed_key]
        == radial_control["morse_contribution_count_by_epsilon"][fixed_key]
        == len(radial_control["fixed_epsilon_stationary_offsets_over_sigma"])
    )
    assert radial_control["all_fixed_epsilon_stationary_points_included_in_morse_sum"] is True
    assert green_tolerances["fixed_epsilon_morse_relative_rms_error"] <= 0.055
    assert green_tolerances["fixed_epsilon_morse_relative_max_error"] <= 0.17
    assert green_tolerances["fixed_epsilon_cancellation_region_normalized_rms"] <= 0.09
    for errors in green_tolerances["registered_complex_response_errors"]:
        assert errors[1] < errors[0]
        assert errors[1] / errors[0] <= 0.5
    assert (
        abs(
            green_tolerances["registered_early_slope_by_grid"][-1]
            - green_tolerances["registered_early_slope_by_grid"][-2]
        )
        <= 0.05
    )
    assert (
        abs(
            green_tolerances["registered_late_slope_by_grid"][-1]
            - green_tolerances["registered_late_slope_by_grid"][-2]
        )
        <= 0.05
    )
    assert np.all(
        np.asarray(green_tolerances["minimum_anisotropic_branch_eigengap_by_epsilon"])
        > 10.0 * np.asarray(green_tolerances["maximum_relative_frequency_uncertainty_by_epsilon"])
    )
    assert (
        np.max(green_tolerances["registered_accumulated_phase_errors"]) <= cfg.phase_error_tolerance
    )

    with np.load(tmp_path / OUTPUTS["convergence"], allow_pickle=False) as data:
        assert data["omega0_error"][-1] < cfg.isotropic_match_tolerance
        assert data["kappa0_error"][-1] < cfg.isotropic_match_tolerance
        assert data["curvature_error"][-1] < cfg.curvature_match_tolerance
        assert np.max(data["eigen_residual"]) < cfg.eigen_residual_tolerance
        assert np.max(data["hermitian_residual"]) < 1.0e-12
        assert np.max(data["mass_orthogonality"]) < 1.0e-10
        assert np.min(data["eigengap"]) > 10.0 * max(
            np.max(data["eigen_residual"]),
            np.finfo(float).eps,
        )
        assert np.all(np.diff(data["quadrature_error"]) < 0.0)
        assert np.max(data["quadrature_error"][1:] / data["quadrature_error"][:-1]) <= 0.5
        assert data["quadrature_error"][-1] <= 0.05
        assert np.all(np.diff(data["interpolation_error"]) < 0.0)
        assert np.max(data["interpolation_error"][1:] / data["interpolation_error"][:-1]) <= 0.5
        assert data["phase_error"][-1] <= cfg.phase_error_tolerance
        assert data["V4_fd_error"][-1] < cfg.sensitivity_match_tolerance
        assert data["B_fd_error"][-1] < cfg.sensitivity_match_tolerance
        assert_allclose(
            data["window_width"],
            [
                cfg.window_sensitivity[0],
                cfg.window_sigma_over_k0,
                cfg.window_sensitivity[1],
            ],
            atol=0.0,
        )
        assert_allclose(data["response_sensitivity"][1, 1], 1.0, atol=5.0e-13)
        assert np.min(data["tracking_mac"]) >= 0.99
        assert np.min(data["tracking_gap"]) > 10.0 * np.max(data["eigen_residual"])
    convergence_metadata = reports["convergence"]["metadata"]
    assert convergence_metadata["no_synthetic_convergence_values"] is True
    direct_provenance = convergence_metadata["direct_measurement_provenance"]
    assert direct_provenance["minimum_anisotropic_relative_eigengap"] > (
        10.0 * direct_provenance["maximum_relative_frequency_uncertainty"]
    )
    assert (
        direct_provenance["response_sensitivity_measurement"]
        == "direct full-wave response RMS-norm ratios over 61 registered times"
    )
    fd_sweep = convergence_metadata["finite_difference_sweep"]
    B_error_matrix = np.asarray(fd_sweep["relative_B_error_matrix"])
    assert B_error_matrix.shape == (4, 4)
    assert np.max(B_error_matrix[-2:, -2:]) < cfg.sensitivity_match_tolerance
    assert min(B_error_matrix[-2:, -1]) < B_error_matrix[0, -1]
    assert min(B_error_matrix[-1, -2:]) < B_error_matrix[-1, 0]

    silicon_metadata = reports["silicon"]["metadata"]
    assert silicon_metadata["scope"] == "finite-anisotropy stress test only"
    assert silicon_metadata["material_source_id"].startswith("doi:")


def test_workflow_contract_rejects_unregistered_profiles_and_paths(tmp_path: Path) -> None:
    cfg = load_reference_config(ROOT / "config/reference.yaml")
    run = STAGES["isotropic"]

    with pytest.raises(ValueError, match="profile"):
        run(cfg, tmp_path, "quick")
    with pytest.raises(TypeError, match="ReferenceConfig"):
        run(object(), tmp_path, "smoke")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="output_dir"):
        run(cfg, str(tmp_path), "smoke")  # type: ignore[arg-type]


def test_workflow_sidecars_are_deterministic_json(tmp_path: Path) -> None:
    cfg = load_reference_config(ROOT / "config/reference.yaml")
    path = STAGES["isotropic"](cfg, tmp_path, "smoke")
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    assert payload["stage"] == "isotropic"
    assert payload["profile"] == "smoke"
    assert payload["artifact"] == "isotropic_zgv"
    assert payload["dimensionless_convention"] == "kappa=k*h, Omega=omega*h/c_T"
    assert sidecar.read_text(encoding="utf-8").endswith("\n")
