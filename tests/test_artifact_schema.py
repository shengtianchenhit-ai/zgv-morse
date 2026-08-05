"""Strict scientific-array contracts layered over artifact integrity checks."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from zgv_morse.artifact_schema import (
    ArtifactValidationError,
    CACHE_SCHEMA_VERSION,
    SCHEMAS,
    compute_cache_key,
    validate_artifact,
)
from zgv_morse.artifacts import write_npz_with_sidecar


EXPECTED_KEYS = {
    "isotropic_zgv": {
        "kappa",
        "omega_symmetric",
        "branch_labels",
        "kappa0",
        "omega0",
        "curvature_a",
        "local_q",
        "local_omega",
        "local_quadratic",
        "mode_z",
        "mode_u",
        "mode_squared_displacement",
    },
    "angular_sensitivity": {
        "theta",
        "V",
        "B",
        "V_reconstruction",
        "harmonic_order",
        "harmonic_amplitude",
        "V0",
        "V4",
        "V8",
        "epsilon",
        "delta_c",
        "physical_V4_shift",
        "V_fd",
        "B_fd",
    },
    "critical_points": {
        "kx",
        "ky",
        "kappa",
        "theta",
        "omega",
        "hessian_eigenvalues",
        "morse_index",
        "kind",
        "kx_pred",
        "ky_pred",
        "omega_pred",
        "gradient_residual",
        "kx_grid",
        "ky_grid",
        "omega_iso_grid",
        "omega_aniso_grid",
    },
    "perturbation_scaling": {
        "epsilon",
        "delta_omega_full",
        "delta_omega_pred",
        "q_min_full",
        "q_min_pred",
        "q_saddle_full",
        "q_saddle_pred",
        "omega_min_error",
        "omega_saddle_error",
        "compensated_splitting",
        "compensated_q_min",
        "compensated_q_saddle",
        "compensated_frequency_error",
        "role_reversal_theta_min",
        "role_reversal_theta_saddle",
        "slope_splitting",
        "slope_radial",
        "slope_remainder",
    },
    "green_crossover": {
        "epsilon",
        "time",
        "G_full",
        "G_normal_form",
        "G_morse",
        "tau",
        "J0",
        "scaled_response",
        "envelope",
        "rms_envelope",
        "local_slope",
        "fit_window_early",
        "fit_window_late",
        "slope_early",
        "slope_late",
        "crossover_time",
        "spectrum_omega",
        "spectrum",
        "omega_min",
        "omega_saddle",
        "phase_error",
    },
    "convergence": {
        "polynomial_order",
        "omega0_error",
        "kappa0_error",
        "curvature_error",
        "eigen_residual",
        "hermitian_residual",
        "mass_orthogonality",
        "eigengap",
        "angular_resolution",
        "radial_resolution",
        "quadrature_error",
        "interpolation_error",
        "phase_error",
        "sensitivity_step",
        "V4_fd_error",
        "B_fd_error",
        "tracking_kappa",
        "tracking_mac",
        "tracking_gap",
        "source_width",
        "window_width",
        "response_sensitivity",
    },
    "silicon_stress_test": {
        "kx_grid",
        "ky_grid",
        "omega_grid",
        "kx",
        "ky",
        "omega",
        "hessian_eigenvalues",
        "kind",
        "tracking_gap",
        "gradient_residual",
    },
}


STAGES = {
    "isotropic_zgv": "isotropic",
    "angular_sensitivity": "sensitivity",
    "critical_points": "critical_points",
    "perturbation_scaling": "scaling",
    "green_crossover": "green",
    "convergence": "convergence",
    "silicon_stress_test": "silicon",
}


def _arrays(name: str) -> dict[str, np.ndarray]:
    one = np.arange(3, dtype=np.float64)
    point = np.arange(8, dtype=np.float64)
    fixtures = {
        "isotropic_zgv": {
            "kappa": one,
            "omega_symmetric": np.ones((2, 3)),
            "branch_labels": np.array(["S1", "S2b"]),
            "kappa0": np.array(1.0),
            "omega0": np.array(2.0),
            "curvature_a": np.array(3.0),
            "local_q": one,
            "local_omega": one,
            "local_quadratic": one,
            "mode_z": one,
            "mode_u": np.ones((3, 3), dtype=np.complex128),
            "mode_squared_displacement": one,
        },
        "angular_sensitivity": {
            "theta": one,
            "V": one,
            "B": one,
            "V_reconstruction": one,
            "harmonic_order": np.arange(2, dtype=np.int64),
            "harmonic_amplitude": np.ones(2),
            "V0": np.array(1.0),
            "V4": np.array(1.0),
            "V8": np.array(1.0),
            "epsilon": one + 1.0,
            "delta_c": one,
            "physical_V4_shift": one,
            "V_fd": one,
            "B_fd": one,
        },
        "critical_points": {
            "kx": point,
            "ky": point,
            "kappa": point,
            "theta": point,
            "omega": point,
            "hessian_eigenvalues": np.ones((8, 2)),
            "morse_index": np.zeros(8, dtype=np.int64),
            "kind": np.array(["minimum", "saddle"] * 4),
            "kx_pred": point,
            "ky_pred": point,
            "omega_pred": point,
            "gradient_residual": point,
            "kx_grid": one,
            "ky_grid": np.arange(4, dtype=np.float64),
            "omega_iso_grid": np.ones((4, 3)),
            "omega_aniso_grid": np.ones((4, 3)),
        },
        "perturbation_scaling": {
            **{
                key: one + 1.0
                for key in (
                    "epsilon",
                    "delta_omega_full",
                    "delta_omega_pred",
                    "q_min_full",
                    "q_min_pred",
                    "q_saddle_full",
                    "q_saddle_pred",
                    "omega_min_error",
                    "omega_saddle_error",
                    "compensated_splitting",
                    "compensated_q_min",
                    "compensated_q_saddle",
                    "compensated_frequency_error",
                )
            },
            "role_reversal_theta_min": np.arange(4, dtype=np.float64),
            "role_reversal_theta_saddle": np.arange(4, dtype=np.float64),
            "slope_splitting": np.array(1.0),
            "slope_radial": np.array(1.0),
            "slope_remainder": np.array(2.0),
        },
        "green_crossover": {
            "epsilon": np.ones(2),
            "time": one + 1.0,
            **{
                key: np.ones((2, 3), dtype=np.complex128)
                for key in ("G_full", "G_normal_form", "G_morse", "scaled_response")
            },
            **{
                key: np.ones((2, 3))
                for key in (
                    "tau",
                    "J0",
                    "envelope",
                    "rms_envelope",
                    "local_slope",
                )
            },
            "fit_window_early": np.ones((2, 3), dtype=np.bool_),
            "fit_window_late": np.ones((2, 3), dtype=np.bool_),
            "slope_early": np.ones(1),
            "slope_late": np.ones(1),
            "crossover_time": np.ones(2),
            "spectrum_omega": np.arange(4, dtype=np.float64),
            "spectrum": np.ones((2, 4)),
            "omega_min": np.ones(2),
            "omega_saddle": np.ones(2),
            "phase_error": np.ones(2),
        },
        "convergence": {
            "polynomial_order": np.arange(3, dtype=np.int64),
            **{
                key: one
                for key in (
                    "omega0_error",
                    "kappa0_error",
                    "curvature_error",
                    "eigen_residual",
                    "hermitian_residual",
                    "mass_orthogonality",
                    "eigengap",
                )
            },
            "angular_resolution": np.arange(2, dtype=np.int64),
            "radial_resolution": np.arange(2, dtype=np.int64),
            "quadrature_error": np.ones(2),
            "interpolation_error": np.ones(2),
            "phase_error": np.ones(2),
            "sensitivity_step": np.ones(4),
            "V4_fd_error": np.ones(4),
            "B_fd_error": np.ones(4),
            "tracking_kappa": np.ones(5),
            "tracking_mac": np.ones(5),
            "tracking_gap": np.ones(5),
            "source_width": np.ones(2),
            "window_width": np.ones(3),
            "response_sensitivity": np.ones((2, 3)),
        },
        "silicon_stress_test": {
            "kx_grid": one,
            "ky_grid": np.arange(4, dtype=np.float64),
            "omega_grid": np.ones((4, 3)),
            "kx": point,
            "ky": point,
            "omega": point,
            "hessian_eigenvalues": np.ones((8, 2)),
            "kind": np.array(["minimum", "saddle"] * 4),
            "tracking_gap": point,
            "gradient_residual": point,
        },
    }
    return {key: np.array(value, copy=True) for key, value in fixtures[name].items()}


def _metadata(name: str, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    stage = STAGES[name]
    profile = "smoke"
    input_artifacts: dict[str, dict[str, str]] = {}
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "artifact": name,
        "stage": stage,
        "profile": profile,
        "command": (f"python -m zgv_morse.workflows --stage {stage} --profile {profile}"),
        "dimensionless_convention": "kappa=k*h, Omega=omega*h/c_T",
        "units": {key: "dimensionless" for key in arrays},
        "config_hash": "1" * 64,
        "source_hash": "2" * 64,
        "code_hash": "3" * 64,
        "uv_lock_hash": "4" * 64,
        "environment": {"python": "3.12"},
        "tolerances": {"absolute": 1.0e-8},
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "input_artifacts": input_artifacts,
    }
    metadata["cache_key"] = compute_cache_key(
        stage=stage,
        profile=profile,
        config_hash=metadata["config_hash"],
        source_hash=metadata["source_hash"],
        code_hash=metadata["code_hash"],
        uv_lock_hash=metadata["uv_lock_hash"],
        input_artifacts=input_artifacts,
    )
    return metadata


def _write(tmp_path: Path, name: str, arrays: dict[str, np.ndarray] | None = None) -> Path:
    values = _arrays(name) if arrays is None else arrays
    path = tmp_path / f"{name}.npz"
    write_npz_with_sidecar(path, values, _metadata(name, values))
    return path


def _rewrite_sidecar(path: Path, mutate) -> None:
    """Mutate and resign a test sidecar through the public artifact writer."""

    with np.load(path, allow_pickle=False) as bundle:
        arrays = {key: np.array(bundle[key], copy=True) for key in bundle.files}
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"arrays", "output_sha256", "metadata_sha256"}
    }
    mutate(metadata)
    path.unlink()
    path.with_suffix(".json").unlink()
    write_npz_with_sidecar(path, arrays, metadata)


def _cache_arguments() -> dict[str, Any]:
    return {
        "stage": "green",
        "profile": "full",
        "config_hash": "a" * 64,
        "source_hash": "b" * 64,
        "code_hash": "c" * 64,
        "uv_lock_hash": "d" * 64,
        "input_artifacts": {
            "sensitivity": {
                "artifact": "angular_sensitivity",
                "output_sha256": "e" * 64,
                "cache_key": "f" * 64,
            }
        },
    }


def _cache_key(arguments: dict[str, Any]) -> str:
    """Select only the public scientific cache inputs from a larger record."""

    return compute_cache_key(
        stage=arguments["stage"],
        profile=arguments["profile"],
        config_hash=arguments["config_hash"],
        source_hash=arguments["source_hash"],
        code_hash=arguments["code_hash"],
        uv_lock_hash=arguments["uv_lock_hash"],
        input_artifacts=arguments["input_artifacts"],
    )


def test_cache_key_is_independent_of_mapping_field_order() -> None:
    arguments = _cache_arguments()
    reversed_arguments = dict(reversed(tuple(arguments.items())))
    input_record = arguments["input_artifacts"]["sensitivity"]
    reversed_arguments["input_artifacts"] = {
        "sensitivity": dict(reversed(tuple(input_record.items())))
    }

    assert _cache_key(reversed_arguments) == _cache_key(arguments)


def test_each_scientific_cache_input_changes_the_cache_key() -> None:
    arguments = _cache_arguments()
    baseline = _cache_key(arguments)
    variants: dict[str, dict[str, Any]] = {}
    for field, value in (
        ("stage", "convergence"),
        ("profile", "smoke"),
        ("config_hash", "0" * 64),
        ("source_hash", "1" * 64),
        ("code_hash", "2" * 64),
        ("uv_lock_hash", "3" * 64),
    ):
        variant = deepcopy(arguments)
        variant[field] = value
        variants[field] = variant

    upstream_output = deepcopy(arguments)
    upstream_output["input_artifacts"]["sensitivity"]["output_sha256"] = "4" * 64
    variants["upstream_output_sha256"] = upstream_output
    upstream_cache = deepcopy(arguments)
    upstream_cache["input_artifacts"]["sensitivity"]["cache_key"] = "5" * 64
    variants["upstream_cache_key"] = upstream_cache

    changed = {name: _cache_key(variant) for name, variant in variants.items()}
    assert all(value != baseline for value in changed.values()), changed


def test_operational_metadata_does_not_participate_in_cache_key() -> None:
    arguments = _cache_arguments()
    arguments.update(
        {
            "environment": {"python": "3.12", "platform": "first"},
            "command": "python -m zgv_morse.workflows --stage green --profile full",
            "path": Path("first/green_crossover.npz"),
            "mtime_ns": 100,
        }
    )
    baseline = _cache_key(arguments)
    changed = deepcopy(arguments)
    changed.update(
        {
            "environment": {"python": "9.99", "platform": "second"},
            "command": "a different invocation",
            "path": Path("elsewhere/result.npz"),
            "mtime_ns": 999_999,
        }
    )

    assert _cache_key(changed) == baseline


def test_schema_declares_the_exact_seven_array_contracts() -> None:
    assert set(SCHEMAS) == set(EXPECTED_KEYS)
    assert {name: set(schema) for name, schema in SCHEMAS.items()} == EXPECTED_KEYS


@pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
def test_each_complete_artifact_validates_and_returns_defensive_copies(
    tmp_path: Path,
    name: str,
) -> None:
    path = _write(tmp_path, name)

    arrays, metadata = validate_artifact(path, path.with_suffix(".json"))

    assert set(arrays) == EXPECTED_KEYS[name]
    assert metadata["artifact"] == name
    arrays[next(iter(arrays))][...] = 99
    metadata["units"][next(iter(arrays))] = "changed"
    reloaded, reloaded_metadata = validate_artifact(path)
    assert not np.all(reloaded[next(iter(arrays))] == 99)
    assert reloaded_metadata["units"][next(iter(arrays))] == "dimensionless"


@pytest.mark.parametrize("change", ["missing", "unexpected"])
def test_missing_or_unexpected_arrays_are_rejected(tmp_path: Path, change: str) -> None:
    arrays = _arrays("isotropic_zgv")
    if change == "missing":
        arrays.pop("curvature_a")
    else:
        arrays["manual_value"] = np.array(1.23)
    path = _write(tmp_path, "isotropic_zgv", arrays)

    with pytest.raises(ArtifactValidationError, match=change):
        validate_artifact(path)


@pytest.mark.parametrize(
    ("key", "replacement", "match"),
    [
        ("kappa", np.ones((1, 3)), "ndim"),
        ("kappa", np.arange(3, dtype=np.int64), "dtype"),
        ("branch_labels", np.arange(2, dtype=np.int64), "dtype"),
    ],
)
def test_array_ndim_and_dtype_are_strict(
    tmp_path: Path,
    key: str,
    replacement: np.ndarray,
    match: str,
) -> None:
    arrays = _arrays("isotropic_zgv")
    arrays[key] = replacement
    path = _write(tmp_path, "isotropic_zgv", arrays)

    with pytest.raises(ArtifactValidationError, match=match):
        validate_artifact(path)


def test_nonfinite_arrays_are_rejected_by_integrity_layer(tmp_path: Path) -> None:
    arrays = _arrays("isotropic_zgv")
    arrays["kappa"][1] = np.nan

    with pytest.raises(ValueError, match="finite"):
        _write(tmp_path, "isotropic_zgv", arrays)


@pytest.mark.parametrize(
    ("name", "key", "replacement", "match"),
    [
        ("isotropic_zgv", "local_omega", np.ones(4), "local arrays"),
        ("isotropic_zgv", "mode_u", np.ones((3, 2), dtype=complex), "mode_u"),
        ("angular_sensitivity", "V_fd", np.ones(4), "angular samples"),
        ("critical_points", "omega", np.ones(7), "eight"),
        ("critical_points", "omega_iso_grid", np.ones((3, 4)), "grid"),
        ("perturbation_scaling", "q_min_full", np.ones(4), "epsilon"),
        ("green_crossover", "G_full", np.ones((3, 2), dtype=complex), "shape"),
        ("green_crossover", "spectrum", np.ones((3, 4)), "spectrum"),
        ("convergence", "omega0_error", np.ones(4), "polynomial"),
        ("convergence", "response_sensitivity", np.ones((3, 2)), "response"),
        ("silicon_stress_test", "tracking_gap", np.ones(7), "eight"),
        ("silicon_stress_test", "omega_grid", np.ones((3, 4)), "grid"),
    ],
)
def test_cross_array_shape_contracts_are_enforced(
    tmp_path: Path,
    name: str,
    key: str,
    replacement: np.ndarray,
    match: str,
) -> None:
    arrays = _arrays(name)
    arrays[key] = replacement
    path = _write(tmp_path, name, arrays)

    with pytest.raises(ArtifactValidationError, match=match):
        validate_artifact(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uv_lock_hash", None),
        ("schema_version", 2),
        ("artifact", "convergence"),
        ("stage", "silicon"),
        ("profile", "draft"),
        ("command", "manual"),
        ("units", {"kappa": "dimensionless"}),
    ],
)
def test_required_metadata_values_are_strict(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = _write(tmp_path, "isotropic_zgv")

    def mutate(metadata: dict[str, object]) -> None:
        if value is None:
            metadata.pop(field)
        else:
            metadata[field] = value

    _rewrite_sidecar(path, mutate)
    with pytest.raises(ArtifactValidationError, match=field.replace("_", ".*")):
        validate_artifact(path)


def test_checksum_and_metadata_checksum_failures_are_wrapped(tmp_path: Path) -> None:
    path = _write(tmp_path, "isotropic_zgv")
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)

    with pytest.raises(ArtifactValidationError, match="checksum"):
        validate_artifact(path)


def test_registered_tolerances_may_be_nested_finite_json(tmp_path: Path) -> None:
    path = _write(tmp_path, "isotropic_zgv")

    def mutate(metadata: dict[str, object]) -> None:
        metadata["tolerances"] = {
            "by_epsilon": {"0.01": 1.0e-4},
            "grid_errors": [[1.0e-5, 2.0e-6]],
        }

    _rewrite_sidecar(path, mutate)
    arrays, metadata = validate_artifact(path)
    assert arrays["kappa"].shape == (3,)
    assert metadata["tolerances"]["by_epsilon"]["0.01"] == 1.0e-4


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("cache_schema_version", 2, "cache_schema_version"),
        ("cache_key", "0" * 64, "cache_key"),
    ],
)
def test_artifact_rejects_invalid_cache_identity_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    path = _write(tmp_path, "isotropic_zgv")

    _rewrite_sidecar(path, lambda metadata: metadata.__setitem__(field, value))

    with pytest.raises(ArtifactValidationError, match=match):
        validate_artifact(path)


def test_artifact_rejects_undeclared_or_malformed_input_lineage(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "critical_points")

    def add_malformed_lineage(metadata: dict[str, object]) -> None:
        metadata["input_artifacts"] = {
            "sensitivity": {
                "artifact": "angular_sensitivity",
                "output_sha256": "0" * 64,
            }
        }

    _rewrite_sidecar(path, add_malformed_lineage)
    with pytest.raises(ArtifactValidationError, match="exactly"):
        validate_artifact(path)

    path = _write(tmp_path, "isotropic_zgv")
    _rewrite_sidecar(
        path,
        lambda metadata: metadata.__setitem__(
            "input_artifacts",
            {
                "sensitivity": {
                    "artifact": "angular_sensitivity",
                    "output_sha256": "0" * 64,
                    "cache_key": "1" * 64,
                }
            },
        ),
    )
    with pytest.raises(ArtifactValidationError, match="undeclared"):
        validate_artifact(path)


def test_artifact_metadata_forbids_manual_figure_values(tmp_path: Path) -> None:
    path = _write(tmp_path, "isotropic_zgv")
    _rewrite_sidecar(
        path,
        lambda metadata: metadata.__setitem__("manual_figure_values", [1.23]),
    )

    with pytest.raises(ArtifactValidationError, match="manual figure values"):
        validate_artifact(path)


def test_unknown_and_noncanonical_sidecar_paths_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unknown.npz"
    np.savez(path, value=np.ones(1))
    with pytest.raises(ArtifactValidationError, match="unknown artifact"):
        validate_artifact(path)

    known = _write(tmp_path, "isotropic_zgv")
    with pytest.raises(ArtifactValidationError, match="canonical sidecar"):
        validate_artifact(known, tmp_path / "renamed.json")
