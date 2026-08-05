"""Integration checks for workflow cache identity and artifact lineage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

import zgv_morse.artifact_schema as artifact_schema
import zgv_morse.workflows.common as common
from zgv_morse.artifact_schema import (
    ArtifactValidationError,
    compute_cache_key,
    validate_artifact,
)
from zgv_morse.config import ReferenceConfig, load_reference_config


ROOT = Path(__file__).resolve().parents[1]


def _isotropic_arrays() -> dict[str, np.ndarray]:
    kappa = np.array([0.7, 0.8, 0.9], dtype=np.float64)
    local_q = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    mode_z = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    return {
        "kappa": kappa,
        "omega_symmetric": np.array([[2.7, 2.8, 2.9], [3.0, 3.1, 3.2]], dtype=np.float64),
        "branch_labels": np.array(["S1", "S2b"]),
        "kappa0": np.array(0.8),
        "omega0": np.array(2.8),
        "curvature_a": np.array(1.5),
        "local_q": local_q,
        "local_omega": 2.8 + 1.5 * local_q**2,
        "local_quadratic": 2.8 + 1.5 * local_q**2,
        "mode_z": mode_z,
        "mode_u": np.ones((3, 3), dtype=np.complex128),
        "mode_squared_displacement": np.ones(3, dtype=np.float64),
    }


def _sensitivity_arrays() -> dict[str, np.ndarray]:
    theta = 2.0 * np.pi * np.arange(16) / 16
    V = 0.2 + 0.03 * np.cos(4.0 * theta)
    B = -0.1 + 0.02 * np.cos(4.0 * theta)
    epsilon = np.array([0.005, 0.01, 0.02], dtype=np.float64)
    return {
        "theta": theta,
        "V": V,
        "B": B,
        "V_reconstruction": V.copy(),
        "harmonic_order": np.array([0, 4, 8], dtype=np.int64),
        "harmonic_amplitude": np.array([0.2, 0.03, 0.0]),
        "V0": np.array(0.2),
        "V4": np.array(0.03),
        "V8": np.array(0.0),
        "epsilon": epsilon,
        "delta_c": 0.01 * epsilon,
        "physical_V4_shift": 0.03 * epsilon,
        "V_fd": V.copy(),
        "B_fd": B.copy(),
    }


def _critical_point_arrays() -> dict[str, np.ndarray]:
    theta = 0.25 * np.pi * np.arange(8)
    kappa = np.full(8, 0.8, dtype=np.float64)
    kx = kappa * np.cos(theta)
    ky = kappa * np.sin(theta)
    grid = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    return {
        "kx": kx,
        "ky": ky,
        "kappa": kappa,
        "theta": theta,
        "omega": np.full(8, 2.8),
        "hessian_eigenvalues": np.array([[1.0, 2.0], [-1.0, 1.0]] * 4),
        "morse_index": np.array([1, -1] * 4, dtype=np.int64),
        "kind": np.array(["minimum", "saddle"] * 4),
        "kx_pred": kx.copy(),
        "ky_pred": ky.copy(),
        "omega_pred": np.full(8, 2.8),
        "gradient_residual": np.zeros(8),
        "kx_grid": grid,
        "ky_grid": grid,
        "omega_iso_grid": np.ones((3, 3)),
        "omega_aniso_grid": np.ones((3, 3)),
    }


def _write(
    stage: str,
    cfg: ReferenceConfig,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    input_artifacts: Mapping[str, Mapping[str, str]] | None = None,
) -> Path:
    return common.write_stage_artifact(
        stage,
        cfg,
        output_dir,
        "smoke",
        arrays,
        {key: "dimensionless" for key in arrays},
        {"synthetic_fixture": 0.0},
        input_artifacts=input_artifacts,
    )


@pytest.fixture
def cfg() -> ReferenceConfig:
    return load_reference_config(ROOT / "config/reference.yaml")


def test_workflow_dependency_graph_is_canonical_shared_and_closed() -> None:
    expected = {
        "isotropic": frozenset(),
        "sensitivity": frozenset(),
        "critical_points": frozenset({"sensitivity"}),
        "scaling": frozenset({"sensitivity"}),
        "green": frozenset({"sensitivity"}),
        "convergence": frozenset({"sensitivity"}),
        "silicon": frozenset(),
    }

    assert common.STAGE_DEPENDENCIES is artifact_schema.STAGE_DEPENDENCIES
    assert dict(common.STAGE_DEPENDENCIES) == expected
    assert all(
        dependency in expected for dependencies in expected.values() for dependency in dependencies
    )
    assert all(
        expected[dependency] == frozenset()
        for dependencies in expected.values()
        for dependency in dependencies
    )


def test_dependency_loader_collects_exact_lineage_for_downstream_writer(
    cfg: ReferenceConfig,
    tmp_path: Path,
) -> None:
    sensitivity = _write("sensitivity", cfg, tmp_path, _sensitivity_arrays())
    _, sensitivity_metadata = validate_artifact(sensitivity)
    input_artifacts: dict[str, dict[str, str]] = {}

    loaded = common.load_stage_dependency(
        "sensitivity",
        cfg,
        tmp_path,
        "smoke",
        {"V0", "V4"},
        input_artifacts,
    )

    assert loaded is not None
    assert set(loaded) == {"V0", "V4"}
    expected_lineage = {
        "sensitivity": {
            "artifact": "angular_sensitivity",
            "output_sha256": sensitivity_metadata["output_sha256"],
            "cache_key": sensitivity_metadata["cache_key"],
        }
    }
    assert input_artifacts == expected_lineage

    downstream = _write(
        "critical_points",
        cfg,
        tmp_path,
        _critical_point_arrays(),
        input_artifacts=input_artifacts,
    )
    _, downstream_metadata = validate_artifact(downstream)
    assert downstream_metadata["input_artifacts"] == expected_lineage
    assert downstream_metadata["cache_key"] == compute_cache_key(
        stage="critical_points",
        profile="smoke",
        config_hash=downstream_metadata["config_hash"],
        source_hash=downstream_metadata["source_hash"],
        code_hash=downstream_metadata["code_hash"],
        uv_lock_hash=downstream_metadata["uv_lock_hash"],
        input_artifacts=expected_lineage,
    )

    manifest = json.loads((tmp_path / "provenance_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == {
        "angular_sensitivity",
        "critical_points",
    }


def test_writer_rejects_claimed_lineage_that_does_not_match_the_loaded_pair(
    cfg: ReferenceConfig,
    tmp_path: Path,
) -> None:
    _write("sensitivity", cfg, tmp_path, _sensitivity_arrays())
    manifest = tmp_path / "provenance_manifest.json"
    before = manifest.read_bytes()
    false_lineage = {
        "sensitivity": {
            "artifact": "angular_sensitivity",
            "output_sha256": "0" * 64,
            "cache_key": "1" * 64,
        }
    }

    with pytest.raises(ValueError, match="lineage does not match"):
        _write(
            "critical_points",
            cfg,
            tmp_path,
            _critical_point_arrays(),
            input_artifacts=false_lineage,
        )

    assert not (tmp_path / "critical_points.npz").exists()
    assert manifest.read_bytes() == before


def test_preflight_failure_preserves_existing_pair_and_manifest(
    cfg: ReferenceConfig,
    tmp_path: Path,
) -> None:
    path = _write("isotropic", cfg, tmp_path, _isotropic_arrays())
    sidecar = path.with_suffix(".json")
    manifest = tmp_path / "provenance_manifest.json"
    before = (path.read_bytes(), sidecar.read_bytes(), manifest.read_bytes())
    invalid = _isotropic_arrays()
    invalid.pop("mode_squared_displacement")

    with pytest.raises(ArtifactValidationError, match="missing"):
        _write("isotropic", cfg, tmp_path, invalid)

    assert (path.read_bytes(), sidecar.read_bytes(), manifest.read_bytes()) == before
    validate_artifact(path)


def test_temporary_output_uses_local_manifest_without_touching_canonical_manifest(
    cfg: ReferenceConfig,
    tmp_path: Path,
) -> None:
    canonical = ROOT / "data" / "provenance_manifest.json"
    canonical_before = canonical.read_bytes() if canonical.exists() else None

    path = _write("isotropic", cfg, tmp_path, _isotropic_arrays())

    assert path.parent == tmp_path
    local = tmp_path / "provenance_manifest.json"
    payload = json.loads(local.read_text(encoding="utf-8"))
    assert set(payload["artifacts"]) == {"isotropic_zgv"}
    canonical_after = canonical.read_bytes() if canonical.exists() else None
    assert canonical_after == canonical_before


def test_stage_without_loaded_artifacts_records_empty_lineage(
    cfg: ReferenceConfig,
    tmp_path: Path,
) -> None:
    path = _write("isotropic", cfg, tmp_path, _isotropic_arrays())

    _, metadata = validate_artifact(path)

    assert metadata["input_artifacts"] == {}
    assert metadata["cache_key"] == compute_cache_key(
        stage="isotropic",
        profile="smoke",
        config_hash=metadata["config_hash"],
        source_hash=metadata["source_hash"],
        code_hash=metadata["code_hash"],
        uv_lock_hash=metadata["uv_lock_hash"],
        input_artifacts={},
    )
