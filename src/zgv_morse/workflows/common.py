"""Shared validation, hashing, and writing for deterministic workflow stages."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from types import MappingProxyType
from typing import Collection, Mapping, MutableMapping

import numpy as np

from ..artifact_schema import (
    CACHE_SCHEMA_VERSION,
    STAGE_ARTIFACTS,
    STAGE_DEPENDENCIES,
    compute_cache_key,
    validate_artifact,
    validate_artifact_payload,
)
from ..artifacts import sha256_file, write_npz_with_sidecar
from ..config import ReferenceConfig
from ..provenance import update_manifest


OUTPUT_FILES = MappingProxyType(
    {stage: f"{artifact}.npz" for stage, artifact in STAGE_ARTIFACTS.items()}
)
PROFILES = ("smoke", "full")
DIMENSIONLESS_CONVENTION = "kappa=k*h, Omega=omega*h/c_T"


def validate_stage_inputs(
    cfg: object,
    output_dir: object,
    profile: object,
) -> tuple[ReferenceConfig, Path, str]:
    """Validate the exact public stage contract without coercing callers."""

    if not isinstance(cfg, ReferenceConfig):
        raise TypeError("cfg must be a ReferenceConfig")
    cfg.validate()
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    return cfg, output_dir, profile


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _config_hash(cfg: ReferenceConfig) -> str:
    payload = json.dumps(
        asdict(cfg),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            "__pycache__" in relative.parts
            or ".pytest_cache" in relative.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _scientific_context(cfg: ReferenceConfig, project_root: Path) -> dict[str, str]:
    return {
        "config_hash": _config_hash(cfg),
        "source_hash": sha256_file(project_root / "config" / "reference.yaml"),
        "code_hash": _tree_hash(project_root / "src"),
        "uv_lock_hash": sha256_file(project_root / "uv.lock"),
    }


def _manifest_path(output_dir: Path, project_root: Path) -> Path:
    canonical = (project_root / "data" / "generated").resolve(strict=False)
    if output_dir.resolve(strict=False) == canonical:
        return project_root / "data" / "provenance_manifest.json"
    return output_dir / "provenance_manifest.json"


def load_stage_dependency(
    stage: str,
    cfg: ReferenceConfig,
    output_dir: Path,
    profile: str,
    required_keys: Collection[str],
    input_artifacts: MutableMapping[str, dict[str, str]] | None = None,
) -> dict[str, np.ndarray] | None:
    """Load one current, checksummed dependency or return ``None`` if absent."""

    config, directory, selected_profile = validate_stage_inputs(
        cfg,
        output_dir,
        profile,
    )
    if stage not in OUTPUT_FILES:
        raise ValueError(f"unknown dependency stage: {stage}")
    keys = frozenset(required_keys)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("required_keys must contain nonempty strings")

    path = directory / OUTPUT_FILES[stage]
    sidecar = path.with_suffix(".json")
    if not path.exists() and not sidecar.exists():
        return None
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"{stage} dependency integrity pair is incomplete")
    try:
        arrays, metadata = validate_artifact(path, sidecar)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"{stage} dependency integrity validation failed: {error}"
        ) from error

    project_root = Path(__file__).resolve().parents[3]
    context = _scientific_context(config, project_root)
    expected = {
        "stage": stage,
        "artifact": path.stem,
        "profile": selected_profile,
        **context,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(f"{stage} dependency {name} does not match this run")

    missing = keys.difference(arrays)
    if missing:
        raise RuntimeError(
            f"{stage} dependency is missing required arrays: {sorted(missing)}"
        )
    if input_artifacts is not None:
        input_artifacts[stage] = {
            "artifact": str(metadata["artifact"]),
            "output_sha256": str(metadata["output_sha256"]),
            "cache_key": str(metadata["cache_key"]),
        }
    return {key: np.array(arrays[key], copy=True) for key in sorted(keys)}


def write_stage_artifact(
    stage: str,
    cfg: ReferenceConfig,
    output_dir: Path,
    profile: str,
    arrays: Mapping[str, np.ndarray],
    units: Mapping[str, str],
    tolerances: Mapping[str, object],
    *,
    extra_metadata: Mapping[str, object] | None = None,
    input_artifacts: Mapping[str, Mapping[str, str]] | None = None,
) -> Path:
    """Write one stage artifact with deterministic scientific provenance."""

    config, directory, selected_profile = validate_stage_inputs(
        cfg,
        output_dir,
        profile,
    )
    if stage not in OUTPUT_FILES:
        raise ValueError(f"unknown workflow stage: {stage}")
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("arrays must be a nonempty mapping")
    if not isinstance(units, Mapping) or set(units) != set(arrays):
        raise ValueError("units must define exactly one entry per array")
    if not isinstance(tolerances, Mapping) or not tolerances:
        raise ValueError("tolerances must be a nonempty mapping")
    inputs = (
        {}
        if input_artifacts is None
        else {key: dict(value) for key, value in input_artifacts.items()}
    )
    unexpected_inputs = set(inputs).difference(STAGE_DEPENDENCIES[stage])
    if unexpected_inputs:
        raise ValueError(
            f"undeclared input artifacts for {stage}: {sorted(unexpected_inputs)}"
        )

    project_root = Path(__file__).resolve().parents[3]
    context = _scientific_context(config, project_root)
    for dependency, claimed in inputs.items():
        dependency_path = directory / OUTPUT_FILES[dependency]
        try:
            _, dependency_metadata = validate_artifact(
                dependency_path,
                dependency_path.with_suffix(".json"),
            )
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                f"input artifact {dependency} is missing or invalid: {error}"
            ) from error
        expected_lineage = {
            "artifact": dependency_metadata["artifact"],
            "output_sha256": dependency_metadata["output_sha256"],
            "cache_key": dependency_metadata["cache_key"],
        }
        if claimed != expected_lineage:
            raise ValueError(f"input artifact lineage does not match {dependency}")
        expected_context = {
            "profile": selected_profile,
            **context,
        }
        if any(
            dependency_metadata.get(key) != value
            for key, value in expected_context.items()
        ):
            raise ValueError(
                f"input artifact scientific context does not match {dependency}"
            )
    cache_key = compute_cache_key(
        stage=stage,
        profile=selected_profile,
        input_artifacts=inputs,
        **context,
    )
    metadata: dict[str, object] = {
        "schema_version": 1,
        "artifact": Path(OUTPUT_FILES[stage]).stem,
        "stage": stage,
        "profile": selected_profile,
        "command": (
            "python -m zgv_morse.workflows "
            f"--stage {stage} --profile {selected_profile}"
        ),
        "units": dict(units),
        "dimensionless_convention": DIMENSIONLESS_CONVENTION,
        **context,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "input_artifacts": inputs,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "mpmath": version("mpmath"),
            "platform": platform.platform(),
        },
        "tolerances": dict(tolerances),
    }
    if extra_metadata is not None:
        if not isinstance(extra_metadata, Mapping):
            raise TypeError("extra_metadata must be a mapping")
        overlap = set(extra_metadata).intersection(metadata)
        if overlap:
            raise ValueError(
                f"extra_metadata cannot replace required keys: {sorted(overlap)}"
            )
        metadata.update(extra_metadata)

    path = directory / OUTPUT_FILES[stage]
    validate_artifact_payload(path.stem, arrays, metadata)
    write_npz_with_sidecar(path, dict(arrays), metadata)
    validate_artifact(path, path.with_suffix(".json"))
    update_manifest(
        _manifest_path(directory, project_root),
        path,
        path.with_suffix(".json"),
    )
    return path
