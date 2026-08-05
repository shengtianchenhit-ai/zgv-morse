"""Strict, deterministic provenance manifests for generated artifacts."""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any

from .artifact_schema import SCHEMAS, ArtifactValidationError, validate_artifact
from .artifacts import sha256_file


SCHEMA_VERSION = 2
ARTIFACT_NAMES = frozenset(SCHEMAS)
FIGURE_NAMES = frozenset(
    {
        "figure_01_geometry_mechanism",
        "figure_02_isotropic_zgv",
        "figure_03_angular_sensitivity",
        "figure_04_morse_points",
        "figure_05_perturbation_scaling",
        "figure_06_decay_crossover",
        "figure_s01_polynomial_two_element",
        "figure_s02_quadrature_phase",
        "figure_s03_mode_tracking",
        "figure_s04_fd_convergence",
        "figure_s05_source_window_sensitivity",
        "figure_s06_silicon_stress_test",
    }
)
_ROOT_KEYS = frozenset({"schema_version", "artifacts", "figures"})
_RECORD_KEYS = frozenset({"path", "sha256", "sidecar", "sidecar_sha256"})
_FIGURE_RECORD_KEYS = frozenset(
    {"script", "script_sha256", "inputs", "source_data", "outputs", "theory_refs"}
)
_FILE_RECORD_KEYS = frozenset({"path", "sha256"})
_FIGURE_FORMATS = frozenset({"svg", "pdf", "png", "tiff"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTEXT_KEYS = (
    "profile",
    "dimensionless_convention",
    "config_hash",
    "source_hash",
    "code_hash",
    "uv_lock_hash",
    "cache_schema_version",
)


def _path(value: object, name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    return value


def _project_root(manifest: Path) -> Path:
    """Anchor portable records at the manifest directory, never at the CWD."""

    return manifest.resolve(strict=False).parent


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ArtifactValidationError(
            f"provenance manifest JSON is malformed: {error}"
        ) from error
    if type(payload) is not dict:
        raise ArtifactValidationError("provenance manifest root must be an object")
    return payload


def _manual_values_present(value: object) -> bool:
    if type(value) is str:
        normalized_value = re.sub(r"[^a-z0-9]", "", value.lower())
        return normalized_value in {"manualfigurevalue", "manualfigurevalues"}
    if type(value) is dict:
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in {"manualfigurevalue", "manualfigurevalues"}:
                return True
            if _manual_values_present(item):
                return True
    elif type(value) is list:
        return any(_manual_values_present(item) for item in value)
    return False


def _relative_record_path(target: Path, root: Path, field: str) -> str:
    if target.is_symlink():
        raise ArtifactValidationError(f"manifest {field} must not be a symbolic link")
    try:
        resolved = target.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ArtifactValidationError(f"manifest {field} is missing or unreadable") from error
    if not resolved.is_file():
        raise ArtifactValidationError(f"manifest {field} must name a regular file")
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ArtifactValidationError(f"manifest {field} escapes the manifest directory") from error
    return relative.as_posix()


def _pure_record_path(raw: object, field: str) -> PurePosixPath:
    if type(raw) is not str or not raw:
        raise ArtifactValidationError(f"manifest {field} must be a nonempty relative path")
    if "\\" in raw or any(ord(character) < 32 for character in raw):
        raise ArtifactValidationError(f"manifest {field} contains an unsafe path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or raw != pure.as_posix() or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise ArtifactValidationError(f"manifest {field} contains an unsafe path")
    return pure


def _resolve_record_path(raw: object, root: Path, field: str) -> Path:
    pure = _pure_record_path(raw, field)
    candidate = root.joinpath(*pure.parts)
    canonical = _relative_record_path(candidate, root, field)
    if canonical != raw:
        raise ArtifactValidationError(f"manifest {field} is not a canonical relative path")
    return candidate.resolve(strict=True)


def _validated_file_record(
    value: object,
    root: Path,
    field: str,
) -> tuple[Path, dict[str, str]]:
    if type(value) is not dict or set(value) != _FILE_RECORD_KEYS:
        raise ArtifactValidationError(
            f"manifest {field} must contain exactly {sorted(_FILE_RECORD_KEYS)}"
        )
    claimed = value["sha256"]
    if type(claimed) is not str or _SHA256.fullmatch(claimed) is None:
        raise ArtifactValidationError(f"manifest {field}.sha256 is malformed")
    path = _resolve_record_path(value["path"], root, f"{field}.path")
    actual = sha256_file(path)
    if not hmac.compare_digest(claimed, actual):
        raise ArtifactValidationError(f"manifest {field}.sha256 checksum mismatch")
    return path, {"path": str(value["path"]), "sha256": claimed}


def _validate_figure_records(
    value: object,
    manifest: Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ArtifactValidationError("manifest figures must be an object")
    names = set(value)
    unexpected = names.difference(FIGURE_NAMES)
    missing = FIGURE_NAMES.difference(names)
    if unexpected or (missing and (require_complete or bool(names))):
        raise ArtifactValidationError(
            "manifest figure set is invalid: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    root = _project_root(manifest).parent
    defensive: dict[str, Any] = {}
    owned_sources: set[Path] = set()
    owned_outputs: set[Path] = set()
    for name in sorted(value):
        record = value[name]
        if type(record) is not dict or set(record) != _FIGURE_RECORD_KEYS:
            raise ArtifactValidationError(
                f"manifest figure record {name} must contain exactly "
                f"{sorted(_FIGURE_RECORD_KEYS)}"
            )
        script_raw = record["script"]
        script = _resolve_record_path(script_raw, root, f"figures.{name}.script")
        if script.suffix != ".py":
            raise ArtifactValidationError(f"manifest figure {name} script must be Python")
        script_sha = record["script_sha256"]
        if type(script_sha) is not str or _SHA256.fullmatch(script_sha) is None:
            raise ArtifactValidationError(
                f"manifest figure {name}.script_sha256 is malformed"
            )
        if not hmac.compare_digest(script_sha, sha256_file(script)):
            raise ArtifactValidationError(
                f"manifest figure {name}.script_sha256 checksum mismatch"
            )

        normalized_lists: dict[str, list[dict[str, str]]] = {}
        for list_name in ("inputs", "source_data"):
            raw_items = record[list_name]
            if type(raw_items) is not list or not raw_items:
                raise ArtifactValidationError(
                    f"manifest figure {name}.{list_name} must be a nonempty list"
                )
            normalized: list[dict[str, str]] = []
            seen: set[Path] = set()
            for index, item in enumerate(raw_items):
                path, file_record = _validated_file_record(
                    item,
                    root,
                    f"figures.{name}.{list_name}[{index}]",
                )
                if path in seen:
                    raise ArtifactValidationError(
                        f"manifest figure {name}.{list_name} contains duplicate paths"
                    )
                if list_name == "source_data":
                    if path.suffix != ".csv":
                        raise ArtifactValidationError(
                            f"manifest figure {name} source data must be CSV"
                        )
                    if path in owned_sources:
                        raise ArtifactValidationError(
                            "manifest figures assign one source-data file to multiple figures"
                        )
                    owned_sources.add(path)
                seen.add(path)
                normalized.append(file_record)
            normalized_lists[list_name] = normalized

        outputs = record["outputs"]
        if type(outputs) is not dict or set(outputs) != _FIGURE_FORMATS:
            raise ArtifactValidationError(
                f"manifest figure {name}.outputs must contain exactly "
                f"{sorted(_FIGURE_FORMATS)}"
            )
        normalized_outputs: dict[str, dict[str, str]] = {}
        for kind in sorted(_FIGURE_FORMATS):
            path, file_record = _validated_file_record(
                outputs[kind], root, f"figures.{name}.outputs.{kind}"
            )
            if path.suffix != f".{kind}" or path.stem != name:
                raise ArtifactValidationError(
                    f"manifest figure {name} output path does not match {kind}"
                )
            if path in owned_outputs:
                raise ArtifactValidationError("manifest figure outputs contain duplicate paths")
            owned_outputs.add(path)
            normalized_outputs[kind] = file_record

        theory_refs = record["theory_refs"]
        if (
            type(theory_refs) is not list
            or not theory_refs
            or any(type(item) is not str or not item.strip() for item in theory_refs)
            or len(set(theory_refs)) != len(theory_refs)
        ):
            raise ArtifactValidationError(
                f"manifest figure {name}.theory_refs must be unique nonempty strings"
            )
        defensive[name] = {
            "script": str(script_raw),
            "script_sha256": script_sha,
            "inputs": normalized_lists["inputs"],
            "source_data": normalized_lists["source_data"],
            "outputs": normalized_outputs,
            "theory_refs": list(theory_refs),
        }
    return defensive


def _artifact_schema_validate(artifact: Path, sidecar: Path) -> dict[str, Any]:
    try:
        _, metadata = validate_artifact(artifact, sidecar)
    except ArtifactValidationError:
        raise
    except (OSError, TypeError, ValueError) as validation_error:
        raise ArtifactValidationError(
            f"artifact schema validation failed: {validation_error}"
        ) from validation_error
    return metadata


def _validated_record(
    name: str,
    value: object,
    root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    artifact_pure, sidecar_pure = _record_structure(name, value)
    assert type(value) is dict  # Narrowed by _record_structure.
    artifact = _resolve_record_path(value["path"], root, f"{name}.path")
    sidecar = _resolve_record_path(value["sidecar"], root, f"{name}.sidecar")
    if artifact_pure.with_suffix(".json") != sidecar_pure:
        raise ArtifactValidationError(f"manifest sidecar path does not match {name}")

    for field, target in (("sha256", artifact), ("sidecar_sha256", sidecar)):
        claimed = value[field]
        assert type(claimed) is str  # Narrowed by _record_structure.
        actual = sha256_file(target)
        if not hmac.compare_digest(claimed, actual):
            raise ArtifactValidationError(f"manifest {name}.{field} checksum mismatch")

    metadata = _artifact_schema_validate(artifact, sidecar)
    return artifact, sidecar, metadata


def _record_structure(
    name: str,
    value: object,
) -> tuple[PurePosixPath, PurePosixPath]:
    if type(value) is not dict or set(value) != _RECORD_KEYS:
        raise ArtifactValidationError(
            f"manifest artifact record {name} must contain exactly {sorted(_RECORD_KEYS)}"
        )
    artifact = _pure_record_path(value["path"], f"{name}.path")
    sidecar = _pure_record_path(value["sidecar"], f"{name}.sidecar")
    if artifact.suffix != ".npz" or artifact.stem != name:
        raise ArtifactValidationError(f"manifest artifact path does not match {name}")
    if sidecar != artifact.with_suffix(".json"):
        raise ArtifactValidationError(f"manifest sidecar path does not match {name}")
    for field in ("sha256", "sidecar_sha256"):
        claimed = value[field]
        if type(claimed) is not str or _SHA256.fullmatch(claimed) is None:
            raise ArtifactValidationError(f"manifest {name}.{field} is malformed")
    return artifact, sidecar


def _validate_structure(
    payload: object,
    *,
    complete: bool,
) -> dict[str, object]:
    if type(payload) is not dict:
        raise ArtifactValidationError("provenance manifest root must be an object")
    if _manual_values_present(payload):
        raise ArtifactValidationError("manual figure values are forbidden")
    if set(payload) != _ROOT_KEYS:
        missing = sorted(_ROOT_KEYS.difference(payload))
        extra = sorted(set(payload).difference(_ROOT_KEYS))
        raise ArtifactValidationError(
            f"manifest root keys are invalid: missing={missing}, unexpected={extra}"
        )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ArtifactValidationError(f"manifest schema_version must equal {SCHEMA_VERSION}")
    artifacts = payload["artifacts"]
    if type(artifacts) is not dict:
        raise ArtifactValidationError("manifest artifacts must be an object")
    names = set(artifacts)
    unexpected = names.difference(ARTIFACT_NAMES)
    missing = ARTIFACT_NAMES.difference(names)
    if unexpected or (complete and missing):
        raise ArtifactValidationError(
            "manifest artifact set is invalid: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name in sorted(artifacts):
        if type(name) is not str:
            raise ArtifactValidationError("manifest artifact names must be strings")
        _record_structure(name, artifacts[name])
    return artifacts


def _validate_payload(
    payload: object,
    manifest: Path,
    *,
    complete: bool,
    require_figures: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    artifacts = _validate_structure(payload, complete=complete)
    assert type(payload) is dict
    figures = _validate_figure_records(
        payload["figures"], manifest, require_complete=require_figures
    )

    root = _project_root(manifest)
    seen_paths: set[Path] = set()
    metadata_by_name: dict[str, dict[str, Any]] = {}
    for name in sorted(artifacts):
        artifact, sidecar, metadata = _validated_record(name, artifacts[name], root)
        metadata_by_name[name] = metadata
        for target in (artifact, sidecar):
            if target in seen_paths:
                raise ArtifactValidationError("manifest records contain duplicate paths")
            seen_paths.add(target)
    _validate_scientific_context(metadata_by_name)
    if complete:
        _validate_dependency_closure(artifacts, metadata_by_name)
    defensive = json.loads(_canonical_json(payload).decode("utf-8"))
    defensive["figures"] = figures
    return defensive, metadata_by_name


def _scientific_context(metadata: dict[str, Any]) -> tuple[object, ...]:
    return tuple(metadata[key] for key in _CONTEXT_KEYS)


def _validate_scientific_context(
    metadata_by_name: dict[str, dict[str, Any]],
) -> None:
    if not metadata_by_name:
        return
    reference_name = min(metadata_by_name)
    reference = metadata_by_name[reference_name]
    for name, metadata in metadata_by_name.items():
        for key in _CONTEXT_KEYS:
            if metadata[key] != reference[key]:
                raise ArtifactValidationError(
                    f"manifest scientific context mismatch for {key}: "
                    f"{reference_name} != {name}"
                )


def _validate_dependency_closure(
    artifacts: dict[str, object],
    metadata_by_name: dict[str, dict[str, Any]],
) -> None:
    by_stage = {
        str(metadata["stage"]): (name, metadata)
        for name, metadata in metadata_by_name.items()
    }
    for name, metadata in metadata_by_name.items():
        inputs = metadata["input_artifacts"]
        assert type(inputs) is dict  # Enforced by the exact artifact schema.
        for dependency, claimed in inputs.items():
            if dependency not in by_stage:
                raise ArtifactValidationError(
                    f"manifest dependency closure is missing stage {dependency}"
                )
            upstream_name, upstream_metadata = by_stage[dependency]
            upstream_record = artifacts[upstream_name]
            assert type(upstream_record) is dict
            assert type(claimed) is dict
            expected = {
                "artifact": upstream_name,
                "output_sha256": upstream_record["sha256"],
                "cache_key": upstream_metadata["cache_key"],
            }
            if claimed != expected:
                raise ArtifactValidationError(
                    f"manifest dependency closure mismatch: {name} -> {dependency}"
                )


def _record_for(
    manifest: Path,
    artifact: Path,
    sidecar: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    root = _project_root(manifest)
    artifact_path = _relative_record_path(artifact, root, "artifact path")
    sidecar_path = _relative_record_path(sidecar, root, "sidecar path")
    if artifact.suffix != ".npz" or artifact.stem not in ARTIFACT_NAMES:
        raise ArtifactValidationError("unknown manifest artifact")
    if sidecar.resolve(strict=True) != artifact.resolve(strict=True).with_suffix(".json"):
        raise ArtifactValidationError("artifact and sidecar paths do not form a pair")
    metadata = _artifact_schema_validate(
        artifact.resolve(strict=True), sidecar.resolve(strict=True)
    )
    record = {
        "path": artifact_path,
        "sha256": sha256_file(artifact),
        "sidecar": sidecar_path,
        "sidecar_sha256": sha256_file(sidecar),
    }
    return record, metadata


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _prune_invalidated_dependents(
    artifacts: dict[str, object],
    metadata_by_name: dict[str, dict[str, Any]],
    incoming_name: str,
    incoming_record: dict[str, str],
    incoming_metadata: dict[str, Any],
) -> tuple[dict[str, object], dict[str, dict[str, Any]]]:
    """Remove stale descendants before publishing a replacement upstream."""

    retained = dict(artifacts)
    retained_metadata = dict(metadata_by_name)
    incoming_stage = str(incoming_metadata["stage"])
    expected_lineage = {
        "artifact": incoming_name,
        "output_sha256": incoming_record["sha256"],
        "cache_key": incoming_metadata["cache_key"],
    }
    removed_stages: set[str] = set()
    for name, metadata in tuple(retained_metadata.items()):
        inputs = metadata["input_artifacts"]
        assert type(inputs) is dict
        claimed = inputs.get(incoming_stage)
        if claimed is not None and claimed != expected_lineage:
            retained.pop(name, None)
            retained_metadata.pop(name, None)
            removed_stages.add(str(metadata["stage"]))

    while removed_stages:
        newly_removed: set[str] = set()
        for name, metadata in tuple(retained_metadata.items()):
            inputs = metadata["input_artifacts"]
            assert type(inputs) is dict
            if set(inputs).intersection(removed_stages):
                retained.pop(name, None)
                retained_metadata.pop(name, None)
                newly_removed.add(str(metadata["stage"]))
        removed_stages = newly_removed
    return retained, retained_metadata


def _upgrade_legacy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Upgrade the closed artifact-only v1 root before an authorized update."""

    if set(payload) == {"schema_version", "artifacts"} and payload.get(
        "schema_version"
    ) == 1:
        upgraded = dict(payload)
        upgraded["schema_version"] = SCHEMA_VERSION
        upgraded["figures"] = {}
        return upgraded
    return payload


def update_manifest(path: Path, artifact: Path, sidecar: Path) -> dict[str, Any]:
    """Atomically insert or replace one validated artifact record."""

    manifest_argument = _path(path, "path")
    if manifest_argument.is_symlink():
        raise ArtifactValidationError("manifest target must not be a symbolic link")
    manifest = manifest_argument.resolve(strict=False)
    artifact_path = _path(artifact, "artifact")
    sidecar_path = _path(sidecar, "sidecar")
    if manifest.suffix != ".json":
        raise ValueError("path must end with .json")
    if manifest.exists() and not manifest.is_file():
        raise ArtifactValidationError("manifest target must be a regular JSON file")

    if manifest.exists():
        payload = _upgrade_legacy_payload(_read_json(manifest))
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifacts": {},
            "figures": {},
        }
    _validate_structure(payload, complete=False)

    name = artifact_path.stem
    prior = dict(payload)
    prior["artifacts"] = dict(payload["artifacts"])
    prior["artifacts"].pop(name, None)
    prior["figures"] = {}
    validated_prior, prior_metadata = _validate_payload(
        prior,
        manifest,
        complete=False,
    )

    record, incoming_metadata = _record_for(manifest, artifact_path, sidecar_path)
    if prior_metadata and any(
        _scientific_context(metadata) != _scientific_context(incoming_metadata)
        for metadata in prior_metadata.values()
    ):
        retained_artifacts: dict[str, object] = {}
    else:
        retained_artifacts, _ = _prune_invalidated_dependents(
            validated_prior["artifacts"],
            prior_metadata,
            name,
            record,
            incoming_metadata,
        )

    candidate = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": retained_artifacts,
        "figures": {},
    }
    candidate["artifacts"][name] = record
    complete = set(candidate["artifacts"]) == ARTIFACT_NAMES
    validated, _ = _validate_payload(candidate, manifest, complete=complete)
    _atomic_write(manifest, _canonical_json(validated))
    return validated


def replace_figure_records(
    path: Path,
    figures: dict[str, Any],
) -> dict[str, Any]:
    """Atomically replace the complete figure closure after artifact validation."""

    manifest_argument = _path(path, "path")
    if manifest_argument.is_symlink():
        raise ArtifactValidationError("manifest target must not be a symbolic link")
    manifest = manifest_argument.resolve(strict=False)
    if not manifest.is_file():
        raise ArtifactValidationError("provenance manifest is missing or unsafe")
    if type(figures) is not dict:
        raise TypeError("figures must be a dict")

    payload = _upgrade_legacy_payload(_read_json(manifest))
    without_figures = dict(payload)
    without_figures["figures"] = {}
    validated_base, _ = _validate_payload(
        without_figures,
        manifest,
        complete=True,
        require_figures=False,
    )
    candidate = dict(validated_base)
    candidate["figures"] = figures
    validated, _ = _validate_payload(
        candidate,
        manifest,
        complete=True,
        require_figures=True,
    )
    _atomic_write(manifest, _canonical_json(validated))
    return validated


def validate_manifest(
    path: Path,
    *,
    require_figures: bool = False,
) -> dict[str, Any]:
    """Validate the closed scientific manifest and optional figure closure."""

    manifest_argument = _path(path, "path")
    if type(require_figures) is not bool:
        raise TypeError("require_figures must be a bool")
    if manifest_argument.is_symlink():
        raise ArtifactValidationError("provenance manifest path is a symbolic link")
    manifest = manifest_argument.resolve(strict=False)
    if manifest.is_symlink() or not manifest.is_file():
        raise ArtifactValidationError("provenance manifest is missing or unsafe")
    payload = _read_json(manifest)
    validated, _ = _validate_payload(
        payload,
        manifest,
        complete=True,
        require_figures=require_figures,
    )
    return validated
