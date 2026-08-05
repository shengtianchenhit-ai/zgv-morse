from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
from typing import Any, Callable

import numpy as np
import pytest

import zgv_morse.provenance as provenance
from zgv_morse.artifact_schema import CACHE_SCHEMA_VERSION, compute_cache_key
from zgv_morse.artifacts import sha256_file, validate_npz_sidecar, write_npz_with_sidecar
from zgv_morse.provenance import ArtifactValidationError
from zgv_morse.provenance import replace_figure_records, update_manifest, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "data" / "generated"
ARTIFACT_NAMES = (
    "angular_sensitivity",
    "convergence",
    "critical_points",
    "green_crossover",
    "isotropic_zgv",
    "perturbation_scaling",
    "silicon_stress_test",
)


def _copy_generated_pair(directory: Path, name: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"{name}.npz"
    sidecar = directory / f"{name}.json"
    source = GENERATED / artifact.name
    report = validate_npz_sidecar(source)
    with np.load(source, allow_pickle=False) as bundle:
        arrays = {key: np.array(bundle[key], copy=True) for key in bundle.files}
    metadata = dict(report["metadata"])
    for generated_key in ("arrays", "output_sha256", "metadata_sha256"):
        metadata.pop(generated_key, None)
    metadata["cache_schema_version"] = CACHE_SCHEMA_VERSION
    metadata["input_artifacts"] = {}
    metadata["cache_key"] = compute_cache_key(
        stage=metadata["stage"],
        profile=metadata["profile"],
        config_hash=metadata["config_hash"],
        source_hash=metadata["source_hash"],
        code_hash=metadata["code_hash"],
        uv_lock_hash=metadata["uv_lock_hash"],
        input_artifacts={},
    )
    write_npz_with_sidecar(artifact, arrays, metadata)
    return artifact, sidecar


def _complete_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "provenance_manifest.json"
    for name in reversed(ARTIFACT_NAMES):
        artifact, sidecar = _copy_generated_pair(tmp_path / "generated", name)
        update_manifest(manifest, artifact, sidecar)
    return manifest


def _file_record(path: Path, project_root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(project_root).as_posix(),
        "sha256": sha256_file(path),
    }


def _complete_figure_records(project_root: Path) -> dict[str, dict[str, Any]]:
    script = project_root / "scripts/generate_figures.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# deterministic test fixture\n", encoding="utf-8")
    shared_input = project_root / "data/generated/figure_input.json"
    shared_input.parent.mkdir(parents=True, exist_ok=True)
    shared_input.write_text('{"fixture": true}\n', encoding="utf-8")

    records: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(sorted(provenance.FIGURE_NAMES)):
        source = project_root / f"data/source_data/{name}.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"x,y\n{index},{index + 1}\n", encoding="utf-8")
        outputs: dict[str, dict[str, str]] = {}
        for kind in ("pdf", "png", "svg", "tiff"):
            output = project_root / f"figures/{name}.{kind}"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"{name}:{kind}\n".encode())
            outputs[kind] = _file_record(output, project_root)
        records[name] = {
            "script": script.relative_to(project_root).as_posix(),
            "script_sha256": sha256_file(script),
            "inputs": [_file_record(shared_input, project_root)],
            "source_data": [_file_record(source, project_root)],
            "outputs": outputs,
            "theory_refs": [f"thm:test-{index:02d}"],
        }
    return records


def _rewrite_manifest(
    manifest: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutate(payload)
    manifest.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _rewrite_pair_metadata(
    artifact: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    report = validate_npz_sidecar(artifact)
    with np.load(artifact, allow_pickle=False) as bundle:
        arrays = {key: np.array(bundle[key], copy=True) for key in bundle.files}
    metadata = dict(report["metadata"])
    for generated_key in ("arrays", "output_sha256", "metadata_sha256"):
        metadata.pop(generated_key)
    mutate(metadata)
    metadata["cache_key"] = compute_cache_key(
        stage=metadata["stage"],
        profile=metadata["profile"],
        config_hash=metadata["config_hash"],
        source_hash=metadata["source_hash"],
        code_hash=metadata["code_hash"],
        uv_lock_hash=metadata["uv_lock_hash"],
        input_artifacts=metadata["input_artifacts"],
    )
    write_npz_with_sidecar(artifact, arrays, metadata)


def test_manifest_is_deterministic_complete_and_independent_of_cwd(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _complete_manifest(tmp_path)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest.read_text(encoding="utf-8") == (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    assert tuple(payload) == ("artifacts", "figures", "schema_version")
    assert tuple(payload["artifacts"]) == ARTIFACT_NAMES
    assert payload["figures"] == {}
    assert all(
        record["path"] == f"generated/{name}.npz"
        and record["sidecar"] == f"generated/{name}.json"
        for name, record in payload["artifacts"].items()
    )

    elsewhere = tmp_path / "unrelated-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    validated = validate_manifest(manifest)

    assert tuple(validated["artifacts"]) == ARTIFACT_NAMES


def test_update_uses_the_exact_artifact_schema_when_pair_name_is_spoofed(tmp_path: Path):
    original, original_sidecar = _copy_generated_pair(
        tmp_path / "generated", "isotropic_zgv"
    )
    artifact = tmp_path / "generated" / "angular_sensitivity.npz"
    sidecar = artifact.with_suffix(".json")
    original.replace(artifact)
    original_sidecar.replace(sidecar)

    with pytest.raises(ArtifactValidationError, match="angular_sensitivity"):
        update_manifest(tmp_path / "provenance_manifest.json", artifact, sidecar)


def test_manifest_rejects_duplicate_json_keys(tmp_path: Path):
    manifest = _complete_manifest(tmp_path)
    text = manifest.read_text(encoding="utf-8").replace(
        '"schema_version": 2',
        '"schema_version": 2, "schema_version": 2',
        1,
    )
    manifest.write_text(text, encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="duplicate"):
        validate_manifest(manifest)


def test_manifest_forbids_manual_figure_values_at_any_depth(tmp_path: Path):
    manifest = _complete_manifest(tmp_path)

    def add_manual_value(payload: dict[str, Any]) -> None:
        record = payload["artifacts"][ARTIFACT_NAMES[0]]
        record["audit"] = {"nested": [{"Manual-Figure-Values": [1.23]}]}

    _rewrite_manifest(manifest, add_manual_value)

    with pytest.raises(ArtifactValidationError, match="manual figure values"):
        validate_manifest(manifest)


def test_manifest_forbids_a_nested_manual_value_marker_string(tmp_path: Path):
    manifest = _complete_manifest(tmp_path)

    def add_manual_marker(payload: dict[str, Any]) -> None:
        record = payload["artifacts"][ARTIFACT_NAMES[0]]
        record["audit"] = {"nested": ["manual_figure_values"]}

    _rewrite_manifest(manifest, add_manual_marker)

    with pytest.raises(ArtifactValidationError, match="manual figure values"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.pop("schema_version"), "root keys"),
        (lambda payload: payload.__setitem__("unexpected", {}), "root keys"),
        (lambda payload: payload.__setitem__("schema_version", True), "schema_version"),
        (
            lambda payload: payload["artifacts"].pop(ARTIFACT_NAMES[0]),
            "artifact set",
        ),
        (
            lambda payload: payload["artifacts"][ARTIFACT_NAMES[0]].__setitem__(
                "unexpected", "value"
            ),
            "exactly",
        ),
        (
            lambda payload: payload["artifacts"][ARTIFACT_NAMES[0]].pop("sha256"),
            "exactly",
        ),
    ],
)
def test_manifest_rejects_missing_extra_or_malformed_structure(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], object],
    match: str,
):
    manifest = _complete_manifest(tmp_path)
    _rewrite_manifest(manifest, mutate)

    with pytest.raises(ArtifactValidationError, match=match):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    ("suffix", "field"),
    [(".npz", "sha256"), (".json", "sidecar_sha256")],
)
def test_manifest_detects_artifact_and_sidecar_tampering(
    tmp_path: Path,
    suffix: str,
    field: str,
):
    manifest = _complete_manifest(tmp_path)
    target = tmp_path / "generated" / f"{ARTIFACT_NAMES[0]}{suffix}"
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(ArtifactValidationError, match=rf"{field} checksum mismatch"):
        validate_manifest(manifest)


@pytest.mark.parametrize("suffix", [".npz", ".json"])
def test_internal_pair_integrity_is_checked_after_manifest_hash_is_recomputed(
    tmp_path: Path,
    suffix: str,
):
    manifest = _complete_manifest(tmp_path)
    name = ARTIFACT_NAMES[0]
    target = tmp_path / "generated" / f"{name}{suffix}"
    target.write_bytes(target.read_bytes() + b"tampered")

    def bless_outer_hash(payload: dict[str, Any]) -> None:
        field = "sha256" if suffix == ".npz" else "sidecar_sha256"
        payload["artifacts"][name][field] = sha256_file(target)

    _rewrite_manifest(manifest, bless_outer_hash)

    with pytest.raises(ArtifactValidationError, match="checksum|JSON"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "unsafe",
    [
        "/absolute/artifact.npz",
        "../generated/angular_sensitivity.npz",
        "generated\\angular_sensitivity.npz",
        "generated/./angular_sensitivity.npz",
    ],
)
def test_manifest_rejects_absolute_escaping_or_noncanonical_paths(
    tmp_path: Path,
    unsafe: str,
):
    manifest = _complete_manifest(tmp_path)

    def replace_path(payload: dict[str, Any]) -> None:
        payload["artifacts"][ARTIFACT_NAMES[0]]["path"] = unsafe

    _rewrite_manifest(manifest, replace_path)

    with pytest.raises(ArtifactValidationError, match="unsafe path"):
        validate_manifest(manifest)


def test_update_is_atomic_when_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = tmp_path / "provenance_manifest.json"
    first, first_sidecar = _copy_generated_pair(
        tmp_path / "generated", ARTIFACT_NAMES[0]
    )
    update_manifest(manifest, first, first_sidecar)
    before = manifest.read_bytes()
    second, second_sidecar = _copy_generated_pair(
        tmp_path / "generated", ARTIFACT_NAMES[1]
    )
    real_replace = provenance.os.replace

    def fail_manifest_publish(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == manifest:
            raise OSError("injected manifest publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(provenance.os, "replace", fail_manifest_publish)

    with pytest.raises(OSError, match="injected"):
        update_manifest(manifest, second, second_sidecar)

    assert manifest.read_bytes() == before
    assert list(tmp_path.glob(f".{manifest.name}.*.tmp")) == []


def test_update_refuses_a_tampered_preexisting_record_without_modifying_manifest(
    tmp_path: Path,
):
    manifest = tmp_path / "provenance_manifest.json"
    first, first_sidecar = _copy_generated_pair(
        tmp_path / "generated", ARTIFACT_NAMES[0]
    )
    update_manifest(manifest, first, first_sidecar)
    second, second_sidecar = _copy_generated_pair(
        tmp_path / "generated", ARTIFACT_NAMES[1]
    )
    first.write_bytes(first.read_bytes() + b"tampered")
    before = manifest.read_bytes()

    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        update_manifest(manifest, second, second_sidecar)

    assert manifest.read_bytes() == before


def test_same_name_update_does_not_erase_forbidden_manual_values(tmp_path: Path):
    manifest = tmp_path / "provenance_manifest.json"
    artifact, sidecar = _copy_generated_pair(
        tmp_path / "generated", ARTIFACT_NAMES[0]
    )
    update_manifest(manifest, artifact, sidecar)

    def add_manual_value(payload: dict[str, Any]) -> None:
        payload["artifacts"][ARTIFACT_NAMES[0]]["manual_figure_values"] = [1.23]

    _rewrite_manifest(manifest, add_manual_value)
    before = manifest.read_bytes()

    with pytest.raises(ArtifactValidationError, match="manual figure values"):
        update_manifest(manifest, artifact, sidecar)

    assert manifest.read_bytes() == before


def test_same_name_update_rejects_a_malformed_old_record(tmp_path: Path):
    manifest = tmp_path / "provenance_manifest.json"
    artifact, sidecar = _copy_generated_pair(
        tmp_path / "generated", ARTIFACT_NAMES[0]
    )
    update_manifest(manifest, artifact, sidecar)

    def add_unexpected_field(payload: dict[str, Any]) -> None:
        payload["artifacts"][ARTIFACT_NAMES[0]]["unexpected"] = "value"

    _rewrite_manifest(manifest, add_unexpected_field)
    before = manifest.read_bytes()

    with pytest.raises(ArtifactValidationError, match="exactly"):
        update_manifest(manifest, artifact, sidecar)

    assert manifest.read_bytes() == before


@pytest.mark.parametrize("value", ["manifest.json", 1, None])
def test_manifest_public_paths_require_path_instances(tmp_path: Path, value: object):
    artifact, sidecar = _copy_generated_pair(
        tmp_path / "generated", ARTIFACT_NAMES[0]
    )
    with pytest.raises(TypeError, match="Path"):
        update_manifest(value, artifact, sidecar)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Path"):
        validate_manifest(value)  # type: ignore[arg-type]


def test_manifest_path_must_not_be_a_symbolic_link(tmp_path: Path):
    manifest = _complete_manifest(tmp_path)
    alias = tmp_path / "manifest-alias.json"
    alias.symlink_to(manifest)
    artifact = tmp_path / "generated" / f"{ARTIFACT_NAMES[0]}.npz"

    with pytest.raises(ArtifactValidationError, match="unsafe|symbolic"):
        validate_manifest(alias)
    with pytest.raises(ArtifactValidationError, match="unsafe|symbolic"):
        update_manifest(alias, artifact, artifact.with_suffix(".json"))


def test_update_rejects_an_existing_nonregular_manifest_target():
    with tempfile.TemporaryDirectory(prefix="zgv-prov-", dir="/tmp") as raw_directory:
        directory = Path(raw_directory)
        manifest = directory / "manifest.json"
        artifact, sidecar = _copy_generated_pair(
            directory / "generated", ARTIFACT_NAMES[0]
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as endpoint:
            endpoint.bind(manifest.as_posix())
            with pytest.raises(ArtifactValidationError, match="regular"):
                update_manifest(manifest, artifact, sidecar)


def test_update_starts_a_new_partial_manifest_when_scientific_context_changes(
    tmp_path: Path,
):
    manifest = _complete_manifest(tmp_path)
    artifact = tmp_path / "generated" / "convergence.npz"
    _rewrite_pair_metadata(
        artifact,
        lambda metadata: metadata.__setitem__("code_hash", "0" * 64),
    )
    updated = update_manifest(manifest, artifact, artifact.with_suffix(".json"))

    assert set(updated["artifacts"]) == {"convergence"}
    with pytest.raises(ArtifactValidationError, match="artifact set.*missing"):
        validate_manifest(manifest)


def test_validator_rejects_a_hash_consistent_mixed_scientific_context(
    tmp_path: Path,
) -> None:
    manifest = _complete_manifest(tmp_path)
    artifact = tmp_path / "generated" / "convergence.npz"
    sidecar = artifact.with_suffix(".json")
    _rewrite_pair_metadata(
        artifact,
        lambda metadata: metadata.__setitem__("code_hash", "0" * 64),
    )

    def bless_changed_pair(payload: dict[str, Any]) -> None:
        record = payload["artifacts"]["convergence"]
        record["sha256"] = sha256_file(artifact)
        record["sidecar_sha256"] = sha256_file(sidecar)

    _rewrite_manifest(manifest, bless_changed_pair)
    with pytest.raises(ArtifactValidationError, match="scientific context.*code_hash"):
        validate_manifest(manifest)


def test_manifest_closes_declared_dependency_against_actual_upstream_record(
    tmp_path: Path,
):
    manifest = _complete_manifest(tmp_path)
    artifact = tmp_path / "generated" / "critical_points.npz"

    def claim_unrelated_sensitivity(metadata: dict[str, Any]) -> None:
        metadata["input_artifacts"] = {
            "sensitivity": {
                "artifact": "angular_sensitivity",
                "output_sha256": "0" * 64,
                "cache_key": "1" * 64,
            }
        }

    _rewrite_pair_metadata(artifact, claim_unrelated_sensitivity)
    before = manifest.read_bytes()

    with pytest.raises(ArtifactValidationError, match="dependency closure mismatch"):
        update_manifest(manifest, artifact, artifact.with_suffix(".json"))
    assert manifest.read_bytes() == before


def test_replace_figure_records_publishes_a_deterministic_complete_closure(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    manifest = _complete_manifest(project_root / "data")
    figures = _complete_figure_records(project_root)

    first = replace_figure_records(manifest, figures)
    first_bytes = manifest.read_bytes()
    second = replace_figure_records(manifest, deepcopy(figures))

    assert set(first["figures"]) == provenance.FIGURE_NAMES
    assert second == first
    assert manifest.read_bytes() == first_bytes
    assert validate_manifest(manifest, require_figures=False) == first
    assert validate_manifest(manifest, require_figures=True) == first


def test_replace_figure_records_rejects_incomplete_or_nonexact_records_atomically(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    manifest = _complete_manifest(project_root / "data")
    figures = _complete_figure_records(project_root)
    before = manifest.read_bytes()
    name = min(figures)

    incomplete = deepcopy(figures)
    incomplete.pop(name)
    unexpected_field = deepcopy(figures)
    unexpected_field[name]["unexpected"] = "forbidden"
    incomplete_outputs = deepcopy(figures)
    incomplete_outputs[name]["outputs"].pop("svg")
    duplicate_theory = deepcopy(figures)
    duplicate_theory[name]["theory_refs"] *= 2

    for malformed, match in (
        (incomplete, "figure set is invalid"),
        (unexpected_field, "must contain exactly"),
        (incomplete_outputs, "outputs must contain exactly"),
        (duplicate_theory, "must be unique"),
    ):
        with pytest.raises(ArtifactValidationError, match=match):
            replace_figure_records(manifest, malformed)
        assert manifest.read_bytes() == before


@pytest.mark.parametrize("require_figures", [False, True])
def test_validate_manifest_rejects_a_nonempty_partial_figure_closure(
    tmp_path: Path,
    require_figures: bool,
) -> None:
    project_root = tmp_path / "project"
    manifest = _complete_manifest(project_root / "data")
    figures = _complete_figure_records(project_root)
    replace_figure_records(manifest, figures)

    def remove_one_figure(payload: dict[str, Any]) -> None:
        payload["figures"].pop(min(payload["figures"]))

    _rewrite_manifest(manifest, remove_one_figure)

    with pytest.raises(ArtifactValidationError, match="figure set is invalid"):
        validate_manifest(manifest, require_figures=require_figures)


def test_manifest_rejects_tampering_inside_the_figure_closure(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    manifest = _complete_manifest(project_root / "data")
    figures = _complete_figure_records(project_root)
    replace_figure_records(manifest, figures)

    source = project_root / figures[min(figures)]["source_data"][0]["path"]
    source.write_bytes(source.read_bytes() + b"tampered")

    with pytest.raises(ArtifactValidationError, match="source_data.*checksum mismatch"):
        validate_manifest(manifest, require_figures=True)


def test_scientific_artifact_update_resets_the_figure_closure(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    manifest = _complete_manifest(project_root / "data")
    figures = _complete_figure_records(project_root)
    replace_figure_records(manifest, figures)
    artifact = project_root / "data/generated/isotropic_zgv.npz"

    updated = update_manifest(manifest, artifact, artifact.with_suffix(".json"))

    assert updated["figures"] == {}
    assert validate_manifest(manifest)["figures"] == {}
    with pytest.raises(ArtifactValidationError, match="figure set is invalid"):
        validate_manifest(manifest, require_figures=True)
