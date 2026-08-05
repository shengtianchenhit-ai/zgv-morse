from __future__ import annotations

from collections.abc import Callable
import hashlib
import io
import json
import os
from pathlib import Path
import warnings
import zipfile

import numpy as np
from numpy.testing import assert_array_equal
import pytest

import zgv_morse.artifacts as artifacts
from zgv_morse.artifacts import sha256_file, validate_npz_sidecar, write_npz_with_sidecar


def _metadata() -> dict[str, object]:
    return {
        "units": {"theta": "rad", "V": "Omega per epsilon"},
        "dimensionless_convention": "kappa=k*h, Omega=omega*h/c_T",
        "config_hash": "config-test",
        "source_hash": "source-test",
        "code_hash": "code-test",
        "tolerances": {"relative_sensitivity": 1.0e-4},
    }


def _arrays(offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "theta": np.linspace(0.0, 1.0, 8) + offset,
        "V": np.linspace(-0.2, 0.2, 8) - offset,
    }


def _metadata_checksum(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _rewrite_valid_sidecar(
    path: Path,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload.pop("metadata_sha256")
    mutate(payload)
    payload["metadata_sha256"] = _metadata_checksum(payload)
    sidecar.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _npy_bytes(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, values, allow_pickle=False)
    return output.getvalue()


def test_npz_sidecar_round_trip_records_keys_shapes_dtypes_and_checksums(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.npz"

    written = write_npz_with_sidecar(path, _arrays(), _metadata())
    report = validate_npz_sidecar(path)

    assert written == report
    assert report["keys"] == ["V", "theta"]
    assert report["shapes"] == {"V": [8], "theta": [8]}
    assert report["dtypes"] == {"V": "float64", "theta": "float64"}
    assert report["metadata"]["output_sha256"] == sha256_file(path)
    assert path.with_suffix(".json").is_file()
    with np.load(path, allow_pickle=False) as bundle:
        assert_array_equal(bundle["theta"], _arrays()["theta"])
        assert_array_equal(bundle["V"], _arrays()["V"])

    # Returned dictionaries are independent validation reports, not live state.
    report["shapes"]["V"][0] = 99
    report["metadata"]["units"]["V"] = "tampered"
    fresh = validate_npz_sidecar(path)
    assert fresh["shapes"]["V"] == [8]
    assert fresh["metadata"]["units"]["V"] == "Omega per epsilon"


def test_npz_payload_tampering_is_detected_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    write_npz_with_sidecar(path, _arrays(), _metadata())

    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="checksum"):
        validate_npz_sidecar(path)


def test_identical_content_produces_a_stable_archive_checksum(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    write_npz_with_sidecar(path, _arrays(), _metadata())
    first = sha256_file(path)

    write_npz_with_sidecar(path, _arrays(), _metadata())

    assert sha256_file(path) == first


def test_sidecar_tampering_missing_and_malformed_json_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    write_npz_with_sidecar(path, _arrays(), _metadata())
    sidecar = path.with_suffix(".json")

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["code_hash"] = "changed-without-updating-integrity"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata checksum"):
        validate_npz_sidecar(path)

    sidecar.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar JSON"):
        validate_npz_sidecar(path)

    sidecar.unlink()
    with pytest.raises(ValueError, match="sidecar"):
        validate_npz_sidecar(path)


def test_sidecar_rejects_duplicate_json_keys_even_when_last_value_is_valid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.npz"
    write_npz_with_sidecar(path, _arrays(), _metadata())
    sidecar = path.with_suffix(".json")
    text = sidecar.read_text(encoding="utf-8").replace(
        '"code_hash": "code-test",',
        '"code_hash": "conflicting-first-value",\n  "code_hash": "code-test",',
        1,
    )
    sidecar.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar JSON"):
        validate_npz_sidecar(path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["arrays"]["V"].__setitem__("shape", [7]),
            "shape metadata mismatch",
        ),
        (
            lambda payload: payload["arrays"]["V"].__setitem__("dtype", "float32"),
            "dtype metadata mismatch",
        ),
        (
            lambda payload: payload["arrays"].pop("V"),
            "key metadata mismatch",
        ),
        (
            lambda payload: payload["arrays"].__setitem__(
                "extra", {"shape": [1], "dtype": "float64"}
            ),
            "key metadata mismatch",
        ),
    ],
)
def test_sidecar_array_schema_must_match_archive_exactly(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    path = tmp_path / "sample.npz"
    write_npz_with_sidecar(path, _arrays(), _metadata())
    _rewrite_valid_sidecar(path, mutate)

    with pytest.raises(ValueError, match=match):
        validate_npz_sidecar(path)


@pytest.mark.parametrize("path_value", ["sample.npz", 3, None])
def test_public_paths_require_path_instances(path_value: object, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Path"):
        write_npz_with_sidecar(path_value, _arrays(), _metadata())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Path"):
        validate_npz_sidecar(path_value)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Path"):
        sha256_file(path_value)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=".npz"):
        write_npz_with_sidecar(tmp_path / "sample.bin", _arrays(), _metadata())


@pytest.mark.parametrize("bad_key", ["", "1theta", "../theta", "a/b", "a.npy", "a b"])
def test_writer_rejects_empty_or_unsafe_array_keys(tmp_path: Path, bad_key: str) -> None:
    with pytest.raises(ValueError, match="array key"):
        write_npz_with_sidecar(
            tmp_path / "sample.npz",
            {bad_key: np.ones(2)},
            _metadata(),
        )


@pytest.mark.parametrize(
    "bad_array",
    [
        np.array([object()], dtype=object),
        np.array([b"one", b"two"]),
        np.array([1.0, np.nan]),
        np.array([1.0 + 0.0j, complex(0.0, np.inf)]),
        np.array([], dtype=float),
    ],
)
def test_writer_accepts_numeric_boolean_or_fixed_width_unicode_arrays(
    tmp_path: Path,
    bad_array: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="array"):
        write_npz_with_sidecar(
            tmp_path / "sample.npz",
            {"bad": bad_array},
            _metadata(),
        )

    report = write_npz_with_sidecar(
        tmp_path / "valid.npz",
        {
            "count": np.arange(3),
            "kind": np.array(["minimum", "saddle"]),
            "mask": np.array([True, False]),
        },
        _metadata(),
    )
    assert report["dtypes"] == {"count": "int64", "kind": "<U7", "mask": "bool"}


def test_writer_requires_nonempty_mapping_inputs_and_required_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    with pytest.raises(TypeError, match="arrays"):
        write_npz_with_sidecar(path, [("x", np.ones(2))], _metadata())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="arrays must not be empty"):
        write_npz_with_sidecar(path, {}, _metadata())
    with pytest.raises(TypeError, match="metadata"):
        write_npz_with_sidecar(path, _arrays(), [])  # type: ignore[arg-type]

    metadata = _metadata()
    metadata.pop("code_hash")
    with pytest.raises(ValueError, match="missing metadata"):
        write_npz_with_sidecar(path, _arrays(), metadata)


def test_numpy_metadata_scalars_are_normalized_to_finite_json_scalars(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["tolerances"] = {
        "accepted": np.bool_(True),
        "count": np.int64(3),
        "relative": np.float64(1.0e-4),
    }

    report = write_npz_with_sidecar(tmp_path / "sample.npz", _arrays(), metadata)

    tolerances = report["metadata"]["tolerances"]
    assert tolerances == {"accepted": True, "count": 3, "relative": 1.0e-4}
    assert type(tolerances["accepted"]) is bool
    assert type(tolerances["count"]) is int
    assert type(tolerances["relative"]) is float


@pytest.mark.parametrize(
    "update",
    [
        {"tolerances": {"bad": np.nan}},
        {"extra": object()},
        {"extra": {1: "non-string key"}},
        {"units": []},
        {"dimensionless_convention": ""},
        {"config_hash": ""},
        {"arrays": {}},
        {"output_sha256": "caller-controlled"},
        {"metadata_sha256": "caller-controlled"},
    ],
)
def test_writer_rejects_non_json_finite_or_reserved_metadata(
    tmp_path: Path,
    update: dict[str, object],
) -> None:
    metadata = _metadata()
    metadata.update(update)
    with pytest.raises((TypeError, ValueError), match="metadata"):
        write_npz_with_sidecar(tmp_path / "sample.npz", _arrays(), metadata)


def test_validation_failure_does_not_modify_an_existing_pair(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    sidecar = path.with_suffix(".json")
    write_npz_with_sidecar(path, _arrays(), _metadata())
    before = (path.read_bytes(), sidecar.read_bytes())

    with pytest.raises(ValueError, match="finite"):
        write_npz_with_sidecar(
            path,
            {"theta": np.array([0.0, np.inf])},
            _metadata(),
        )

    assert (path.read_bytes(), sidecar.read_bytes()) == before
    validate_npz_sidecar(path)


def test_publish_failure_rolls_back_both_existing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sample.npz"
    sidecar = path.with_suffix(".json")
    write_npz_with_sidecar(path, _arrays(), _metadata())
    before = (path.read_bytes(), sidecar.read_bytes())
    real_replace = os.replace
    failed = False

    def fail_first_sidecar_publish(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        if Path(destination) == sidecar and not failed:
            failed = True
            raise OSError("injected sidecar publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_first_sidecar_publish)

    with pytest.raises(OSError, match="injected"):
        write_npz_with_sidecar(path, _arrays(offset=0.5), _metadata())

    assert failed
    assert (path.read_bytes(), sidecar.read_bytes()) == before
    validate_npz_sidecar(path)


def test_failed_rollback_restoration_retains_the_recoverable_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sample.npz"
    sidecar = path.with_suffix(".json")
    write_npz_with_sidecar(path, _arrays(), _metadata())
    original_npz = path.read_bytes()
    original_sidecar = sidecar.read_bytes()
    real_replace = os.replace
    publish_failed = False
    restoration_failed = False

    def fail_publish_then_sidecar_restore(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        nonlocal publish_failed, restoration_failed
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == sidecar and not publish_failed:
            publish_failed = True
            raise OSError("injected sidecar publish failure")
        if (
            destination_path == sidecar
            and source_path.suffix == ".backup"
            and not restoration_failed
        ):
            restoration_failed = True
            raise OSError("injected sidecar restoration failure")
        real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_publish_then_sidecar_restore)

    with pytest.raises(RuntimeError, match="rollback was incomplete") as caught:
        write_npz_with_sidecar(path, _arrays(offset=0.5), _metadata())

    retained = list(tmp_path.glob(".sample.json.*.backup"))
    assert publish_failed and restoration_failed
    assert path.read_bytes() == original_npz
    assert not sidecar.exists()
    assert len(retained) == 1
    assert retained[0].read_bytes() == original_sidecar
    assert str(retained[0]) in str(caught.value)


def test_validator_rejects_unsafe_external_object_archive(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    write_npz_with_sidecar(path, {"x": np.ones(2)}, _metadata())
    with path.open("wb") as handle:
        np.savez_compressed(handle, x=np.array([object()], dtype=object))

    def update(payload: dict[str, object]) -> None:
        payload["output_sha256"] = sha256_file(path)
        payload["arrays"] = {"x": {"shape": [1], "dtype": "object"}}

    _rewrite_valid_sidecar(path, update)
    with pytest.raises(ValueError, match="unsafe array dtype"):
        validate_npz_sidecar(path)


def test_validator_rejects_duplicate_or_unsafe_zip_member_names(tmp_path: Path) -> None:
    for filename, duplicate in (("x.npy", True), ("../x.npy", False)):
        path = tmp_path / ("duplicate.npz" if duplicate else "unsafe.npz")
        write_npz_with_sidecar(path, {"x": np.ones(2)}, _metadata())
        encoded = _npy_bytes(np.ones(2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(filename, encoded)
                if duplicate:
                    archive.writestr(filename, encoded)

        def update(payload: dict[str, object]) -> None:
            payload["output_sha256"] = sha256_file(path)

        _rewrite_valid_sidecar(path, update)
        match = "duplicate" if duplicate else "unsafe"
        with pytest.raises(ValueError, match=match):
            validate_npz_sidecar(path)
