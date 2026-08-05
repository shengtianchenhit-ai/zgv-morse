"""Deterministic, checksummed NPZ artifacts with validated JSON sidecars."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import zipfile

import numpy as np
from numpy.typing import NDArray


REQUIRED_METADATA = frozenset(
    {
        "units",
        "dimensionless_convention",
        "config_hash",
        "source_hash",
        "code_hash",
        "tolerances",
    }
)
_GENERATED_METADATA = frozenset({"arrays", "output_sha256", "metadata_sha256"})
_SAFE_ARRAY_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_KINDS = frozenset("buifc")
_SAFE_ARRAY_KINDS = _NUMERIC_KINDS | {"U"}
_ALLOWED_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})


def _path(value: object, name: str, *, npz: bool = False) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    if npz and value.suffix != ".npz":
        raise ValueError(f"{name} must end with .npz")
    return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a readable regular file."""

    target = _path(path, "path")
    if not target.is_file():
        raise ValueError(f"path is not a readable file: {target}")
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"path is not a readable file: {target}") from error
    return digest.hexdigest()


def _safe_array_key(value: object) -> str:
    if type(value) is not str or _SAFE_ARRAY_KEY.fullmatch(value) is None:
        raise ValueError(
            "array key must start with a letter and contain only letters, digits, or underscores"
        )
    return value


def _numeric_array(value: object, key: str) -> NDArray[np.generic]:
    try:
        candidate = np.asarray(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"array {key} must be a nonempty finite numeric array") from error
    if candidate.size == 0:
        raise ValueError(f"array {key} must not be empty")
    if candidate.dtype.kind not in _SAFE_ARRAY_KINDS:
        raise ValueError(f"array {key} has unsafe array dtype {candidate.dtype}")
    if candidate.dtype.kind in _NUMERIC_KINDS:
        try:
            finite = bool(np.isfinite(candidate).all())
        except (TypeError, ValueError) as error:
            raise ValueError(f"array {key} must be a finite numeric array") from error
        if not finite:
            raise ValueError(f"array {key} entries must be finite")
    return np.array(candidate, copy=True, subok=False)


def _normalized_arrays(values: object) -> dict[str, NDArray[np.generic]]:
    if type(values) is not dict:
        raise TypeError("arrays must be a dict with unique string keys")
    if not values:
        raise ValueError("arrays must not be empty")
    result: dict[str, NDArray[np.generic]] = {}
    for raw_key, raw_array in values.items():
        key = _safe_array_key(raw_key)
        if key in result:  # Defensive for unusual dict subclasses.
            raise ValueError(f"duplicate array key: {key}")
        result[key] = _numeric_array(raw_array, key)
    return result


def _json_safe(value: object, path: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.str_):
        return str(value)
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON values")
        return value
    if isinstance(value, np.floating):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"{path} must contain only finite JSON values")
        return scalar
    if type(value) in (list, tuple):
        return [_json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if type(value) is dict:
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} object keys must be strings")
            normalized[key] = _json_safe(item, f"{path}.{key}")
        return normalized
    raise TypeError(f"{path} contains a value that is not JSON-safe")


def _normalized_metadata(values: object) -> dict[str, Any]:
    if type(values) is not dict:
        raise TypeError("metadata must be a dict")
    missing = REQUIRED_METADATA - values.keys()
    if missing:
        raise ValueError(f"missing metadata: {sorted(missing)}")
    reserved = _GENERATED_METADATA & values.keys()
    if reserved:
        raise ValueError(f"metadata contains reserved fields: {sorted(reserved)}")
    normalized = _json_safe(values, "metadata")
    if type(normalized["units"]) is not dict or not normalized["units"]:
        raise ValueError("metadata units must be a nonempty JSON object")
    if type(normalized["tolerances"]) is not dict:
        raise ValueError("metadata tolerances must be a JSON object")
    for key in ("dimensionless_convention", "config_hash", "source_hash", "code_hash"):
        if type(normalized[key]) is not str or not normalized[key]:
            raise ValueError(f"metadata {key} must be a nonempty string")
    return normalized


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain only finite JSON-safe values") from error
    return text.encode("utf-8")


def _metadata_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sidecar_bytes(payload: dict[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain only finite JSON-safe values") from error
    return (text + "\n").encode("utf-8")


def _write_deterministic_npz(
    path: Path,
    arrays: dict[str, NDArray[np.generic]],
) -> None:
    with path.open("wb") as raw:
        with zipfile.ZipFile(raw, "w") as archive:
            for key in sorted(arrays):
                encoded = io.BytesIO()
                np.lib.format.write_array(encoded, arrays[key], allow_pickle=False)
                member = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.create_system = 3
                member.external_attr = 0o600 << 16
                archive.writestr(
                    member,
                    encoded.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        raw.flush()
        os.fsync(raw.fileno())


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _read_sidecar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"artifact sidecar is missing: {path}")

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
        raise ValueError("artifact sidecar JSON is malformed") from error
    if type(payload) is not dict:
        raise ValueError("artifact sidecar JSON root must be an object")
    return payload


def _validated_sidecar(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_sidecar(path)
    claimed_metadata_checksum = payload.get("metadata_sha256")
    if type(claimed_metadata_checksum) is not str or _SHA256.fullmatch(
        claimed_metadata_checksum
    ) is None:
        raise ValueError("artifact sidecar metadata checksum is missing or malformed")
    unsigned = dict(payload)
    unsigned.pop("metadata_sha256")
    actual_metadata_checksum = _metadata_checksum(unsigned)
    if not hmac.compare_digest(claimed_metadata_checksum, actual_metadata_checksum):
        raise ValueError("artifact sidecar metadata checksum mismatch")

    if "arrays" not in unsigned or "output_sha256" not in unsigned:
        raise ValueError("artifact sidecar is missing generated metadata")
    user_metadata = {
        key: value
        for key, value in unsigned.items()
        if key not in {"arrays", "output_sha256"}
    }
    _normalized_metadata(user_metadata)
    output_checksum = unsigned["output_sha256"]
    if type(output_checksum) is not str or _SHA256.fullmatch(output_checksum) is None:
        raise ValueError("artifact sidecar output checksum is malformed")
    return payload, unsigned


def _validated_array_metadata(value: object) -> dict[str, dict[str, object]]:
    if type(value) is not dict or not value:
        raise ValueError("artifact sidecar arrays metadata must be a nonempty object")
    result: dict[str, dict[str, object]] = {}
    for raw_key, raw_record in value.items():
        key = _safe_array_key(raw_key)
        if type(raw_record) is not dict or set(raw_record) != {"shape", "dtype"}:
            raise ValueError(f"array metadata for {key} must contain shape and dtype")
        raw_shape = raw_record["shape"]
        raw_dtype = raw_record["dtype"]
        if type(raw_shape) is not list or any(
            type(dimension) is not int or dimension < 0 for dimension in raw_shape
        ):
            raise ValueError(f"array metadata shape for {key} is malformed")
        if type(raw_dtype) is not str or not raw_dtype:
            raise ValueError(f"array metadata dtype for {key} is malformed")
        result[key] = {"shape": list(raw_shape), "dtype": raw_dtype}
    return result


def _archive_keys(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError("artifact NPZ archive is malformed") from error
    if not members:
        raise ValueError("artifact NPZ archive is empty")
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise ValueError("artifact NPZ archive contains duplicate array keys")
    keys: list[str] = []
    for member in members:
        if (
            member.is_dir()
            or member.flag_bits & 0x1
            or member.compress_type not in _ALLOWED_COMPRESSION
            or not member.filename.endswith(".npy")
        ):
            raise ValueError("artifact NPZ archive contains an unsafe member")
        key = member.filename[:-4]
        if _SAFE_ARRAY_KEY.fullmatch(key) is None:
            raise ValueError("artifact NPZ archive contains an unsafe member name")
        keys.append(key)
    return sorted(keys)


def _loaded_archive_arrays(
    path: Path,
    keys: list[str],
) -> tuple[dict[str, list[int]], dict[str, str]]:
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    try:
        with np.load(path, allow_pickle=False) as bundle:
            if sorted(bundle.files) != keys:
                raise ValueError("artifact NPZ key listing is inconsistent")
            for key in keys:
                try:
                    value = bundle[key]
                except ValueError as error:
                    raise ValueError(f"artifact {key} has an unsafe array dtype") from error
                _numeric_array(value, key)
                shapes[key] = list(value.shape)
                dtypes[key] = str(value.dtype)
    except ValueError:
        raise
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError("artifact NPZ archive is malformed") from error
    return shapes, dtypes


def validate_npz_sidecar(path: Path) -> dict[str, Any]:
    """Validate a complete NPZ/JSON pair and return a defensive report."""

    target = _path(path, "path", npz=True)
    if not target.is_file():
        raise ValueError(f"artifact NPZ is missing: {target}")
    sidecar = target.with_suffix(".json")
    payload, unsigned = _validated_sidecar(sidecar)
    expected_checksum = unsigned["output_sha256"]
    actual_checksum = sha256_file(target)
    if not hmac.compare_digest(expected_checksum, actual_checksum):
        raise ValueError("artifact NPZ checksum mismatch")

    metadata_arrays = _validated_array_metadata(unsigned["arrays"])
    archive_keys = _archive_keys(target)
    metadata_keys = sorted(metadata_arrays)
    if archive_keys != metadata_keys:
        raise ValueError("artifact key metadata mismatch")
    shapes, dtypes = _loaded_archive_arrays(target, archive_keys)
    expected_shapes = {key: metadata_arrays[key]["shape"] for key in metadata_keys}
    expected_dtypes = {key: metadata_arrays[key]["dtype"] for key in metadata_keys}
    if shapes != expected_shapes:
        raise ValueError("artifact shape metadata mismatch")
    if dtypes != expected_dtypes:
        raise ValueError("artifact dtype metadata mismatch")

    # Round-trip through JSON produces a deep copy containing only JSON values.
    defensive_payload = json.loads(_sidecar_bytes(payload).decode("utf-8"))
    return {
        "keys": archive_keys,
        "shapes": shapes,
        "dtypes": dtypes,
        "metadata": defensive_payload,
    }


def _temporary_npz(parent: Path, stem: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{stem}.",
        suffix=".npz",
        dir=parent,
    )
    os.close(descriptor)
    return Path(name)


def _backup_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".backup",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        # File contents themselves have already been fsynced.  Some platforms
        # do not permit fsync on a directory descriptor.
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_pair(
    temporary_npz: Path,
    temporary_sidecar: Path,
    target_npz: Path,
    target_sidecar: Path,
) -> dict[str, Any]:
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    rollback_errors: list[OSError] = []
    retained_backups: set[Path] = set()
    try:
        for target in (target_npz, target_sidecar):
            if _lexists(target):
                if target.is_dir():
                    raise ValueError(f"artifact target must not be a directory: {target}")
                backup = _backup_path(target)
                os.replace(target, backup)
                backups[target] = backup
        os.replace(temporary_npz, target_npz)
        installed.append(target_npz)
        os.replace(temporary_sidecar, target_sidecar)
        installed.append(target_sidecar)
        _fsync_directory(target_npz.parent)
        report = validate_npz_sidecar(target_npz)
    except Exception as error:
        for target in reversed(installed):
            try:
                _unlink(target)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        for target, backup in reversed(tuple(backups.items())):
            try:
                os.replace(backup, target)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
                retained_backups.add(backup)
        _fsync_directory(target_npz.parent)
        if rollback_errors:
            retained = ", ".join(str(path) for path in sorted(retained_backups))
            detail = retained if retained else "none"
            raise RuntimeError(
                "artifact publication failed and rollback was incomplete; "
                f"retained backups: {detail}"
            ) from error
        raise
    else:
        for backup in backups.values():
            _unlink(backup)
        _fsync_directory(target_npz.parent)
        return report
    finally:
        _unlink(temporary_npz)
        _unlink(temporary_sidecar)
        for backup in backups.values():
            if backup not in retained_backups:
                _unlink(backup)


def write_npz_with_sidecar(
    path: Path,
    arrays: dict[str, NDArray[np.generic]],
    metadata: dict[str, object],
) -> dict[str, Any]:
    """Atomically publish a deterministic NPZ and its checksummed sidecar."""

    target = _path(path, "path", npz=True)
    normalized_arrays = _normalized_arrays(arrays)
    normalized_metadata = _normalized_metadata(metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = _temporary_npz(target.parent, target.stem)
    temporary_sidecar = temporary_npz.with_suffix(".json")
    try:
        _write_deterministic_npz(temporary_npz, normalized_arrays)
        unsigned_payload: dict[str, Any] = dict(normalized_metadata)
        unsigned_payload["arrays"] = {
            key: {
                "shape": list(normalized_arrays[key].shape),
                "dtype": str(normalized_arrays[key].dtype),
            }
            for key in sorted(normalized_arrays)
        }
        unsigned_payload["output_sha256"] = sha256_file(temporary_npz)
        payload = dict(unsigned_payload)
        payload["metadata_sha256"] = _metadata_checksum(unsigned_payload)
        _write_bytes(temporary_sidecar, _sidecar_bytes(payload))
        validate_npz_sidecar(temporary_npz)
        return _publish_pair(
            temporary_npz,
            temporary_sidecar,
            target,
            target.with_suffix(".json"),
        )
    except Exception:
        _unlink(temporary_npz)
        _unlink(temporary_sidecar)
        raise
