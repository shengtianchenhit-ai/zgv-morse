"""Strict scientific schemas for the seven reproducible paper artifacts.

The low-level :mod:`zgv_morse.artifacts` module owns archive safety, checksum,
sidecar-signature, and atomic-I/O concerns.  This module deliberately builds on
that validated report and adds the paper-specific array and provenance
contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from .artifacts import validate_npz_sidecar


class ArtifactValidationError(ValueError):
    """Raised when an integrity-valid pair violates its scientific schema."""


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """Required rank and NumPy dtype kinds for one named array."""

    ndim: int
    kinds: str
    finite: bool = True


A = ArraySpec
SCHEMAS: Mapping[str, Mapping[str, ArraySpec]] = {
    "isotropic_zgv": {
        "kappa": A(1, "f"),
        "omega_symmetric": A(2, "f"),
        "branch_labels": A(1, "U"),
        "kappa0": A(0, "f"),
        "omega0": A(0, "f"),
        "curvature_a": A(0, "f"),
        "local_q": A(1, "f"),
        "local_omega": A(1, "f"),
        "local_quadratic": A(1, "f"),
        "mode_z": A(1, "f"),
        "mode_u": A(2, "c"),
        "mode_squared_displacement": A(1, "f"),
    },
    "angular_sensitivity": {
        "theta": A(1, "f"),
        "V": A(1, "f"),
        "B": A(1, "f"),
        "V_reconstruction": A(1, "f"),
        "harmonic_order": A(1, "i"),
        "harmonic_amplitude": A(1, "f"),
        "V0": A(0, "f"),
        "V4": A(0, "f"),
        "V8": A(0, "f"),
        "epsilon": A(1, "f"),
        "delta_c": A(1, "f"),
        "physical_V4_shift": A(1, "f"),
        "V_fd": A(1, "f"),
        "B_fd": A(1, "f"),
    },
    "critical_points": {
        "kx": A(1, "f"),
        "ky": A(1, "f"),
        "kappa": A(1, "f"),
        "theta": A(1, "f"),
        "omega": A(1, "f"),
        "hessian_eigenvalues": A(2, "f"),
        "morse_index": A(1, "i"),
        "kind": A(1, "U"),
        "kx_pred": A(1, "f"),
        "ky_pred": A(1, "f"),
        "omega_pred": A(1, "f"),
        "gradient_residual": A(1, "f"),
        "kx_grid": A(1, "f"),
        "ky_grid": A(1, "f"),
        "omega_iso_grid": A(2, "f"),
        "omega_aniso_grid": A(2, "f"),
    },
    "perturbation_scaling": {
        "epsilon": A(1, "f"),
        "delta_omega_full": A(1, "f"),
        "delta_omega_pred": A(1, "f"),
        "q_min_full": A(1, "f"),
        "q_min_pred": A(1, "f"),
        "q_saddle_full": A(1, "f"),
        "q_saddle_pred": A(1, "f"),
        "omega_min_error": A(1, "f"),
        "omega_saddle_error": A(1, "f"),
        "compensated_splitting": A(1, "f"),
        "compensated_q_min": A(1, "f"),
        "compensated_q_saddle": A(1, "f"),
        "compensated_frequency_error": A(1, "f"),
        "role_reversal_theta_min": A(1, "f"),
        "role_reversal_theta_saddle": A(1, "f"),
        "slope_splitting": A(0, "f"),
        "slope_radial": A(0, "f"),
        "slope_remainder": A(0, "f"),
    },
    "green_crossover": {
        "epsilon": A(1, "f"),
        "time": A(1, "f"),
        "G_full": A(2, "c"),
        "G_normal_form": A(2, "c"),
        "G_morse": A(2, "c"),
        "tau": A(2, "f"),
        "J0": A(2, "f"),
        "scaled_response": A(2, "c"),
        "envelope": A(2, "f"),
        "rms_envelope": A(2, "f"),
        "local_slope": A(2, "f"),
        "fit_window_early": A(2, "b"),
        "fit_window_late": A(2, "b"),
        "slope_early": A(1, "f"),
        "slope_late": A(1, "f"),
        "crossover_time": A(1, "f"),
        "spectrum_omega": A(1, "f"),
        "spectrum": A(2, "f"),
        "omega_min": A(1, "f"),
        "omega_saddle": A(1, "f"),
        "phase_error": A(1, "f"),
    },
    "convergence": {
        "polynomial_order": A(1, "i"),
        "omega0_error": A(1, "f"),
        "kappa0_error": A(1, "f"),
        "curvature_error": A(1, "f"),
        "eigen_residual": A(1, "f"),
        "hermitian_residual": A(1, "f"),
        "mass_orthogonality": A(1, "f"),
        "eigengap": A(1, "f"),
        "angular_resolution": A(1, "i"),
        "radial_resolution": A(1, "i"),
        "quadrature_error": A(1, "f"),
        "interpolation_error": A(1, "f"),
        "phase_error": A(1, "f"),
        "sensitivity_step": A(1, "f"),
        "V4_fd_error": A(1, "f"),
        "B_fd_error": A(1, "f"),
        "tracking_kappa": A(1, "f"),
        "tracking_mac": A(1, "f"),
        "tracking_gap": A(1, "f"),
        "source_width": A(1, "f"),
        "window_width": A(1, "f"),
        "response_sensitivity": A(2, "f"),
    },
    "silicon_stress_test": {
        "kx_grid": A(1, "f"),
        "ky_grid": A(1, "f"),
        "omega_grid": A(2, "f"),
        "kx": A(1, "f"),
        "ky": A(1, "f"),
        "omega": A(1, "f"),
        "hessian_eigenvalues": A(2, "f"),
        "kind": A(1, "U"),
        "tracking_gap": A(1, "f"),
        "gradient_residual": A(1, "f"),
    },
}


REQUIRED_METADATA = frozenset(
    {
        "schema_version",
        "artifact",
        "stage",
        "profile",
        "command",
        "dimensionless_convention",
        "units",
        "config_hash",
        "source_hash",
        "code_hash",
        "uv_lock_hash",
        "environment",
        "tolerances",
        "arrays",
        "output_sha256",
        "metadata_sha256",
        "cache_schema_version",
        "cache_key",
        "input_artifacts",
    }
)
ARTIFACT_STAGES = MappingProxyType(
    {
        "isotropic_zgv": "isotropic",
        "angular_sensitivity": "sensitivity",
        "critical_points": "critical_points",
        "perturbation_scaling": "scaling",
        "green_crossover": "green",
        "convergence": "convergence",
        "silicon_stress_test": "silicon",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
CACHE_SCHEMA_VERSION = 1
_CACHE_INPUT_KEYS = frozenset({"artifact", "output_sha256", "cache_key"})
STAGE_DEPENDENCIES = MappingProxyType(
    {
        "isotropic": frozenset(),
        "sensitivity": frozenset(),
        "critical_points": frozenset({"sensitivity"}),
        "scaling": frozenset({"sensitivity"}),
        "green": frozenset({"sensitivity"}),
        "convergence": frozenset({"sensitivity"}),
        "silicon": frozenset(),
    }
)
STAGE_ARTIFACTS = MappingProxyType(
    {stage: artifact for artifact, stage in ARTIFACT_STAGES.items()}
)


def _error(message: str) -> None:
    raise ArtifactValidationError(message)


def _same_shape(
    name: str,
    arrays: Mapping[str, NDArray[np.generic]],
    keys: tuple[str, ...],
) -> None:
    shapes = {arrays[key].shape for key in keys}
    if len(shapes) != 1:
        _error(f"{name}: {', '.join(keys)} do not have matching shapes")


def _validate_isotropic(arrays: Mapping[str, NDArray[np.generic]]) -> None:
    if arrays["omega_symmetric"].shape != (
        arrays["branch_labels"].size,
        arrays["kappa"].size,
    ):
        _error("isotropic_zgv: branch arrays do not align with kappa")
    _same_shape(
        "isotropic_zgv local arrays",
        arrays,
        ("local_q", "local_omega", "local_quadratic"),
    )
    if arrays["mode_u"].shape != (arrays["mode_z"].size, 3):
        _error("isotropic_zgv.mode_u must have one three-component row per mode_z")
    if arrays["mode_squared_displacement"].shape != arrays["mode_z"].shape:
        _error("isotropic_zgv: mode_squared_displacement does not align with mode_z")


def _validate_sensitivity(arrays: Mapping[str, NDArray[np.generic]]) -> None:
    _same_shape(
        "angular_sensitivity angular samples",
        arrays,
        ("theta", "V", "B", "V_reconstruction", "V_fd", "B_fd"),
    )
    _same_shape(
        "angular_sensitivity harmonics",
        arrays,
        ("harmonic_order", "harmonic_amplitude"),
    )
    _same_shape(
        "angular_sensitivity epsilon samples",
        arrays,
        ("epsilon", "delta_c", "physical_V4_shift"),
    )


_CRITICAL_POINT_KEYS = (
    "kx",
    "ky",
    "kappa",
    "theta",
    "omega",
    "morse_index",
    "kind",
    "kx_pred",
    "ky_pred",
    "omega_pred",
    "gradient_residual",
)


def _validate_critical(arrays: Mapping[str, NDArray[np.generic]]) -> None:
    if any(arrays[key].shape != (8,) for key in _CRITICAL_POINT_KEYS):
        _error("critical_points: weak cubic critical-point count is not eight")
    if arrays["hessian_eigenvalues"].shape != (8, 2):
        _error("critical_points: eight Hessians must each have two eigenvalues")
    grid_shape = (arrays["ky_grid"].size, arrays["kx_grid"].size)
    if arrays["omega_iso_grid"].shape != grid_shape:
        _error("critical_points: isotropic grid shape does not align with grid axes")
    if arrays["omega_aniso_grid"].shape != grid_shape:
        _error("critical_points: anisotropic grid shape does not align with grid axes")


_SCALING_EPSILON_KEYS = (
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


def _validate_scaling(arrays: Mapping[str, NDArray[np.generic]]) -> None:
    _same_shape("perturbation_scaling epsilon arrays", arrays, _SCALING_EPSILON_KEYS)
    _same_shape(
        "perturbation_scaling role-reversal arrays",
        arrays,
        ("role_reversal_theta_min", "role_reversal_theta_saddle"),
    )


_GREEN_TIME_KEYS = (
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
)


def _validate_green(arrays: Mapping[str, NDArray[np.generic]]) -> None:
    target = (arrays["epsilon"].size, arrays["time"].size)
    for key in _GREEN_TIME_KEYS:
        if arrays[key].shape != target:
            _error(f"green_crossover: response shape mismatch for {key}")
    for key in ("crossover_time", "omega_min", "omega_saddle", "phase_error"):
        if arrays[key].shape != (arrays["epsilon"].size,):
            _error(f"green_crossover: epsilon shape mismatch for {key}")
    if arrays["spectrum"].shape != (
        arrays["epsilon"].size,
        arrays["spectrum_omega"].size,
    ):
        _error("green_crossover: spectrum does not align with epsilon and frequency")
    if arrays["slope_early"].shape != arrays["slope_late"].shape:
        _error("green_crossover: early and late slope arrays do not align")


def _validate_convergence(arrays: Mapping[str, NDArray[np.generic]]) -> None:
    _same_shape(
        "convergence polynomial arrays",
        arrays,
        (
            "polynomial_order",
            "omega0_error",
            "kappa0_error",
            "curvature_error",
            "eigen_residual",
            "hermitian_residual",
            "mass_orthogonality",
            "eigengap",
        ),
    )
    _same_shape(
        "convergence quadrature arrays",
        arrays,
        (
            "angular_resolution",
            "radial_resolution",
            "quadrature_error",
            "interpolation_error",
            "phase_error",
        ),
    )
    _same_shape(
        "convergence sensitivity arrays",
        arrays,
        ("sensitivity_step", "V4_fd_error", "B_fd_error"),
    )
    _same_shape(
        "convergence tracking arrays",
        arrays,
        ("tracking_kappa", "tracking_mac", "tracking_gap"),
    )
    if arrays["response_sensitivity"].shape != (
        arrays["source_width"].size,
        arrays["window_width"].size,
    ):
        _error("convergence: response sensitivity does not align with width axes")


_SILICON_POINT_KEYS = (
    "kx",
    "ky",
    "omega",
    "kind",
    "tracking_gap",
    "gradient_residual",
)


def _validate_silicon(arrays: Mapping[str, NDArray[np.generic]]) -> None:
    if any(arrays[key].shape != (8,) for key in _SILICON_POINT_KEYS):
        _error("silicon_stress_test: critical-point count is not eight")
    if arrays["hessian_eigenvalues"].shape != (8, 2):
        _error("silicon_stress_test: eight Hessians must each have two eigenvalues")
    if arrays["omega_grid"].shape != (
        arrays["ky_grid"].size,
        arrays["kx_grid"].size,
    ):
        _error("silicon_stress_test: grid shape does not align with grid axes")


_CROSS_VALIDATORS: Mapping[
    str,
    Callable[[Mapping[str, NDArray[np.generic]]], None],
] = {
    "isotropic_zgv": _validate_isotropic,
    "angular_sensitivity": _validate_sensitivity,
    "critical_points": _validate_critical,
    "perturbation_scaling": _validate_scaling,
    "green_crossover": _validate_green,
    "convergence": _validate_convergence,
    "silicon_stress_test": _validate_silicon,
}


def _validate_hash(metadata: Mapping[str, Any], key: str) -> None:
    value = metadata[key]
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _error(f"metadata {key} must be a lowercase SHA-256 digest")


def _manual_values_present(value: object) -> bool:
    if type(value) is str:
        normalized = re.sub(r"[^a-z0-9]", "", value.lower())
        return normalized in {"manualfigurevalue", "manualfigurevalues"}
    if type(value) is dict:
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in {"manualfigurevalue", "manualfigurevalues"}:
                return True
            if _manual_values_present(item):
                return True
    elif type(value) in (list, tuple):
        return any(_manual_values_present(item) for item in value)
    return False


def _normalized_input_artifacts(
    stage: str,
    value: object,
) -> dict[str, dict[str, str]]:
    if type(value) is not dict:
        _error("metadata input_artifacts must be an object")
    if any(type(dependency) is not str for dependency in value):
        _error("metadata input_artifacts keys must be stage names")
    unexpected = set(value).difference(STAGE_DEPENDENCIES[stage])
    if unexpected:
        _error(
            "metadata input_artifacts contains undeclared dependencies: "
            f"{sorted(unexpected)}"
        )
    result: dict[str, dict[str, str]] = {}
    for dependency in sorted(value):
        record = value[dependency]
        if type(record) is not dict or set(record) != _CACHE_INPUT_KEYS:
            _error(
                f"metadata input_artifacts.{dependency} must contain exactly "
                f"{sorted(_CACHE_INPUT_KEYS)}"
            )
        if record["artifact"] != STAGE_ARTIFACTS[dependency]:
            _error(
                f"metadata input_artifacts.{dependency}.artifact does not match stage"
            )
        normalized: dict[str, str] = {"artifact": record["artifact"]}
        for key in ("output_sha256", "cache_key"):
            digest = record[key]
            if type(digest) is not str or _SHA256.fullmatch(digest) is None:
                _error(
                    f"metadata input_artifacts.{dependency}.{key} must be a SHA-256"
                )
            normalized[key] = digest
        result[dependency] = normalized
    return result


def compute_cache_key(
    *,
    stage: str,
    profile: str,
    config_hash: str,
    source_hash: str,
    code_hash: str,
    uv_lock_hash: str,
    input_artifacts: Mapping[str, Mapping[str, str]],
) -> str:
    """Return the canonical scientific cache identity for one workflow stage."""

    if stage not in STAGE_DEPENDENCIES:
        raise ValueError(f"unknown cache stage: {stage}")
    if profile not in {"smoke", "full"}:
        raise ValueError("cache profile must be smoke or full")
    hashes = {
        "config_hash": config_hash,
        "source_hash": source_hash,
        "code_hash": code_hash,
        "uv_lock_hash": uv_lock_hash,
    }
    for name, digest in hashes.items():
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    inputs = _normalized_input_artifacts(stage, dict(input_artifacts))
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "stage": stage,
        "profile": profile,
        **hashes,
        "input_artifacts": inputs,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_metadata(
    name: str,
    metadata: Mapping[str, Any],
    expected_keys: set[str],
) -> None:
    if _manual_values_present(metadata):
        _error("manual figure values are forbidden in artifact metadata")
    missing = REQUIRED_METADATA.difference(metadata)
    if missing:
        _error(f"missing metadata: {sorted(missing)}")
    if metadata["schema_version"] != 1 or type(metadata["schema_version"]) is not int:
        _error("metadata schema_version must be integer 1")
    if metadata["artifact"] != name:
        _error(f"metadata artifact must be {name}")
    stage = ARTIFACT_STAGES[name]
    if metadata["stage"] != stage:
        _error(f"metadata stage must be {stage}")
    profile = metadata["profile"]
    if type(profile) is not str or profile not in {"smoke", "full"}:
        _error("metadata profile must be smoke or full")
    expected_command = f"python -m zgv_morse.workflows --stage {stage} --profile {profile}"
    if metadata["command"] != expected_command:
        _error(f"metadata command must be {expected_command!r}")
    convention = metadata["dimensionless_convention"]
    if type(convention) is not str or not convention:
        _error("metadata dimensionless_convention must be a nonempty string")
    units = metadata["units"]
    if type(units) is not dict or set(units) != expected_keys:
        _error("metadata units must define exactly one entry per schema array")
    if any(type(value) is not str or not value for value in units.values()):
        _error("metadata units values must be nonempty strings")
    array_records = metadata["arrays"]
    if type(array_records) is not dict or set(array_records) != expected_keys:
        _error("metadata arrays must define exactly one entry per schema array")
    for key, record in array_records.items():
        if type(record) is not dict or set(record) != {"shape", "dtype"}:
            _error(f"metadata arrays.{key} must contain exactly shape and dtype")
        if type(record["shape"]) is not list or any(
            type(dimension) is not int or dimension < 0
            for dimension in record["shape"]
        ):
            _error(f"metadata arrays.{key}.shape is malformed")
        if type(record["dtype"]) is not str or not record["dtype"]:
            _error(f"metadata arrays.{key}.dtype is malformed")
    for key in ("config_hash", "source_hash", "code_hash", "uv_lock_hash"):
        _validate_hash(metadata, key)
    for key in ("output_sha256", "metadata_sha256", "cache_key"):
        _validate_hash(metadata, key)
    if (
        type(metadata["cache_schema_version"]) is not int
        or metadata["cache_schema_version"] != CACHE_SCHEMA_VERSION
    ):
        _error(f"metadata cache_schema_version must equal {CACHE_SCHEMA_VERSION}")
    input_artifacts = _normalized_input_artifacts(stage, metadata["input_artifacts"])
    expected_cache_key = compute_cache_key(
        stage=stage,
        profile=profile,
        config_hash=metadata["config_hash"],
        source_hash=metadata["source_hash"],
        code_hash=metadata["code_hash"],
        uv_lock_hash=metadata["uv_lock_hash"],
        input_artifacts=input_artifacts,
    )
    if metadata["cache_key"] != expected_cache_key:
        _error("metadata cache_key does not match the canonical scientific inputs")
    environment = metadata["environment"]
    if type(environment) is not dict or not environment:
        _error("metadata environment must be a nonempty object")
    tolerances = metadata["tolerances"]
    if type(tolerances) is not dict or not tolerances:
        _error("metadata tolerances must be a nonempty object")
    # Nested per-epsilon and per-grid evidence is intentional.  The underlying
    # integrity validator has already recursively rejected non-JSON and
    # non-finite values before this paper-specific contract is reached.


def validate_artifact_payload(
    name: str,
    arrays: Mapping[str, object],
    metadata: Mapping[str, object],
) -> dict[str, NDArray[np.generic]]:
    """Preflight an artifact before atomic publication replaces an existing pair."""

    if name not in SCHEMAS:
        _error(f"unknown artifact: {name}")
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping")
    expected = SCHEMAS[name]
    actual_keys = set(arrays)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        _error(
            f"{name}: missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    normalized = {key: np.asarray(arrays[key]) for key in sorted(expected)}
    candidate_metadata = dict(metadata)
    candidate_metadata.setdefault(
        "arrays",
        {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in normalized.items()
        },
    )
    candidate_metadata.setdefault("output_sha256", "0" * 64)
    candidate_metadata.setdefault("metadata_sha256", "0" * 64)
    _validate_metadata(name, candidate_metadata, expected_keys)
    for key, value in normalized.items():
        expected_record = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if candidate_metadata["arrays"][key] != expected_record:
            _error(f"{name}.{key}: preflight array metadata mismatch")
    for key, spec in expected.items():
        value = normalized[key]
        if value.ndim != spec.ndim:
            _error(f"{name}.{key}: invalid ndim {value.ndim}; expected {spec.ndim}")
        if value.dtype.kind not in spec.kinds:
            _error(
                f"{name}.{key}: invalid dtype {value.dtype}; "
                f"expected kind in {spec.kinds!r}"
            )
        if spec.finite and value.dtype.kind in "fc" and not np.isfinite(value).all():
            _error(f"{name}.{key}: non-finite values")
    _CROSS_VALIDATORS[name](normalized)
    return {key: np.array(value, copy=True) for key, value in normalized.items()}


def _load_defensive_arrays(
    path: Path,
    keys: tuple[str, ...],
) -> dict[str, NDArray[np.generic]]:
    try:
        with np.load(path, allow_pickle=False) as bundle:
            return {key: np.array(bundle[key], copy=True) for key in keys}
    except (OSError, EOFError, ValueError) as error:
        raise ArtifactValidationError(f"artifact archive load failed: {error}") from error


def validate_artifact(
    npz_path: Path,
    sidecar_path: Path | None = None,
) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
    """Validate one exact paper artifact and return defensive data copies.

    A sidecar argument is accepted for API clarity, but a pair is valid only
    when the JSON file is the canonical sibling of the NPZ.  This preserves the
    low-level integrity validator's guarantee that a checksum cannot be paired
    with a different file by the caller.
    """

    if not isinstance(npz_path, Path):
        raise TypeError("npz_path must be a pathlib.Path")
    canonical_sidecar = npz_path.with_suffix(".json")
    if sidecar_path is not None:
        if not isinstance(sidecar_path, Path):
            raise TypeError("sidecar_path must be a pathlib.Path")
        if sidecar_path != canonical_sidecar:
            _error("artifact must use its canonical sidecar path")

    name = npz_path.stem
    if name not in SCHEMAS:
        _error(f"unknown artifact: {name}")
    try:
        report = validate_npz_sidecar(npz_path)
    except (OSError, TypeError, ValueError) as error:
        raise ArtifactValidationError(str(error)) from error

    expected = SCHEMAS[name]
    actual_keys = set(report["keys"])
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        _error(
            f"{name}: missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )

    metadata = report["metadata"]
    if type(metadata) is not dict:
        _error("artifact metadata report must be an object")
    _validate_metadata(name, metadata, expected_keys)
    arrays = _load_defensive_arrays(npz_path, tuple(sorted(expected)))
    for key, spec in expected.items():
        value = arrays[key]
        if value.ndim != spec.ndim:
            _error(f"{name}.{key}: invalid ndim {value.ndim}; expected {spec.ndim}")
        if value.dtype.kind not in spec.kinds:
            _error(f"{name}.{key}: invalid dtype {value.dtype}; expected kind in {spec.kinds!r}")
        if spec.finite and value.dtype.kind in "fc" and not np.isfinite(value).all():
            _error(f"{name}.{key}: non-finite values")

    _CROSS_VALIDATORS[name](arrays)
    # The integrity layer already JSON-round-trips the report.  Repeat the
    # round-trip here so callers cannot mutate any object retained internally.
    defensive_metadata = json.loads(json.dumps(metadata, allow_nan=False))
    return arrays, defensive_metadata
