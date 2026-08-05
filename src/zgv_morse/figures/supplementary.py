"""Six artifact-backed supplementary validation figures and two TeX tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Final

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
from numpy.typing import NDArray

from zgv_morse.artifacts import sha256_file
from zgv_morse.config import ReferenceConfig, load_reference_config

from .common import (
    FigureSpec,
    MARKERS,
    PALETTE,
    apply_publication_style,
    load_figure_artifact,
    save_publication_figure,
    write_source_csv,
)


FIGURE_STEMS: Final = (
    "figure_s01_polynomial_two_element",
    "figure_s02_quadrature_phase",
    "figure_s03_mode_tracking",
    "figure_s04_fd_convergence",
    "figure_s05_source_window_sensitivity",
    "figure_s06_silicon_stress_test",
)
SOURCE_FILES: Final = (
    "s01_panel_a_polynomial.csv",
    "s01_panel_b_independent.csv",
    "s01_panel_c_two_element.csv",
    "s02_panel_a_quadrature.csv",
    "s02_panel_b_dispersion.csv",
    "s02_panel_c_phase.csv",
    "s03_panel_a_mac.csv",
    "s03_panel_b_tracking_gap.csv",
    "s03_panel_c_gap_residual.csv",
    "s04_panel_a_v4.csv",
    "s04_panel_b_b_diagonal.csv",
    "s04_panel_c_b_matrix.csv",
    "s05_panel_a_heatmap.csv",
    "s05_panel_b_profiles.csv",
    "s06_panel_a_points.csv",
    "s06_panel_a_surface.csv",
    "s06_panel_b_hessian.csv",
    "s06_panel_c_diagnostics.csv",
    "s06_material_record.csv",
)
_ARTIFACT_NAMES: Final = (
    "isotropic_zgv",
    "angular_sensitivity",
    "convergence",
    "silicon_stress_test",
)
_SPECS: Final = {
    FIGURE_STEMS[0]: FigureSpec(
        "S1",
        "spectral results are stable in polynomial order and an independent two-element mesh",
        "quantitative grid",
        183.0,
        104.0,
    ),
    FIGURE_STEMS[1]: FigureSpec(
        "S2",
        "nested angular-radial quadrature resolves dispersion and accumulated phase",
        "quantitative grid",
        183.0,
        104.0,
    ),
    FIGURE_STEMS[2]: FigureSpec(
        "S3",
        "mode tracking remains separated and continuous through the ZGV window",
        "quantitative grid",
        183.0,
        104.0,
    ),
    FIGURE_STEMS[3]: FigureSpec(
        "S4",
        "independent centered differences converge to the analytic V4 and B coefficients",
        "quantitative grid",
        183.0,
        104.0,
    ),
    FIGURE_STEMS[4]: FigureSpec(
        "S5",
        "the registered response is stable to source radius and spectral-window width",
        "quantitative grid",
        183.0,
        104.0,
    ),
    FIGURE_STEMS[5]: FigureSpec(
        "S6",
        "finite-anisotropy silicon is a stress test outside the weak-anisotropy proof",
        "quantitative grid",
        183.0,
        104.0,
    ),
}
_CONTEXT_FIELDS: Final = (
    "profile",
    "config_hash",
    "source_hash",
    "code_hash",
    "uv_lock_hash",
    "dimensionless_convention",
)
_BENCHMARK_FIELDS: Final = (
    "order",
    "elements",
    "k_zgv",
    "omega_zgv",
    "curvature",
    "relative_k_error",
    "relative_omega_error",
    "relative_curvature_error",
    "maximum_eigen_residual",
    "hermitian_defect",
    "mass_orthogonality_defect",
    "rotational_frequency_defect",
    "minimum_relative_eigengap",
)
_EXACT_FIELDS: Final = (
    "branch_index",
    "curvature_a",
    "det_residual",
    "group_velocity",
    "kappa0",
    "omega0",
)


@dataclass(frozen=True, slots=True)
class _Bundle:
    arrays: dict[str, dict[str, NDArray[np.generic]]]
    metadata: dict[str, dict[str, Any]]
    validation: dict[str, Any]
    config: ReferenceConfig
    B_error_matrix: NDArray[np.float64]
    B_epsilon_steps: NDArray[np.float64]
    B_radial_steps: NDArray[np.float64]
    material_constants: dict[str, float]
    material_source_id: str
    silicon_orientation: str


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _metadata_number(metadata: Mapping[str, Any], section: str, name: str) -> float:
    values = _mapping(metadata.get(section), f"metadata {section}")
    if name not in values:
        raise ValueError(f"metadata {section} must contain {name}")
    return _finite_number(values[name], f"metadata {section}.{name}")


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate auxiliary validation key: {key}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate,
    )
    if type(payload) is not dict:
        raise ValueError("auxiliary validation root must be an object")
    return payload


def _validate_benchmark_record(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(_BENCHMARK_FIELDS):
        raise ValueError(f"{label} has an invalid benchmark schema")
    result: dict[str, Any] = {}
    for name in _BENCHMARK_FIELDS:
        item = value[name]
        if name in {"order", "elements"}:
            if type(item) is not int or item <= 0:
                raise ValueError(f"{label}.{name} must be a positive integer")
            result[name] = item
        else:
            number = _finite_number(item, f"{label}.{name}")
            if name in {
                "k_zgv",
                "omega_zgv",
                "curvature",
                "minimum_relative_eigengap",
            } and number <= 0.0:
                raise ValueError(f"{label}.{name} must be positive")
            if name not in {"k_zgv", "omega_zgv", "curvature"} and number < 0.0:
                raise ValueError(f"{label}.{name} must be nonnegative")
            result[name] = number
    return result


def _load_isotropic_validation(json_path: Path, csv_path: Path) -> dict[str, Any]:
    if not isinstance(json_path, Path) or not isinstance(csv_path, Path):
        raise TypeError("auxiliary validation paths must be pathlib.Path values")
    payload = _strict_json(json_path)
    if set(payload) != {"exact", "single_element", "two_element"}:
        raise ValueError("auxiliary validation root keys are invalid")
    exact = payload["exact"]
    if type(exact) is not dict or set(exact) != set(_EXACT_FIELDS):
        raise ValueError("auxiliary validation exact-point schema is invalid")
    exact_result: dict[str, Any] = {}
    for name in _EXACT_FIELDS:
        value = exact[name]
        if name == "branch_index":
            if type(value) is not int or value <= 0:
                raise ValueError("auxiliary validation branch_index must be positive")
            exact_result[name] = value
        else:
            exact_result[name] = _finite_number(value, f"auxiliary exact.{name}")

    single_raw = payload["single_element"]
    if type(single_raw) is not list or not single_raw:
        raise ValueError("auxiliary validation single-element rows must be nonempty")
    single = [
        _validate_benchmark_record(row, f"single_element[{index}]")
        for index, row in enumerate(single_raw)
    ]
    if any(row["elements"] != 1 for row in single) or any(
        left["order"] >= right["order"]
        for left, right in zip(single, single[1:], strict=False)
    ):
        raise ValueError("auxiliary single-element orders are invalid")
    two = _validate_benchmark_record(payload["two_element"], "two_element")
    if two["elements"] != 2 or two["order"] != single[-1]["order"]:
        raise ValueError("auxiliary two-element record must match the final order")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _BENCHMARK_FIELDS:
            raise ValueError("auxiliary validation CSV header is invalid")
        csv_rows = list(reader)
    records = [*single, two]
    if len(csv_rows) != len(records):
        raise ValueError("auxiliary validation CSV row count is invalid")
    for row_index, (csv_row, record) in enumerate(zip(csv_rows, records, strict=True)):
        for name in _BENCHMARK_FIELDS:
            observed: int | float
            if name in {"order", "elements"}:
                observed = int(csv_row[name])
            else:
                observed = float(csv_row[name])
            if observed != record[name]:
                raise ValueError(
                    f"auxiliary validation CSV disagrees at row {row_index}, field {name}"
                )
    return {"exact": exact_result, "single_element": single, "two_element": two}


def _validate_context(
    metadata: dict[str, dict[str, Any]],
    config_path: Path,
) -> None:
    reference = metadata[_ARTIFACT_NAMES[0]]
    reference_hash = sha256_file(config_path)
    for name in _ARTIFACT_NAMES:
        record = metadata[name]
        if record.get("artifact") != name:
            raise ValueError(f"scientific context: artifact identity differs for {name}")
        for field in _CONTEXT_FIELDS:
            if record.get(field) != reference.get(field):
                raise ValueError(
                    f"scientific context: {field} differs between registered artifacts"
                )
        if record.get("source_hash") != reference_hash:
            raise ValueError("scientific context: reference configuration hash is not registered")

    inputs = _mapping(
        metadata["convergence"].get("input_artifacts"),
        "convergence input_artifacts",
    )
    sensitivity_input = _mapping(
        inputs.get("sensitivity"), "convergence sensitivity lineage"
    )
    sensitivity_metadata = metadata["angular_sensitivity"]
    for key in ("artifact", "cache_key", "output_sha256"):
        if sensitivity_input.get(key) != sensitivity_metadata.get(key):
            raise ValueError("scientific context: convergence sensitivity lineage differs")


def _validate_auxiliary(
    validation: dict[str, Any],
    isotropic: dict[str, NDArray[np.generic]],
    config: ReferenceConfig,
) -> None:
    exact = validation["exact"]
    for record_name, artifact_name in (
        ("kappa0", "kappa0"),
        ("omega0", "omega0"),
        ("curvature_a", "curvature_a"),
    ):
        if exact[record_name] != float(isotropic[artifact_name]):
            raise ValueError(
                "auxiliary validation exact point disagrees with isotropic_zgv"
            )
    records = [*validation["single_element"], validation["two_element"]]
    for row in records:
        if (
            row["relative_k_error"] >= config.isotropic_match_tolerance
            or row["relative_omega_error"] >= config.isotropic_match_tolerance
            or row["relative_curvature_error"] >= config.curvature_match_tolerance
            or row["maximum_eigen_residual"] >= config.eigen_residual_tolerance
            or row["minimum_relative_eigengap"]
            <= 10.0 * max(row["maximum_eigen_residual"], np.finfo(float).eps)
        ):
            raise ValueError("auxiliary validation convergence gate failed")


def _validate_convergence(
    convergence: dict[str, NDArray[np.generic]],
    metadata: dict[str, Any],
    config: ReferenceConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    angular = np.asarray(convergence["angular_resolution"], dtype=np.int64)
    radial = np.asarray(convergence["radial_resolution"], dtype=np.int64)
    quadrature = np.asarray(convergence["quadrature_error"], dtype=np.float64)
    dispersion = np.asarray(convergence["interpolation_error"], dtype=np.float64)
    phase = np.asarray(convergence["phase_error"], dtype=np.float64)
    if (
        not np.all(angular[1:] > angular[:-1])
        or not np.all(radial[1:] > radial[:-1])
        or not np.all(quadrature[1:] < quadrature[:-1])
        or not np.all(dispersion[1:] < dispersion[:-1])
        or not np.all(phase[1:] < phase[:-1])
        or phase[-1] > config.phase_error_tolerance
    ):
        raise ValueError("nested quadrature must satisfy the phase-discrepancy threshold")
    if not np.isclose(
        float(phase[-1]),
        _metadata_number(metadata, "tolerances", "finest_accumulated_phase_error"),
        rtol=2.0e-12,
        atol=2.0e-15,
    ):
        raise ValueError("phase-discrepancy threshold disagrees with convergence metadata")

    tracking_mac = np.asarray(convergence["tracking_mac"], dtype=np.float64)
    tracking_gap = np.asarray(convergence["tracking_gap"], dtype=np.float64)
    if not np.isclose(
        float(np.min(tracking_mac)),
        _metadata_number(metadata, "tolerances", "minimum_tracking_mac"),
        rtol=2.0e-12,
        atol=2.0e-15,
    ):
        raise ValueError("minimum tracking MAC disagrees with metadata")
    if not np.isclose(
        float(np.min(tracking_gap)),
        _metadata_number(metadata, "tolerances", "minimum_tracking_gap"),
        rtol=2.0e-12,
        atol=2.0e-15,
    ):
        raise ValueError("minimum tracking eigengap disagrees with metadata")
    eigengap = np.asarray(convergence["eigengap"], dtype=np.float64)
    residual = np.asarray(convergence["eigen_residual"], dtype=np.float64)
    if not np.all(eigengap > 10.0 * np.maximum(residual, np.finfo(float).eps)):
        raise ValueError("polynomial eigengap does not resolve the tracked branch")

    fd = _mapping(metadata.get("finite_difference_sweep"), "finite_difference_sweep")
    epsilon_steps = np.asarray(fd.get("epsilon_steps"), dtype=np.float64)
    radial_steps = np.asarray(fd.get("radial_steps"), dtype=np.float64)
    B_matrix = np.asarray(fd.get("relative_B_error_matrix"), dtype=np.float64)
    steps = np.asarray(convergence["sensitivity_step"], dtype=np.float64)
    if (
        B_matrix.shape != (steps.size, steps.size)
        or not np.isfinite(B_matrix).all()
        or not np.all(B_matrix > 0.0)
        or not np.array_equal(epsilon_steps, steps)
        or not np.array_equal(radial_steps, steps)
    ):
        raise ValueError("B finite-difference matrix has an invalid registered sweep")
    np.testing.assert_allclose(
        np.diag(B_matrix),
        convergence["B_fd_error"],
        rtol=0.0,
        atol=0.0,
        err_msg="B finite-difference matrix diagonal must equal B_fd_error",
    )
    if (
        not np.all(convergence["V4_fd_error"][1:] < convergence["V4_fd_error"][:-1])
        or not np.all(convergence["B_fd_error"][1:] < convergence["B_fd_error"][:-1])
        or np.max(B_matrix[-2:, -2:]) >= config.sensitivity_match_tolerance
    ):
        raise ValueError("finite-difference convergence plateau is unresolved")

    response = np.asarray(convergence["response_sensitivity"], dtype=np.float64)
    if not np.isclose(response[1, 1], 1.0, rtol=5.0e-13, atol=5.0e-13):
        raise ValueError("response-sensitivity baseline must equal one")
    provenance = _mapping(
        metadata.get("direct_measurement_provenance"),
        "direct_measurement_provenance",
    )
    boundary_weight = _finite_number(
        provenance.get("widest_window_boundary_weight"),
        "widest_window_boundary_weight",
    )
    boundary_tolerance = _finite_number(
        provenance.get("boundary_weight_tolerance"),
        "boundary_weight_tolerance",
    )
    widest_window = _finite_number(
        provenance.get("widest_window_sigma_over_k0"),
        "widest_window_sigma_over_k0",
    )
    if (
        provenance.get("response_sensitivity_measurement")
        != "direct full-wave response RMS-norm ratios over 61 registered times"
        or provenance.get("response_sensitivity_scope")
        != "source/window response-norm robustness only; not evidence for the crossover exponents or Morse theorem"
        or boundary_tolerance <= 0.0
        or boundary_weight >= boundary_tolerance
        or not np.isclose(
            widest_window,
            float(np.max(convergence["window_width"])),
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError("response-sensitivity provenance is invalid")
    return B_matrix, epsilon_steps, radial_steps


def _validate_silicon(
    silicon: dict[str, NDArray[np.generic]],
    metadata: dict[str, Any],
) -> tuple[dict[str, float], str, str]:
    kinds = np.asarray(silicon["kind"])
    minimum = kinds == "minimum"
    saddle = kinds == "saddle"
    if np.count_nonzero(minimum) != 4 or np.count_nonzero(saddle) != 4:
        raise ValueError("silicon count must be four minima and four saddles")
    theta = np.mod(np.arctan2(silicon["ky"], silicon["kx"]), 2.0 * np.pi)
    order = np.argsort(theta, kind="stable")
    if not np.all(kinds[order] != np.roll(kinds[order], 1)):
        raise ValueError("silicon critical points must alternate in angle")
    hessian = np.asarray(silicon["hessian_eigenvalues"], dtype=np.float64)
    if not (
        np.all(hessian[minimum] > 0.0)
        and np.all(hessian[saddle, 0] < 0.0)
        and np.all(hessian[saddle, 1] > 0.0)
    ):
        raise ValueError("silicon Hessian inertia is invalid")
    if metadata.get("scope") != "finite-anisotropy stress test only":
        raise ValueError("silicon must remain outside the weak-anisotropy proof")
    orientation = metadata.get("orientation")
    if type(orientation) is not str or orientation != "[001] plate normal":
        raise ValueError("silicon orientation record is invalid")
    source_id = metadata.get("material_source_id")
    if type(source_id) is not str or not source_id.startswith("doi:"):
        raise ValueError("silicon material source identifier is invalid")
    raw_constants = _mapping(metadata.get("material_constants_GPa"), "silicon constants")
    if set(raw_constants) != {"C11", "C12", "C44"}:
        raise ValueError("silicon material constants are incomplete")
    constants = {
        name: _finite_number(value, f"silicon {name}")
        for name, value in raw_constants.items()
    }
    if any(value <= 0.0 for value in constants.values()):
        raise ValueError("silicon material constants must be positive")
    max_gradient = _metadata_number(metadata, "tolerances", "maximum_gradient_residual")
    min_gap = _metadata_number(metadata, "tolerances", "minimum_tracking_gap")
    if not np.isclose(
        float(np.max(silicon["gradient_residual"])),
        max_gradient,
        rtol=2.0e-12,
        atol=2.0e-15,
    ) or not np.isclose(
        float(np.min(silicon["tracking_gap"])),
        min_gap,
        rtol=2.0e-12,
        atol=2.0e-15,
    ):
        raise ValueError("silicon diagnostics disagree with metadata")
    boundary = _metadata_number(metadata, "tolerances", "minimum_boundary_gradient")
    uncertainty = _metadata_number(
        metadata, "tolerances", "maximum_boundary_gradient_uncertainty"
    )
    if boundary <= 10.0 * uncertainty:
        raise ValueError("silicon boundary certificate is unresolved")
    certificate = _mapping(
        metadata.get("annular_boundary_certificate"),
        "silicon annular boundary certificate",
    )
    center = _finite_number(
        certificate.get("annulus_center_kappa"), "annulus_center_kappa"
    )
    half_width = _finite_number(
        certificate.get("annulus_half_width"), "annulus_half_width"
    )
    coarse_nodes = certificate.get("coarse_boundary_nodes")
    fine_nodes = certificate.get("fine_boundary_nodes")
    if (
        center <= 0.0
        or half_width <= 0.0
        or half_width >= center
        or type(coarse_nodes) is not int
        or type(fine_nodes) is not int
        or coarse_nodes <= 0
        or fine_nodes != 2 * coarse_nodes
        or certificate.get("boundary_is_noncritical") is not True
        or certificate.get("index_closes") is not True
    ):
        raise ValueError("silicon annular boundary certificate is invalid")
    return constants, source_id, orientation


def _load_bundle(
    data_dir: Path,
    validation_json: Path,
    validation_csv: Path,
    config_path: Path,
) -> _Bundle:
    arrays: dict[str, dict[str, NDArray[np.generic]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for name in _ARTIFACT_NAMES:
        values, sidecar = load_figure_artifact(data_dir, name)
        arrays[name] = values
        metadata[name] = sidecar
    config = load_reference_config(config_path)
    _validate_context(metadata, config_path)
    validation = _load_isotropic_validation(validation_json, validation_csv)
    _validate_auxiliary(validation, arrays["isotropic_zgv"], config)
    B_matrix, epsilon_steps, radial_steps = _validate_convergence(
        arrays["convergence"], metadata["convergence"], config
    )
    constants, source_id, orientation = _validate_silicon(
        arrays["silicon_stress_test"], metadata["silicon_stress_test"]
    )
    return _Bundle(
        arrays=arrays,
        metadata=metadata,
        validation=validation,
        config=config,
        B_error_matrix=B_matrix,
        B_epsilon_steps=epsilon_steps,
        B_radial_steps=radial_steps,
        material_constants=constants,
        material_source_id=source_id,
        silicon_orientation=orientation,
    )


def _write_source_data(source_dir: Path, bundle: _Bundle) -> None:
    convergence = bundle.arrays["convergence"]
    polynomial_keys = (
        "polynomial_order",
        "omega0_error",
        "kappa0_error",
        "curvature_error",
        "eigen_residual",
        "hermitian_residual",
        "mass_orthogonality",
        "eigengap",
    )
    write_source_csv(
        source_dir / SOURCE_FILES[0],
        {name: convergence[name] for name in polynomial_keys},
    )

    independent = [
        *bundle.validation["single_element"],
        bundle.validation["two_element"],
    ]
    independent_columns: dict[str, object] = {
        "discretization": np.array(
            ["single_element"] * len(bundle.validation["single_element"])
            + ["two_element"]
        )
    }
    for name in _BENCHMARK_FIELDS:
        independent_columns[name] = np.array([row[name] for row in independent])
    write_source_csv(source_dir / SOURCE_FILES[1], independent_columns)
    final_records = [bundle.validation["single_element"][-1], independent[-1]]
    final_columns: dict[str, object] = {
        "discretization": np.array(["single_element", "two_element"])
    }
    for name in _BENCHMARK_FIELDS:
        final_columns[name] = np.array([row[name] for row in final_records])
    write_source_csv(source_dir / SOURCE_FILES[2], final_columns)

    grid_columns = {
        "angular_resolution": convergence["angular_resolution"],
        "radial_resolution": convergence["radial_resolution"],
    }
    write_source_csv(
        source_dir / SOURCE_FILES[3],
        {**grid_columns, "quadrature_error": convergence["quadrature_error"]},
    )
    write_source_csv(
        source_dir / SOURCE_FILES[4],
        {**grid_columns, "interpolation_error": convergence["interpolation_error"]},
    )
    phase_tolerance = bundle.config.phase_error_tolerance
    write_source_csv(
        source_dir / SOURCE_FILES[5],
        {
            **grid_columns,
            "phase_error": convergence["phase_error"],
            "phase_error_tolerance": np.full(
                convergence["phase_error"].size, phase_tolerance
            ),
        },
    )

    write_source_csv(
        source_dir / SOURCE_FILES[6],
        {
            "tracking_kappa": convergence["tracking_kappa"],
            "tracking_mac": convergence["tracking_mac"],
        },
    )
    write_source_csv(
        source_dir / SOURCE_FILES[7],
        {
            "tracking_kappa": convergence["tracking_kappa"],
            "tracking_gap": convergence["tracking_gap"],
        },
    )
    write_source_csv(
        source_dir / SOURCE_FILES[8],
        {
            "polynomial_order": convergence["polynomial_order"],
            "eigengap": convergence["eigengap"],
            "eigen_residual": convergence["eigen_residual"],
        },
    )

    write_source_csv(
        source_dir / SOURCE_FILES[9],
        {
            "sensitivity_step": convergence["sensitivity_step"],
            "V4_fd_error": convergence["V4_fd_error"],
        },
    )
    write_source_csv(
        source_dir / SOURCE_FILES[10],
        {
            "sensitivity_step": convergence["sensitivity_step"],
            "B_fd_error": convergence["B_fd_error"],
        },
    )
    write_source_csv(
        source_dir / SOURCE_FILES[11],
        {
            "epsilon_step": np.repeat(
                bundle.B_epsilon_steps, bundle.B_radial_steps.size
            ),
            "radial_step": np.tile(
                bundle.B_radial_steps, bundle.B_epsilon_steps.size
            ),
            "relative_B_error": bundle.B_error_matrix.reshape(-1),
        },
    )

    source_width = convergence["source_width"]
    window_width = convergence["window_width"]
    response = convergence["response_sensitivity"]
    response_provenance = _mapping(
        bundle.metadata["convergence"].get("direct_measurement_provenance"),
        "direct_measurement_provenance",
    )
    response_columns = {
        "source_radius_over_h": np.repeat(source_width, window_width.size),
        "window_sigma_over_k0": np.tile(window_width, source_width.size),
        "response_ratio": response.reshape(-1),
        "widest_window_boundary_weight": np.full(
            response.size,
            _finite_number(
                response_provenance.get("widest_window_boundary_weight"),
                "widest_window_boundary_weight",
            ),
        ),
        "boundary_weight_tolerance": np.full(
            response.size,
            _finite_number(
                response_provenance.get("boundary_weight_tolerance"),
                "boundary_weight_tolerance",
            ),
        ),
        "widest_window_sigma_over_k0": np.full(
            response.size,
            _finite_number(
                response_provenance.get("widest_window_sigma_over_k0"),
                "widest_window_sigma_over_k0",
            ),
        ),
    }
    write_source_csv(source_dir / SOURCE_FILES[12], response_columns)
    write_source_csv(source_dir / SOURCE_FILES[13], response_columns)

    silicon = bundle.arrays["silicon_stress_test"]
    surface_kx, surface_ky = np.meshgrid(silicon["kx_grid"], silicon["ky_grid"])
    write_source_csv(
        source_dir / SOURCE_FILES[15],
        {
            "kx": surface_kx,
            "ky": surface_ky,
            "omega": silicon["omega_grid"],
        },
    )
    point_count = silicon["kind"].size
    write_source_csv(
        source_dir / SOURCE_FILES[14],
        {
            "point_number": np.arange(1, point_count + 1, dtype=np.int64),
            "kind": silicon["kind"],
            "kx": silicon["kx"],
            "ky": silicon["ky"],
            "omega": silicon["omega"],
            "tracking_gap": silicon["tracking_gap"],
            "gradient_residual": silicon["gradient_residual"],
        },
    )
    write_source_csv(
        source_dir / SOURCE_FILES[16],
        {
            "point_number": np.arange(1, point_count + 1, dtype=np.int64),
            "kind": silicon["kind"],
            "hessian_eigenvalue_1": silicon["hessian_eigenvalues"][:, 0],
            "hessian_eigenvalue_2": silicon["hessian_eigenvalues"][:, 1],
        },
    )
    write_source_csv(
        source_dir / SOURCE_FILES[17],
        {
            "point_number": np.arange(1, point_count + 1, dtype=np.int64),
            "kind": silicon["kind"],
            "tracking_gap": silicon["tracking_gap"],
            "gradient_residual": silicon["gradient_residual"],
            "annulus_center_kappa": np.full(
                point_count,
                bundle.metadata["silicon_stress_test"][
                    "annular_boundary_certificate"
                ]["annulus_center_kappa"],
            ),
            "annulus_half_width": np.full(
                point_count,
                bundle.metadata["silicon_stress_test"][
                    "annular_boundary_certificate"
                ]["annulus_half_width"],
            ),
            "coarse_boundary_nodes": np.full(
                point_count,
                bundle.metadata["silicon_stress_test"][
                    "annular_boundary_certificate"
                ]["coarse_boundary_nodes"],
            ),
            "fine_boundary_nodes": np.full(
                point_count,
                bundle.metadata["silicon_stress_test"][
                    "annular_boundary_certificate"
                ]["fine_boundary_nodes"],
            ),
            "boundary_is_noncritical": np.full(
                point_count,
                int(
                    bundle.metadata["silicon_stress_test"][
                        "annular_boundary_certificate"
                    ]["boundary_is_noncritical"]
                ),
            ),
            "index_closes": np.full(
                point_count,
                int(
                    bundle.metadata["silicon_stress_test"][
                        "annular_boundary_certificate"
                    ]["index_closes"]
                ),
            ),
            "minimum_boundary_gradient": np.full(
                point_count,
                bundle.metadata["silicon_stress_test"]["tolerances"][
                    "minimum_boundary_gradient"
                ],
            ),
            "maximum_boundary_gradient_uncertainty": np.full(
                point_count,
                bundle.metadata["silicon_stress_test"]["tolerances"][
                    "maximum_boundary_gradient_uncertainty"
                ],
            ),
            "boundary_resolution_ratio": np.full(
                point_count,
                bundle.metadata["silicon_stress_test"]["tolerances"][
                    "minimum_boundary_gradient"
                ]
                / bundle.metadata["silicon_stress_test"]["tolerances"][
                    "maximum_boundary_gradient_uncertainty"
                ],
            ),
        },
    )
    write_source_csv(
        source_dir / SOURCE_FILES[18],
        {
            "material_source_id": np.array([bundle.material_source_id]),
            "orientation": np.array([bundle.silicon_orientation]),
            **{
                name: np.array([value])
                for name, value in bundle.material_constants.items()
            },
        },
    )


def _panel_label(ax: Axes, label: str) -> None:
    """Panel tag, set outside the axes at the bottom left in parentheses."""

    ax.text(
        -0.02,
        -0.16,
        f"({label})",
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        ha="right",
        va="top",
    )


def _freeze_constrained_layout(fig: Figure, spec: FigureSpec) -> None:
    """Use constrained layout, then freeze rounded axes positions for all formats."""

    fig.set_size_inches(spec.width_mm / 25.4, spec.height_mm / 25.4)
    previous: tuple[tuple[float, float, float, float], ...] | None = None
    for _iteration in range(12):
        fig.canvas.draw()
        current = tuple(
            tuple(float(value) for value in ax.get_position().bounds)
            for ax in fig.axes
        )
        if previous is not None:
            difference = max(
                abs(left - right)
                for old, new in zip(previous, current, strict=True)
                for left, right in zip(old, new, strict=True)
            )
            if difference <= 1.0e-12:
                frozen = tuple(
                    tuple(round(value, 12) for value in bounds) for bounds in current
                )
                fig.set_layout_engine(None)
                for ax, bounds in zip(fig.axes, frozen, strict=True):
                    ax.set_position(bounds, which="both")
                return
        previous = current
    raise RuntimeError("supplementary constrained layout did not converge")


def _save(fig: Figure, output_dir: Path, stem: str) -> dict[str, Path]:
    spec = _SPECS[stem]
    _freeze_constrained_layout(fig, spec)
    return save_publication_figure(fig, output_dir / stem, spec)


def _style_log_axis(ax: Axes) -> None:
    ax.set_yscale("log")
    ax.grid(which="major", color="#E8E8E8", linewidth=0.45)


def _set_step_ticks(ax: Axes, steps: NDArray[np.generic]) -> None:
    values = np.asarray(steps, dtype=np.float64)
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.xaxis.set_major_formatter(
        FixedFormatter(tuple(format(float(value), ".4g") for value in values))
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=5.8)


def _heatmap_text_color(mappable: Any, value: float) -> str:
    red, green, blue, _alpha = mappable.cmap(mappable.norm(float(value)))

    def linearize(component: float) -> float:
        if component <= 0.04045:
            return component / 12.92
        return ((component + 0.055) / 1.055) ** 2.4

    luminance = (
        0.2126 * linearize(float(red))
        + 0.7152 * linearize(float(green))
        + 0.0722 * linearize(float(blue))
    )
    # The crossover equalizes black-on-cell and white-on-cell WCAG contrast.
    return "white" if luminance < 0.179 else "black"


def _figure_s01(bundle: _Bundle, output_dir: Path) -> dict[str, Path]:
    convergence = bundle.arrays["convergence"]
    fig, axes = plt.subplots(1, 3, layout="constrained", facecolor="white")
    order = convergence["polynomial_order"]
    for name, label, color, marker in (
        ("kappa0_error", r"$\kappa_0$", PALETTE["minimum"], "o"),
        ("omega0_error", r"$\Omega_0$", PALETTE["saddle"], "s"),
        ("curvature_error", r"$a$", PALETTE["anisotropic"], "D"),
    ):
        axes[0].plot(order, convergence[name], color=color, marker=marker, label=label)
    _style_log_axis(axes[0])
    axes[0].set_xlabel("reported polynomial order $p$")
    axes[0].set_ylabel("relative error")
    axes[0].legend(fontsize=8.4)
    _panel_label(axes[0], "a")

    single = bundle.validation["single_element"]
    two = bundle.validation["two_element"]
    single_order = np.array([row["order"] for row in single])
    for name, label, color, marker in (
        ("relative_k_error", r"$\kappa_0$", PALETTE["minimum"], "o"),
        ("relative_omega_error", r"$\Omega_0$", PALETTE["saddle"], "s"),
        ("relative_curvature_error", r"$a$", PALETTE["anisotropic"], "D"),
    ):
        axes[1].plot(
            single_order,
            np.array([row[name] for row in single]),
            color=color,
            marker=marker,
            label=label,
        )
        axes[1].scatter(
            [two["order"]],
            [two[name]],
            color=color,
            marker="*",
            s=38,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
    _style_log_axis(axes[1])
    axes[1].set_xlabel("independent validation order $p$")
    axes[1].set_ylabel("relative error")
    axes[1].legend(fontsize=8.4)
    _panel_label(axes[1], "b")

    final_single = single[-1]
    categories = (r"$\kappa_0$", r"$\Omega_0$", r"$a$")
    positions = np.arange(len(categories), dtype=np.float64)
    single_values = np.array(
        [
            final_single["relative_k_error"],
            final_single["relative_omega_error"],
            final_single["relative_curvature_error"],
        ]
    )
    two_values = np.array(
        [
            two["relative_k_error"],
            two["relative_omega_error"],
            two["relative_curvature_error"],
        ]
    )
    axes[2].scatter(
        positions - 0.09,
        single_values,
        color=PALETTE["isotropic"],
        marker="o",
        label="one element",
    )
    axes[2].scatter(
        positions + 0.09,
        two_values,
        color=PALETTE["anisotropic"],
        marker="*",
        s=42,
        label="two elements",
    )
    axes[2].set_xticks(positions, categories)
    _style_log_axis(axes[2])
    axes[2].set_ylabel("relative error at shared final $p$")
    axes[2].legend(fontsize=8.4)
    _panel_label(axes[2], "c")
    return _save(fig, output_dir, FIGURE_STEMS[0])


def _grid_ticks(ax: Axes, angular: NDArray[np.generic], radial: NDArray[np.generic]) -> None:
    positions = np.arange(angular.size)
    ax.set_xticks(positions, [f"{a}/{r}" for a, r in zip(angular, radial, strict=True)])
    ax.set_xlabel(r"nested grid $N_\theta/N_q$")


def _figure_s02(bundle: _Bundle, output_dir: Path) -> dict[str, Path]:
    convergence = bundle.arrays["convergence"]
    angular = convergence["angular_resolution"]
    radial = convergence["radial_resolution"]
    positions = np.arange(angular.size)
    fig, axes = plt.subplots(1, 3, layout="constrained", facecolor="white")
    panels = (
        (
            "quadrature_error",
            "relative response error",
            "Angular-radial quadrature",
            PALETTE["anisotropic"],
        ),
        (
            "interpolation_error",
            r"maximum $|\delta\Omega|$",
            "Dispersion interpolation",
            PALETTE["minimum"],
        ),
        (
            "phase_error",
            "accumulated phase error (rad)",
            "Phase-discrepancy threshold",
            PALETTE["saddle"],
        ),
    )
    for index, (name, ylabel, title, color) in enumerate(panels):
        axes[index].plot(
            positions,
            convergence[name],
            color=color,
            marker="o",
            linewidth=1.2,
        )
        if name == "phase_error":
            axes[index].axhline(
                bundle.config.phase_error_tolerance,
                color=PALETTE["prediction"],
                linestyle=(0, (3, 2)),
                linewidth=0.85,
                label="declared threshold",
            )
            axes[index].legend(fontsize=8.4)
        _style_log_axis(axes[index])
        _grid_ticks(axes[index], angular, radial)
        axes[index].set_ylabel(ylabel)
        axes[index].set_title(title)
        _panel_label(axes[index], chr(ord("a") + index))
    return _save(fig, output_dir, FIGURE_STEMS[1])


def _figure_s03(bundle: _Bundle, output_dir: Path) -> dict[str, Path]:
    convergence = bundle.arrays["convergence"]
    fig, axes = plt.subplots(1, 3, layout="constrained", facecolor="white")
    axes[0].plot(
        convergence["tracking_kappa"],
        convergence["tracking_mac"],
        color=PALETTE["minimum"],
        marker="o",
    )
    axes[0].set_xlabel(r"$\kappa$")
    axes[0].set_ylabel("mass MAC")
    axes[0].grid(color="#E8E8E8", linewidth=0.45)
    _panel_label(axes[0], "a")

    axes[1].plot(
        convergence["tracking_kappa"],
        convergence["tracking_gap"],
        color=PALETTE["saddle"],
        marker="D",
    )
    axes[1].set_xlabel(r"$\kappa$")
    axes[1].set_ylabel("relative eigengap")
    axes[1].grid(color="#E8E8E8", linewidth=0.45)
    _panel_label(axes[1], "b")

    axes[2].plot(
        convergence["polynomial_order"],
        convergence["eigengap"],
        color=PALETTE["prediction"],
        marker="o",
        label="eigengap",
    )
    axes[2].plot(
        convergence["polynomial_order"],
        convergence["eigen_residual"],
        color=PALETTE["anisotropic"],
        marker="s",
        label="eigen residual",
    )
    _style_log_axis(axes[2])
    axes[2].set_xlabel("polynomial order $p$")
    axes[2].set_ylabel("relative diagnostic")
    axes[2].legend(fontsize=8.4)
    _panel_label(axes[2], "c")
    return _save(fig, output_dir, FIGURE_STEMS[2])


def _figure_s04(bundle: _Bundle, output_dir: Path) -> dict[str, Path]:
    convergence = bundle.arrays["convergence"]
    step = convergence["sensitivity_step"]
    fig, axes = plt.subplots(1, 3, layout="constrained", facecolor="white")
    for ax, values, color, title, ylabel in (
        (
            axes[0],
            convergence["V4_fd_error"],
            PALETTE["minimum"],
            r"$V_4$ centered difference",
            r"relative $V_4$ error",
        ),
        (
            axes[1],
            convergence["B_fd_error"],
            PALETTE["saddle"],
            r"$B$ paired-step diagonal",
            r"relative $B$ error",
        ),
    ):
        ax.loglog(step, values, color=color, marker="o", linewidth=1.2)
        ax.invert_xaxis()
        _set_step_ticks(ax, step)
        ax.set_xlabel("centered-difference step")
        ax.set_ylabel(ylabel)
        ax.grid(which="major", color="#E8E8E8", linewidth=0.45)
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")

    rows, columns = bundle.B_error_matrix.shape
    image = axes[2].pcolormesh(
        np.arange(columns + 1, dtype=np.float64) - 0.5,
        np.arange(rows + 1, dtype=np.float64) - 0.5,
        bundle.B_error_matrix,
        shading="flat",
        cmap="Blues",
        norm=LogNorm(
            vmin=float(np.min(bundle.B_error_matrix)),
            vmax=float(np.max(bundle.B_error_matrix)),
        ),
    )
    axes[2].set_xlim(-0.5, columns - 0.5)
    axes[2].set_ylim(rows - 0.5, -0.5)
    for row in range(bundle.B_error_matrix.shape[0]):
        for column in range(bundle.B_error_matrix.shape[1]):
            axes[2].text(
                column,
                row,
                f"{bundle.B_error_matrix[row, column]:.1e}",
                ha="center",
                va="center",
                fontsize=8.4,
                color=_heatmap_text_color(
                    image,
                    float(bundle.B_error_matrix[row, column]),
                ),
            )
    axes[2].set_xticks(
        np.arange(bundle.B_radial_steps.size),
        [f"{value:g}" for value in bundle.B_radial_steps],
    )
    axes[2].set_yticks(
        np.arange(bundle.B_epsilon_steps.size),
        [f"{value:g}" for value in bundle.B_epsilon_steps],
    )
    axes[2].set_xlabel("radial step")
    axes[2].set_ylabel(r"$\varepsilon$ step")
    colorbar = fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.035)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("relative error", fontsize=8.4)
    colorbar.ax.tick_params(labelsize=5)
    _panel_label(axes[2], "c")
    return _save(fig, output_dir, FIGURE_STEMS[3])


def _figure_s05(bundle: _Bundle, output_dir: Path) -> dict[str, Path]:
    convergence = bundle.arrays["convergence"]
    source = convergence["source_width"]
    window = convergence["window_width"]
    response = convergence["response_sensitivity"]
    fig, axes = plt.subplots(1, 2, layout="constrained", facecolor="white")
    deviation = float(np.max(np.abs(response - 1.0)))
    rows, columns = response.shape
    image = axes[0].pcolormesh(
        np.arange(columns + 1, dtype=np.float64) - 0.5,
        np.arange(rows + 1, dtype=np.float64) - 0.5,
        response,
        shading="flat",
        cmap="coolwarm",
        vmin=1.0 - deviation,
        vmax=1.0 + deviation,
    )
    axes[0].set_xlim(-0.5, columns - 0.5)
    axes[0].set_ylim(-0.5, rows - 0.5)
    for row in range(response.shape[0]):
        for column in range(response.shape[1]):
            axes[0].text(
                column,
                row,
                f"{response[row, column]:.4f}",
                ha="center",
                va="center",
                fontsize=8.4,
                color=_heatmap_text_color(image, float(response[row, column])),
            )
    axes[0].set_xticks(np.arange(window.size), [f"{value:g}" for value in window])
    axes[0].set_yticks(np.arange(source.size), [f"{value:g}" for value in source])
    axes[0].set_xlabel(r"window width $\sigma/k_0$")
    axes[0].set_ylabel(r"source radius $R/h$")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.035)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("ratio to registered baseline", fontsize=8.4)
    colorbar.ax.tick_params(labelsize=5)
    _panel_label(axes[0], "a")

    colors = (PALETTE["minimum"], PALETTE["anisotropic"], PALETTE["saddle"])
    for column, (window_value, color) in enumerate(zip(window, colors, strict=True)):
        axes[1].plot(
            source,
            response[:, column],
            color=color,
            marker="o",
            label=rf"$\sigma/k_0={window_value:g}$",
        )
    axes[1].axhline(1.0, color=PALETTE["prediction"], linewidth=0.7)
    axes[1].set_xlabel(r"source radius $R/h$")
    axes[1].set_ylabel("RMS response ratio")
    axes[1].grid(color="#E8E8E8", linewidth=0.45)
    axes[1].legend(fontsize=8.4)
    _panel_label(axes[1], "b")
    return _save(fig, output_dir, FIGURE_STEMS[4])


def _figure_s06(bundle: _Bundle, output_dir: Path) -> dict[str, Path]:
    silicon = bundle.arrays["silicon_stress_test"]
    fig, axes = plt.subplots(1, 3, layout="constrained", facecolor="white")
    # Figure-level title, not a panel heading: it carries the scope caveat
    # that this material lies outside the weak-anisotropy proof.
    fig.suptitle(
        "stress test outside the weak-anisotropy proof",
        fontsize=8.4,
        fontweight="bold",
    )
    kx, ky = np.meshgrid(silicon["kx_grid"], silicon["ky_grid"])
    axes[0].contourf(kx, ky, silicon["omega_grid"], levels=9, cmap="Blues")
    axes[0].contour(
        kx,
        ky,
        silicon["omega_grid"],
        levels=9,
        colors=PALETTE["anisotropic"],
        linewidths=0.5,
        alpha=0.7,
    )
    kinds = np.asarray(silicon["kind"])
    for kind in ("minimum", "saddle"):
        selected = kinds == kind
        axes[0].scatter(
            silicon["kx"][selected],
            silicon["ky"][selected],
            color=PALETTE[kind],
            marker=MARKERS[kind],
            s=26,
            edgecolor="white",
            linewidth=0.5,
            label=kind,
            zorder=4,
        )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(r"$k_xh$")
    axes[0].set_ylabel(r"$k_yh$")
    # Panel a is locked to a square aspect, so a legend stacked above it
    # competed with the title for vertical room and collapsed the axes once
    # the type grew.  Anchoring it to the figure keeps it clear of every
    # panel and out of the filled contour field.
    axes[0].legend(
        fontsize=8.4,
        loc="upper right",
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0.3,
    )
    # Provenance note: anchored to the figure, not to the axes.  As axes
    # text below the panel, constrained layout counted this long string as
    # panel decoration and squeezed the axes to nothing once the type grew.
    axes[0].figure.text(
        0.01,
        0.002,
        f"material record: {bundle.material_source_id}; not theorem evidence",
        fontsize=7.0,
        color=PALETTE["neutral"],
        ha="left",
        va="bottom",
    )
    _panel_label(axes[0], "a")

    points = np.arange(1, silicon["kind"].size + 1)
    axes[1].plot(
        points,
        silicon["hessian_eigenvalues"][:, 0],
        color=PALETTE["saddle"],
        marker="o",
        label=r"$\lambda_1$",
    )
    axes[1].plot(
        points,
        silicon["hessian_eigenvalues"][:, 1],
        color=PALETTE["minimum"],
        marker="s",
        label=r"$\lambda_2$",
    )
    axes[1].axhline(0.0, color=PALETTE["prediction"], linewidth=0.65)
    axes[1].set_xticks(points)
    axes[1].set_xlabel("critical-point number")
    axes[1].set_ylabel("Cartesian Hessian eigenvalue")
    axes[1].grid(color="#E8E8E8", linewidth=0.45)
    axes[1].legend(fontsize=8.4)
    _panel_label(axes[1], "b")

    axes[2].plot(
        points,
        silicon["tracking_gap"],
        color=PALETTE["minimum"],
        marker="o",
        label="tracking gap",
    )
    axes[2].plot(
        points,
        silicon["gradient_residual"],
        color=PALETTE["saddle"],
        marker="D",
        label="gradient residual",
    )
    axes[2].set_yscale("log")
    axes[2].set_xticks(points)
    axes[2].set_xlabel("critical-point number")
    axes[2].set_ylabel("diagnostic magnitude")
    axes[2].grid(which="major", color="#E8E8E8", linewidth=0.45)
    axes[2].legend(fontsize=8.4)
    constants = bundle.material_constants
    axes[2].text(
        0.04,
        0.38,
        (
            f"C11={constants['C11']:g}, C12={constants['C12']:g} GPa\n"
            f"C44={constants['C44']:g} GPa"
        ),
        transform=axes[2].transAxes,
        fontsize=8.4,
        color=PALETTE["neutral"],
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
    )
    _panel_label(axes[2], "c")
    return _save(fig, output_dir, FIGURE_STEMS[5])


def build_all(
    data_dir: Path,
    validation_json: Path,
    validation_csv: Path,
    config_path: Path,
    output_dir: Path,
    source_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Validate every input first, then build exactly six supplementary figures."""

    bundle = _load_bundle(data_dir, validation_json, validation_csv, config_path)
    _write_source_data(source_dir, bundle)
    apply_publication_style()
    plt.rcParams["path.simplify"] = False
    builders = (
        _figure_s01,
        _figure_s02,
        _figure_s03,
        _figure_s04,
        _figure_s05,
        _figure_s06,
    )
    return {
        stem: builder(bundle, output_dir)
        for stem, builder in zip(FIGURE_STEMS, builders, strict=True)
    }


def format_table_number(value: object) -> str:
    """Format one finite input-backed table value without hidden rounding state."""

    if type(value) is int:
        return str(value)
    number = _finite_number(value, "table value")
    magnitude = abs(number)
    if number == 0.0 or 1.0e-3 <= magnitude < 1.0e4:
        return format(number, ".8g")
    return format(number, ".6e")


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _convergence_table(bundle: _Bundle) -> str:
    rows = (
        ("one element", bundle.validation["single_element"][-1]),
        ("two elements", bundle.validation["two_element"]),
    )
    lines = [
        "% Generated from the deterministic auxiliary isotropic validation record.",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"discretization & $p$ & elements & $|\delta\kappa_0|/\kappa_0$ & $|\delta\Omega_0|/\Omega_0$ & $|\delta a|/a$ & eigen residual & minimum eigengap \\",
        r"\hline",
    ]
    for label, record in rows:
        values = (
            record["order"],
            record["elements"],
            record["relative_k_error"],
            record["relative_omega_error"],
            record["relative_curvature_error"],
            record["maximum_eigen_residual"],
            record["minimum_relative_eigengap"],
        )
        lines.append(
            f"{label} & "
            + " & ".join(format_table_number(value) for value in values)
            + r" \\"
        )
    lines.extend((r"\hline", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _parameters_table(bundle: _Bundle) -> str:
    config = bundle.config
    rows: list[tuple[str, object, str]] = [
        (r"$h$", config.h, "reference configuration"),
        (r"$\rho$", config.rho, "reference configuration"),
        (r"$\lambda$", config.lam, "reference configuration"),
        (r"$\mu$", config.mu, "reference configuration"),
        (r"$\delta$", config.delta, "reference configuration"),
        (r"$R/h$", config.source_radius_over_h, "reference configuration"),
        (r"$\sigma/k_0$", config.window_sigma_over_k0, "reference configuration"),
        ("annulus fraction", config.annulus_fraction, "reference configuration"),
        ("eigen residual tolerance", config.eigen_residual_tolerance, "reference configuration"),
        ("isotropic match tolerance", config.isotropic_match_tolerance, "reference configuration"),
        ("curvature match tolerance", config.curvature_match_tolerance, "reference configuration"),
        ("sensitivity match tolerance", config.sensitivity_match_tolerance, "reference configuration"),
        ("phase error tolerance", config.phase_error_tolerance, "reference configuration"),
    ]
    for name in ("C11", "C12", "C44"):
        rows.append(
            (
                rf"${name[0]}_{{{name[1:]}}}$ (GPa)",
                bundle.material_constants[name],
                bundle.material_source_id,
            )
        )
    lines = [
        "% Generated only from reference.yaml and silicon_stress_test metadata.",
        r"\begin{tabular}{lll}",
        r"\hline",
        r"parameter & value & provenance \\",
        r"\hline",
    ]
    lines.extend(
        f"{label} & {format_table_number(value)} & {provenance} \\\\"
        for label, value, provenance in rows
    )
    lines.extend((r"\hline", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def export_tables(
    data_dir: Path,
    validation_json: Path,
    validation_csv: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Export two deterministic TeX tables after the same strict evidence checks."""

    bundle = _load_bundle(data_dir, validation_json, validation_csv, config_path)
    return {
        "table_s01_convergence": _write_text_atomic(
            output_dir / "table_s01_convergence.tex", _convergence_table(bundle)
        ),
        "table_s02_parameters": _write_text_atomic(
            output_dir / "table_s02_parameters.tex", _parameters_table(bundle)
        ),
    }
