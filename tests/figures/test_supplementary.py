"""Evidence, provenance, export, and table contract for supplementary assets."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
from pypdf import PdfReader
import pytest

from zgv_morse.config import load_reference_config
from zgv_morse.figures import common
from zgv_morse.figures.supplementary import (
    FIGURE_STEMS,
    SOURCE_FILES,
    build_all,
    export_tables,
)
import zgv_morse.figures.supplementary as supplementary


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/generated"
VALIDATION_JSON = DATA_DIR / "isotropic_validation.json"
VALIDATION_CSV = DATA_DIR / "isotropic_convergence.csv"
CONFIG_PATH = ROOT / "config/reference.yaml"
ARTIFACT_NAMES = (
    "isotropic_zgv",
    "angular_sensitivity",
    "convergence",
    "silicon_stress_test",
)
EXPECTED_STEMS = (
    "figure_s01_polynomial_two_element",
    "figure_s02_quadrature_phase",
    "figure_s03_mode_tracking",
    "figure_s04_fd_convergence",
    "figure_s05_source_window_sensitivity",
    "figure_s06_silicon_stress_test",
)
EXPECTED_SOURCE_FILES = (
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


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return tuple(reader.fieldnames), list(reader)


def _float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(row[name]) for row in rows], dtype=np.float64)


def _artifact_copies():
    bundles = {
        name: common.load_figure_artifact(DATA_DIR, name) for name in ARTIFACT_NAMES
    }
    arrays = {
        name: {key: np.array(value, copy=True) for key, value in values.items()}
        for name, (values, _metadata) in bundles.items()
    }
    metadata = {
        name: copy.deepcopy(sidecar)
        for name, (_values, sidecar) in bundles.items()
    }
    return arrays, metadata


def _count_pdf_image_xobjects(path: Path) -> int:
    """Count raster image objects, including images nested inside form objects."""

    def count_resources(resources: object) -> int:
        resolved = resources.get_object()  # type: ignore[union-attr]
        xobjects = resolved.get("/XObject")
        if xobjects is None:
            return 0
        count = 0
        for reference in xobjects.get_object().values():
            xobject = reference.get_object()
            if xobject.get("/Subtype") == "/Image":
                count += 1
            nested = xobject.get("/Resources")
            if nested is not None:
                count += count_resources(nested)
        return count

    return sum(
        count_resources(page["/Resources"]) for page in PdfReader(path).pages
    )


def _validation_record() -> dict[str, object]:
    return json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))


def test_heatmap_annotations_switch_to_white_on_dark_cells() -> None:
    grayscale = SimpleNamespace(
        norm=lambda value: value,
        cmap=lambda value: (value, value, value, 1.0),
    )

    assert supplementary._heatmap_text_color(grayscale, 0.05) == "white"
    assert supplementary._heatmap_text_color(grayscale, 0.95) == "black"
    for channel in np.linspace(0.0, 1.0, 1001):
        selected = supplementary._heatmap_text_color(grayscale, float(channel))
        linear = (
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
        )
        contrast = (
            1.05 / (linear + 0.05)
            if selected == "white"
            else (linear + 0.05) / 0.05
        )
        assert contrast >= 4.5


def test_silicon_figure_title_band_has_print_safe_horizontal_margin() -> None:
    path = ROOT / "figures/supplementary/figure_s06_silicon_stress_test.png"
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    ink = np.any(rgb[:700] < 245, axis=2)
    _rows, columns = np.nonzero(ink)

    assert columns.size > 0
    assert int(columns.min()) >= 8
    assert rgb.shape[1] - 1 - int(columns.max()) >= 8


@pytest.mark.filterwarnings("error:Glyph .* missing from font")
def test_builds_exactly_six_deterministic_editable_supplementary_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    real_loader = common.load_figure_artifact

    def tracked_loader(data_dir: Path, name: str):
        calls.append(name)
        return real_loader(data_dir, name)

    monkeypatch.setattr(supplementary, "load_figure_artifact", tracked_loader)
    bundles: list[tuple[dict[str, dict[str, Path]], Path]] = []
    for run in ("first", "second"):
        output_dir = tmp_path / run / "figures"
        source_dir = tmp_path / run / "source"
        outputs = build_all(
            DATA_DIR,
            VALIDATION_JSON,
            VALIDATION_CSV,
            CONFIG_PATH,
            output_dir,
            source_dir,
        )
        bundles.append((outputs, source_dir))

    assert calls == list(ARTIFACT_NAMES) * 2
    assert FIGURE_STEMS == EXPECTED_STEMS
    assert SOURCE_FILES == EXPECTED_SOURCE_FILES
    first, first_source = bundles[0]
    second, second_source = bundles[1]
    assert tuple(first) == EXPECTED_STEMS
    assert tuple(second) == EXPECTED_STEMS
    assert tuple(sorted(path.name for path in first_source.glob("*.csv"))) == tuple(
        sorted(EXPECTED_SOURCE_FILES)
    )

    for stem in EXPECTED_STEMS:
        spec = supplementary._SPECS[stem]
        expected_pixels = (
            int(spec.width_mm / 25.4 * 600),
            int(spec.height_mm / 25.4 * 600),
        )
        assert set(first[stem]) == set(common.PUBLICATION_FORMATS)
        for kind in common.PUBLICATION_FORMATS:
            first_path = first[stem][kind]
            second_path = second[stem][kind]
            assert first_path.name == f"{stem}.{kind}"
            assert first_path.read_bytes() == second_path.read_bytes()
        svg = first[stem]["svg"].read_text(encoding="utf-8")
        assert "<text" in svg
        assert "dc:date" not in svg
        assert b"/CreationDate" not in first[stem]["pdf"].read_bytes()
        for kind in ("png", "tiff"):
            with Image.open(first[stem][kind]) as image:
                assert image.size == expected_pixels
                assert image.info["dpi"] == pytest.approx((600.0, 600.0), abs=0.1)
                if kind == "tiff":
                    assert image.tag_v2[259] == 5

    for stem in (EXPECTED_STEMS[3], EXPECTED_STEMS[4]):
        assert _count_pdf_image_xobjects(first[stem]["pdf"]) == 0

    silicon_svg = first[EXPECTED_STEMS[-1]]["svg"].read_text(encoding="utf-8")
    assert "stress test outside the weak-anisotropy proof" in silicon_svg
    assert "doi:10.1107/S1600577514004962" in silicon_svg
    assert "not theorem evidence" in silicon_svg

    for filename in EXPECTED_SOURCE_FILES:
        assert (first_source / filename).read_bytes() == (
            second_source / filename
        ).read_bytes()


def test_every_supplementary_panel_csv_traces_to_registered_inputs(
    tmp_path: Path,
) -> None:
    outputs = build_all(
        DATA_DIR,
        VALIDATION_JSON,
        VALIDATION_CSV,
        CONFIG_PATH,
        tmp_path / "figures",
        tmp_path / "source",
    )
    assert tuple(outputs) == EXPECTED_STEMS
    source = tmp_path / "source"
    convergence, convergence_metadata = common.load_figure_artifact(
        DATA_DIR, "convergence"
    )
    silicon, silicon_metadata = common.load_figure_artifact(
        DATA_DIR, "silicon_stress_test"
    )
    validation = _validation_record()

    _names, rows = _read_csv(source / "s01_panel_a_polynomial.csv")
    for name in (
        "polynomial_order",
        "omega0_error",
        "kappa0_error",
        "curvature_error",
        "eigen_residual",
        "hermitian_residual",
        "mass_orthogonality",
        "eigengap",
    ):
        assert_allclose(_float_column(rows, name), convergence[name], rtol=0, atol=0)

    _names, rows = _read_csv(source / "s01_panel_b_independent.csv")
    independent = [*validation["single_element"], validation["two_element"]]
    assert len(rows) == len(independent)
    for name in (
        "order",
        "elements",
        "relative_k_error",
        "relative_omega_error",
        "relative_curvature_error",
        "maximum_eigen_residual",
        "minimum_relative_eigengap",
    ):
        assert_allclose(
            _float_column(rows, name),
            np.array([record[name] for record in independent]),
            rtol=0,
            atol=0,
        )

    for filename, keys in (
        (
            "s02_panel_a_quadrature.csv",
            ("angular_resolution", "radial_resolution", "quadrature_error"),
        ),
        (
            "s02_panel_b_dispersion.csv",
            ("angular_resolution", "radial_resolution", "interpolation_error"),
        ),
        (
            "s02_panel_c_phase.csv",
            ("angular_resolution", "radial_resolution", "phase_error"),
        ),
        (
            "s03_panel_a_mac.csv",
            ("tracking_kappa", "tracking_mac"),
        ),
        (
            "s03_panel_b_tracking_gap.csv",
            ("tracking_kappa", "tracking_gap"),
        ),
        (
            "s03_panel_c_gap_residual.csv",
            ("polynomial_order", "eigengap", "eigen_residual"),
        ),
        (
            "s04_panel_a_v4.csv",
            ("sensitivity_step", "V4_fd_error"),
        ),
        (
            "s04_panel_b_b_diagonal.csv",
            ("sensitivity_step", "B_fd_error"),
        ),
    ):
        _names, rows = _read_csv(source / filename)
        for name in keys:
            assert_allclose(_float_column(rows, name), convergence[name], rtol=0, atol=0)

    fd = convergence_metadata["finite_difference_sweep"]
    B_matrix = np.asarray(fd["relative_B_error_matrix"])
    epsilon_steps = np.asarray(fd["epsilon_steps"])
    radial_steps = np.asarray(fd["radial_steps"])
    _names, rows = _read_csv(source / "s04_panel_c_b_matrix.csv")
    assert_allclose(
        _float_column(rows, "epsilon_step"),
        np.repeat(epsilon_steps, radial_steps.size),
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float_column(rows, "radial_step"),
        np.tile(radial_steps, epsilon_steps.size),
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float_column(rows, "relative_B_error"),
        B_matrix.reshape(-1),
        rtol=0,
        atol=0,
    )

    _names, rows = _read_csv(source / "s05_panel_a_heatmap.csv")
    expected_response = convergence["response_sensitivity"].reshape(-1)
    assert_allclose(
        _float_column(rows, "response_ratio"), expected_response, rtol=0, atol=0
    )
    assert_allclose(
        _float_column(rows, "source_radius_over_h"),
        np.repeat(convergence["source_width"], convergence["window_width"].size),
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float_column(rows, "window_sigma_over_k0"),
        np.tile(convergence["window_width"], convergence["source_width"].size),
        rtol=0,
        atol=0,
    )
    provenance = convergence_metadata["direct_measurement_provenance"]
    for filename in ("s05_panel_a_heatmap.csv", "s05_panel_b_profiles.csv"):
        _names, certificate_rows = _read_csv(source / filename)
        for column, value in (
            (
                "widest_window_boundary_weight",
                provenance["widest_window_boundary_weight"],
            ),
            ("boundary_weight_tolerance", provenance["boundary_weight_tolerance"]),
            (
                "widest_window_sigma_over_k0",
                provenance["widest_window_sigma_over_k0"],
            ),
        ):
            assert_allclose(
                _float_column(certificate_rows, column), value, rtol=0, atol=0
            )
    assert provenance["widest_window_boundary_weight"] < provenance[
        "boundary_weight_tolerance"
    ]
    assert provenance["widest_window_sigma_over_k0"] == pytest.approx(
        float(np.max(convergence["window_width"])), abs=0.0
    )

    _names, rows = _read_csv(source / "s06_panel_a_surface.csv")
    kx, ky = np.meshgrid(silicon["kx_grid"], silicon["ky_grid"])
    assert_allclose(_float_column(rows, "kx"), kx.reshape(-1), rtol=0, atol=0)
    assert_allclose(_float_column(rows, "ky"), ky.reshape(-1), rtol=0, atol=0)
    assert_allclose(
        _float_column(rows, "omega"), silicon["omega_grid"].reshape(-1), rtol=0, atol=0
    )
    _names, point_rows = _read_csv(source / "s06_panel_a_points.csv")
    for name in ("kx", "ky", "omega", "tracking_gap", "gradient_residual"):
        assert_allclose(
            _float_column(point_rows, name), silicon[name], rtol=0, atol=0
        )
    assert [row["kind"] for row in point_rows] == silicon["kind"].tolist()

    _names, diagnostic_rows = _read_csv(source / "s06_panel_c_diagnostics.csv")
    certificate = silicon_metadata["annular_boundary_certificate"]
    boundary = silicon_metadata["tolerances"]["minimum_boundary_gradient"]
    uncertainty = silicon_metadata["tolerances"][
        "maximum_boundary_gradient_uncertainty"
    ]
    for column, value in (
        ("annulus_center_kappa", certificate["annulus_center_kappa"]),
        ("annulus_half_width", certificate["annulus_half_width"]),
        ("coarse_boundary_nodes", certificate["coarse_boundary_nodes"]),
        ("fine_boundary_nodes", certificate["fine_boundary_nodes"]),
        ("boundary_is_noncritical", int(certificate["boundary_is_noncritical"])),
        ("index_closes", int(certificate["index_closes"])),
        ("minimum_boundary_gradient", boundary),
        ("maximum_boundary_gradient_uncertainty", uncertainty),
        ("boundary_resolution_ratio", boundary / uncertainty),
    ):
        assert_allclose(_float_column(diagnostic_rows, column), value, rtol=0, atol=0)
    assert certificate["boundary_is_noncritical"] is True
    assert certificate["index_closes"] is True
    assert boundary / uncertainty > 10.0

    _names, material_rows = _read_csv(source / "s06_material_record.csv")
    assert len(material_rows) == 1
    assert material_rows[0]["material_source_id"] == silicon_metadata["material_source_id"]
    assert material_rows[0]["orientation"] == silicon_metadata["orientation"]
    for name, value in silicon_metadata["material_constants_GPa"].items():
        assert float(material_rows[0][name]) == pytest.approx(value, abs=0.0)


def test_tables_are_deterministic_and_every_value_is_input_backed(tmp_path: Path) -> None:
    first = export_tables(
        DATA_DIR,
        VALIDATION_JSON,
        VALIDATION_CSV,
        CONFIG_PATH,
        tmp_path / "first",
    )
    second = export_tables(
        DATA_DIR,
        VALIDATION_JSON,
        VALIDATION_CSV,
        CONFIG_PATH,
        tmp_path / "second",
    )
    assert tuple(first) == ("table_s01_convergence", "table_s02_parameters")
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()

    validation = _validation_record()
    convergence_text = first["table_s01_convergence"].read_text(encoding="utf-8")
    for record in (validation["single_element"][-1], validation["two_element"]):
        for name in (
            "order",
            "elements",
            "relative_k_error",
            "relative_omega_error",
            "relative_curvature_error",
            "maximum_eigen_residual",
            "minimum_relative_eigengap",
        ):
            assert supplementary.format_table_number(record[name]) in convergence_text

    config = load_reference_config(CONFIG_PATH)
    _silicon, silicon_metadata = common.load_figure_artifact(
        DATA_DIR, "silicon_stress_test"
    )
    parameter_text = first["table_s02_parameters"].read_text(encoding="utf-8")
    for value in (
        config.h,
        config.rho,
        config.lam,
        config.mu,
        config.delta,
        config.source_radius_over_h,
        config.window_sigma_over_k0,
        config.annulus_fraction,
        config.eigen_residual_tolerance,
        config.isotropic_match_tolerance,
        config.curvature_match_tolerance,
        config.sensitivity_match_tolerance,
        config.phase_error_tolerance,
        *silicon_metadata["material_constants_GPa"].values(),
    ):
        assert supplementary.format_table_number(value) in parameter_text
    assert silicon_metadata["material_source_id"] in parameter_text


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("profile", "scientific context"),
        ("validation_exact", "auxiliary validation exact point"),
        ("phase", "phase-discrepancy threshold"),
        ("mac", "minimum tracking MAC"),
        ("B_matrix", "B finite-difference matrix"),
        ("response", "response-sensitivity baseline"),
        ("boundary_weight", "response-sensitivity provenance"),
        ("silicon_count", "silicon count"),
        ("silicon_scope", "outside the weak-anisotropy proof"),
        ("material_source", "material source identifier"),
        ("silicon_certificate", "boundary certificate"),
    ),
)
def test_corruption_fails_before_any_supplementary_output(
    corruption: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays, metadata = _artifact_copies()
    validation = _validation_record()
    if corruption == "profile":
        metadata["convergence"]["profile"] = "mismatch"
    elif corruption == "validation_exact":
        validation["exact"]["kappa0"] += 1.0e-3
    elif corruption == "phase":
        arrays["convergence"]["phase_error"][-1] = 1.0
    elif corruption == "mac":
        metadata["convergence"]["tolerances"]["minimum_tracking_mac"] = 0.5
    elif corruption == "B_matrix":
        metadata["convergence"]["finite_difference_sweep"][
            "relative_B_error_matrix"
        ][0][0] = 1.0
    elif corruption == "response":
        arrays["convergence"]["response_sensitivity"][1, 1] = 0.5
    elif corruption == "boundary_weight":
        metadata["convergence"]["direct_measurement_provenance"][
            "boundary_weight_tolerance"
        ] = 0.0
    elif corruption == "silicon_count":
        arrays["silicon_stress_test"]["kind"][0] = "saddle"
    elif corruption == "silicon_scope":
        metadata["silicon_stress_test"]["scope"] = "theorem proof"
    elif corruption == "material_source":
        metadata["silicon_stress_test"]["material_source_id"] = ""
    elif corruption == "silicon_certificate":
        metadata["silicon_stress_test"]["annular_boundary_certificate"][
            "boundary_is_noncritical"
        ] = False
    else:  # pragma: no cover - parameter guard
        raise AssertionError(corruption)

    def corrupted_loader(_data_dir: Path, name: str):
        return arrays[name], metadata[name]

    monkeypatch.setattr(supplementary, "load_figure_artifact", corrupted_loader)
    monkeypatch.setattr(
        supplementary,
        "_load_isotropic_validation",
        lambda _json_path, _csv_path: validation,
    )
    with pytest.raises((AssertionError, ValueError), match=message):
        build_all(
            DATA_DIR,
            VALIDATION_JSON,
            VALIDATION_CSV,
            CONFIG_PATH,
            tmp_path / "figures",
            tmp_path / "source",
        )
    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "source").exists()


def test_supplementary_code_has_no_plot_time_solver_fit_mask_or_manual_results() -> None:
    source = Path(supplementary.__file__).read_text(encoding="utf-8")
    make_source = (ROOT / "scripts/make_supplementary_figures.py").read_text(
        encoding="utf-8"
    )
    compact = "".join(source.split())
    assert "load_figure_artifact(data_dir,name)" in compact
    for artifact in ARTIFACT_NAMES:
        assert f'"{artifact}"' in source
    assert "stress test outside the weak-anisotropy proof" in source
    assert "doi:10.1107/S1600577514004962" not in source
    assert "165.7" not in source
    assert "63.9" not in source
    assert "79.6" not in source
    for forbidden in (
        "np.load",
        "numpy.load",
        "polyfit",
        "curve_fit",
        "lstsq",
        "np.gradient",
        "np.diff",
        "minimize",
        "root(",
        "run_isotropic_validation",
        "solve_plate_modes",
        "assemble_plate_matrices",
        "scipy",
    ):
        assert forbidden not in source
        assert forbidden not in make_source
