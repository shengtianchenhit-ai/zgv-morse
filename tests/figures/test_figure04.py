"""Evidence, certificate, and export contract for main Figure 4."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.text import Annotation
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image
import pytest

from zgv_morse.config import load_reference_config
from zgv_morse.figures import common
import zgv_morse.figures.figure04_morse as figure04


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/generated"
CONFIG_PATH = ROOT / "config/reference.yaml"
CONFIG = load_reference_config(CONFIG_PATH)
EXPECTED_STEM = "figure_04_morse_points"
EXPECTED_SOURCE_FILES = {
    "panel_a_isotropic_surface.csv",
    "panel_b_anisotropic_surface.csv",
    "panel_b_points.csv",
    "panel_c_hessian.csv",
    "panel_d_prediction_errors.csv",
}
LOCAL_COUNT_TEXT = "eight resolved roots in the declared local annulus"


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return tuple(reader.fieldnames), list(reader)


def _float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(row[name]) for row in rows], dtype=np.float64)


def _artifact_bundles():
    return {
        name: common.load_figure_artifact(DATA_DIR, name)
        for name in ("isotropic_zgv", "critical_points")
    }


@pytest.mark.filterwarnings("error:Glyph .* missing from font")
def test_build_exports_exact_local_morse_evidence_and_source_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = _artifact_bundles()
    isotropic, _isotropic_metadata = bundles["isotropic_zgv"]
    points, points_metadata = bundles["critical_points"]
    calls: list[tuple[Path, str]] = []
    scatter_calls: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []
    validated_loader = figure04.load_figure_artifact
    original_scatter = Axes.scatter

    def tracking_loader(data_dir: Path, name: str):
        calls.append((data_dir, name))
        return validated_loader(data_dir, name)

    def tracking_scatter(self, x, y, *args, **kwargs):
        scatter_calls.append((np.asarray(x), np.asarray(y), dict(kwargs)))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(figure04, "load_figure_artifact", tracking_loader)
    monkeypatch.setattr(Axes, "scatter", tracking_scatter)

    bundles_out: list[tuple[dict[str, Path], Path]] = []
    for run in ("first", "second"):
        output_dir = tmp_path / run / "figures"
        source_dir = tmp_path / run / "source"
        bundles_out.append((figure04.build(DATA_DIR, output_dir, source_dir), source_dir))

    assert calls == [
        (DATA_DIR, "isotropic_zgv"),
        (DATA_DIR, "critical_points"),
        (DATA_DIR, "isotropic_zgv"),
        (DATA_DIR, "critical_points"),
    ]
    (first, first_source), (second, second_source) = bundles_out
    assert set(first) == set(common.PUBLICATION_FORMATS)
    assert set(second) == set(common.PUBLICATION_FORMATS)
    for kind in common.PUBLICATION_FORMATS:
        assert first[kind].name == f"{EXPECTED_STEM}.{kind}"
        assert first[kind].read_bytes() == second[kind].read_bytes()

    svg = first["svg"].read_text(encoding="utf-8")
    assert "<text" in svg
    assert svg.count(LOCAL_COUNT_TEXT) == 1
    assert "entire dispersion surface" not in svg
    assert "Cartesian Hessian eigenvalue" in svg
    assert "relative signed error" in svg
    assert "dc:date" not in svg
    assert b"/CreationDate" not in first["pdf"].read_bytes()
    expected_pixels = (
        int(figure04._SPEC.width_mm / 25.4 * 600),
        int(figure04._SPEC.height_mm / 25.4 * 600),
    )
    for kind in ("png", "tiff"):
        with Image.open(first[kind]) as image:
            assert image.size == expected_pixels
            assert image.info["dpi"] == pytest.approx((600.0, 600.0), abs=0.1)
            if kind == "tiff":
                assert image.info["compression"] == "tiff_lzw"
                assert image.tag_v2[259] == 5

    kinds = np.asarray(points["kind"])
    for kind in ("minimum", "saddle"):
        selected = kinds == kind
        assert any(
            assert_x.shape == points["kx"][selected].shape
            and assert_y.shape == points["ky"][selected].shape
            and np.array_equal(assert_x, points["kx"][selected])
            and np.array_equal(assert_y, points["ky"][selected])
            and kwargs.get("marker") == common.MARKERS[kind]
            and kwargs.get("facecolor") == common.PALETTE[kind]
            for assert_x, assert_y, kwargs in scatter_calls
        )

    assert {path.name for path in first_source.iterdir()} == EXPECTED_SOURCE_FILES
    for name in EXPECTED_SOURCE_FILES:
        assert (first_source / name).read_bytes() == (second_source / name).read_bytes()

    kappa0 = float(isotropic["kappa0"])
    omega0 = float(isotropic["omega0"])
    half_width = CONFIG.annulus_fraction * kappa0
    inner_radius = kappa0 - half_width
    outer_radius = kappa0 + half_width
    kx_mesh, ky_mesh = np.meshgrid(points["kx_grid"], points["ky_grid"])
    circle_theta = np.linspace(
        0.0,
        2.0 * np.pi,
        points["omega_iso_grid"].size,
        endpoint=True,
    )

    names_a, rows_a = _read_csv(first_source / "panel_a_isotropic_surface.csv")
    assert set(names_a) == {
        "annulus_fraction",
        "annulus_inner_kx",
        "annulus_inner_ky",
        "annulus_outer_kx",
        "annulus_outer_ky",
        "kx",
        "ky",
        "omega_iso",
        "ring_kx",
        "ring_ky",
    }
    assert len(rows_a) == points["omega_iso_grid"].size
    assert_array_equal(_float_column(rows_a, "kx"), kx_mesh.ravel(order="C"))
    assert_array_equal(_float_column(rows_a, "ky"), ky_mesh.ravel(order="C"))
    assert_array_equal(
        _float_column(rows_a, "omega_iso"),
        points["omega_iso_grid"].ravel(order="C"),
    )
    assert_array_equal(
        _float_column(rows_a, "ring_kx"), kappa0 * np.cos(circle_theta)
    )
    assert_array_equal(
        _float_column(rows_a, "ring_ky"), kappa0 * np.sin(circle_theta)
    )
    assert_allclose(
        np.hypot(
            _float_column(rows_a, "annulus_inner_kx"),
            _float_column(rows_a, "annulus_inner_ky"),
        ),
        inner_radius,
        rtol=0.0,
        atol=5.0e-16,
    )
    assert_allclose(
        np.hypot(
            _float_column(rows_a, "annulus_outer_kx"),
            _float_column(rows_a, "annulus_outer_ky"),
        ),
        outer_radius,
        rtol=0.0,
        atol=5.0e-16,
    )
    assert_array_equal(
        _float_column(rows_a, "annulus_fraction"),
        np.full(circle_theta.size, CONFIG.annulus_fraction),
    )
    assert CONFIG.annulus_fraction == pytest.approx(0.15, abs=0.0)

    names_b, rows_b = _read_csv(first_source / "panel_b_anisotropic_surface.csv")
    assert set(names_b) == {
        "annulus_fraction",
        "annulus_inner_kx",
        "annulus_inner_ky",
        "annulus_outer_kx",
        "annulus_outer_ky",
        "kx",
        "ky",
        "omega_aniso",
    }
    assert_array_equal(_float_column(rows_b, "kx"), kx_mesh.ravel(order="C"))
    assert_array_equal(_float_column(rows_b, "ky"), ky_mesh.ravel(order="C"))
    assert_array_equal(
        _float_column(rows_b, "omega_aniso"),
        points["omega_aniso_grid"].ravel(order="C"),
    )
    assert_array_equal(
        _float_column(rows_b, "annulus_fraction"),
        np.full(circle_theta.size, CONFIG.annulus_fraction),
    )

    names_points, rows_points = _read_csv(first_source / "panel_b_points.csv")
    point_columns = {
        "artifact_row",
        "gradient_residual",
        "gradient_residual_over_uncertainty",
        "gradient_uncertainty_bound",
        "hessian_eigenvalue_1",
        "hessian_eigenvalue_2",
        "kind",
        "kappa",
        "kx",
        "kx_pred",
        "ky",
        "ky_pred",
        "morse_index",
        "negative_boundary_resolution_ratio",
        "omega",
        "omega_pred",
        "positive_boundary_resolution_ratio",
        "theta",
    }
    assert set(names_points) == point_columns
    assert len(rows_points) == 8
    assert [row["kind"] for row in rows_points].count("minimum") == 4
    assert [row["kind"] for row in rows_points].count("saddle") == 4
    assert_array_equal(_float_column(rows_points, "artifact_row"), np.arange(8))
    for name in (
        "gradient_residual",
        "kappa",
        "kx",
        "kx_pred",
        "ky",
        "ky_pred",
        "morse_index",
        "omega",
        "omega_pred",
        "theta",
    ):
        assert_array_equal(_float_column(rows_points, name), points[name])
    for column, index in (("hessian_eigenvalue_1", 0), ("hessian_eigenvalue_2", 1)):
        assert_array_equal(
            _float_column(rows_points, column), points["hessian_eigenvalues"][:, index]
        )

    tolerances = points_metadata["tolerances"]
    assert isinstance(tolerances, dict)
    independent_gradient_uncertainty = min(
        float(tolerances["positive_maximum_boundary_gradient_uncertainty"]),
        float(tolerances["negative_maximum_boundary_gradient_uncertainty"]),
    )
    assert_array_equal(
        _float_column(rows_points, "gradient_uncertainty_bound"),
        np.full(8, independent_gradient_uncertainty),
    )
    assert_allclose(
        _float_column(rows_points, "gradient_residual_over_uncertainty"),
        points["gradient_residual"] / independent_gradient_uncertainty,
        rtol=0.0,
        atol=0.0,
    )

    order = np.argsort(np.mod(points["theta"], 2.0 * np.pi), kind="stable")
    names_c, rows_c = _read_csv(first_source / "panel_c_hessian.csv")
    assert set(names_c) == {
        "artifact_row",
        "hessian_eigenvalue_1",
        "hessian_eigenvalue_2",
        "hessian_uncertainty_upper_bound",
        "kind",
        "minimum_hessian_to_uncertainty",
        "morse_index",
        "point_number",
        "theta",
    }
    assert_array_equal(_float_column(rows_c, "artifact_row"), order)
    assert_array_equal(_float_column(rows_c, "point_number"), np.arange(1, 9))
    assert [row["kind"] for row in rows_c] == list(points["kind"][order])
    ratio = float(tolerances["minimum_hessian_to_uncertainty"])
    hessian = points["hessian_eigenvalues"][order]
    uncertainty_upper_bound = np.min(np.abs(hessian), axis=1) / ratio
    assert_array_equal(_float_column(rows_c, "hessian_eigenvalue_1"), hessian[:, 0])
    assert_array_equal(_float_column(rows_c, "hessian_eigenvalue_2"), hessian[:, 1])
    assert_allclose(
        _float_column(rows_c, "hessian_uncertainty_upper_bound"),
        uncertainty_upper_bound,
        rtol=0.0,
        atol=0.0,
    )
    assert_array_equal(
        _float_column(rows_c, "minimum_hessian_to_uncertainty"), np.full(8, ratio)
    )

    names_d, rows_d = _read_csv(first_source / "panel_d_prediction_errors.csv")
    assert set(names_d) == {
        "artifact_row",
        "cartesian_location_error_over_kappa0",
        "delta_kappa_over_kappa0",
        "delta_omega_over_omega0",
        "kind",
        "point_number",
        "theta",
    }
    assert_array_equal(_float_column(rows_d, "artifact_row"), order)
    assert_array_equal(_float_column(rows_d, "point_number"), np.arange(1, 9))
    predicted_kappa = np.hypot(points["kx_pred"], points["ky_pred"])
    delta_kappa = (points["kappa"] - predicted_kappa) / kappa0
    delta_omega = (points["omega"] - points["omega_pred"]) / omega0
    cartesian_error = np.hypot(
        points["kx"] - points["kx_pred"],
        points["ky"] - points["ky_pred"],
    ) / kappa0
    assert_array_equal(
        _float_column(rows_d, "delta_kappa_over_kappa0"), delta_kappa[order]
    )
    assert_array_equal(
        _float_column(rows_d, "delta_omega_over_omega0"), delta_omega[order]
    )
    assert_allclose(
        _float_column(rows_d, "cartesian_location_error_over_kappa0"),
        cartesian_error[order],
        rtol=0.0,
        atol=0.0,
    )


def _corrupt_bundle(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    corruption: str,
) -> None:
    tolerances = metadata["tolerances"]
    assert isinstance(tolerances, dict)
    if corruption == "count":
        arrays["kind"][0] = "minimum"
    elif corruption == "alternation":
        arrays["theta"][[1, 2]] = arrays["theta"][[2, 1]]
    elif corruption == "index sum":
        arrays["morse_index"][0] = 1
    elif corruption == "pointwise index":
        arrays["morse_index"][[0, 1]] = arrays["morse_index"][[1, 0]]
    elif corruption == "Hessian inertia":
        arrays["hessian_eigenvalues"][0] = np.abs(
            arrays["hessian_eigenvalues"][0]
        )
    elif corruption == "resolved Hessian":
        tolerances["minimum_hessian_to_uncertainty"] = 1.0
    elif corruption == "finite gradient":
        arrays["gradient_residual"][0] = np.nan
    elif corruption == "bounded gradient":
        arrays["gradient_residual"][0] = 1.0
        tolerances["maximum_gradient_residual"] = 1.0
    elif corruption == "positive boundary":
        tolerances["positive_minimum_boundary_gradient"] = 0.0
    elif corruption == "negative boundary":
        tolerances["negative_minimum_boundary_gradient"] = 0.0
    elif corruption == "registered annulus":
        arrays["kx_grid"][[0, -1]] *= 1.01
        arrays["ky_grid"][[0, -1]] *= 1.01
    elif corruption == "scientific context":
        metadata["config_hash"] = "0" * 64
    else:  # pragma: no cover - test helper guard
        raise AssertionError(corruption)


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("count", "count"),
        ("alternation", "alternation"),
        ("index sum", "index sum"),
        ("pointwise index", "pointwise index"),
        ("Hessian inertia", "Hessian inertia"),
        ("resolved Hessian", "resolved Hessian"),
        ("finite gradient", "finite gradient"),
        ("bounded gradient", "bounded gradient"),
        ("positive boundary", "positive boundary"),
        ("negative boundary", "negative boundary"),
        ("registered annulus", "registered annulus"),
        ("scientific context", "scientific context"),
    ),
)
def test_certificate_corruptions_fail_before_any_output(
    corruption: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = _artifact_bundles()
    arrays = {
        name: {key: np.array(value, copy=True) for key, value in values.items()}
        for name, (values, _metadata) in bundles.items()
    }
    metadata = {
        name: copy.deepcopy(sidecar)
        for name, (_values, sidecar) in bundles.items()
    }
    _corrupt_bundle(arrays["critical_points"], metadata["critical_points"], corruption)

    monkeypatch.setattr(
        figure04,
        "load_figure_artifact",
        lambda _data_dir, name: (arrays[name], metadata[name]),
    )
    output_dir = tmp_path / "figures"
    source_dir = tmp_path / "source"
    with pytest.raises((AssertionError, ValueError), match=message):
        figure04.build(DATA_DIR, output_dir, source_dir)

    assert not output_dir.exists()
    assert not source_dir.exists()


def test_figure04_uses_registered_inputs_without_plot_time_inference() -> None:
    source = Path(figure04.__file__).read_text(encoding="utf-8")

    assert 'load_figure_artifact(data_dir, "isotropic_zgv")' in source
    assert 'load_figure_artifact(data_dir, "critical_points")' in source
    assert LOCAL_COUNT_TEXT in source
    assert "entire dispersion surface" not in source
    assert "0.15" not in source
    for forbidden in (
        "np.load",
        "numpy.load",
        "scipy",
        "gradient(",
        "root(",
        "minimize(",
        "polyfit(",
        "curve_fit(",
        "lstsq(",
    ):
        assert forbidden not in source


def test_geometry_label_and_quantitative_panels_have_visual_headroom() -> None:
    isotropic, isotropic_metadata = common.load_figure_artifact(
        DATA_DIR, "isotropic_zgv"
    )
    points, points_metadata = common.load_figure_artifact(DATA_DIR, "critical_points")
    evidence = figure04._validate_evidence(
        isotropic,
        isotropic_metadata,
        points,
        points_metadata,
    )
    surface_kx, surface_ky = np.meshgrid(points["kx_grid"], points["ky_grid"])
    circles = figure04._circle_arrays(points["omega_iso_grid"].size, evidence)
    levels = np.linspace(
        min(float(np.min(points["omega_iso_grid"])), float(np.min(points["omega_aniso_grid"]))),
        max(float(np.max(points["omega_iso_grid"])), float(np.max(points["omega_aniso_grid"]))),
        9,
    )
    common.apply_publication_style()
    fig, axes = plt.subplots(1, 3)
    try:
        figure04._draw_panel_a(
            axes[0],
            surface_kx,
            surface_ky,
            points["omega_iso_grid"],
            points,
            circles,
            levels,
        )
        figure04._draw_panel_c(axes[1], points, evidence)
        figure04._draw_panel_d(axes[2], points, evidence)
        fig.canvas.draw()

        ring_labels = [
            text for text in axes[0].texts if text.get_text() == "critical ring"
        ]
        assert len(ring_labels) == 1 and isinstance(ring_labels[0], Annotation)
        assert_allclose(
            np.hypot(*ring_labels[0].xy),
            evidence.kappa0,
            rtol=0.0,
            atol=5.0e-16,
        )
        assert axes[1].get_ylim()[1] > 1.80 * float(
            np.max(points["hessian_eigenvalues"])
        )
        assert abs(axes[1].get_ylim()[0]) > 1.80 * abs(
            float(np.min(points["hessian_eigenvalues"]))
        )
        eigenvalue_key = next(
            text for text in axes[1].texts if text.get_text().startswith("filled:")
        )
        assert eigenvalue_key.get_position()[1] <= 0.85
        assert axes[2].get_ylim()[1] > 0.15 * abs(axes[2].get_ylim()[0])
    finally:
        plt.close(fig)
