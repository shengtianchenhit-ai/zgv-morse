"""Evidence contract for Figure 1: critical-ring geometry and mechanism."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
import numpy as np
import pytest

import zgv_morse.figures.figure01_geometry as figure01
from zgv_morse.figures.common import load_figure_inputs


DATA_DIR = Path("data/generated")
EXPECTED_STEM = "figure_01_geometry_mechanism"
EXPECTED_SOURCE_FILES = {
    "panel_a_ring.csv",
    "panel_b_surface.csv",
    "panel_b_points.csv",
}


def _numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    return {
        name: np.array([float(row[name]) for row in rows])
        for name in rows[0]
        if name != "kind"
    }


@pytest.mark.filterwarnings("error:Glyph .* missing from font")
def test_build_exports_validated_geometry_source_data_and_exact_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str]] = []
    annotations: list[tuple[str, tuple[float, float], str | None]] = []
    validated_loader = figure01.load_figure_artifact
    original_annotate = figure01.Axes.annotate

    def tracking_loader(data_dir: Path, name: str):
        calls.append((data_dir, name))
        return validated_loader(data_dir, name)

    def tracking_annotate(self, text, xy, *args, **kwargs):
        if text == "minimum":
            annotations.append((text, kwargs["xytext"], kwargs.get("ha")))
        return original_annotate(self, text, xy, *args, **kwargs)

    monkeypatch.setattr(figure01, "load_figure_artifact", tracking_loader)
    monkeypatch.setattr(figure01.Axes, "annotate", tracking_annotate)
    output_dir = tmp_path / "figures"
    source_dir = tmp_path / "source"

    outputs = figure01.build(DATA_DIR, output_dir, source_dir)

    assert calls == [
        (DATA_DIR, "isotropic_zgv"),
        (DATA_DIR, "angular_sensitivity"),
        (DATA_DIR, "critical_points"),
    ]
    # Morse classes are keyed by a legend now, not by leader labels.
    assert annotations == []
    assert set(outputs) == {"svg", "pdf", "png", "tiff"}
    assert {path.name for path in outputs.values()} == {
        f"{EXPECTED_STEM}.{kind}" for kind in outputs
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    assert {path.name for path in source_dir.iterdir()} == EXPECTED_SOURCE_FILES

    svg = outputs["svg"].read_text(encoding="utf-8")
    assert "<text" in svg
    assert "eight resolved roots in the declared local annulus" in svg
    assert "stationary dimension 1" in svg
    assert "stationary dimension 0" in svg
    assert "t⁻¹⁄²" in svg
    assert "t⁻¹" in svg

    # Matplotlib raster canvases use integer truncation after the exact physical
    # size is set in inches; this is the exporter contract at 600 dpi.
    expected_pixels = (int(figure01._SPEC.width_mm / 25.4 * 600), int(figure01._SPEC.height_mm / 25.4 * 600))
    for kind in ("png", "tiff"):
        with Image.open(outputs[kind]) as image:
            assert image.size == expected_pixels
            assert image.info["dpi"] == pytest.approx((600.0, 600.0), abs=0.1)

    iso = load_figure_inputs(DATA_DIR, "isotropic_zgv")
    points = load_figure_inputs(DATA_DIR, "critical_points")
    kx_mesh, ky_mesh = np.meshgrid(points["kx_grid"], points["ky_grid"])

    panel_a = _numeric_csv(source_dir / "panel_a_ring.csv")
    assert set(panel_a) == {"kx", "ky", "omega_iso", "ring_kx", "ring_ky"}
    np.testing.assert_array_equal(panel_a["kx"], kx_mesh.ravel(order="C"))
    np.testing.assert_array_equal(panel_a["ky"], ky_mesh.ravel(order="C"))
    np.testing.assert_array_equal(
        panel_a["omega_iso"], points["omega_iso_grid"].ravel(order="C")
    )
    np.testing.assert_allclose(
        np.hypot(panel_a["ring_kx"], panel_a["ring_ky"]),
        float(iso["kappa0"]),
        rtol=0.0,
        atol=5e-16,
    )

    panel_b = _numeric_csv(source_dir / "panel_b_surface.csv")
    assert set(panel_b) == {"kx", "ky", "omega_aniso", "angular_anomaly"}
    np.testing.assert_array_equal(panel_b["kx"], kx_mesh.ravel(order="C"))
    np.testing.assert_array_equal(panel_b["ky"], ky_mesh.ravel(order="C"))
    np.testing.assert_array_equal(
        panel_b["omega_aniso"], points["omega_aniso_grid"].ravel(order="C")
    )

    point_csv = source_dir / "panel_b_points.csv"
    with point_csv.open(newline="", encoding="utf-8") as handle:
        point_rows = list(csv.DictReader(handle))
    assert len(point_rows) == 8
    assert [row["kind"] for row in point_rows].count("minimum") == 4
    assert [row["kind"] for row in point_rows].count("saddle") == 4
    for key in ("kx", "ky", "theta", "omega", "morse_index", "gradient_residual"):
        np.testing.assert_array_equal(
            np.array([float(row[key]) for row in point_rows]), points[key]
        )
    for column, index in (("hessian_eigenvalue_1", 0), ("hessian_eigenvalue_2", 1)):
        np.testing.assert_array_equal(
            np.array([float(row[column]) for row in point_rows]),
            points["hessian_eigenvalues"][:, index],
        )


def test_build_is_byte_deterministic_and_avoids_plot_time_inference(tmp_path: Path) -> None:
    bundles: list[tuple[dict[str, bytes], dict[str, bytes]]] = []
    for run in ("first", "second"):
        outputs = figure01.build(DATA_DIR, tmp_path / run / "figures", tmp_path / run / "source")
        bundles.append(
            (
                {kind: path.read_bytes() for kind, path in outputs.items()},
                {
                    path.name: path.read_bytes()
                    for path in sorted((tmp_path / run / "source").iterdir())
                },
            )
        )

    assert bundles[0] == bundles[1]
    source = Path(figure01.__file__).read_text(encoding="utf-8")
    assert "np.load" not in source
    assert "scipy" not in source
    assert "gradient(" not in source
    assert "minimize(" not in source
    assert "root(" not in source


def _corrupt_certificate(
    points: dict[str, np.ndarray],
    metadata: dict[str, object],
    invariant: str,
) -> None:
    tolerances = metadata["tolerances"]
    assert isinstance(tolerances, dict)
    if invariant == "count":
        points["kind"][0] = "minimum"
    elif invariant == "alternation":
        points["theta"][[1, 2]] = points["theta"][[2, 1]]
    elif invariant == "index sum":
        points["morse_index"][0] = 1
    elif invariant == "pointwise index":
        points["morse_index"][[0, 1]] = points["morse_index"][[1, 0]]
    elif invariant == "Hessian inertia":
        points["hessian_eigenvalues"][0] = np.abs(points["hessian_eigenvalues"][0])
    elif invariant == "resolved Hessian":
        tolerances["minimum_hessian_to_uncertainty"] = 1.0
    elif invariant == "finite gradient":
        points["gradient_residual"][0] = np.nan
    elif invariant == "bounded gradient":
        tolerances["maximum_gradient_residual"] = 1.0
        points["gradient_residual"][0] = 1.0
    elif invariant == "noncritical boundary":
        tolerances["positive_minimum_boundary_gradient"] = 0.0
    else:  # pragma: no cover - test helper guard
        raise AssertionError(invariant)


@pytest.mark.parametrize(
    "invariant",
    (
        "count",
        "alternation",
        "index sum",
        "pointwise index",
        "Hessian inertia",
        "resolved Hessian",
        "finite gradient",
        "bounded gradient",
        "noncritical boundary",
    ),
)
def test_scientific_invariants_fail_before_render_or_source_export(
    invariant: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = {
        name: figure01.load_figure_artifact(DATA_DIR, name)
        for name in ("isotropic_zgv", "angular_sensitivity", "critical_points")
    }
    artifacts = {
        name: {key: np.array(value, copy=True) for key, value in arrays.items()}
        for name, (arrays, _metadata) in bundles.items()
    }
    metadata = {
        name: {
            **sidecar,
            "tolerances": dict(sidecar.get("tolerances", {})),
        }
        for name, (_arrays, sidecar) in bundles.items()
    }
    _corrupt_certificate(
        artifacts["critical_points"], metadata["critical_points"], invariant
    )
    monkeypatch.setattr(
        figure01,
        "load_figure_artifact",
        lambda _data_dir, name: (artifacts[name], metadata[name]),
    )

    with pytest.raises(AssertionError, match=invariant):
        figure01.build(DATA_DIR, tmp_path / "figures", tmp_path / "source")

    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "source").exists()
