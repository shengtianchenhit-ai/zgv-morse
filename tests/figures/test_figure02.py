"""Evidence and export contract for main Figure 2."""

from __future__ import annotations

import csv
from pathlib import Path

from matplotlib.axes import Axes
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image
import pytest

from zgv_morse.config import load_reference_config
from zgv_morse.figures import common
from zgv_morse.figures.figure02_isotropic import build
import zgv_morse.figures.figure02_isotropic as figure02


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/generated"
CONFIG = load_reference_config(ROOT / "config/reference.yaml")
SOURCE_FILENAMES = (
    "panel_a_branches.csv",
    "panel_b_local_quadratic.csv",
    "panel_c_convergence.csv",
    "panel_d_mode_profile.csv",
)


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return tuple(reader.fieldnames), list(reader)


def _float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(row[name]) for row in rows], dtype=np.float64)


def _artifact_copies() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    isotropic = common.load_figure_inputs(DATA_DIR, "isotropic_zgv")
    convergence = common.load_figure_inputs(DATA_DIR, "convergence")
    return (
        {name: np.array(value, copy=True) for name, value in isotropic.items()},
        {name: np.array(value, copy=True) for name, value in convergence.items()},
    )


def test_build_exports_exact_evidence_and_is_byte_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isotropic, convergence = _artifact_copies()
    calls: list[str] = []
    scatter_points: list[tuple[np.ndarray, np.ndarray]] = []
    real_loader = common.load_figure_inputs
    real_scatter = Axes.scatter

    def tracked_loader(data_dir: Path, name: str) -> dict[str, np.ndarray]:
        calls.append(name)
        return real_loader(data_dir, name)

    def tracked_scatter(self: Axes, x, y, *args, **kwargs):
        scatter_points.append((np.asarray(x), np.asarray(y)))
        return real_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(figure02, "load_figure_inputs", tracked_loader)
    monkeypatch.setattr(Axes, "scatter", tracked_scatter)

    first_output = tmp_path / "first/figures"
    first_source = tmp_path / "first/source"
    second_output = tmp_path / "second/figures"
    second_source = tmp_path / "second/source"
    first = build(DATA_DIR, first_output, first_source)
    second = build(DATA_DIR, second_output, second_source)

    assert calls == [
        "isotropic_zgv",
        "convergence",
        "isotropic_zgv",
        "convergence",
    ]
    assert set(first) == set(common.PUBLICATION_FORMATS)
    assert set(second) == set(common.PUBLICATION_FORMATS)
    for kind in common.PUBLICATION_FORMATS:
        assert first[kind].name == f"figure_02_isotropic_zgv.{kind}"
        assert second[kind].name == f"figure_02_isotropic_zgv.{kind}"
        assert first[kind].read_bytes() == second[kind].read_bytes()

    svg = first["svg"].read_text(encoding="utf-8")
    assert "<text" in svg
    assert "ZGV" in svg
    assert str(isotropic["branch_labels"][0]) in svg
    assert "quadratic prediction" in svg
    assert "a = " in svg
    assert "&gt; 0" in svg
    assert "dimensionless wavenumber" in svg
    assert "dimensionless frequency" in svg
    assert "radial offset" in svg
    assert "dc:date" not in svg
    assert b"/CreationDate" not in first["pdf"].read_bytes()
    expected_pixels = (
        int(figure02._SPEC.width_mm / 25.4 * 600),
        int(figure02._SPEC.height_mm / 25.4 * 600),
    )
    for kind in ("png", "tiff"):
        with Image.open(first[kind]) as image:
            assert image.size == expected_pixels
            assert image.info["dpi"] == pytest.approx((600.0, 600.0), abs=0.1)

    kappa0 = float(isotropic["kappa0"])
    omega0 = float(isotropic["omega0"])
    assert any(
        x.size == y.size == 1
        and float(x.ravel()[0]) == pytest.approx(kappa0, abs=0.0)
        and float(y.ravel()[0]) == pytest.approx(omega0, abs=0.0)
        for x, y in scatter_points
    )
    assert np.isfinite(float(isotropic["curvature_a"]))
    assert float(isotropic["curvature_a"]) > 0.0

    assert tuple(sorted(path.name for path in first_source.glob("*.csv"))) == tuple(
        sorted(SOURCE_FILENAMES)
    )
    for filename in SOURCE_FILENAMES:
        assert (first_source / filename).read_bytes() == (
            second_source / filename
        ).read_bytes()

    names_a, rows_a = _read_csv(first_source / "panel_a_branches.csv")
    assert set(names_a) == {
        "branch_label",
        "kappa",
        "omega",
        "zgv_kappa0",
        "zgv_omega0",
    }
    branch_count, sample_count = isotropic["omega_symmetric"].shape
    assert len(rows_a) == branch_count * sample_count
    assert_array_equal(
        np.array([row["branch_label"] for row in rows_a]),
        np.repeat(isotropic["branch_labels"].astype(str), sample_count),
    )
    assert_allclose(
        _float_column(rows_a, "kappa"),
        np.tile(isotropic["kappa"], branch_count),
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(
        _float_column(rows_a, "omega"),
        isotropic["omega_symmetric"].reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    assert_allclose(_float_column(rows_a, "zgv_kappa0"), kappa0, rtol=0.0, atol=0.0)
    assert_allclose(_float_column(rows_a, "zgv_omega0"), omega0, rtol=0.0, atol=0.0)

    names_b, rows_b = _read_csv(first_source / "panel_b_local_quadratic.csv")
    assert set(names_b) == {
        "curvature_a",
        "local_q",
        "local_omega",
        "local_quadratic",
        "omega0",
    }
    formula = omega0 + 0.5 * float(isotropic["curvature_a"]) * isotropic["local_q"] ** 2
    assert_allclose(isotropic["local_quadratic"], formula, rtol=2e-15, atol=2e-15)
    assert_allclose(_float_column(rows_b, "local_q"), isotropic["local_q"], rtol=0, atol=0)
    assert_allclose(
        _float_column(rows_b, "local_omega"), isotropic["local_omega"], rtol=0, atol=0
    )
    assert_allclose(_float_column(rows_b, "local_quadratic"), formula, rtol=0, atol=0)
    assert_allclose(
        _float_column(rows_b, "curvature_a"),
        float(isotropic["curvature_a"]),
        rtol=0,
        atol=0,
    )

    names_c, rows_c = _read_csv(first_source / "panel_c_convergence.csv")
    convergence_keys = (
        "polynomial_order",
        "omega0_error",
        "kappa0_error",
        "curvature_error",
        "eigen_residual",
        "hermitian_residual",
        "mass_orthogonality",
        "eigengap",
    )
    assert set(names_c) == set(convergence_keys)
    for name in convergence_keys:
        assert_allclose(_float_column(rows_c, name), convergence[name], rtol=0, atol=0)
    assert np.all(convergence["omega0_error"][-2:] < CONFIG.isotropic_match_tolerance)
    assert np.all(convergence["kappa0_error"][-2:] < CONFIG.isotropic_match_tolerance)
    assert np.all(convergence["curvature_error"][-2:] < CONFIG.curvature_match_tolerance)
    assert np.all(convergence["eigen_residual"][-2:] < CONFIG.eigen_residual_tolerance)
    assert np.all(
        convergence["eigengap"][-2:]
        > 10.0 * np.maximum(convergence["eigen_residual"][-2:], np.finfo(float).eps)
    )

    names_d, rows_d = _read_csv(first_source / "panel_d_mode_profile.csv")
    profile_keys = {
        "z_over_h",
        "u_x_magnitude_normalized",
        "u_y_magnitude_normalized",
        "u_z_magnitude_normalized",
        "squared_displacement_proxy_normalized",
    }
    assert set(names_d) == profile_keys
    displacement = np.abs(isotropic["mode_u"])
    displacement /= np.max(displacement)
    squared_displacement = isotropic["mode_squared_displacement"]
    squared_displacement /= np.max(squared_displacement)
    assert_allclose(_float_column(rows_d, "z_over_h"), isotropic["mode_z"], rtol=0, atol=0)
    for index, component in enumerate(("x", "y", "z")):
        assert_allclose(
            _float_column(rows_d, f"u_{component}_magnitude_normalized"),
            displacement[:, index],
            rtol=0,
            atol=0,
        )
    assert_allclose(
        _float_column(rows_d, "squared_displacement_proxy_normalized"),
        squared_displacement,
        rtol=0,
        atol=0,
    )
    assert np.isfinite(displacement).all()
    assert np.isfinite(squared_displacement).all()


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("quadratic", "local_quadratic"),
        ("curvature", "positive radial curvature"),
        ("convergence", "final two polynomial orders"),
    ),
)
def test_build_rejects_unverified_scientific_evidence_before_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    message: str,
) -> None:
    isotropic, convergence = _artifact_copies()
    if corruption == "quadratic":
        isotropic["local_quadratic"][0] += 1.0e-3
    elif corruption == "curvature":
        isotropic["curvature_a"] = np.array(-1.0)
    else:
        convergence["curvature_error"][-2:] = 1.0

    def corrupted_loader(_data_dir: Path, name: str) -> dict[str, np.ndarray]:
        return isotropic if name == "isotropic_zgv" else convergence

    monkeypatch.setattr(figure02, "load_figure_inputs", corrupted_loader)
    with pytest.raises((AssertionError, ValueError), match=message):
        build(DATA_DIR, tmp_path / "figures", tmp_path / "source")

    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "source").exists()


def test_figure02_uses_validated_inputs_without_loading_or_fitting() -> None:
    source = Path(figure02.__file__).read_text(encoding="utf-8")

    assert "load_figure_inputs(data_dir, \"isotropic_zgv\")" in source
    assert "load_figure_inputs(data_dir, \"convergence\")" in source
    assert "np.load" not in source
    assert "numpy.load" not in source
    assert "polyfit" not in source
    assert "curve_fit" not in source
    assert "lstsq" not in source
    assert r'wavenumber $\kappa h$' not in source
    assert r'radial offset $q h$' not in source
