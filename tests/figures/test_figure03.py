"""Evidence, convention, provenance, and export contract for main Figure 3."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image
import pytest

from zgv_morse.figures import common
from zgv_morse.figures.figure03_sensitivity import build
import zgv_morse.figures.figure03_sensitivity as figure03


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/generated"
EXPECTED_STEM = "figure_03_angular_sensitivity"
SOURCE_FILENAMES = (
    "panel_a_polar.csv",
    "panel_b_harmonics.csv",
    "panel_c_physical_shift.csv",
    "panel_d_angular_fd.csv",
    "panel_d_step_convergence.csv",
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
        name: common.load_figure_artifact(DATA_DIR, name)
        for name in ("angular_sensitivity", "convergence")
    }
    arrays = {
        name: {key: np.array(value, copy=True) for key, value in values.items()}
        for name, (values, _metadata) in bundles.items()
    }
    metadata = {name: copy.deepcopy(sidecar) for name, (_values, sidecar) in bundles.items()}
    return arrays, metadata


@pytest.mark.filterwarnings("error:Glyph .* missing from font")
def test_build_exports_exact_artifact_evidence_and_is_byte_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    angular, _angular_metadata = common.load_figure_artifact(DATA_DIR, "angular_sensitivity")
    convergence, _convergence_metadata = common.load_figure_artifact(DATA_DIR, "convergence")
    calls: list[str] = []
    real_loader = common.load_figure_artifact

    def tracked_loader(data_dir: Path, name: str):
        calls.append(name)
        return real_loader(data_dir, name)

    monkeypatch.setattr(figure03, "load_figure_artifact", tracked_loader)
    bundles: list[tuple[dict[str, Path], Path]] = []
    for run in ("first", "second"):
        output_dir = tmp_path / run / "figures"
        source_dir = tmp_path / run / "source"
        bundles.append((build(DATA_DIR, output_dir, source_dir), source_dir))

    assert calls == [
        "angular_sensitivity",
        "convergence",
        "angular_sensitivity",
        "convergence",
    ]
    first, first_source = bundles[0]
    second, second_source = bundles[1]
    assert set(first) == set(common.PUBLICATION_FORMATS)
    assert set(second) == set(common.PUBLICATION_FORMATS)
    for kind in common.PUBLICATION_FORMATS:
        assert first[kind].name == f"{EXPECTED_STEM}.{kind}"
        assert first[kind].read_bytes() == second[kind].read_bytes()

    svg = first["svg"].read_text(encoding="utf-8")
    assert "<text" in svg
    # Panel titles removed; axis labels identify these panels.
    assert "angular harmonic order" in svg
    assert "harmonic amplitude" in svg
    assert "physical frequency shift" in svg
    assert "centered FD" in svg
    assert "one application of" in svg
    assert "dc:date" not in svg
    assert b"/CreationDate" not in first["pdf"].read_bytes()

    expected_pixels = (
        int(figure03._SPEC.width_mm / 25.4 * 600),
        int(figure03._SPEC.height_mm / 25.4 * 600),
    )
    with Image.open(first["png"]) as image:
        assert image.size == expected_pixels
        assert image.info["dpi"] == pytest.approx((600.0, 600.0), abs=0.1)
    with Image.open(first["tiff"]) as image:
        assert image.size == expected_pixels
        assert image.info["dpi"] == pytest.approx((600.0, 600.0), abs=0.1)
        assert image.tag_v2[259] == 5  # lossless LZW compression

    assert tuple(sorted(path.name for path in first_source.glob("*.csv"))) == tuple(
        sorted(SOURCE_FILENAMES)
    )
    for filename in SOURCE_FILENAMES:
        assert (first_source / filename).read_bytes() == (second_source / filename).read_bytes()

    theta = angular["theta"]
    angular_step = float(theta[1] - theta[0])
    period = angular_step * theta.size
    closed_theta = np.concatenate((theta, theta[:1] + period))
    closed_V = np.concatenate((angular["V"], angular["V"][:1]))
    closed_reconstruction = np.concatenate(
        (angular["V_reconstruction"], angular["V_reconstruction"][:1])
    )
    names_a, rows_a = _read_csv(first_source / "panel_a_polar.csv")
    assert set(names_a) == {
        "theta",
        "V_minus_V0",
        "fourfold_reconstruction",
    }
    assert_allclose(_float_column(rows_a, "theta"), closed_theta, rtol=0, atol=0)
    assert_allclose(
        _float_column(rows_a, "V_minus_V0"),
        closed_V - float(angular["V0"]),
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float_column(rows_a, "fourfold_reconstruction"),
        closed_reconstruction - float(angular["V0"]),
        rtol=0,
        atol=0,
    )

    names_b, rows_b = _read_csv(first_source / "panel_b_harmonics.csv")
    assert set(names_b) == {"harmonic_order", "harmonic_amplitude", "is_m4"}
    assert_array_equal(_float_column(rows_b, "harmonic_order"), angular["harmonic_order"])
    assert_allclose(
        _float_column(rows_b, "harmonic_amplitude"),
        angular["harmonic_amplitude"],
        rtol=0,
        atol=0,
    )
    is_m4 = angular["harmonic_order"] == 4
    assert_array_equal(_float_column(rows_b, "is_m4"), is_m4.astype(int))
    assert np.count_nonzero(is_m4) == 1
    assert float(angular["V4"]) != 0.0
    assert_allclose(
        angular["harmonic_amplitude"][is_m4],
        abs(float(angular["V4"])),
        rtol=2e-14,
        atol=2e-15,
    )

    names_c, rows_c = _read_csv(first_source / "panel_c_physical_shift.csv")
    assert set(names_c) == {
        "delta_c",
        "epsilon",
        "physical_epsilon_V4_shift",
        "Q4",
        "Q4_delta_c_prediction",
    }
    for name, expected in (
        ("epsilon", angular["epsilon"]),
        ("delta_c", angular["delta_c"]),
        ("physical_epsilon_V4_shift", angular["physical_V4_shift"]),
    ):
        assert_allclose(_float_column(rows_c, name), expected, rtol=0, atol=0)
    assert_allclose(
        angular["physical_V4_shift"],
        angular["epsilon"] * float(angular["V4"]),
        rtol=2e-14,
        atol=2e-15,
    )
    assert_allclose(
        _float_column(rows_c, "Q4_delta_c_prediction"),
        angular["physical_V4_shift"],
        rtol=2e-14,
        atol=2e-15,
    )
    q4 = _float_column(rows_c, "Q4")
    assert_allclose(q4, q4[0], rtol=0, atol=0)
    assert_allclose(
        q4 * angular["delta_c"],
        angular["physical_V4_shift"],
        rtol=2e-14,
        atol=2e-15,
    )

    names_d, rows_d = _read_csv(first_source / "panel_d_angular_fd.csv")
    assert set(names_d) == {"theta", "V", "V_fd", "B", "B_fd"}
    for name in ("theta", "V", "V_fd", "B", "B_fd"):
        assert_allclose(_float_column(rows_d, name), angular[name], rtol=0, atol=0)

    names_step, rows_step = _read_csv(first_source / "panel_d_step_convergence.csv")
    assert set(names_step) == {"sensitivity_step", "V4_fd_error", "B_fd_error"}
    for name in ("sensitivity_step", "V4_fd_error", "B_fd_error"):
        assert_allclose(_float_column(rows_step, name), convergence[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("V4", "V4 must be nonzero"),
        ("reconstruction", "fourfold reconstruction"),
        ("harmonic", "m=4 harmonic"),
        ("physical_shift", "epsilon exactly once"),
        ("delta_c", "Q4 coefficient relation"),
        ("V_fd", "analytic/finite-difference V"),
        ("B_fd", "analytic/finite-difference B"),
        ("step", "finite-difference step convergence"),
        ("lineage", "convergence lineage"),
    ),
)
def test_corrupt_scientific_evidence_fails_before_any_output(
    corruption: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays, metadata = _artifact_copies()
    angular = arrays["angular_sensitivity"]
    convergence = arrays["convergence"]
    if corruption == "V4":
        angular["V4"] = np.array(0.0)
    elif corruption == "reconstruction":
        angular["V_reconstruction"][0] += 1.0e-3
    elif corruption == "harmonic":
        angular["harmonic_amplitude"][angular["harmonic_order"] == 4] *= 2.0
    elif corruption == "physical_shift":
        angular["physical_V4_shift"] *= angular["epsilon"]
    elif corruption == "delta_c":
        angular["delta_c"][0] *= 1.1
    elif corruption == "V_fd":
        angular["V_fd"] += 1.0
    elif corruption == "B_fd":
        angular["B_fd"] += 1.0
    elif corruption == "step":
        convergence["V4_fd_error"][-2:] = 1.0
    elif corruption == "lineage":
        metadata["convergence"]["input_artifacts"]["sensitivity"]["output_sha256"] = "0" * 64
    else:  # pragma: no cover - parameter guard
        raise AssertionError(corruption)

    def corrupted_loader(_data_dir: Path, name: str):
        return arrays[name], metadata[name]

    monkeypatch.setattr(figure03, "load_figure_artifact", corrupted_loader)
    with pytest.raises((AssertionError, ValueError), match=message):
        build(DATA_DIR, tmp_path / "figures", tmp_path / "source")

    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "source").exists()


def test_figure03_has_unambiguous_physical_convention_and_no_plot_time_inference() -> None:
    source = Path(figure03.__file__).read_text(encoding="utf-8")
    compact_source = "".join(source.split())

    assert 'load_figure_artifact(data_dir,"angular_sensitivity")' in compact_source
    assert 'load_figure_artifact(data_dir,"convergence")' in compact_source
    assert 'sensitivity["V_fd"]' in source
    assert 'sensitivity["B_fd"]' in source
    assert r"physical frequency shift $\varepsilon V_4$" in source
    assert r"cubic stiffness perturbation $\Delta_C(\varepsilon)$" in source
    assert r"physical frequency shift $V_4$" not in source
    for forbidden in (
        "np.load",
        "numpy.load",
        "polyfit",
        "curve_fit",
        "lstsq",
        "np.gradient",
        "np.diff",
        "savgol",
        "scipy",
    ):
        assert forbidden not in source


def test_dense_validation_content_uses_flat_dedicated_axes_without_overlays() -> None:
    source = Path(figure03.__file__).read_text(encoding="utf-8")

    assert "subgridspec" not in source
    assert "inset_axes" not in source
    assert "bbox_to_anchor=(0.5, -0.16)" not in source
