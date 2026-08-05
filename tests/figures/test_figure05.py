"""Scientific, semantic, and export contract for main Figure 5."""

from __future__ import annotations

import csv
from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
import pytest

from zgv_morse.figures import common
import zgv_morse.figures.figure05_scaling as figure05
from zgv_morse.config import load_reference_config
from zgv_morse.workflows.scaling import run as run_scaling


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/generated"
EXPECTED_STEM = "figure_05_perturbation_scaling"
SOURCE_FILENAMES = (
    "panel_a_splitting.csv",
    "panel_b_compensated_radial_shift.csv",
    "panel_c_frequency_remainder.csv",
    "panel_d_role_reversal.csv",
)


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return tuple(reader.fieldnames), list(reader)


def _float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(row[name]) for row in rows], dtype=np.float64)


def _artifact_copy() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays, metadata = common.load_figure_artifact(DATA_DIR, "perturbation_scaling")
    copied_arrays = {
        name: np.array(value, copy=True) for name, value in arrays.items()
    }
    copied_metadata = {
        **metadata,
        "tolerances": dict(metadata["tolerances"]),
    }
    return copied_arrays, copied_metadata


@pytest.mark.filterwarnings("error:Glyph .* missing from font")
def test_build_exports_exact_source_evidence_and_is_byte_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaling, _metadata = _artifact_copy()
    calls: list[tuple[Path, str]] = []
    scatter_semantics: list[tuple[str | None, str | None]] = []
    real_loader = figure05.load_figure_artifact
    real_scatter = Axes.scatter

    def tracked_loader(data_dir: Path, name: str):
        calls.append((data_dir, name))
        return real_loader(data_dir, name)

    def tracked_scatter(self: Axes, *args, **kwargs):
        scatter_semantics.append((kwargs.get("marker"), kwargs.get("color")))
        return real_scatter(self, *args, **kwargs)

    monkeypatch.setattr(figure05, "load_figure_artifact", tracked_loader)
    monkeypatch.setattr(Axes, "scatter", tracked_scatter)

    bundles: list[tuple[dict[str, bytes], dict[str, bytes]]] = []
    first_outputs: dict[str, Path] | None = None
    first_source: Path | None = None
    for run in ("first", "second"):
        output_dir = tmp_path / run / "figures"
        source_dir = tmp_path / run / "source"
        outputs = figure05.build(DATA_DIR, output_dir, source_dir)
        if first_outputs is None:
            first_outputs = outputs
            first_source = source_dir
        bundles.append(
            (
                {kind: path.read_bytes() for kind, path in outputs.items()},
                {
                    path.name: path.read_bytes()
                    for path in sorted(source_dir.glob("*.csv"))
                },
            )
        )

    assert calls == [
        (DATA_DIR, "perturbation_scaling"),
        (DATA_DIR, "perturbation_scaling"),
    ]
    assert bundles[0] == bundles[1]
    assert first_outputs is not None and first_source is not None
    assert set(first_outputs) == set(common.PUBLICATION_FORMATS)
    assert {
        path.name for path in first_outputs.values()
    } == {f"{EXPECTED_STEM}.{kind}" for kind in common.PUBLICATION_FORMATS}
    assert tuple(sorted(path.name for path in first_source.glob("*.csv"))) == tuple(
        sorted(SOURCE_FILENAMES)
    )

    svg = first_outputs["svg"].read_text(encoding="utf-8")
    assert "<text" in svg
    assert "first-order prediction" in svg
    assert "compensated radial shift" in svg
    assert "frequency error" in svg
    assert "anisotropy sign" in svg
    assert "minimum" in svg and "saddle" in svg
    assert "stored slope" in svg
    assert "dc:date" not in svg
    assert b"/CreationDate" not in first_outputs["pdf"].read_bytes()

    expected_pixels = (
        int(figure05._SPEC.width_mm / 25.4 * 600),
        int(figure05._SPEC.height_mm / 25.4 * 600),
    )
    for kind in ("png", "tiff"):
        with Image.open(first_outputs[kind]) as image:
            assert image.size == expected_pixels
            assert image.info["dpi"] == pytest.approx((600.0, 600.0), abs=0.1)
            if kind == "tiff":
                assert image.info["compression"] == "tiff_lzw"
                assert image.tag_v2[259] == 5

    semantic_pairs = set(scatter_semantics)
    assert (common.MARKERS["minimum"], common.PALETTE["minimum"]) in semantic_pairs
    assert (common.MARKERS["saddle"], common.PALETTE["saddle"]) in semantic_pairs

    names_a, rows_a = _read_csv(first_source / "panel_a_splitting.csv")
    assert set(names_a) == {
        "delta_omega_full",
        "delta_omega_pred",
        "epsilon_abs",
        "slope_splitting",
    }
    assert_allclose(_float_column(rows_a, "epsilon_abs"), scaling["epsilon"], rtol=0, atol=0)
    assert_allclose(
        _float_column(rows_a, "delta_omega_full"),
        scaling["delta_omega_full"],
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float_column(rows_a, "delta_omega_pred"),
        scaling["delta_omega_pred"],
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float_column(rows_a, "slope_splitting"),
        float(scaling["slope_splitting"]),
        rtol=0,
        atol=0,
    )

    names_b, rows_b = _read_csv(
        first_source / "panel_b_compensated_radial_shift.csv"
    )
    assert set(names_b) == {
        "compensated_q_min",
        "compensated_q_saddle",
        "epsilon_abs",
        "q_min_limit_prediction",
        "q_saddle_limit_prediction",
        "slope_radial",
    }
    assert_allclose(
        _float_column(rows_b, "compensated_q_min"),
        scaling["compensated_q_min"],
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float_column(rows_b, "compensated_q_saddle"),
        scaling["compensated_q_saddle"],
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float_column(rows_b, "q_min_limit_prediction"),
        scaling["q_min_pred"] / scaling["epsilon"],
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float_column(rows_b, "q_saddle_limit_prediction"),
        scaling["q_saddle_pred"] / scaling["epsilon"],
        rtol=0,
        atol=0,
    )

    names_c, rows_c = _read_csv(first_source / "panel_c_frequency_remainder.csv")
    assert set(names_c) == {
        "epsilon_abs",
        "minimum_error_over_epsilon_squared",
        "saddle_error_over_epsilon_squared",
        "slope_remainder",
        "stored_max_error_over_epsilon_squared",
    }
    minimum_error = scaling["omega_min_error"] / scaling["epsilon"] ** 2
    saddle_error = scaling["omega_saddle_error"] / scaling["epsilon"] ** 2
    assert_allclose(
        _float_column(rows_c, "minimum_error_over_epsilon_squared"),
        minimum_error,
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float_column(rows_c, "saddle_error_over_epsilon_squared"),
        saddle_error,
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float_column(rows_c, "stored_max_error_over_epsilon_squared"),
        scaling["compensated_frequency_error"],
        rtol=0,
        atol=0,
    )

    names_d, rows_d = _read_csv(first_source / "panel_d_role_reversal.csv")
    assert set(names_d) == {"epsilon_sign", "kind", "theta", "theta_over_pi"}
    assert len(rows_d) == 16
    negative_min = np.array(
        [float(row["theta"]) for row in rows_d if row["epsilon_sign"] == "-1" and row["kind"] == "minimum"]
    )
    negative_saddle = np.array(
        [float(row["theta"]) for row in rows_d if row["epsilon_sign"] == "-1" and row["kind"] == "saddle"]
    )
    positive_min = np.array(
        [float(row["theta"]) for row in rows_d if row["epsilon_sign"] == "1" and row["kind"] == "minimum"]
    )
    positive_saddle = np.array(
        [float(row["theta"]) for row in rows_d if row["epsilon_sign"] == "1" and row["kind"] == "saddle"]
    )
    assert_allclose(negative_min, scaling["role_reversal_theta_min"], rtol=0, atol=0)
    assert_allclose(negative_saddle, scaling["role_reversal_theta_saddle"], rtol=0, atol=0)
    assert_allclose(positive_min, scaling["role_reversal_theta_saddle"], rtol=0, atol=0)
    assert_allclose(positive_saddle, scaling["role_reversal_theta_min"], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("splitting_slope", "splitting slope"),
        ("remainder_slope", "remainder slope"),
        ("prediction", "stored radial prediction"),
        ("agreement", "radial compensated limit"),
        ("roles", "role reversal"),
        ("zero_coefficient", "first nonzero order"),
        ("compensated_remainder", "stored compensated frequency error"),
        ("profile", "registered smoke or full profile"),
    ),
)
def test_scientific_preflight_fails_before_source_or_rendering(
    corruption: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays, metadata = _artifact_copy()
    if corruption == "splitting_slope":
        arrays["slope_splitting"] = np.array(1.11)
    elif corruption == "remainder_slope":
        arrays["slope_remainder"] = np.array(2.21)
    elif corruption == "prediction":
        arrays["q_min_pred"][0] *= 0.5
    elif corruption == "agreement":
        arrays["q_min_full"][0] = -0.2 * arrays["epsilon"][0]
        arrays["compensated_q_min"][0] = -0.2
    elif corruption == "roles":
        arrays["role_reversal_theta_min"] = np.array(
            arrays["role_reversal_theta_saddle"], copy=True
        )
    elif corruption == "zero_coefficient":
        arrays["q_min_pred"][:] = 0.0
        arrays["q_min_full"][:] = 0.0
        arrays["compensated_q_min"][:] = 0.0
    elif corruption == "compensated_remainder":
        arrays["compensated_frequency_error"][0] *= 2.0
    else:
        metadata["profile"] = "preview"

    monkeypatch.setattr(
        figure05,
        "load_figure_artifact",
        lambda _data_dir, _name: (arrays, metadata),
    )
    with pytest.raises((AssertionError, ValueError), match=message):
        figure05.build(DATA_DIR, tmp_path / "figures", tmp_path / "source")

    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "source").exists()


def test_registered_smoke_scaling_artifact_passes_the_figure_evidence_gate(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data" / "generated"
    run_scaling(load_reference_config(ROOT / "config/reference.yaml"), data_dir, "smoke")
    arrays, metadata = common.load_figure_artifact(data_dir, "perturbation_scaling")

    evidence = figure05._validate_evidence(arrays, metadata)

    assert metadata["profile"] == "smoke"
    assert evidence.epsilon.size == 3


def test_figure05_has_no_plot_time_fit_mask_or_unvalidated_input_path() -> None:
    source = Path(figure05.__file__).read_text(encoding="utf-8")

    assert 'load_figure_artifact(data_dir, "perturbation_scaling")' in source
    assert "angular_sensitivity" not in source
    assert "isotropic_zgv" not in source
    assert "np.load" not in source
    assert "numpy.load" not in source
    assert "polyfit" not in source
    assert "curve_fit" not in source
    assert "lstsq" not in source
    assert "scipy" not in source
    assert "fit_window" not in source
    assert "plot_mask" not in source
    assert 'MARKERS["minimum"]' in source
    assert 'MARKERS["saddle"]' in source
    assert 'PALETTE["minimum"]' in source
    assert 'PALETTE["saddle"]' in source
    assert 'layout="constrained"' in source
    assert figure05._SPEC.width_mm == 183.0
    assert figure05._SPEC.height_mm > 0.0  # size is a design choice


def test_make_figure05_script_is_root_anchored_and_one_purpose() -> None:
    source = (ROOT / "scripts/make_figure_05.py").read_text(encoding="utf-8")

    assert "ROOT = Path(__file__).resolve().parents[1]" in source
    assert "figure_05_perturbation_scaling" not in source
    assert "figure05_scaling import build" in source
    assert 'ROOT / "data/generated"' in source
    assert 'ROOT / "figures/main"' in source
    assert 'ROOT / "data/source_data/figure_05"' in source


def test_registered_epsilon_ticks_suppress_overlapping_minor_labels() -> None:
    scaling, _metadata = _artifact_copy()
    fig, ax = plt.subplots()
    try:
        ax.set_xscale("log")
        figure05._set_epsilon_ticks(ax, scaling["epsilon"])
        fig.canvas.draw()
        major = tuple(
            tick.label1.get_text()
            for tick in ax.xaxis.get_major_ticks()
            if tick.label1.get_visible()
        )
        minor = tuple(
            tick.label1.get_text()
            for tick in ax.xaxis.get_minor_ticks()
            if tick.label1.get_visible() and tick.label1.get_text()
        )
    finally:
        plt.close(fig)

    assert major == ("0.0025", "0.01", "0.04", "0.08")
    assert minor == ()
