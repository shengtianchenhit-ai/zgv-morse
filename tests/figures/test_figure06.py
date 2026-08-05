"""Evidence, provenance, and export contract for main Figure 6."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image
import pytest

from zgv_morse.figures import common
import zgv_morse.figures.figure06_crossover as figure06


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/generated"
SOURCE_FILENAMES = (
    "panel_a_collapse.csv",
    "panel_b_absolute_error.csv",
    "panel_c_envelopes.csv",
    "panel_c_late_fit.csv",
    "panel_d_crossover.csv",
    "panel_e_fixed_morse.csv",
    "panel_f_frequency_features.csv",
)


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return tuple(reader.fieldnames), list(reader)


def _float(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def _copies():
    bundles = {
        name: common.load_figure_artifact(DATA_DIR, name)
        for name in ("green_crossover", "angular_sensitivity", "convergence")
    }
    return {
        name: (
            {key: np.array(value, copy=True) for key, value in arrays.items()},
            copy.deepcopy(metadata),
        )
        for name, (arrays, metadata) in bundles.items()
    }


def test_frequency_feature_axis_label_has_print_safe_right_margin() -> None:
    path = ROOT / "figures/main/figure_06_decay_crossover.png"
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    ink = np.any(rgb < 245, axis=2)
    _rows, columns = np.nonzero(ink)

    assert columns.size > 0
    assert rgb.shape[1] - 1 - int(columns.max()) >= 8


@pytest.mark.filterwarnings("error:Glyph .* missing from font")
def test_build_exports_exact_crossover_evidence_and_is_byte_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    real_loader = common.load_figure_artifact

    def tracked_loader(data_dir: Path, name: str):
        calls.append(name)
        return real_loader(data_dir, name)

    monkeypatch.setattr(figure06, "load_figure_artifact", tracked_loader)
    first = figure06.build(DATA_DIR, tmp_path / "first/figures", tmp_path / "first/source")
    second = figure06.build(
        DATA_DIR, tmp_path / "second/figures", tmp_path / "second/source"
    )

    assert calls == [
        "green_crossover",
        "angular_sensitivity",
        "convergence",
        "green_crossover",
        "angular_sensitivity",
        "convergence",
    ]
    assert set(first) == set(common.PUBLICATION_FORMATS)
    assert set(second) == set(common.PUBLICATION_FORMATS)
    for kind in common.PUBLICATION_FORMATS:
        assert first[kind].name == f"figure_06_decay_crossover.{kind}"
        assert first[kind].read_bytes() == second[kind].read_bytes()

    first_source = tmp_path / "first/source"
    second_source = tmp_path / "second/source"
    assert tuple(sorted(path.name for path in first_source.glob("*.csv"))) == tuple(
        sorted(SOURCE_FILENAMES)
    )
    for filename in SOURCE_FILENAMES:
        assert (first_source / filename).read_bytes() == (
            second_source / filename
        ).read_bytes()

    svg = first["svg"].read_text(encoding="utf-8")
    svg_lower = svg.lower()
    for text in (
        "without fitted alignment",
        "absolute complex error",
        "joint-limit slice",
        "response envelope",
        "crossover time",
        "exact-Morse stationary sum",
        "normalized spectrum",
        "angular-frequency detuning",
    ):
        assert text.lower() in svg_lower
    assert "parameter-free" not in svg_lower
    assert "observable decay:" not in svg_lower
    assert "spectral lines" not in svg_lower
    assert "dc:date" not in svg
    assert b"/CreationDate" not in first["pdf"].read_bytes()

    expected_pixels = (int(figure06._SPEC.width_mm / 25.4 * 600), int(figure06._SPEC.height_mm / 25.4 * 600))
    for kind in ("png", "tiff"):
        with Image.open(first[kind]) as image:
            assert image.size == expected_pixels
            assert image.info["dpi"] == pytest.approx((600.0, 600.0), abs=0.1)
            if kind == "png":
                rgb = np.asarray(image.convert("RGB"))
                ink = np.any(rgb < 245, axis=2)
                _rows, columns = np.nonzero(ink)
                assert columns.size > 0
                assert rgb.shape[1] - 1 - int(columns.max()) >= 8
            if kind == "tiff":
                assert image.tag_v2[259] == 5

    green, green_meta = common.load_figure_artifact(DATA_DIR, "green_crossover")
    angular, _ = common.load_figure_artifact(DATA_DIR, "angular_sensitivity")
    uniform_epsilon = np.asarray(green_meta["uniform_rows"], dtype=np.float64)
    uniform_indices = np.array(
        [int(np.flatnonzero(green["epsilon"] == value)[0]) for value in uniform_epsilon]
    )
    row_count = uniform_indices.size * green["time"].size

    names_a, rows_a = _read_csv(first_source / "panel_a_collapse.csv")
    assert set(names_a) == {
        "J0",
        "epsilon",
        "scaled_imaginary",
        "scaled_magnitude",
        "scaled_real",
        "tau",
        "time",
    }
    assert len(rows_a) == row_count
    assert_allclose(
        _float(rows_a, "epsilon"),
        np.repeat(uniform_epsilon, green["time"].size),
        rtol=0,
        atol=0,
    )
    selected_scaled = green["scaled_response"][uniform_indices]
    for name, expected in (
        ("time", np.tile(green["time"], uniform_indices.size)),
        ("tau", green["tau"][uniform_indices].reshape(-1)),
        ("J0", green["J0"][uniform_indices].reshape(-1)),
        ("scaled_real", selected_scaled.real.reshape(-1)),
        ("scaled_imaginary", selected_scaled.imag.reshape(-1)),
        ("scaled_magnitude", np.abs(selected_scaled).reshape(-1)),
    ):
        assert_allclose(_float(rows_a, name), expected, rtol=0, atol=0)

    names_b, rows_b = _read_csv(first_source / "panel_b_absolute_error.csv")
    assert {
        "absolute_complex_error",
        "registered_joint_limit_window",
        "registered_minimum_time",
        "registered_maximum_abs_tau",
    }.issubset(names_b)
    absolute_error = np.abs(selected_scaled - green["J0"][uniform_indices])
    assert_allclose(
        _float(rows_b, "absolute_complex_error"),
        absolute_error.reshape(-1),
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float(rows_b, "absolute_complex_error"),
        np.hypot(
            _float(rows_b, "scaled_real") - _float(rows_b, "J0"),
            _float(rows_b, "scaled_imaginary"),
        ),
        rtol=2e-15,
        atol=2e-15,
    )
    selected_time = np.tile(green["time"], uniform_indices.size)
    selected_tau = green["tau"][uniform_indices].reshape(-1)
    expected_window = (
        (selected_time >= figure06._REGISTERED_UNIFORM_TIME_START)
        & (np.abs(selected_tau) <= figure06._REGISTERED_UNIFORM_TAU_MAXIMUM)
    )
    assert_array_equal(
        _float(rows_b, "registered_joint_limit_window").astype(np.int64),
        expected_window.astype(np.int64),
    )
    assert_allclose(
        _float(rows_b, "registered_minimum_time"),
        figure06._REGISTERED_UNIFORM_TIME_START,
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float(rows_b, "registered_maximum_abs_tau"),
        figure06._REGISTERED_UNIFORM_TAU_MAXIMUM,
        rtol=0,
        atol=0,
    )

    names_c, rows_c = _read_csv(first_source / "panel_c_envelopes.csv")
    assert set(names_c) == {
        "envelope",
        "epsilon",
        "fit_window_early",
        "fit_window_late",
        "local_slope",
        "rms_envelope",
        "time",
    }
    fixed_epsilon = float(green_meta["fixed_morse_epsilon"])
    fixed_index = int(np.flatnonzero(green["epsilon"] == fixed_epsilon)[0])
    early_index = uniform_indices[0]
    source_indices = np.array([early_index, fixed_index])
    assert len(rows_c) == 2 * green["time"].size
    assert_array_equal(
        _float(rows_c, "fit_window_early").astype(np.int64),
        green["fit_window_early"][source_indices].reshape(-1).astype(np.int64),
    )
    assert_array_equal(
        _float(rows_c, "fit_window_late").astype(np.int64),
        green["fit_window_late"][source_indices].reshape(-1).astype(np.int64),
    )
    names_c_fit, rows_c_fit = _read_csv(first_source / "panel_c_late_fit.csv")
    assert set(names_c_fit) == {
        "direct_response_rms",
        "epsilon",
        "geometric_time_center",
        "stored_late_slope",
    }
    late_evidence = green_meta["late_fit_evidence"]
    assert_allclose(
        _float(rows_c_fit, "geometric_time_center"),
        late_evidence["geometric_time_centers"],
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float(rows_c_fit, "direct_response_rms"),
        late_evidence["direct_response_rms"],
        rtol=0,
        atol=0,
    )

    names_d, rows_d = _read_csv(first_source / "panel_d_crossover.csv")
    assert set(names_d) == {
        "crossover_time",
        "epsilon",
        "epsilon_log_slope",
        "inverse_abs_epsilon_V4",
        "inverse_rate_log_slope",
        "phase_error",
    }
    v4 = float(angular["V4"])
    assert_allclose(
        _float(rows_d, "inverse_abs_epsilon_V4"),
        1.0 / np.abs(green["epsilon"] * v4),
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float(rows_d, "crossover_time"), green["crossover_time"], rtol=0, atol=0
    )
    epsilon_slope = float(green_meta["tolerances"]["measured_crossover_log_slope"])
    assert_allclose(_float(rows_d, "epsilon_log_slope"), epsilon_slope, rtol=0, atol=0)
    assert_allclose(
        _float(rows_d, "inverse_rate_log_slope"), -epsilon_slope, rtol=0, atol=0
    )

    names_e, rows_e = _read_csv(first_source / "panel_e_fixed_morse.csv")
    assert {
        "full_real",
        "full_imaginary",
        "full_magnitude",
        "morse_real",
        "morse_imaginary",
        "morse_magnitude",
        "comparison_window",
        "cancellation_region",
        "certified_noncancellation_window",
        "morse_phasor_coherence",
        "coherence_threshold",
        "comparison_start_time",
        "comparison_stop_time",
        "noncancellation_morse_rms_normalization",
    }.issubset(names_e)
    assert_allclose(
        _float(rows_e, "full_magnitude"),
        np.abs(green["G_full"][fixed_index]),
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float(rows_e, "morse_magnitude"),
        np.abs(green["G_morse"][fixed_index]),
        rtol=0,
        atol=0,
    )
    cancellation = green_meta["fixed_morse_cancellation_evidence"]
    coherence = np.asarray(cancellation["coherence"], dtype=np.float64)
    threshold = float(cancellation["coherence_threshold"])
    comparison_window = (
        (green["time"] >= float(cancellation["comparison_start_time"]))
        & (green["time"] <= float(cancellation["comparison_stop_time"]))
    )
    expected_cancellation = (comparison_window & (coherence < threshold)).astype(
        np.int64
    )
    assert_array_equal(
        _float(rows_e, "cancellation_region").astype(np.int64),
        expected_cancellation,
    )
    assert_allclose(
        _float(rows_e, "morse_phasor_coherence"), coherence, rtol=0, atol=0
    )
    for column, value in (
        ("coherence_threshold", threshold),
        ("comparison_start_time", float(cancellation["comparison_start_time"])),
        ("comparison_stop_time", float(cancellation["comparison_stop_time"])),
        (
            "noncancellation_morse_rms_normalization",
            float(cancellation["noncancellation_morse_rms_normalization"]),
        ),
    ):
        assert_allclose(_float(rows_e, column), value, rtol=0, atol=0)

    names_f, rows_f = _read_csv(first_source / "panel_f_frequency_features.csv")
    assert {
        "spectrum_omega",
        "spectrum",
        "omega_min",
        "omega_saddle",
        "predicted_separation_2_abs_epsilon_V4",
        "signed_modulation_rate_abs_epsilon_V4",
    }.issubset(names_f)
    assert_allclose(
        _float(rows_f, "spectrum_omega"), green["spectrum_omega"], rtol=0, atol=0
    )
    assert_allclose(
        _float(rows_f, "spectrum"), green["spectrum"][fixed_index], rtol=0, atol=0
    )
    rate = abs(fixed_epsilon * v4)
    assert_allclose(
        _float(rows_f, "predicted_separation_2_abs_epsilon_V4"),
        2.0 * rate,
        rtol=0,
        atol=0,
    )
    assert_allclose(
        _float(rows_f, "signed_modulation_rate_abs_epsilon_V4"),
        rate,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("early_slope", "early slope"),
        ("late_slope", "late slope"),
        ("collapse", "Bessel collapse"),
        ("crossover", "crossover"),
        ("fixed_morse", "exact-Morse"),
        ("morse_arrays", "exact-Morse metrics"),
        ("cancellation", "cancellation fraction"),
        ("coherence", "cancellation coherence"),
        ("late_fit", "late-fit centers"),
        ("phase", "phase error"),
        ("v4", "V4"),
        ("context", "scientific context"),
    ),
)
def test_scientific_gates_fail_before_any_output(
    corruption: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = _copies()
    green, green_meta = bundles["green_crossover"]
    angular, _angular_meta = bundles["angular_sensitivity"]
    convergence, _convergence_meta = bundles["convergence"]
    tolerances = green_meta["tolerances"]
    assert isinstance(tolerances, dict)
    if corruption == "early_slope":
        green["slope_early"][0] = -0.4
    elif corruption == "late_slope":
        green["slope_late"][0] = -0.9
    elif corruption == "collapse":
        tolerances["maximum_bessel_collapse_error"] = 0.2
    elif corruption == "crossover":
        tolerances["measured_crossover_log_slope"] = -0.7
    elif corruption == "fixed_morse":
        tolerances["fixed_epsilon_morse_relative_rms_error"] = 0.2
    elif corruption == "morse_arrays":
        green["G_morse"][-1] *= 1.1
    elif corruption == "cancellation":
        tolerances["fixed_epsilon_cancellation_fraction"] = 0.5
    elif corruption == "coherence":
        green_meta["fixed_morse_cancellation_evidence"]["coherence_threshold"] = 0.9
    elif corruption == "late_fit":
        green_meta["late_fit_evidence"]["geometric_time_centers"] = [2.0, 1.0, 3.0]
        green_meta["late_fit_evidence"]["direct_response_rms"] = [1.0, 1.0, 1.0]
    elif corruption == "phase":
        green["phase_error"][0] = 0.2
    elif corruption == "v4":
        angular["V4"] = np.asarray(0.0)
    elif corruption == "context":
        bundles["angular_sensitivity"][1]["config_hash"] = "0" * 64
    else:  # pragma: no cover
        raise AssertionError(corruption)

    monkeypatch.setattr(
        figure06,
        "load_figure_artifact",
        lambda _data_dir, name: bundles[name],
    )
    with pytest.raises((AssertionError, ValueError), match=message):
        figure06.build(DATA_DIR, tmp_path / "figures", tmp_path / "source")
    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "source").exists()


def test_figure06_has_no_plot_time_fit_transform_or_raw_archive_load() -> None:
    source = Path(figure06.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "np.load",
        "numpy.load",
        "polyfit",
        "curve_fit",
        "lstsq",
        "brentq",
        "np.gradient",
        "np.fft",
        "scipy",
    ):
        assert forbidden not in source
    assert "_UNIFORM_TAU_MAXIMUM as _REGISTERED_UNIFORM_TAU_MAXIMUM" in source
