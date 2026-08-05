"""Contracts for the publication-figure framework."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

import zgv_morse.figures.common as common
from zgv_morse.figures.common import (
    FigureSpec,
    MARKERS,
    PALETTE,
    PUBLICATION_FORMATS,
    apply_publication_style,
    load_figure_artifact,
    load_figure_inputs,
    save_publication_figure,
    write_source_csv,
)


def _old_figure_outputs(stem: Path) -> dict[str, bytes]:
    expected = {
        kind: f"old-{kind}".encode("ascii") for kind in PUBLICATION_FORMATS
    }
    stem.parent.mkdir(parents=True, exist_ok=True)
    for kind, content in expected.items():
        stem.with_suffix(f".{kind}").write_bytes(content)
    return expected


def _assert_old_figure_outputs(stem: Path, expected: dict[str, bytes]) -> None:
    for kind, content in expected.items():
        assert stem.with_suffix(f".{kind}").read_bytes() == content
    assert not [path for path in stem.parent.iterdir() if path.name.startswith(".")]


def test_publication_export_preserves_svg_text_and_all_formats(tmp_path: Path) -> None:
    apply_publication_style()
    fig, ax = plt.subplots()
    ax.set_xlabel("wave number")
    ax.plot([0, 1], [0, 1])
    outputs = save_publication_figure(
        fig,
        tmp_path / "figure_test",
        FigureSpec("T", "export test", "quantitative grid", 89.0, 60.0),
    )

    assert set(outputs) == {"svg", "pdf", "png", "tiff"}
    assert all(path.exists() for path in outputs.values())
    svg = outputs["svg"].read_text(encoding="utf-8")
    pdf = outputs["pdf"].read_bytes()
    assert "<text" in svg
    assert "dc:date" not in svg
    assert all(line == line.rstrip(" \t") for line in svg.splitlines())
    assert b"/CreationDate" not in pdf
    assert b"/ModDate" not in pdf
    expected_pixels = (round(89.0 / 25.4 * 600), round(60.0 / 25.4 * 600))
    with Image.open(outputs["png"]) as image:
        assert image.info["dpi"] == pytest.approx((600.0, 600.0), abs=0.1)
        assert image.size == expected_pixels
    with Image.open(outputs["tiff"]) as image:
        assert image.info["dpi"] == pytest.approx((600.0, 600.0), abs=0.1)
        assert image.size == expected_pixels
        assert image.info["compression"] == "tiff_lzw"
        assert image.tag_v2[259] == 5
    assert fig.number not in plt.get_fignums()


def test_publication_style_fixes_vector_identity_and_morse_redundancy() -> None:
    mpl.rcParams["font.family"] = ["fantasy"]
    mpl.rcParams["axes.linewidth"] = 9.0
    apply_publication_style()

    assert mpl.get_backend().lower() == "agg"
    assert mpl.rcParams["font.family"] == ["serif"]
    assert mpl.rcParams["font.serif"] == ["STIXGeneral"]
    assert mpl.rcParams["mathtext.fontset"] == "stix"
    assert mpl.rcParams["text.antialiased"] is False
    assert mpl.rcParams["text.hinting"] == "none"
    assert mpl.rcParams["lines.antialiased"] is False
    assert mpl.rcParams["patch.antialiased"] is False
    assert mpl.rcParams["axes.linewidth"] == pytest.approx(0.8)
    assert mpl.rcParams["svg.fonttype"] == "none"
    assert mpl.rcParams["svg.hashsalt"] == "zgv-morse-publication-v1"
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert PUBLICATION_FORMATS == ("svg", "pdf", "png", "tiff")
    assert PALETTE["minimum"] == "#3775BA"
    assert PALETTE["saddle"] == "#E28E2C"
    assert PALETTE["minimum"] != PALETTE["saddle"]
    assert MARKERS == {"minimum": "o", "saddle": "D"}


@pytest.mark.parametrize("field", ("number", "conclusion"))
@pytest.mark.parametrize("value", ("", "   "))
def test_figure_spec_rejects_blank_identity_fields(field: str, value: str) -> None:
    values = {
        "number": "1",
        "conclusion": "a defensible conclusion",
        "archetype": "quantitative grid",
        "width_mm": 89.0,
        "height_mm": 60.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        FigureSpec(**values)


@pytest.mark.parametrize("field", ("number", "conclusion", "archetype"))
def test_figure_spec_rejects_non_string_text(field: str) -> None:
    values = {
        "number": "1",
        "conclusion": "a defensible conclusion",
        "archetype": "quantitative grid",
        "width_mm": 89.0,
        "height_mm": 60.0,
    }
    values[field] = 1

    with pytest.raises(TypeError, match=field):
        FigureSpec(**values)


def test_figure_spec_accepts_only_registered_archetypes() -> None:
    with pytest.raises(ValueError, match="archetype"):
        FigureSpec("1", "a conclusion", "dashboard", 89.0, 60.0)

    for archetype in (
        "quantitative grid",
        "schematic-led composite",
        "image plate + quant",
        "asymmetric mixed-modality figure",
    ):
        spec = FigureSpec("1", "a conclusion", archetype, 89, np.float64(60.0))
        assert spec.archetype == archetype
        assert type(spec.width_mm) is float
        assert type(spec.height_mm) is float


@pytest.mark.parametrize("field", ("width_mm", "height_mm"))
@pytest.mark.parametrize("value", (0.0, -1.0, np.nan, np.inf))
def test_figure_spec_rejects_nonpositive_or_nonfinite_dimensions(
    field: str,
    value: float,
) -> None:
    values = {
        "number": "1",
        "conclusion": "a defensible conclusion",
        "archetype": "quantitative grid",
        "width_mm": 89.0,
        "height_mm": 60.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        FigureSpec(**values)


@pytest.mark.parametrize("value", (True, "89", 1 + 2j))
def test_figure_spec_rejects_nonreal_dimensions(value: object) -> None:
    with pytest.raises(TypeError, match="width_mm"):
        FigureSpec("1", "a conclusion", "quantitative grid", value, 60.0)


def test_publication_export_is_byte_deterministic(tmp_path: Path) -> None:
    apply_publication_style()
    spec = FigureSpec("T", "determinism test", "quantitative grid", 40.0, 28.0)

    def render(stem: str) -> dict[str, Path]:
        fig, ax = plt.subplots()
        ax.plot([0.0, 0.5, 1.0], [0.0, 1.0, 0.0])
        ax.set_ylabel("response")
        return save_publication_figure(fig, tmp_path / stem, spec)

    first = render("first")
    second = render("second")

    for kind in PUBLICATION_FORMATS:
        assert first[kind].read_bytes() == second[kind].read_bytes()


def test_publication_export_is_deterministic_across_processes(tmp_path: Path) -> None:
    script = """
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from zgv_morse.figures.common import FigureSpec, apply_publication_style, save_publication_figure
apply_publication_style()
fig, ax = plt.subplots()
ax.plot([0.0, 0.5, 1.0], [0.0, 1.0, 0.0])
ax.set_xlabel('wave number')
ax.set_ylabel('response')
save_publication_figure(
    fig,
    Path(sys.argv[1]),
    FigureSpec('T', 'cross-process determinism', 'quantitative grid', 40.0, 28.0),
)
"""
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    stems = (tmp_path / "process_one", tmp_path / "process_two")
    for stem in stems:
        subprocess.run(
            [sys.executable, "-c", script, str(stem)],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )

    for kind in PUBLICATION_FORMATS:
        assert stems[0].with_suffix(f".{kind}").read_bytes() == stems[1].with_suffix(
            f".{kind}"
        ).read_bytes()


def test_publication_export_rolls_back_when_rendering_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stem = tmp_path / "figure"
    old = _old_figure_outputs(stem)
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    real_savefig = fig.savefig
    calls = 0

    def fail_on_third_render(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected rendering failure")
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(fig, "savefig", fail_on_third_render)
    with pytest.raises(OSError, match="injected rendering failure"):
        save_publication_figure(
            fig,
            stem,
            FigureSpec("T", "transaction test", "quantitative grid", 40.0, 28.0),
        )

    _assert_old_figure_outputs(stem, old)
    assert fig.number not in plt.get_fignums()


def test_publication_export_rejects_empty_temporary_renderings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stem = tmp_path / "figure"
    old = _old_figure_outputs(stem)
    fig = plt.figure()

    def render_empty(path: Path, **_kwargs) -> None:
        Path(path).write_bytes(b"")

    monkeypatch.setattr(fig, "savefig", render_empty)
    with pytest.raises(ValueError, match="empty"):
        save_publication_figure(
            fig,
            stem,
            FigureSpec("T", "transaction test", "quantitative grid", 40.0, 28.0),
        )

    _assert_old_figure_outputs(stem, old)
    assert fig.number not in plt.get_fignums()


def test_publication_export_rolls_back_when_publication_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stem = tmp_path / "figure"
    old = _old_figure_outputs(stem)
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [1.0, 0.0])
    real_replace = common.os.replace
    failed = False

    def fail_once_on_pdf(source: object, destination: object) -> None:
        nonlocal failed
        if Path(destination) == stem.with_suffix(".pdf") and not failed:
            failed = True
            raise OSError("injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(common.os, "replace", fail_once_on_pdf)
    with pytest.raises(OSError, match="injected publication failure"):
        save_publication_figure(
            fig,
            stem,
            FigureSpec("T", "transaction test", "quantitative grid", 40.0, 28.0),
        )

    _assert_old_figure_outputs(stem, old)
    assert fig.number not in plt.get_fignums()


def test_publication_export_retains_backup_when_rollback_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stem = tmp_path / "figure"
    old = _old_figure_outputs(stem)
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [1.0, 0.0])
    real_replace = common.os.replace
    publication_failed = False
    rollback_failed = False

    def fail_publication_and_rollback(source: object, destination: object) -> None:
        nonlocal publication_failed, rollback_failed
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            destination_path == stem.with_suffix(".pdf")
            and source_path.suffix == ".tmp"
            and not publication_failed
        ):
            publication_failed = True
            raise OSError("injected publication failure")
        if (
            destination_path == stem.with_suffix(".svg")
            and source_path.suffix == ".backup"
            and not rollback_failed
        ):
            rollback_failed = True
            raise OSError("injected rollback failure")
        real_replace(source, destination)

    monkeypatch.setattr(common.os, "replace", fail_publication_and_rollback)
    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        save_publication_figure(
            fig,
            stem,
            FigureSpec("T", "transaction test", "quantitative grid", 40.0, 28.0),
        )

    retained = list(tmp_path.glob(".figure.svg.*.backup"))
    assert len(retained) == 1
    assert retained[0].read_bytes() == old["svg"]
    for kind in ("pdf", "png", "tiff"):
        assert stem.with_suffix(f".{kind}").read_bytes() == old[kind]
    assert not list(tmp_path.glob("*.tmp"))
    assert fig.number not in plt.get_fignums()


@pytest.mark.parametrize(
    ("fig", "stem", "spec", "error", "message"),
    (
        (object(), Path("figure"), FigureSpec("1", "c", "quantitative grid", 1, 1), TypeError, "fig"),
        (None, "figure", FigureSpec("1", "c", "quantitative grid", 1, 1), TypeError, "stem"),
        (None, Path("figure"), object(), TypeError, "spec"),
    ),
)
def test_publication_export_rejects_invalid_parameter_types(
    fig: object,
    stem: object,
    spec: object,
    error: type[Exception],
    message: str,
) -> None:
    if fig is None:
        fig = plt.figure()
    try:
        with pytest.raises(error, match=message):
            save_publication_figure(fig, stem, spec)
    finally:
        if isinstance(fig, mpl.figure.Figure):
            plt.close(fig)


def test_publication_export_requires_an_extensionless_file_stem(tmp_path: Path) -> None:
    spec = FigureSpec("1", "a conclusion", "quantitative grid", 40.0, 28.0)
    invalid_stems = (tmp_path / "figure.svg", tmp_path / ".hidden", tmp_path)

    for stem in invalid_stems:
        fig = plt.figure()
        with pytest.raises(ValueError, match="stem"):
            save_publication_figure(fig, stem, spec)
        plt.close(fig)


def test_publication_export_refuses_existing_nonfiles_and_symlinks(tmp_path: Path) -> None:
    spec = FigureSpec("1", "a conclusion", "quantitative grid", 40.0, 28.0)
    stem = tmp_path / "figure"
    stem.mkdir()
    fig = plt.figure()
    with pytest.raises(ValueError, match="stem"):
        save_publication_figure(fig, stem, spec)
    plt.close(fig)

    stem.rmdir()
    target = tmp_path / "target.svg"
    target.write_text("do not overwrite through a symlink", encoding="utf-8")
    stem.with_suffix(".svg").symlink_to(target)
    fig = plt.figure()
    with pytest.raises(ValueError, match="symlink"):
        save_publication_figure(fig, stem, spec)
    plt.close(fig)


def test_publication_export_refuses_a_symlinked_parent_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    fig = plt.figure()

    with pytest.raises(ValueError, match="symlink"):
        save_publication_figure(
            fig,
            linked_parent / "nested" / "figure",
            FigureSpec("T", "symlink test", "quantitative grid", 40.0, 28.0),
        )

    plt.close(fig)
    assert not (real_parent / "nested").exists()


def test_figure_input_loader_uses_the_strict_artifact_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path]] = []
    expected = {"kappa": np.array([1.0, 2.0])}

    def validate(npz_path: Path, sidecar_path: Path):
        calls.append((npz_path, sidecar_path))
        return expected, {"artifact": "isotropic_zgv"}

    monkeypatch.setattr(common, "validate_artifact", validate)
    arrays = load_figure_inputs(tmp_path, "isotropic_zgv")

    npz_path = tmp_path / "isotropic_zgv.npz"
    assert calls == [(npz_path, npz_path.with_suffix(".json"))]
    assert arrays is expected


def test_figure_artifact_loader_preserves_validated_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_arrays = {"kappa": np.array([1.0, 2.0])}
    expected_metadata = {
        "artifact": "critical_points",
        "tolerances": {"minimum_hessian_to_uncertainty": 20.0},
    }

    def validate(npz_path: Path, sidecar_path: Path):
        assert npz_path == tmp_path / "critical_points.npz"
        assert sidecar_path == tmp_path / "critical_points.json"
        return expected_arrays, expected_metadata

    monkeypatch.setattr(common, "validate_artifact", validate)

    arrays, metadata = load_figure_artifact(tmp_path, "critical_points")

    assert arrays is expected_arrays
    assert metadata is expected_metadata


@pytest.mark.parametrize(
    ("data_dir", "name", "error", "message"),
    (
        ("data", "isotropic_zgv", TypeError, "data_dir"),
        (Path("data"), 1, TypeError, "name"),
        (Path("data"), "../isotropic_zgv", ValueError, "artifact name"),
        (Path("data"), "unknown", ValueError, "artifact name"),
    ),
)
def test_figure_input_loader_rejects_invalid_parameters(
    data_dir: object,
    name: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        load_figure_inputs(data_dir, name)


def test_figure_input_loader_rejects_a_non_directory(tmp_path: Path) -> None:
    not_a_directory = tmp_path / "artifact-store"
    not_a_directory.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="data_dir"):
        load_figure_inputs(not_a_directory, "isotropic_zgv")


def test_source_csv_is_canonical_precise_and_unicode_safe(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "source.csv"
    precise = np.nextafter(np.float64(1.0), np.float64(2.0))
    columns = {
        "value": np.array([precise, -0.0]),
        "label": np.array(["最小值", "鞍点"]),
        "index": np.array([2, 1], dtype=np.int64),
    }

    result = write_source_csv(path, columns)

    assert result == path
    assert path.read_bytes().endswith(b"\n")
    assert "1.0000000000000002" in path.read_text(encoding="utf-8")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["index", "label", "value"],
        ["2", "最小值", "1.0000000000000002"],
        ["1", "鞍点", "-0"],
    ]


def test_source_csv_is_independent_of_mapping_insertion_order(tmp_path: Path) -> None:
    first = write_source_csv(
        tmp_path / "first.csv",
        {"z": np.array([1.25]), "a": np.array(["α"])},
    )
    second = write_source_csv(
        tmp_path / "second.csv",
        {"a": np.array(["α"]), "z": np.array([1.25])},
    )

    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    ("columns", "message"),
    (
        ({}, "must not be empty"),
        ({"x": np.array([])}, "must not be empty"),
        ({"x": np.array([1.0]), "y": np.array([1.0, 2.0])}, "equal"),
        ({"x": np.array([np.nan])}, "finite"),
        ({"x": np.array([np.inf])}, "finite"),
        ({"x": np.array([1 + 2j])}, "dtype"),
        ({"x": np.array([b"bytes"])}, "dtype"),
        ({"x": np.array([object()], dtype=object)}, "dtype"),
        ({"x": np.array([True])}, "dtype"),
        ({"": np.array([1.0])}, "column name"),
        ({"bad\nname": np.array([1.0])}, "column name"),
    ),
)
def test_source_csv_rejects_invalid_columns(
    tmp_path: Path,
    columns: dict[str, np.ndarray],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        write_source_csv(tmp_path / "source.csv", columns)


@pytest.mark.parametrize(
    "value",
    (
        "line\nbreak",
        "carriage\rreturn",
        "null\0value",
        "zero\u200bwidth",
        "line\u2028separator",
        "paragraph\u2029separator",
        "surrogate\ud800value",
        "=SUM(A1:A2)",
        "+cmd",
        "-cmd",
        "@cmd",
    ),
)
def test_source_csv_rejects_unsafe_unicode_values(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="Unicode"):
        write_source_csv(tmp_path / "source.csv", {"label": np.array([value])})


def test_source_csv_rejects_non_dict_columns(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="columns"):
        write_source_csv(tmp_path / "source.csv", [("x", np.array([1.0]))])


@pytest.mark.parametrize("path", ("source.csv", Path("source.txt"), Path(".csv")))
def test_source_csv_requires_a_visible_csv_path(tmp_path: Path, path: object) -> None:
    target = tmp_path / path if isinstance(path, Path) else path
    with pytest.raises((TypeError, ValueError), match="path"):
        write_source_csv(target, {"x": np.array([1.0])})


def test_source_csv_refuses_directories_and_symlinks(tmp_path: Path) -> None:
    directory = tmp_path / "source.csv"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        write_source_csv(directory, {"x": np.array([1.0])})

    directory.rmdir()
    target = tmp_path / "target.csv"
    target.write_text("do not overwrite", encoding="utf-8")
    directory.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        write_source_csv(directory, {"x": np.array([1.0])})


def test_source_csv_refuses_a_symlinked_parent_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        write_source_csv(
            linked_parent / "nested" / "source.csv",
            {"x": np.array([1.0])},
        )

    assert not (real_parent / "nested").exists()


def test_source_csv_preserves_old_file_when_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.csv"
    path.write_bytes(b"old-source-data\n")
    real_fsync = common.os.fsync
    failed = False

    def fail_once(descriptor: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected source-data fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(common.os, "fsync", fail_once)
    with pytest.raises(OSError, match="injected source-data fsync failure"):
        write_source_csv(path, {"x": np.array([2.0])})

    assert path.read_bytes() == b"old-source-data\n"
    assert list(tmp_path.iterdir()) == [path]


def test_source_csv_preserves_old_file_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.csv"
    path.write_bytes(b"old-source-data\n")
    real_replace = common.os.replace
    failed = False

    def fail_once(source: object, destination: object) -> None:
        nonlocal failed
        if Path(destination) == path and not failed:
            failed = True
            raise OSError("injected source-data publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(common.os, "replace", fail_once)
    with pytest.raises(OSError, match="injected source-data publication failure"):
        write_source_csv(path, {"x": np.array([2.0])})

    assert path.read_bytes() == b"old-source-data\n"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("number", "1\n"),
        ("number", "1\x00"),
        ("conclusion", "unsafe\rconclusion"),
        ("conclusion", "unsafe\x7fconclusion"),
        ("conclusion", "unsafe\u200bconclusion"),
        ("conclusion", "unsafe\u2028conclusion"),
        ("conclusion", "unsafe\u2029conclusion"),
        ("conclusion", "unsafe\ud800conclusion"),
        ("archetype", "quantitative\tgrid"),
    ),
)
def test_figure_spec_rejects_control_characters(field: str, value: str) -> None:
    values = {
        "number": "1",
        "conclusion": "a defensible conclusion",
        "archetype": "quantitative grid",
        "width_mm": 89.0,
        "height_mm": 60.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        FigureSpec(**values)


def test_all_six_figures_have_complete_five_point_contracts() -> None:
    contract_path = Path(__file__).resolve().parents[2] / "docs/figures/figure_contracts.md"
    text = contract_path.read_text(encoding="utf-8")
    sections = text.split("## Figure ")[1:]
    conclusions = (
        "anisotropy lifts the tangential Hessian kernel, changing a one-dimensional stationary set into isolated points.",
        "the chosen ZGV ring has a nondegenerate positive radial curvature.",
        "cubic anisotropy produces the predicted fourfold unfolding potential.",
        "the full three-dimensional elastic calculation resolves a stable eight-root set with the Morse classes predicted by the sufficiently-small theorem.",
        "the computed unfolding is genuinely perturbative and quantitatively predicted.",
        "a joint-limit Bessel slice and a fixed-anisotropy Morse slice are connected by the separately proved growing-`|tau|` overlap.",
    )
    archetypes = (
        "schematic-led composite",
        "quantitative grid",
        "quantitative grid",
        "quantitative grid",
        "quantitative grid",
        "asymmetric mixed-modality figure",
    )

    assert len(sections) == 6
    for number, (section, conclusion, archetype) in enumerate(
        zip(sections, conclusions, archetypes, strict=True),
        start=1,
    ):
        assert section.startswith(f"{number}")
        for field in (
            "Core conclusion",
            "Archetype",
            "Target/output",
            "Final size",
            "Panel map",
            "Hero evidence",
            "Validation evidence",
            "Source data",
            "Statistics",
            "Image-integrity",
            "Reviewer risk",
        ):
            assert f"**{field}:**" in section
        assert f"**Core conclusion:** {conclusion}" in section
        assert f"**Archetype:** `{archetype}`" in section
        assert "Physical Review Research" in section
        assert "editable SVG" in section
        assert "PDF, PNG, and TIFF" in section
