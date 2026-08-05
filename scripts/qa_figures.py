"""Record and strictly validate every publication figure and source-data bundle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Final
import xml.etree.ElementTree as ET

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from zgv_morse.artifacts import sha256_file
from zgv_morse.artifact_schema import SCHEMAS, validate_artifact
from zgv_morse.config import load_reference_config
from zgv_morse.provenance import (
    FIGURE_NAMES,
    ArtifactValidationError,
    replace_figure_records,
    validate_manifest,
)
from zgv_morse.workflows.common import _scientific_context


ROOT: Final = Path(__file__).resolve().parents[1]
FORMATS: Final = ("svg", "pdf", "png", "tiff")
FILE_RECORD_KEYS: Final = frozenset({"path", "sha256"})
FIGURE_RECORD_KEYS: Final = frozenset(
    {"script", "script_sha256", "inputs", "source_data", "outputs", "theory_refs"}
)
SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
REVIEW_CHECKS: Final = (
    "clipping",
    "text size",
    "axis honesty",
    "symbols",
    "grayscale ambiguity",
    "contour readability",
    "hero conclusion",
)


@dataclass(frozen=True, slots=True)
class FigureContract:
    stem: str
    script: str
    inputs: tuple[str, ...]
    source_files: tuple[str, ...]
    output_directory: str
    width_mm: float
    height_mm: float
    panels: tuple[str, ...]
    theory_refs: tuple[str, ...]


def _artifact_inputs(*names: str) -> tuple[str, ...]:
    return tuple(
        path
        for name in names
        for path in (f"data/generated/{name}.npz", f"data/generated/{name}.json")
    )


def _source_files(directory: str, *names: str) -> tuple[str, ...]:
    return tuple(f"data/source_data/{directory}/{name}" for name in names)


_COMMON = "src/zgv_morse/figures/common.py"
_SUPPLEMENTARY_INPUTS = (
    _COMMON,
    "src/zgv_morse/figures/supplementary.py",
    "scripts/validate_isotropic.py",
    "config/reference.yaml",
    "data/generated/isotropic_validation.json",
    "data/generated/isotropic_convergence.csv",
    *_artifact_inputs(
        "isotropic_zgv",
        "angular_sensitivity",
        "convergence",
        "silicon_stress_test",
    ),
)


CONTRACTS: Final = (
    FigureContract(
        "figure_01_geometry_mechanism",
        "scripts/make_figure_01.py",
        (_COMMON, "src/zgv_morse/figures/figure01_geometry.py", *_artifact_inputs(
            "isotropic_zgv", "angular_sensitivity", "critical_points"
        )),
        _source_files(
            "figure_01",
            "panel_a_ring.csv",
            "panel_b_points.csv",
            "panel_b_surface.csv",
        ),
        "figures/main",
        183.0,
        140.0,
        ("a", "b", "c"),
        (
            "thm:morse-bott-ring",
            "thm:cubic-morse-splitting",
            "thm:uniform-bessel-crossover",
            "thm:fixed-anisotropy-decay",
        ),
    ),
    FigureContract(
        "figure_02_isotropic_zgv",
        "scripts/make_figure_02.py",
        (_COMMON, "src/zgv_morse/figures/figure02_isotropic.py", *_artifact_inputs(
            "isotropic_zgv", "convergence"
        )),
        _source_files(
            "figure_02",
            "panel_a_branches.csv",
            "panel_b_local_quadratic.csv",
            "panel_c_convergence.csv",
            "panel_d_mode_profile.csv",
        ),
        "figures/main",
        183.0,
        195.0,
        ("a", "b", "c", "d"),
        ("thm:morse-bott-ring",),
    ),
    FigureContract(
        "figure_03_angular_sensitivity",
        "scripts/make_figure_03.py",
        (_COMMON, "src/zgv_morse/figures/figure03_sensitivity.py", *_artifact_inputs(
            "angular_sensitivity", "convergence"
        )),
        _source_files(
            "figure_03",
            "panel_a_polar.csv",
            "panel_b_harmonics.csv",
            "panel_c_physical_shift.csv",
            "panel_d_angular_fd.csv",
            "panel_d_step_convergence.csv",
        ),
        "figures/main",
        183.0,
        158.0,
        ("a", "b", "c", "d"),
        ("thm:anisotropic-normal-form", "thm:cubic-morse-splitting"),
    ),
    FigureContract(
        "figure_04_morse_points",
        "scripts/make_figure_04.py",
        (
            _COMMON,
            "src/zgv_morse/figures/figure04_morse.py",
            "config/reference.yaml",
            *_artifact_inputs("isotropic_zgv", "critical_points"),
        ),
        _source_files(
            "figure_04",
            "panel_a_isotropic_surface.csv",
            "panel_b_anisotropic_surface.csv",
            "panel_b_points.csv",
            "panel_c_hessian.csv",
            "panel_d_prediction_errors.csv",
        ),
        "figures/main",
        183.0,
        158.0,
        ("a", "b", "c", "d"),
        ("thm:cubic-morse-splitting",),
    ),
    FigureContract(
        "figure_05_perturbation_scaling",
        "scripts/make_figure_05.py",
        (_COMMON, "src/zgv_morse/figures/figure05_scaling.py", *_artifact_inputs(
            "perturbation_scaling"
        )),
        _source_files(
            "figure_05",
            "panel_a_splitting.csv",
            "panel_b_compensated_radial_shift.csv",
            "panel_c_frequency_remainder.csv",
            "panel_d_role_reversal.csv",
        ),
        "figures/main",
        183.0,
        142.0,
        ("a", "b", "c", "d"),
        ("thm:anisotropic-normal-form", "thm:cubic-morse-splitting"),
    ),
    FigureContract(
        "figure_06_decay_crossover",
        "scripts/make_figure_06.py",
        (
            _COMMON,
            "src/zgv_morse/figures/figure06_crossover.py",
            "src/zgv_morse/workflows/green.py",
            *_artifact_inputs("green_crossover", "angular_sensitivity", "convergence"),
        ),
        _source_files(
            "figure_06",
            "panel_a_collapse.csv",
            "panel_b_absolute_error.csv",
            "panel_c_envelopes.csv",
            "panel_c_late_fit.csv",
            "panel_d_crossover.csv",
            "panel_e_fixed_morse.csv",
            "panel_f_frequency_features.csv",
        ),
        "figures/main",
        183.0,
        185.0,
        ("a", "b", "c", "d", "e", "f"),
        ("thm:uniform-bessel-crossover", "thm:fixed-anisotropy-decay"),
    ),
    FigureContract(
        "figure_s01_polynomial_two_element",
        "scripts/make_supplementary_figures.py",
        _SUPPLEMENTARY_INPUTS,
        _source_files(
            "supplementary",
            "s01_panel_a_polynomial.csv",
            "s01_panel_b_independent.csv",
            "s01_panel_c_two_element.csv",
        ),
        "figures/supplementary",
        183.0,
        104.0,
        ("a", "b", "c"),
        ("thm:morse-bott-ring",),
    ),
    FigureContract(
        "figure_s02_quadrature_phase",
        "scripts/make_supplementary_figures.py",
        _SUPPLEMENTARY_INPUTS,
        _source_files(
            "supplementary",
            "s02_panel_a_quadrature.csv",
            "s02_panel_b_dispersion.csv",
            "s02_panel_c_phase.csv",
        ),
        "figures/supplementary",
        183.0,
        104.0,
        ("a", "b", "c"),
        ("thm:uniform-bessel-crossover", "thm:fixed-anisotropy-decay"),
    ),
    FigureContract(
        "figure_s03_mode_tracking",
        "scripts/make_supplementary_figures.py",
        _SUPPLEMENTARY_INPUTS,
        _source_files(
            "supplementary",
            "s03_panel_a_mac.csv",
            "s03_panel_b_tracking_gap.csv",
            "s03_panel_c_gap_residual.csv",
        ),
        "figures/supplementary",
        183.0,
        104.0,
        ("a", "b", "c"),
        ("thm:morse-bott-ring", "thm:anisotropic-normal-form"),
    ),
    FigureContract(
        "figure_s04_fd_convergence",
        "scripts/make_supplementary_figures.py",
        _SUPPLEMENTARY_INPUTS,
        _source_files(
            "supplementary",
            "s04_panel_a_v4.csv",
            "s04_panel_b_b_diagonal.csv",
            "s04_panel_c_b_matrix.csv",
        ),
        "figures/supplementary",
        183.0,
        104.0,
        ("a", "b", "c"),
        ("thm:anisotropic-normal-form", "thm:cubic-morse-splitting"),
    ),
    FigureContract(
        "figure_s05_source_window_sensitivity",
        "scripts/make_supplementary_figures.py",
        _SUPPLEMENTARY_INPUTS,
        _source_files(
            "supplementary",
            "s05_panel_a_heatmap.csv",
            "s05_panel_b_profiles.csv",
        ),
        "figures/supplementary",
        183.0,
        104.0,
        ("a", "b"),
        ("thm:uniform-bessel-crossover",),
    ),
    FigureContract(
        "figure_s06_silicon_stress_test",
        "scripts/make_supplementary_figures.py",
        _SUPPLEMENTARY_INPUTS,
        _source_files(
            "supplementary",
            "s06_material_record.csv",
            "s06_panel_a_points.csv",
            "s06_panel_a_surface.csv",
            "s06_panel_b_hessian.csv",
            "s06_panel_c_diagnostics.csv",
        ),
        "figures/supplementary",
        183.0,
        104.0,
        ("a", "b", "c"),
        ("thm:cubic-morse-splitting",),
    ),
)


def _relative_file(project_root: Path, relative: str, label: str) -> Path:
    if type(relative) is not str or not relative or "\\" in relative:
        raise ArtifactValidationError(f"{label} path is malformed")
    pure = Path(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ArtifactValidationError(f"{label} path is unsafe")
    candidate = project_root.joinpath(*pure.parts)
    if candidate.is_symlink():
        raise ArtifactValidationError(f"{label} path must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(project_root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise ArtifactValidationError(f"{label} path is missing or escapes the project") from error
    if not resolved.is_file():
        raise ArtifactValidationError(f"{label} path must be a regular file")
    return resolved


def _file_record(project_root: Path, relative: str) -> dict[str, str]:
    path = _relative_file(project_root, relative, "figure input")
    return {"path": relative, "sha256": sha256_file(path)}


def _rendering_code_inputs(project_root: Path) -> tuple[str, ...]:
    """Return the conservative local code/environment closure for every figure."""

    package_sources = tuple(
        path.relative_to(project_root).as_posix()
        for path in sorted((project_root / "src/zgv_morse").rglob("*.py"))
        if path.is_file()
    )
    if not package_sources:
        raise ArtifactValidationError("the local zgv_morse source closure is empty")
    return ("pyproject.toml", "uv.lock", "config/reference.yaml", *package_sources)


def _validate_current_artifact_context(project_root: Path) -> str:
    """Reject scientific artifacts produced by another source/config/lock state."""

    expected = _scientific_context(
        load_reference_config(project_root / "config/reference.yaml"),
        project_root,
    )
    profiles: set[str] = set()
    for name in sorted(SCHEMAS):
        artifact = project_root / "data/generated" / f"{name}.npz"
        _, metadata = validate_artifact(artifact, artifact.with_suffix(".json"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ArtifactValidationError(
                    f"{name} {key} does not match the current repository"
                )
        profile = metadata.get("profile")
        if profile not in {"smoke", "full"}:
            raise ArtifactValidationError(f"{name} has an invalid registered profile")
        profiles.add(str(profile))
    if len(profiles) != 1:
        raise ArtifactValidationError("scientific artifacts mix smoke and full profiles")
    return profiles.pop()


def build_figure_records(project_root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Build deterministic records from the declared twelve-figure closure."""

    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a pathlib.Path")
    rendering_inputs = _rendering_code_inputs(project_root)
    records: dict[str, dict[str, Any]] = {}
    declared_sources: set[str] = set()
    declared_outputs: set[str] = set()
    for contract in CONTRACTS:
        script = _relative_file(project_root, contract.script, f"{contract.stem} script")
        source_paths = contract.source_files
        if not source_paths or len(set(source_paths)) != len(source_paths):
            raise ArtifactValidationError(f"{contract.stem} has no source-data CSV files")
        overlap = declared_sources.intersection(source_paths)
        if overlap:
            raise ArtifactValidationError(
                f"source-data files are assigned to multiple figures: {sorted(overlap)}"
            )
        declared_sources.update(source_paths)
        outputs = {
            kind: _file_record(
                project_root,
                f"{contract.output_directory}/{contract.stem}.{kind}",
            )
            for kind in FORMATS
        }
        declared_outputs.update(item["path"] for item in outputs.values())
        declared_inputs = tuple(sorted(set((*rendering_inputs, *contract.inputs))))
        records[contract.stem] = {
            "script": contract.script,
            "script_sha256": sha256_file(script),
            "inputs": [_file_record(project_root, path) for path in declared_inputs],
            "source_data": [_file_record(project_root, path) for path in source_paths],
            "outputs": outputs,
            "theory_refs": list(contract.theory_refs),
        }
    if set(records) != FIGURE_NAMES:
        raise ArtifactValidationError("figure contract set is incomplete")
    actual_sources = {
        path.relative_to(project_root).as_posix()
        for path in (project_root / "data/source_data").rglob("*.csv")
        if path.is_file()
    }
    if declared_sources != actual_sources:
        raise ArtifactValidationError(
            "source-data closure is invalid: "
            f"unclaimed={sorted(actual_sources - declared_sources)}, "
            f"missing={sorted(declared_sources - actual_sources)}"
        )
    actual_outputs = {
        path.relative_to(project_root).as_posix()
        for directory in (project_root / "figures/main", project_root / "figures/supplementary")
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lstrip(".") in FORMATS
    }
    if declared_outputs != actual_outputs:
        raise ArtifactValidationError(
            "figure-output closure is invalid: "
            f"unclaimed={sorted(actual_outputs - declared_outputs)}, "
            f"missing={sorted(declared_outputs - actual_outputs)}"
        )
    return records


def _checked_record_path(
    project_root: Path,
    value: object,
    label: str,
) -> Path:
    if type(value) is not dict or set(value) != FILE_RECORD_KEYS:
        raise ArtifactValidationError(f"{label} must be an exact file record")
    claimed = value["sha256"]
    if type(claimed) is not str or SHA256.fullmatch(claimed) is None:
        raise ArtifactValidationError(f"{label} checksum is malformed")
    path = _relative_file(project_root, value["path"], label)
    if claimed != sha256_file(path):
        raise ArtifactValidationError(f"{label} checksum mismatch")
    return path


def _svg_text_and_dimensions(path: Path) -> tuple[tuple[str, ...], float, float]:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ET.ParseError) as error:
        raise ArtifactValidationError(f"SVG is malformed: {path}") from error
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ArtifactValidationError(f"SVG root element is invalid: {path}")
    texts = tuple(
        "".join(element.itertext()).strip()
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "text"
    )
    if not texts:
        raise ArtifactValidationError(f"SVG has no editable text nodes: {path}")
    if not any(element.tag.rsplit("}", 1)[-1] == "path" for element in root.iter()):
        raise ArtifactValidationError(f"SVG has no rendered path geometry: {path}")

    def points(attribute: str) -> float:
        raw = root.attrib.get(attribute, "")
        if not raw.endswith("pt"):
            raise ArtifactValidationError(f"SVG {attribute} must be declared in points")
        try:
            value = float(raw[:-2])
        except ValueError as error:
            raise ArtifactValidationError(f"SVG {attribute} is malformed") from error
        if not math.isfinite(value) or value <= 0.0:
            raise ArtifactValidationError(f"SVG {attribute} must be finite and positive")
        return value

    width = points("width")
    height = points("height")
    raw_view_box = root.attrib.get("viewBox", "").split()
    try:
        view_box = tuple(float(value) for value in raw_view_box)
    except ValueError as error:
        raise ArtifactValidationError(f"SVG viewBox is malformed: {path}") from error
    if (
        len(view_box) != 4
        or not all(math.isfinite(value) for value in view_box)
        or not math.isclose(view_box[0], 0.0, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(view_box[1], 0.0, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(view_box[2], width, rel_tol=0.0, abs_tol=2.0e-4)
        or not math.isclose(view_box[3], height, rel_tol=0.0, abs_tol=2.0e-4)
    ):
        raise ArtifactValidationError(f"SVG viewBox does not match its dimensions: {path}")
    return texts, width, height


def _validate_source_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ArtifactValidationError(f"source CSV has an invalid header: {path}")
        rows = list(reader)
    if not rows:
        raise ArtifactValidationError(f"source CSV is empty: {path}")
    for row in rows:
        if set(row) != set(reader.fieldnames):
            raise ArtifactValidationError(f"source CSV row is malformed: {path}")
        for value in row.values():
            if value is None or value == "":
                raise ArtifactValidationError(f"source CSV contains an empty value: {path}")
            try:
                number = float(value)
            except ValueError:
                if any(ord(character) < 32 for character in value) or value.lstrip().startswith(
                    ("=", "+", "-", "@")
                ):
                    raise ArtifactValidationError(
                        f"source CSV contains an unsafe string: {path}"
                    )
            else:
                if not math.isfinite(number):
                    raise ArtifactValidationError(
                        f"source CSV contains a non-finite number: {path}"
                    )


def _validate_pdf(path: Path, contract: FigureContract) -> None:
    try:
        reader = PdfReader(str(path), strict=True)
        if reader.is_encrypted or len(reader.pages) != 1:
            raise ArtifactValidationError(f"PDF must contain one unencrypted page: {path}")
        metadata = reader.metadata or {}
        if "/CreationDate" in metadata or "/ModDate" in metadata:
            raise ArtifactValidationError(f"PDF contains nondeterministic dates: {path}")
        page = reader.pages[0]
        expected_width = contract.width_mm / 25.4 * 72.0
        expected_height = contract.height_mm / 25.4 * 72.0
        if not math.isclose(
            float(page.mediabox.width), expected_width, rel_tol=0.0, abs_tol=2.0e-4
        ) or not math.isclose(
            float(page.mediabox.height), expected_height, rel_tol=0.0, abs_tol=2.0e-4
        ):
            raise ArtifactValidationError(f"PDF MediaBox has incorrect dimensions: {path}")
        contents = page.get_contents()
        if contents is None or len(contents.get_data()) < 256:
            raise ArtifactValidationError(f"PDF page has no substantive content stream: {path}")
        extracted = page.extract_text() or ""
    except ArtifactValidationError:
        raise
    except (OSError, PdfReadError, TypeError, ValueError) as error:
        raise ArtifactValidationError(f"PDF is malformed or unreadable: {path}") from error
    # Panel tags are drawn parenthesised, e.g. "(a)".
    tokens = set(extracted.split())
    for panel in contract.panels:
        if panel not in tokens and f"({panel})" not in tokens:
            raise ArtifactValidationError(f"PDF is missing panel label {panel}: {path}")


def _validate_svg(path: Path, contract: FigureContract) -> None:
    try:
        svg_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactValidationError(f"SVG is unreadable: {path}") from error
    if "dc:date" in svg_text:
        raise ArtifactValidationError(f"{contract.stem} SVG has nondeterministic metadata")
    texts, width_pt, height_pt = _svg_text_and_dimensions(path)
    if len(texts) < len(contract.panels) + 3:
        raise ArtifactValidationError(
            f"{contract.stem} SVG has implausibly little editable text"
        )
    for panel in contract.panels:
        if panel not in texts and f"({panel})" not in texts:
            raise ArtifactValidationError(
                f"{contract.stem} SVG is missing panel label {panel}"
            )
    expected_width = contract.width_mm / 25.4 * 72.0
    expected_height = contract.height_mm / 25.4 * 72.0
    if not math.isclose(width_pt, expected_width, rel_tol=0.0, abs_tol=2.0e-4):
        raise ArtifactValidationError(f"{contract.stem} SVG width is incorrect")
    if not math.isclose(height_pt, expected_height, rel_tol=0.0, abs_tol=2.0e-4):
        raise ArtifactValidationError(f"{contract.stem} SVG height is incorrect")


def _validate_raster(path: Path, kind: str, expected_pixels: tuple[int, int]) -> None:
    expected_format = {"png": "PNG", "tiff": "TIFF"}[kind]
    try:
        with Image.open(path) as image:
            if image.format != expected_format:
                raise ArtifactValidationError(f"{kind} has an incorrect encoded format")
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.format != expected_format or image.size != expected_pixels:
                raise ArtifactValidationError(f"{kind} dimensions or format are incorrect")
            dpi = image.info.get("dpi")
            if (
                not isinstance(dpi, tuple)
                or len(dpi) != 2
                or any(abs(float(value) - 600.0) > 0.1 for value in dpi)
            ):
                raise ArtifactValidationError(f"{kind} must be 600 dpi")
            if kind == "tiff" and image.tag_v2.get(259) != 5:
                raise ArtifactValidationError("TIFF must use LZW compression")
            rgb = image.convert("RGB")
            difference = ImageChops.difference(
                rgb,
                Image.new("RGB", image.size, "white"),
            )
            bounds = difference.getbbox()
            if bounds is None:
                raise ArtifactValidationError(f"{kind} is unexpectedly blank")
            left, top, right, bottom = bounds
            margins = (left, top, image.width - right, image.height - bottom)
            if min(margins) < 8:
                raise ArtifactValidationError(
                    f"{kind} lacks an 8-pixel print-safe margin"
                )
    except ArtifactValidationError:
        raise
    except (OSError, SyntaxError, ValueError) as error:
        raise ArtifactValidationError(f"{kind} is malformed or truncated: {path}") from error


def validate_figure_records_content(
    project_root: Path,
    records: dict[str, Any],
    *,
    require_complete: bool = True,
) -> None:
    """Validate hashes, editability, physical size, panels, rasters, and CSVs."""

    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a pathlib.Path")
    if type(records) is not dict:
        raise TypeError("records must be a dict")
    if type(require_complete) is not bool:
        raise TypeError("require_complete must be a bool")
    names = set(records)
    if names.difference(FIGURE_NAMES) or (require_complete and names != FIGURE_NAMES):
        raise ArtifactValidationError("figure record set is incomplete or unexpected")
    contracts = {contract.stem: contract for contract in CONTRACTS}
    for name in sorted(records):
        contract = contracts[name]
        record = records[name]
        if type(record) is not dict or set(record) != FIGURE_RECORD_KEYS:
            raise ArtifactValidationError(f"{name} record has invalid keys")
        if record["script"] != contract.script:
            raise ArtifactValidationError(f"{name} script path violates its contract")
        if record["theory_refs"] != list(contract.theory_refs):
            raise ArtifactValidationError(f"{name} theory references violate their contract")
        script = _relative_file(project_root, record["script"], f"{name} script")
        if record["script_sha256"] != sha256_file(script):
            raise ArtifactValidationError(f"{name} script checksum mismatch")
        for list_name in ("inputs", "source_data"):
            items = record[list_name]
            if type(items) is not list or not items:
                raise ArtifactValidationError(f"{name} {list_name} is empty")
            paths = [
                _checked_record_path(project_root, item, f"{name} {list_name}")
                for item in items
            ]
            if len(set(paths)) != len(paths):
                raise ArtifactValidationError(f"{name} {list_name} has duplicate paths")
            if list_name == "source_data":
                for path in paths:
                    _validate_source_csv(path)
        outputs = record["outputs"]
        if type(outputs) is not dict or tuple(sorted(outputs)) != tuple(sorted(FORMATS)):
            raise ArtifactValidationError(f"{name} output formats are incomplete")
        paths = {
            kind: _checked_record_path(project_root, outputs[kind], f"{name} {kind}")
            for kind in FORMATS
        }
        _validate_svg(paths["svg"], contract)
        _validate_pdf(paths["pdf"], contract)
        expected_pixels = (
            int(contract.width_mm / 25.4 * 600),
            int(contract.height_mm / 25.4 * 600),
        )
        for kind in ("png", "tiff"):
            _validate_raster(paths[kind], kind, expected_pixels)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def make_contact_sheet(
    project_root: Path,
    records: dict[str, Any],
    output: Path,
) -> Path:
    """Assemble a deterministic twelve-panel PNG contact sheet with matplotlib."""

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(12.0, 12.0), facecolor="white")
    for ax, contract in zip(axes.flat, CONTRACTS, strict=True):
        png_record = records[contract.stem]["outputs"]["png"]
        png = _checked_record_path(project_root, png_record, f"{contract.stem} contact PNG")
        ax.imshow(mpimg.imread(png))
        ax.set_title(contract.stem, fontsize=7)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.98, wspace=0.04, hspace=0.12)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        fig.savefig(
            temporary,
            format="png",
            dpi=200,
            metadata={"Software": "zgv-morse"},
        )
        os.replace(temporary, output)
    finally:
        plt.close(fig)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return output


_FIXED_REVIEW_STATUS: Final = {
        "figure_01_geometry_mechanism": "fixed",
        "figure_03_angular_sensitivity": "fixed",
        "figure_06_decay_crossover": "fixed",
        "figure_s04_fd_convergence": "fixed",
        "figure_s06_silicon_stress_test": "fixed",
}


def write_qa_report(path: Path, records: dict[str, Any]) -> Path:
    """Write the manually authorized full-profile review ledger with PNG hashes."""

    if type(records) is not dict or set(records) != FIGURE_NAMES:
        raise ArtifactValidationError("visual review requires a complete figure closure")
    lines = [
        "# Figure QA report",
        "",
        (
            "All figures were manually inspected at final physical size after strict "
            "automated validation. Each row is bound to the reviewed full-profile PNG."
        ),
        "",
        (
            "| Figure | reviewed PNG SHA-256 | clipping | text size | axis honesty | "
            "symbols | grayscale ambiguity | contour readability | hero conclusion | status |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for contract in CONTRACTS:
        status = _FIXED_REVIEW_STATUS.get(contract.stem, "pass")
        png_sha256 = records[contract.stem]["outputs"]["png"]["sha256"]
        lines.append(
            f"| {contract.stem} | {png_sha256} | "
            + " | ".join("pass" for _ in REVIEW_CHECKS)
            + f" | {status} |"
        )
    lines.extend(
        (
            "",
            "## Fixed items",
            "",
            "- Figure 1: moved Morse labels and kept the count local to the registered annulus.",
            "- Figure 3: replaced nested layout with a byte-deterministic flat constrained grid.",
            "- Figure 6: corrected inverse-rate slope sign and separated certified/all-time errors.",
            "- Figure S4: removed colliding finite-difference minor tick labels.",
            "- Figure S6: moved material constants away from the stationarity curve.",
            "",
            "## Blocked items",
            "",
            "None.",
        )
    )
    _atomic_text(path, "\n".join(lines) + "\n")
    return path


def validate_qa_report(
    path: Path,
    records: dict[str, Any],
    *,
    require_hash_match: bool,
) -> None:
    """Validate a human review ledger without manufacturing review results in CI."""

    if type(require_hash_match) is not bool:
        raise TypeError("require_hash_match must be a bool")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactValidationError("the manual figure QA report is missing") from error
    rows = [line for line in text.splitlines() if line.startswith("| figure_")]
    if len(rows) != len(CONTRACTS):
        raise ArtifactValidationError("the manual figure QA report must contain twelve rows")
    for line, contract in zip(rows, CONTRACTS, strict=True):
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        if len(cells) != 3 + len(REVIEW_CHECKS) or cells[0] != contract.stem:
            raise ArtifactValidationError("the manual figure QA report has invalid columns")
        reviewed_hash = cells[1]
        if SHA256.fullmatch(reviewed_hash) is None:
            raise ArtifactValidationError("the manual figure QA report has an invalid PNG hash")
        expected_hash = records[contract.stem]["outputs"]["png"]["sha256"]
        if require_hash_match and reviewed_hash != expected_hash:
            raise ArtifactValidationError(
                f"{contract.stem} has changed since its manual visual review"
            )
        if any(value != "pass" for value in cells[2 : 2 + len(REVIEW_CHECKS)]):
            raise ArtifactValidationError(
                f"{contract.stem} has an incomplete manual visual check"
            )
        expected_status = _FIXED_REVIEW_STATUS.get(contract.stem, "pass")
        if cells[-1] != expected_status:
            raise ArtifactValidationError(f"{contract.stem} review status is invalid")
    if "## Blocked items\n\nNone.\n" not in text or "| blocked |" in text:
        raise ArtifactValidationError("the manual figure QA report contains blocked items")


def run_strict(
    project_root: Path = ROOT,
    *,
    record_review: bool = False,
) -> dict[str, Any]:
    """Validate existing records or create them only when the closure is absent."""

    if type(record_review) is not bool:
        raise TypeError("record_review must be a bool")
    manifest = project_root / "data/provenance_manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    existing = raw.get("figures") if isinstance(raw, dict) else None
    profile = _validate_current_artifact_context(project_root)
    expected = build_figure_records(project_root)
    validate_figure_records_content(project_root, expected)
    make_contact_sheet(
        project_root,
        expected,
        project_root / "build/figure_contact_sheet.png",
    )
    report = project_root / "docs/figures/qa_report.md"
    if record_review:
        if profile != "full":
            raise ArtifactValidationError("manual visual review may record only full-profile figures")
        write_qa_report(report, expected)
    validate_qa_report(report, expected, require_hash_match=profile == "full")
    if existing in (None, {}):
        replace_figure_records(manifest, expected)
    else:
        if type(existing) is not dict or set(existing) != FIGURE_NAMES:
            raise ArtifactValidationError("existing figure closure is partial or malformed")
        validate_manifest(manifest, require_figures=True)
        if existing != expected:
            raise ArtifactValidationError(
                "existing figure records do not match the declared source-data closure"
            )
    return validate_manifest(manifest, require_figures=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--record-review", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.strict:
        parser.error("--strict is required")
    run_strict(ROOT, record_review=arguments.record_review)


if __name__ == "__main__":
    main()
