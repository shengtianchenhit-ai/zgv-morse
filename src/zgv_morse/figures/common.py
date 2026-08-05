"""Shared matplotlib publication style and export helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Final
import unicodedata

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

from zgv_morse.artifact_schema import SCHEMAS, validate_artifact

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


PUBLICATION_FORMATS: Final = ("svg", "pdf", "png", "tiff")
FIGURE_ARCHETYPES: Final = frozenset(
    {
        "quantitative grid",
        "schematic-led composite",
        "image plate + quant",
        "asymmetric mixed-modality figure",
    }
)
PALETTE: Final = {
    # Refined tones: same semantics (grey = isotropic, blue = anisotropic,
    # amber = saddle), tuned for softer contrast and colour-blind safety.
    "neutral": "#55565A",
    "isotropic": "#8A8B8F",
    "anisotropic": "#1F5FA8",
    "minimum": "#3775BA",
    "saddle": "#E28E2C",
    "prediction": "#2B2B2B",
    "uncertainty": "#D6D5D4",
}
# Color is never the sole Morse-class cue: circles denote minima and diamonds
# denote saddles in every panel, including grayscale reproductions.
MARKERS: Final = {"minimum": "o", "saddle": "D"}
_UNSAFE_UNICODE_CATEGORIES: Final = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """Immutable journal figure identity and final physical dimensions."""

    number: str
    conclusion: str
    archetype: str
    width_mm: float
    height_mm: float

    def __post_init__(self) -> None:
        for field_name in ("number", "conclusion", "archetype"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(f"{field_name} must be a string")
            if any(
                unicodedata.category(character) in _UNSAFE_UNICODE_CATEGORIES
                for character in value
            ):
                raise ValueError(f"{field_name} must not contain control characters")
            normalized = value.strip()
            if not normalized:
                raise ValueError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, normalized)
        if self.archetype not in FIGURE_ARCHETYPES:
            raise ValueError(f"archetype must be one of {sorted(FIGURE_ARCHETYPES)}")
        for field_name in ("width_mm", "height_mm"):
            value = getattr(self, field_name)
            if isinstance(value, (bool, complex)) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a real number")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
            object.__setattr__(self, field_name, normalized)


_DEFAULT_LINESPACING: Final = 1.35


def _apply_default_linespacing() -> None:
    """Open the leading on every multi-line text artist.

    Matplotlib exposes no rcParam for line spacing, and STIX sits on a
    smaller body than DejaVu, so the stock single spacing let consecutive
    lines of a multi-line annotation overlap by about 3 pt.  Patching the
    Text constructor default keeps every call site consistent without
    editing each annotation.
    """

    from matplotlib.text import Text

    if getattr(Text, "_zgv_linespacing_patched", False):
        return
    original = Text.__init__

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("linespacing", _DEFAULT_LINESPACING)
        original(self, *args, **kwargs)

    Text.__init__ = __init__  # type: ignore[method-assign]
    Text._zgv_linespacing_patched = True  # type: ignore[attr-defined]


def apply_publication_style() -> None:
    """Apply the journal-neutral matplotlib style used by every main figure."""

    # Ignore user/site matplotlibrc files and use Matplotlib's bundled font so
    # Linux CI and macOS produce the same typography and geometry.
    mpl.rcdefaults()
    _apply_default_linespacing()
    mpl.rcParams.update(
        {
            # Times-compatible serif, matching the journal body text.
            # STIXGeneral ships inside matplotlib, so Linux CI and macOS
            # resolve the same file; a system Times clone such as Liberation
            # Serif would silently fall back to a different face on machines
            # that lack it and break the cross-platform geometry contract.
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            # Match the maths to the text: the panels are dense with symbols
            # like kappa_0 and Omega, and STIX maths is the companion face.
            "mathtext.fontset": "stix",
            # STIX has shorter ascenders and descenders than DejaVu, so the
            # default single spacing let consecutive lines of a multi-line
            # annotation touch.  Open the leading to keep them apart.
            "axes.titlepad": 4.0,
            # Antialiasing stays off: tests/figures/test_common.py pins this
            # as part of the vector-identity contract (raster and vector
            # exports must agree stroke for stroke).
            "text.antialiased": False,
            "text.hinting": "none",
            "svg.fonttype": "none",
            "svg.hashsalt": "zgv-morse-publication-v1",
            "pdf.fonttype": 42,
            "lines.antialiased": False,
            "patch.antialiased": False,
            # Typographic hierarchy: quiet near-black ink, semibold titles,
            # slightly smaller ticks than labels.
            "font.size": 8.6,
            "text.color": "#262626",
            "axes.titlesize": 9.4,
            "axes.titleweight": "medium",
            "axes.titlecolor": "#1A1A1A",
            "axes.labelsize": 8.6,
            "axes.labelcolor": "#262626",
            "axes.edgecolor": "#3C3C3C",
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "xtick.color": "#3C3C3C",
            "ytick.color": "#3C3C3C",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8.4,
            "legend.handlelength": 1.5,
            "savefig.facecolor": "white",
        }
    )


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        # File contents are already fsynced. Some platforms reject directory fsync.
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _reject_symlink_ancestors(path: Path, label: str) -> None:
    candidate = path if path.is_absolute() else Path.cwd() / path
    parent_parts = candidate.parent.parts
    current = Path(parent_parts[0])
    for component in parent_parts[1:]:
        if component == "..":
            current = current.parent
            continue
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink ancestor: {current}")


def _temporary_path(parent: Path, prefix: str, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=parent)
    os.close(descriptor)
    return Path(name)


def _render_publication_format(fig: Figure, path: Path, kind: str) -> None:
    options: dict[str, object] = {"format": kind}
    if kind == "svg":
        options["metadata"] = {"Creator": "zgv-morse", "Date": None}
    elif kind == "pdf":
        options["metadata"] = {
            "Creator": "zgv-morse",
            "Producer": "zgv-morse",
            "CreationDate": None,
            "ModDate": None,
        }
    elif kind == "png":
        options.update(dpi=600, metadata={"Software": "zgv-morse"})
    elif kind == "tiff":
        options.update(
            dpi=600,
            pil_kwargs={
                "compression": "tiff_lzw",
                "tiffinfo": {305: "zgv-morse"},
            },
        )
    else:  # Defensive: PUBLICATION_FORMATS is immutable module configuration.
        raise ValueError(f"unsupported publication format: {kind}")
    fig.savefig(path, **options)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"temporary {kind} rendering is empty")
    if kind == "svg":
        # Matplotlib emits spaces before many SVG path newlines.  They have no
        # rendering meaning, but make generated masters fail whitespace audits.
        data = path.read_bytes()
        path.write_bytes(b"\n".join(line.rstrip(b" \t\r") for line in data.split(b"\n")))
    _fsync_file(path)


def _publish_figure_transaction(
    temporary: dict[str, Path],
    outputs: dict[str, Path],
) -> None:
    parent = next(iter(outputs.values())).parent
    backups: dict[Path, Path] = {}
    backup_candidates: list[Path] = []
    installed: list[Path] = []
    rollback_errors: list[OSError] = []
    retained_backups: set[Path] = set()
    try:
        for output in outputs.values():
            if os.path.lexists(output):
                backup = _temporary_path(
                    parent,
                    prefix=f".{output.name}.",
                    suffix=".backup",
                )
                backup_candidates.append(backup)
                os.replace(output, backup)
                backups[output] = backup
        _fsync_directory(parent)
        for kind in PUBLICATION_FORMATS:
            output = outputs[kind]
            os.replace(temporary[kind], output)
            installed.append(output)
        _fsync_directory(parent)
    except Exception as error:
        for output in reversed(installed):
            try:
                _unlink(output)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        for output, backup in reversed(tuple(backups.items())):
            try:
                os.replace(backup, output)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
                retained_backups.add(backup)
        _fsync_directory(parent)
        if rollback_errors:
            retained = [str(path) for path in sorted(retained_backups)]
            raise RuntimeError(
                "figure publication failed and rollback was incomplete; "
                f"retained backups: {retained}"
            ) from error
        raise
    else:
        for backup in backups.values():
            _unlink(backup)
        _fsync_directory(parent)
    finally:
        for path in temporary.values():
            _unlink(path)
        for path in backup_candidates:
            if path not in retained_backups:
                _unlink(path)


def save_publication_figure(
    fig: Figure,
    stem: Path,
    spec: FigureSpec,
) -> dict[str, Path]:
    """Export an editable vector master and three publication derivatives."""

    if not isinstance(fig, Figure):
        raise TypeError("fig must be a matplotlib Figure")
    if not isinstance(stem, Path):
        raise TypeError("stem must be a pathlib.Path")
    if not isinstance(spec, FigureSpec):
        raise TypeError("spec must be a FigureSpec")
    _reject_symlink_ancestors(stem, "stem")
    if (
        not stem.name
        or stem.name in {".", ".."}
        or stem.name.startswith(".")
        or stem.suffix
        or os.path.lexists(stem)
    ):
        raise ValueError("stem must be a nonexistent, extensionless file stem")
    if stem.parent.exists() and not stem.parent.is_dir():
        raise ValueError("stem parent must be a directory")

    stem.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(stem, "stem")

    outputs = {
        kind: stem.with_suffix(f".{kind}") for kind in PUBLICATION_FORMATS
    }
    for kind, output in outputs.items():
        if output.suffix != f".{kind}":
            raise ValueError(f"invalid output format path for {kind}")
        if output.is_symlink():
            raise ValueError(f"refusing to overwrite output symlink: {output}")
        if output.exists() and not output.is_file():
            raise ValueError(f"output path must be a regular file: {output}")

    fig.set_size_inches(spec.width_mm / 25.4, spec.height_mm / 25.4)
    temporary: dict[str, Path] = {}
    try:
        for kind in PUBLICATION_FORMATS:
            path = _temporary_path(
                stem.parent,
                prefix=f".{stem.name}.{kind}.",
                suffix=".tmp",
            )
            temporary[kind] = path
            _render_publication_format(fig, path, kind)
        _fsync_directory(stem.parent)
        _publish_figure_transaction(temporary, outputs)
    finally:
        plt.close(fig)
        for path in temporary.values():
            _unlink(path)
    return outputs


def load_figure_artifact(
    data_dir: Path,
    name: str,
) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
    """Load arrays and evidence metadata only after full artifact validation."""

    if not isinstance(data_dir, Path):
        raise TypeError("data_dir must be a pathlib.Path")
    if type(name) is not str:
        raise TypeError("name must be a string")
    if name not in SCHEMAS:
        raise ValueError(f"artifact name must be one of {sorted(SCHEMAS)}")
    if not data_dir.is_dir():
        raise ValueError("data_dir must be an existing directory")
    path = data_dir / f"{name}.npz"
    return validate_artifact(path, path.with_suffix(".json"))


def load_figure_inputs(
    data_dir: Path,
    name: str,
) -> dict[str, NDArray[np.generic]]:
    """Load one named figure input only after full artifact validation."""

    arrays, _ = load_figure_artifact(data_dir, name)
    return arrays


def _source_csv_path(path: object) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if path.name.startswith(".") or path.suffix != ".csv":
        raise ValueError("path must name a visible .csv file")
    _reject_symlink_ancestors(path, "path")
    if path.is_symlink():
        raise ValueError(f"refusing to overwrite a source-data symlink: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"source-data path must be a regular file: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise ValueError("path parent must be a directory")
    return path


def _source_columns(
    columns: object,
) -> tuple[tuple[str, ...], dict[str, NDArray[np.generic]]]:
    if type(columns) is not dict:
        raise TypeError("columns must be a dict")
    if not columns:
        raise ValueError("source-data columns must not be empty")
    if any(type(name) is not str for name in columns):
        raise TypeError("column names must be strings")
    names = tuple(sorted(columns))
    normalized: dict[str, NDArray[np.generic]] = {}
    for name in names:
        if not name.strip():
            raise ValueError("column name must not be blank")
        _validate_source_unicode(name, "column name")
        try:
            values = np.asarray(columns[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"source-data column {name!r} has an invalid dtype") from error
        if values.size == 0:
            raise ValueError(f"source-data column {name!r} must not be empty")
        if values.dtype.kind not in "iufU":
            raise ValueError(f"source-data column {name!r} has an unsupported dtype")
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise ValueError(f"source-data column {name!r} must contain only finite values")
        if values.dtype.kind == "U":
            for value in values.ravel(order="C"):
                _validate_source_unicode(str(value), f"column {name!r} Unicode value")
        normalized[name] = np.array(values.ravel(order="C"), copy=True)
    lengths = {values.size for values in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("source-data columns must have equal flattened length")
    return names, normalized


def _validate_source_unicode(value: str, label: str) -> None:
    if any(
        unicodedata.category(character) in _UNSAFE_UNICODE_CATEGORIES
        for character in value
    ):
        raise ValueError(f"{label} contains unsafe Unicode control characters")
    if value.lstrip().startswith(("=", "+", "-", "@")):
        raise ValueError(f"{label} contains an unsafe Unicode formula prefix")


def _source_cell(value: np.generic, kind: str) -> str:
    if kind in "iu":
        return str(int(value))
    if kind == "f":
        return format(float(value), ".17g")
    return str(value)


def write_source_csv(path: Path, columns: dict[str, object]) -> Path:
    """Write deterministic, finite, machine-readable source data as UTF-8 CSV."""

    target = _source_csv_path(path)
    names, normalized = _source_columns(columns)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(target, "path")
    _source_csv_path(target)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            descriptor = -1
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(names)
            for row in zip(*(normalized[name] for name in names), strict=True):
                writer.writerow(
                    [
                        _source_cell(value, normalized[name].dtype.kind)
                        for name, value in zip(names, row, strict=True)
                    ]
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _unlink(temporary)
    return target
