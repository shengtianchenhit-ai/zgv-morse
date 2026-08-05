"""Reject unresolved or materially broken final LaTeX compilation logs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
import re


DEFAULT_MAX_OVERFULL_PT = 3.0


class LatexLogError(RuntimeError):
    """Raised when a final LaTeX log violates a registered publication gate."""


_HARD_FAILURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "undefined citation",
        re.compile(r"(?:LaTeX|Package\s+\S+)\s+Warning:.*Citation.*undefined", re.I),
    ),
    (
        "undefined reference",
        re.compile(
            r"LaTeX\s+Warning:\s+(?:Reference\s+.*undefined|There were undefined references)",
            re.I,
        ),
    ),
    (
        "multiply defined label",
        re.compile(r"(?:multiply defined|multiply-defined labels)", re.I),
    ),
    (
        "missing file",
        re.compile(
            r"(?:!\s+(?:LaTeX|pdftex\.def|Package\s+\S+)\s+Error:\s+"
            r"File\s+.+?\s+not found|I can't find file)",
            re.I,
        ),
    ),
    (
        "missing bibliography output",
        re.compile(r"^No file\s+\S+\.bbl\.", re.I | re.M),
    ),
    (
        "missing auxiliary output",
        re.compile(r"^No file\s+.+\.\s*$", re.I | re.M),
    ),
    (
        "rerun required",
        re.compile(
            r"(?:Label\(s\) may have changed|Rerun to get cross-references right|"
            r"Package rerunfilecheck Warning: File .* has changed|"
            r"Please\s+\(re\)run\s+(?:LaTeX|Biber|BibTeX))",
            re.I,
        ),
    ),
    (
        "fatal TeX error",
        re.compile(
            r"Fatal error occurred|no output PDF file produced|No pages of output|"
            r"TeX capacity exceeded",
            re.I,
        ),
    ),
    (
        "LaTeX warning",
        re.compile(r"^LaTeX(?:\s+Font)?\s+Warning:", re.I | re.M),
    ),
    (
        "package warning",
        re.compile(r"^(?:Package|Class)\s+\S+\s+Warning:", re.I | re.M),
    ),
    (
        "engine warning",
        re.compile(r"^(?:pdfTeX|LuaTeX|XeTeX)\s+warning", re.I | re.M),
    ),
    (
        "TeX error",
        re.compile(
            r"^(?:\.{0,2}/|/|[A-Za-z0-9_.-])[^\r\n]*"
            r"\.(?:tex|sty|cls|bib):\d+:\s+(?:"
            r".*\bError:|Undefined control sequence|Emergency stop|"
            r"Fatal error occurred|TeX capacity exceeded|Missing .+ inserted|"
            r"Extra .+|Runaway argument|Paragraph ended before|"
            r"File .+ not found|Use of .+ doesn't match|You can't use|"
            r"Illegal unit of measure|Dimension too large|Misplaced .+|"
            r"Incomplete \\if.+|End occurred when.+)[^\r\n]*$",
            re.I | re.M,
        ),
    ),
    (
        "TeX error",
        re.compile(r"^!\s*.+", re.M),
    ),
)

_OVERFULL = re.compile(
    r"Overfull\s+\\[hv]box\s+\(\s*"
    r"(?P<points>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"pt\s+too\s+(?:wide|high)\s*\)",
    re.I,
)
_OVERFULL_START = re.compile(r"Overfull\s+\\[hv]box", re.I)
_UNDERFULL = re.compile(r"Underfull\s+\\[hv]box", re.I)
_OUTPUT_RECORD = re.compile(
    r"^Output written on (?P<path>.*?\.pdf)\s+\(\d+\s+pages?,",
    re.I | re.M | re.S,
)


def _source_name(source: object) -> str:
    if not isinstance(source, str) or not source:
        raise TypeError("source must be a nonempty string")
    return source


def _threshold(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("max_overfull_pt must be numeric")
    threshold = float(value)
    if not 0.0 <= threshold < float("inf"):
        raise ValueError("max_overfull_pt must be finite and nonnegative")
    return Decimal(str(threshold))


def check_log_text(
    text: object,
    *,
    source: str = "<log>",
    max_overfull_pt: float = DEFAULT_MAX_OVERFULL_PT,
    expected_engine: str | None = None,
    expected_pdf: Path | None = None,
) -> tuple[str, ...]:
    """Validate one final log and return nonfatal small-overfull diagnostics."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    name = _source_name(source)
    threshold = _threshold(max_overfull_pt)
    if expected_engine is not None and (
        not isinstance(expected_engine, str) or not expected_engine
    ):
        raise TypeError("expected_engine must be None or a nonempty string")
    if expected_pdf is not None:
        if not isinstance(expected_pdf, Path):
            raise TypeError("expected_pdf must be None or a pathlib.Path")
        if not expected_pdf.is_absolute():
            raise ValueError("expected_pdf must be absolute")
    failures: list[str] = []
    notes: list[str] = []

    for label, pattern in _HARD_FAILURES:
        if pattern.search(text) is not None:
            failures.append(label)

    outputs = tuple(_OUTPUT_RECORD.finditer(text))
    if not outputs:
        failures.append("missing successful PDF output marker")
    elif len(outputs) > 1:
        failures.append("multiple PDF output records")
    if expected_engine is not None and not text.startswith(expected_engine):
        failures.append("unexpected TeX engine or version")
    if expected_pdf is not None and len(outputs) == 1:
        recorded = re.sub(r"\r?\n", "", outputs[0].group("path")).strip()
        actual = Path(recorded)
        if not actual.is_absolute() or actual.resolve(strict=False) != expected_pdf.resolve(
            strict=False
        ):
            failures.append("unexpected PDF output identity")

    overfull_matches = tuple(_OVERFULL.finditer(text))
    if len(overfull_matches) != len(_OVERFULL_START.findall(text)):
        failures.append("malformed overfull box diagnostic")

    for match in overfull_matches:
        raw_points = match.group("points")
        points = Decimal(raw_points)
        diagnostic = (
            f"{name}: overfull box {raw_points}pt (limit {threshold}pt)"
        )
        if points > threshold:
            failures.append(diagnostic)
        else:
            notes.append(diagnostic)

    notes.extend(
        f"{name}: underfull box diagnostic" for _ in _UNDERFULL.finditer(text)
    )

    if failures:
        unique = tuple(dict.fromkeys(failures))
        raise LatexLogError(f"{name}: " + "; ".join(unique))
    return tuple(notes)


def check_log_file(
    path: Path,
    *,
    max_overfull_pt: float = DEFAULT_MAX_OVERFULL_PT,
    expected_engine: str | None = None,
    expected_pdf: Path | None = None,
) -> tuple[str, ...]:
    """Read and validate one UTF-8 LaTeX log file."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not path.is_file():
        raise LatexLogError(f"missing log: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise LatexLogError(f"LaTeX log is not UTF-8: {path}") from error
    except OSError as error:
        raise LatexLogError(f"cannot read LaTeX log: {path}") from error
    return check_log_text(
        text,
        source=str(path),
        max_overfull_pt=max_overfull_pt,
        expected_engine=expected_engine,
        expected_pdf=expected_pdf,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the strict gate for one or more final logs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--max-overfull-pt",
        type=float,
        default=DEFAULT_MAX_OVERFULL_PT,
    )
    arguments = parser.parse_args(argv)
    try:
        notes = tuple(
            note
            for log in arguments.logs
            for note in check_log_file(
                log,
                max_overfull_pt=arguments.max_overfull_pt,
            )
        )
    except (LatexLogError, TypeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    for note in notes:
        print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
