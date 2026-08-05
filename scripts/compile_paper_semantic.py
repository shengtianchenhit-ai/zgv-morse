"""Compile the paper once with strict semantic gates on a nonreference TeX host.

This entry point intentionally does not claim PDF byte reproduction.  The
reference-only byte gate remains ``scripts/compile_paper.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys

from pypdf import PdfReader
from pypdf.errors import PdfReadError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import compile_paper as reference_compile  # noqa: E402
from scripts.check_latex_log import LatexLogError, check_log_file  # noqa: E402


PAPER_DIR = reference_compile.PAPER_DIR
BUILD_DIR = reference_compile.BUILD_DIR


def _tail(output: str, lines: int = 30) -> str:
    return "\n".join(output.splitlines()[-lines:])


def _run_command(command: tuple[str, ...]) -> None:
    try:
        subprocess.run(
            command,
            cwd=PAPER_DIR,
            env=reference_compile.deterministic_environment(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            timeout=reference_compile.COMPILE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise reference_compile.PaperCompileError(
            "semantic compile timed out after "
            f"{reference_compile.COMPILE_TIMEOUT_SECONDS}s: {' '.join(command)}"
        ) from error
    except subprocess.CalledProcessError as error:
        output = error.stdout if isinstance(error.stdout, str) else ""
        detail = f"\n{_tail(output)}" if output else ""
        raise reference_compile.PaperCompileError(
            f"semantic compile failed with exit code {error.returncode}: "
            f"{' '.join(command)}{detail}"
        ) from error
    except OSError as error:
        raise reference_compile.PaperCompileError(
            f"cannot execute {' '.join(command)}: {error}"
        ) from error


def _validate_pdf_structure(path: Path) -> None:
    """Require a strict, unencrypted PDF with at least one readable page."""

    try:
        reader = PdfReader(path, strict=True)
        if reader.is_encrypted:
            raise reference_compile.PaperCompileError(
                f"compiled PDF must not be encrypted: {path}"
            )
        if len(reader.pages) < 1:
            raise reference_compile.PaperCompileError(
                f"compiled PDF has no readable pages: {path}"
            )
        _ = reader.metadata
    except reference_compile.PaperCompileError:
        raise
    except (PdfReadError, OSError, TypeError, ValueError) as error:
        raise reference_compile.PaperCompileError(
            f"invalid PDF structure: {path}: {error}"
        ) from error


def compile_paper_semantic() -> tuple[Path, ...]:
    """Build Supplement then main once and enforce all non-version semantic gates."""

    reference_compile.prepare_clean_build_dir()
    outputs: list[Path] = []
    for document, command in zip(
        reference_compile.DOCUMENTS,
        reference_compile.build_commands(),
        strict=True,
    ):
        _run_command(command)
        stem = Path(document).stem
        pdf = BUILD_DIR / f"{stem}.pdf"
        log = BUILD_DIR / f"{stem}.log"
        if stem == "supplement":
            auxiliary = BUILD_DIR / "supplement.aux"
            if not auxiliary.is_file() or auxiliary.stat().st_size == 0:
                raise reference_compile.PaperCompileError(
                    "supplement label file is missing before main compilation"
                )
        try:
            notes = check_log_file(log, expected_pdf=pdf)
            data = pdf.read_bytes()
            reference_compile.validate_pdf_bytes(data, source=str(pdf))
            _validate_pdf_structure(pdf)
        except LatexLogError as error:
            raise reference_compile.PaperCompileError(str(error)) from error
        except OSError as error:
            raise reference_compile.PaperCompileError(
                f"compiled PDF is missing or unreadable: {pdf}"
            ) from error
        for note in notes:
            print(note)
        print(f"semantic PDF gate passed: {pdf.relative_to(reference_compile.ROOT)}")
        outputs.append(pdf)
    return tuple(outputs)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the nonreference semantic PDF build."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        compile_paper_semantic()
    except (reference_compile.PaperCompileError, TypeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
