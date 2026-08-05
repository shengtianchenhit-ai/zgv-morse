"""Clean-build both paper PDFs twice and require byte-identical outputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_latex_log import LatexLogError, check_log_file  # noqa: E402


PAPER_DIR: Final = ROOT / "paper"
BUILD_DIR: Final = ROOT / "build/paper"
DOCUMENTS: Final = ("supplement.tex", "main.tex")
SOURCE_DATE_EPOCH: Final = "1783612800"
EXPECTED_LATEXMK_BANNER: Final = (
    "Latexmk, John Collins, 9 March 2026. Version 4.88"
)
EXPECTED_PDFTEX_VERSION_LINE: Final = (
    "pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)"
)
EXPECTED_PDFTEX_BANNER: Final = (
    "This is pdfTeX, Version 3.141592653-2.6-1.40.29 (TeX Live 2026)"
)
EXPECTED_BIBTEX_VERSION_LINE: Final = "BibTeX 0.99e (TeX Live 2026)"
TOOLCHAIN_TIMEOUT_SECONDS: Final = 30
COMPILE_TIMEOUT_SECONDS: Final = 600
_LATEXMK_OPTIONS: Final = (
    "-norc",
    "-pdf",
    "-pdflatex=pdflatex %O %S",
    "-interaction=nonstopmode",
    "-halt-on-error",
    "-file-line-error",
    "-outdir=../build/paper",
)
_FORBIDDEN_METADATA: Final = (
    b"/CreationDate",
    b"/ModDate",
    b"/PTEX.",
    b"/PTEX_",
)
_PASSTHROUGH_ENVIRONMENT: Final = (
    "COMSPEC",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "HOME",
    "LD_LIBRARY_PATH",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "WINDIR",
)


class PaperCompileError(RuntimeError):
    """Raised when compilation, reproducibility, or output validation fails."""


def build_commands() -> tuple[tuple[str, ...], ...]:
    """Return the registered supplement-first compilation commands."""

    return tuple(("latexmk", *_LATEXMK_OPTIONS, document) for document in DOCUMENTS)


def deterministic_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a copied subprocess environment with deterministic TeX metadata."""

    source = os.environ if base is None else base
    if not isinstance(source, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in source.items()
    ):
        raise TypeError("base environment must map strings to strings")
    environment = {
        variable: source[variable]
        for variable in _PASSTHROUGH_ENVIRONMENT
        if variable in source
    }
    environment.update(
        {
            "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
            "FORCE_SOURCE_DATE": "1",
            "TZ": "UTC",
            "LC_ALL": "C",
            "LANG": "C",
            "TEXMFHOME": str(BUILD_DIR / ".texmf/home"),
            "TEXMFCONFIG": str(BUILD_DIR / ".texmf/config"),
            "TEXMFVAR": str(BUILD_DIR / ".texmf/var"),
            "TEXMFCACHE": str(BUILD_DIR / ".texmf/var"),
            "TEXMFLOCAL": str(BUILD_DIR / ".texmf/local"),
        }
    )
    return environment


def _tail(output: str, lines: int = 30) -> str:
    return "\n".join(output.splitlines()[-lines:])


def _run_command(command: tuple[str, ...], *, timeout: int) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=PAPER_DIR,
            env=deterministic_environment(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise PaperCompileError(
            f"command timed out after {timeout}s: {' '.join(command)}"
        ) from error
    except subprocess.CalledProcessError as error:
        output = error.stdout if isinstance(error.stdout, str) else ""
        detail = f"\n{_tail(output)}" if output else ""
        raise PaperCompileError(
            f"command failed with exit code {error.returncode}: {' '.join(command)}"
            f"{detail}"
        ) from error
    except OSError as error:
        raise PaperCompileError(f"cannot execute {' '.join(command)}: {error}") from error
    return completed.stdout


def verify_toolchain() -> None:
    """Require the exact external TeX toolchain recorded in the Methods."""

    checks = (
        (("latexmk", "-version"), EXPECTED_LATEXMK_BANNER),
        (("pdflatex", "--version"), EXPECTED_PDFTEX_VERSION_LINE),
        (("bibtex", "--version"), EXPECTED_BIBTEX_VERSION_LINE),
    )
    for command, expected in checks:
        output = _run_command(command, timeout=TOOLCHAIN_TIMEOUT_SECONDS)
        if expected not in output:
            observed = output.splitlines()[0] if output.splitlines() else "<no output>"
            raise PaperCompileError(
                f"unsupported TeX toolchain for byte reproduction: {observed}; "
                f"expected {expected}"
            )


def _require_safe_build_location() -> None:
    try:
        relative = BUILD_DIR.relative_to(ROOT)
    except ValueError as error:
        raise PaperCompileError("build directory escapes the project root") from error
    if relative != Path("build/paper"):
        raise PaperCompileError("refusing to clean an unregistered build directory")

    current = ROOT
    for component in relative.parts:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise PaperCompileError(
                f"build directory traverses a symbolic link: {current}"
            )

    resolved_root = ROOT.resolve(strict=True)
    resolved_build = BUILD_DIR.resolve(strict=False)
    if resolved_root not in resolved_build.parents:
        raise PaperCompileError("resolved build directory escapes the project root")


def prepare_clean_build_dir() -> None:
    """Safely replace only the registered in-project generated build directory."""

    _require_safe_build_location()
    try:
        if os.path.lexists(BUILD_DIR):
            if not BUILD_DIR.is_dir():
                raise PaperCompileError("build output exists but is not a directory")
            build_device = BUILD_DIR.stat().st_dev
            if BUILD_DIR.is_mount() or build_device != BUILD_DIR.parent.stat().st_dev:
                raise PaperCompileError(
                    f"refusing to clean a mount point: {BUILD_DIR}"
                )
            for directory, names, files in os.walk(BUILD_DIR, followlinks=False):
                parent = Path(directory)
                for name in (*names, *files):
                    candidate = parent / name
                    if candidate.is_symlink():
                        continue
                    if candidate.is_mount() or candidate.stat().st_dev != build_device:
                        raise PaperCompileError(
                            f"refusing to cross a mount point: {candidate}"
                        )
            shutil.rmtree(BUILD_DIR)
        BUILD_DIR.mkdir(parents=True, exist_ok=False)
        for directory in ("home", "config", "var", "local"):
            (BUILD_DIR / ".texmf" / directory).mkdir(parents=True)
    except PaperCompileError:
        raise
    except OSError as error:
        raise PaperCompileError(f"cannot prepare clean build directory: {error}") from error
    _require_safe_build_location()


def validate_pdf_bytes(data: object, *, source: str) -> None:
    """Reject malformed PDFs and engine-generated nondeterministic metadata."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if not isinstance(source, str) or not source:
        raise TypeError("source must be a nonempty string")
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-1024:]:
        raise PaperCompileError(f"compiled output is not a complete PDF: {source}")
    markers = [marker.decode("ascii") for marker in _FORBIDDEN_METADATA if marker in data]
    if re.search(rb"/ID\s*\[", data) is not None:
        markers.append("/ID")
    if markers:
        raise PaperCompileError(
            f"nondeterministic PDF metadata in {source}: {', '.join(markers)}"
        )


def _read_validated_pdf(path: Path) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise PaperCompileError(f"compiled PDF is missing or unreadable: {path}") from error
    validate_pdf_bytes(data, source=str(path))
    return data


def _one_clean_build() -> dict[str, bytes]:
    prepare_clean_build_dir()
    outputs: dict[str, bytes] = {}
    for document, command in zip(DOCUMENTS, build_commands(), strict=True):
        _run_command(command, timeout=COMPILE_TIMEOUT_SECONDS)
        stem = Path(document).stem
        log = BUILD_DIR / f"{stem}.log"
        pdf = BUILD_DIR / f"{stem}.pdf"
        if stem == "supplement":
            aux = BUILD_DIR / "supplement.aux"
            if not aux.is_file() or aux.stat().st_size == 0:
                raise PaperCompileError(
                    "supplement label file is missing before main compilation"
                )
        try:
            notes = check_log_file(
                log,
                expected_engine=EXPECTED_PDFTEX_BANNER,
                expected_pdf=pdf,
            )
        except LatexLogError as error:
            raise PaperCompileError(str(error)) from error
        for note in notes:
            print(note)
        outputs[pdf.name] = _read_validated_pdf(pdf)
    return outputs


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compile_paper() -> tuple[Path, ...]:
    """Run two independent clean builds and retain only identical second outputs."""

    verify_toolchain()
    first = _one_clean_build()
    second = _one_clean_build()
    for document in DOCUMENTS:
        name = f"{Path(document).stem}.pdf"
        if first[name] != second[name]:
            prepare_clean_build_dir()
            raise PaperCompileError(f"{name} is not byte reproducible across clean builds")
        print(f"build/paper/{name}  sha256={_sha256(second[name])}")
    return tuple(BUILD_DIR / f"{Path(document).stem}.pdf" for document in DOCUMENTS)


def main(argv: Sequence[str] | None = None) -> int:
    """Compile the two registered paper documents under the locked toolchain."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        compile_paper()
    except (PaperCompileError, TypeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
