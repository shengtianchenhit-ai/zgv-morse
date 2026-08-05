from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.check_latex_log as check_latex_log  # noqa: E402
import scripts.compile_paper as compile_paper  # noqa: E402


CLEAN_LOG = r"""
This is pdfTeX, Version 3.141592653.
Output written on ../build/paper/main.pdf (22 pages, 852442 bytes).
Transcript written on ../build/paper/main.log.
"""


@pytest.mark.parametrize(
    ("fragment", "message"),
    (
        ("LaTeX Warning: Citation `missing' on page 1 undefined.", "undefined citation"),
        ("Package natbib Warning: Citation `missing' on page 2 undefined", "undefined citation"),
        ("LaTeX Warning: Reference `eq:missing' on page 3 undefined.", "undefined reference"),
        ("LaTeX Warning: There were undefined references.", "undefined reference"),
        ("LaTeX Warning: Label `eq:dup' multiply defined.", "multiply defined"),
        ("LaTeX Warning: There were multiply-defined labels.", "multiply defined"),
        ("! LaTeX Error: File `missing.sty' not found.", "missing file"),
        ("! Undefined control sequence.", "TeX error"),
        ("! Emergency stop.", "TeX error"),
        ("Fatal error occurred, no output PDF file produced!", "fatal TeX error"),
        ("No file main.bbl.", "missing bibliography output"),
        ("No file main.aux.", "missing auxiliary output"),
        ("LaTeX Warning: Label(s) may have changed. Rerun", "rerun required"),
        ("Rerun to get cross-references right.", "rerun required"),
        ("Package rerunfilecheck Warning: File `main.out' has changed.", "rerun required"),
        ("LaTeX Warning: Float too large for page by 4.0pt.", "LaTeX warning"),
        ("Package hyperref Warning: Token not allowed in a PDF string.", "package warning"),
        ("pdfTeX warning (dest): name{figure.1} does not exist", "engine warning"),
        ("LaTeX Font Warning: Font shape unavailable.", "LaTeX warning"),
        ("Please (re)run Biber on the file: main", "rerun required"),
        ("I can't find file `missing.tex'.", "missing file"),
        ("! pdftex.def Error: File `missing.pdf' not found", "missing file"),
        ("./main.tex:42: Undefined control sequence.", "TeX error"),
        ("No pages of output.", "fatal TeX error"),
        ("!pdfTeX error: corrupted object stream", "TeX error"),
    ),
)
def test_registered_log_failures_raise(fragment: str, message: str) -> None:
    with pytest.raises(check_latex_log.LatexLogError, match=message):
        check_latex_log.check_log_text(CLEAN_LOG + fragment, source="fixture.log")


def test_material_overfull_box_fails_at_registered_threshold() -> None:
    fragment = r"Overfull \hbox (4.0pt too wide) in paragraph at lines 10--12"

    with pytest.raises(check_latex_log.LatexLogError, match=r"4\.0.*3\.0"):
        check_latex_log.check_log_text(
            CLEAN_LOG + fragment,
            source="fixture.log",
            max_overfull_pt=3.0,
        )


def test_small_overfull_box_is_reported_without_failing() -> None:
    fragment = r"Overfull \hbox (2.0pt too wide) in paragraph at lines 4--5"

    notes = check_latex_log.check_log_text(
        CLEAN_LOG + fragment,
        source="fixture.log",
        max_overfull_pt=3.0,
    )

    assert len(notes) == 1
    assert "2.0pt" in notes[0]
    assert "fixture.log" in notes[0]


def test_exact_threshold_and_small_vbox_are_nonfatal() -> None:
    text = CLEAN_LOG + "\n".join(
        (
            r"Overfull \hbox (3.0pt too wide) in paragraph at lines 4--5",
            "Overfull \\vbox (\n.5pt too high) has occurred while \\output is active",
            "Package rerunfilecheck Info: File `main.out' has not changed.",
            r"Underfull \hbox (badness 10000) in paragraph at lines 6--7",
        )
    )

    notes = check_latex_log.check_log_text(text, source="fixture.log")

    assert len(notes) == 3


def test_fractionally_material_and_malformed_overfull_boxes_fail_closed() -> None:
    with pytest.raises(check_latex_log.LatexLogError, match="3.0001"):
        check_latex_log.check_log_text(
            CLEAN_LOG + r"Overfull \hbox (3.0001pt too wide)",
            source="fixture.log",
        )
    with pytest.raises(check_latex_log.LatexLogError, match="malformed overfull"):
        check_latex_log.check_log_text(
            CLEAN_LOG + r"Overfull \hbox (unknown amount) in paragraph",
            source="fixture.log",
        )


def test_truncated_log_without_successful_pdf_marker_fails() -> None:
    with pytest.raises(check_latex_log.LatexLogError, match="successful PDF output"):
        check_latex_log.check_log_text(
            "This is pdfTeX, Version 3.141592653.",
            source="truncated.log",
        )


def test_concatenated_logs_with_two_output_records_fail() -> None:
    with pytest.raises(check_latex_log.LatexLogError, match="multiple PDF output"):
        check_latex_log.check_log_text(CLEAN_LOG + CLEAN_LOG, source="joined.log")


def test_nonerror_file_location_note_is_allowed() -> None:
    notes = check_latex_log.check_log_text(
        CLEAN_LOG + "main.tex:42: informational note\n",
        source="fixture.log",
    )

    assert notes == ()


def test_expected_engine_and_wrapped_output_path_are_verified(tmp_path: Path) -> None:
    expected = tmp_path / "a-long-build-directory" / "main.pdf"
    output = str(expected)
    wrapped = output[: len(output) // 2] + "\n" + output[len(output) // 2 :]
    text = (
        compile_paper.EXPECTED_PDFTEX_BANNER
        + " (preloaded format=pdflatex)\n"
        + f"Output written on {wrapped} (1 page, 123 bytes).\n"
    )

    check_latex_log.check_log_text(
        text,
        source="main.log",
        expected_engine=compile_paper.EXPECTED_PDFTEX_BANNER,
        expected_pdf=expected,
    )

    with pytest.raises(check_latex_log.LatexLogError, match="unexpected TeX engine"):
        check_latex_log.check_log_text(
            text.replace("pdfTeX", "LuaTeX", 1),
            source="main.log",
            expected_engine=compile_paper.EXPECTED_PDFTEX_BANNER,
            expected_pdf=expected,
        )
    with pytest.raises(check_latex_log.LatexLogError, match="unexpected PDF output"):
        check_latex_log.check_log_text(
            text.replace("main.pdf", "unrelated.pdf"),
            source="main.log",
            expected_engine=compile_paper.EXPECTED_PDFTEX_BANNER,
            expected_pdf=expected,
        )


def test_log_file_must_exist_and_be_utf8(tmp_path: Path) -> None:
    with pytest.raises(check_latex_log.LatexLogError, match="missing log"):
        check_latex_log.check_log_file(tmp_path / "missing.log")

    invalid = tmp_path / "invalid.log"
    invalid.write_bytes(b"\xff\xfe")
    with pytest.raises(check_latex_log.LatexLogError, match="UTF-8"):
        check_latex_log.check_log_file(invalid)


def test_compile_order_commands_and_environment_are_registered() -> None:
    assert compile_paper.DOCUMENTS == ("supplement.tex", "main.tex")
    assert compile_paper.build_commands() == (
        (
            "latexmk",
            "-norc",
            "-pdf",
            "-pdflatex=pdflatex %O %S",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-outdir=../build/paper",
            "supplement.tex",
        ),
        (
            "latexmk",
            "-norc",
            "-pdf",
            "-pdflatex=pdflatex %O %S",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-outdir=../build/paper",
            "main.tex",
        ),
    )

    base = {
        "PATH": "/bin",
        "SOURCE_DATE_EPOCH": "wrong",
        "TEXINPUTS": "/unregistered/tex",
        "TEXMFCNF": "/unregistered/config",
        "TEXMF": "/unregistered/tree",
        "TEXPICTS": "/unregistered/pictures",
        "WEB2C": "/unregistered/web2c",
        "OSFONTDIR": "/unregistered/fonts",
        "PERL5LIB": "/unregistered/perl",
    }
    environment = compile_paper.deterministic_environment(base)
    assert base["SOURCE_DATE_EPOCH"] == "wrong"
    assert environment["PATH"] == "/bin"
    assert environment["SOURCE_DATE_EPOCH"] == "1783612800"
    assert environment["FORCE_SOURCE_DATE"] == "1"
    assert environment["TZ"] == "UTC"
    assert environment["LC_ALL"] == "C"
    assert environment["LANG"] == "C"
    assert environment["TEXMFHOME"] == str(
        compile_paper.BUILD_DIR / ".texmf/home"
    )
    assert environment["TEXMFCONFIG"] == str(
        compile_paper.BUILD_DIR / ".texmf/config"
    )
    assert environment["TEXMFVAR"] == str(compile_paper.BUILD_DIR / ".texmf/var")
    assert environment["TEXMFLOCAL"] == str(
        compile_paper.BUILD_DIR / ".texmf/local"
    )
    for variable in (
        "TEXINPUTS",
        "TEXMFCNF",
        "TEXMF",
        "TEXPICTS",
        "WEB2C",
        "OSFONTDIR",
        "PERL5LIB",
    ):
        assert variable not in environment


def _fake_complete_compile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    change_second_build: bool = False,
) -> tuple[list[tuple[tuple[str, ...], dict[str, object]]], Path]:
    root = tmp_path / "project"
    paper = root / "paper"
    build = root / "build/paper"
    paper.mkdir(parents=True)
    build.mkdir(parents=True)
    (build / "stale.pdf").write_bytes(b"stale output")
    monkeypatch.setattr(compile_paper, "ROOT", root)
    monkeypatch.setattr(compile_paper, "PAPER_DIR", paper)
    monkeypatch.setattr(compile_paper, "BUILD_DIR", build)
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    document_calls = 0

    def fake_run(command: tuple[str, ...], **kwargs: object):
        nonlocal document_calls
        calls.append((tuple(command), kwargs))
        if tuple(command) == ("latexmk", "-version"):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=compile_paper.EXPECTED_LATEXMK_BANNER + "\n",
            )
        if tuple(command) == ("pdflatex", "--version"):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=compile_paper.EXPECTED_PDFTEX_VERSION_LINE + "\n",
            )
        if tuple(command) == ("bibtex", "--version"):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=compile_paper.EXPECTED_BIBTEX_VERSION_LINE + "\n",
            )

        document = Path(command[-1])
        stem = document.stem
        cycle = document_calls // len(compile_paper.DOCUMENTS)
        suffix = b"-changed" if change_second_build and cycle == 1 else b""
        payload = b"%PDF-1.7\nfixture" + suffix + b"\n%%EOF\n"
        build.mkdir(parents=True, exist_ok=True)
        pdf = build / f"{stem}.pdf"
        pdf.write_bytes(payload)
        if stem == "supplement":
            (build / "supplement.aux").write_text("fixture", encoding="utf-8")
        (build / f"{stem}.log").write_text(
            compile_paper.EXPECTED_PDFTEX_BANNER
            + " (preloaded format=pdflatex)\n"
            + f"Output written on {pdf} (1 page, {len(payload)} bytes).\n",
            encoding="utf-8",
        )
        document_calls += 1
        return subprocess.CompletedProcess(command, 0, stdout="compiled\n")

    monkeypatch.setattr(compile_paper.subprocess, "run", fake_run)
    return calls, build


def test_compiler_runs_two_clean_locked_builds_and_compares_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls, build = _fake_complete_compile(monkeypatch, tmp_path)

    outputs = compile_paper.compile_paper()

    assert outputs == (build / "supplement.pdf", build / "main.pdf")
    assert not (build / "stale.pdf").exists()
    assert [call[0] for call in calls] == [
        ("latexmk", "-version"),
        ("pdflatex", "--version"),
        ("bibtex", "--version"),
        *compile_paper.build_commands(),
        *compile_paper.build_commands(),
    ]
    for _command, kwargs in calls:
        assert kwargs["cwd"] == compile_paper.PAPER_DIR
        assert kwargs["timeout"] <= compile_paper.COMPILE_TIMEOUT_SECONDS
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["SOURCE_DATE_EPOCH"] == compile_paper.SOURCE_DATE_EPOCH


def test_compiler_rejects_nonidentical_clean_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _calls, build = _fake_complete_compile(
        monkeypatch,
        tmp_path,
        change_second_build=True,
    )

    with pytest.raises(compile_paper.PaperCompileError, match="not byte reproducible"):
        compile_paper.compile_paper()
    assert not tuple(build.glob("*.pdf"))


def test_clean_build_rejects_symlinked_output_ancestor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    external = tmp_path / "external"
    root.mkdir()
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (root / "build").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(compile_paper, "ROOT", root)
    monkeypatch.setattr(compile_paper, "BUILD_DIR", root / "build/paper")

    with pytest.raises(compile_paper.PaperCompileError, match="symbolic link"):
        compile_paper.prepare_clean_build_dir()
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_clean_build_rejects_nested_mount_point(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    mounted = root / "build/paper/mounted"
    mounted.mkdir(parents=True)
    sentinel = mounted / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    real_is_mount = Path.is_mount

    def fake_is_mount(path: Path) -> bool:
        return path == mounted or real_is_mount(path)

    monkeypatch.setattr(Path, "is_mount", fake_is_mount)
    monkeypatch.setattr(compile_paper, "ROOT", root)
    monkeypatch.setattr(compile_paper, "BUILD_DIR", root / "build/paper")

    with pytest.raises(compile_paper.PaperCompileError, match="mount point"):
        compile_paper.prepare_clean_build_dir()
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "marker",
    (b"/CreationDate", b"/ModDate", b"/ID [", b"/PTEX.Fullbanner"),
)
def test_pdf_validator_rejects_nondeterministic_metadata(marker: bytes) -> None:
    with pytest.raises(compile_paper.PaperCompileError, match="metadata"):
        compile_paper.validate_pdf_bytes(
            b"%PDF-1.7\n" + marker + b"\n%%EOF\n",
            source="fixture.pdf",
        )


def test_shared_macros_suppress_engine_generated_pdf_metadata() -> None:
    macros = (ROOT / "paper/shared_macros.tex").read_text(encoding="utf-8")
    for assignment in (
        r"\pdfinfoomitdate=1",
        r"\pdftrailerid{}",
        r"\pdfsuppressptexinfo=-1",
    ):
        assert assignment in macros
