from pathlib import Path
import subprocess
import sys

from pypdf import PdfWriter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.compile_paper as reference_compile  # noqa: E402
import scripts.compile_paper_semantic as semantic_compile  # noqa: E402


def _write_valid_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72.0, height=72.0)
    with path.open("wb") as handle:
        writer.write(handle)


def test_semantic_compiler_accepts_nonreference_engine_and_validates_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repository"
    paper = root / "paper"
    build = root / "build/paper"
    paper.mkdir(parents=True)
    for document in reference_compile.DOCUMENTS:
        (paper / document).write_text("document", encoding="utf-8")

    monkeypatch.setattr(reference_compile, "ROOT", root)
    monkeypatch.setattr(reference_compile, "PAPER_DIR", paper)
    monkeypatch.setattr(reference_compile, "BUILD_DIR", build)
    monkeypatch.setattr(semantic_compile, "PAPER_DIR", paper)
    monkeypatch.setattr(semantic_compile, "BUILD_DIR", build)

    calls: list[tuple[str, ...]] = []

    def fake_run(command, **kwargs):
        command = tuple(command)
        calls.append(command)
        assert kwargs["cwd"] == paper
        assert kwargs["env"]["SOURCE_DATE_EPOCH"] == reference_compile.SOURCE_DATE_EPOCH
        assert kwargs["check"] is True
        assert kwargs["timeout"] <= reference_compile.COMPILE_TIMEOUT_SECONDS
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.STDOUT
        if command[-1] == "supplement.tex":
            (build / "supplement.aux").write_text("labels", encoding="utf-8")
        if command[-1].endswith(".tex"):
            stem = Path(command[-1]).stem
            pdf = build / f"{stem}.pdf"
            _write_valid_pdf(pdf)
            payload = pdf.read_bytes()
            (build / f"{stem}.log").write_text(
                "This is pdfTeX, Version 3.141592653 (TeX Live nonreference)\n"
                f"Output written on {pdf} (1 page, {len(payload)} bytes).\n",
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(semantic_compile.subprocess, "run", fake_run)

    outputs = semantic_compile.compile_paper_semantic()

    assert calls == list(reference_compile.build_commands())
    assert outputs == (build / "supplement.pdf", build / "main.pdf")
    assert all(path.is_file() for path in outputs)


def test_semantic_compiler_retains_strict_log_gate(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repository"
    paper = root / "paper"
    build = root / "build/paper"
    paper.mkdir(parents=True)
    for document in reference_compile.DOCUMENTS:
        (paper / document).write_text("document", encoding="utf-8")

    monkeypatch.setattr(reference_compile, "ROOT", root)
    monkeypatch.setattr(reference_compile, "PAPER_DIR", paper)
    monkeypatch.setattr(reference_compile, "BUILD_DIR", build)
    monkeypatch.setattr(semantic_compile, "PAPER_DIR", paper)
    monkeypatch.setattr(semantic_compile, "BUILD_DIR", build)

    def fake_run(command, **kwargs):
        stem = Path(command[-1]).stem
        pdf = build / f"{stem}.pdf"
        pdf.write_bytes(b"%PDF-1.7\nsemantic output\n%%EOF\n")
        (build / f"{stem}.log").write_text(
            "This is pdfTeX, Version 3.141592653 (TeX Live nonreference)\n"
            "LaTeX Warning: There were undefined references.\n"
            f"Output written on {pdf} (1 page, 32 bytes).\n",
            encoding="utf-8",
        )
        if stem == "supplement":
            (build / "supplement.aux").write_text("labels", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(semantic_compile.subprocess, "run", fake_run)

    try:
        semantic_compile.compile_paper_semantic()
    except reference_compile.PaperCompileError as error:
        assert "undefined reference" in str(error)
    else:
        raise AssertionError("semantic compile accepted a broken final LaTeX log")


def test_semantic_compiler_rejects_a_pdf_without_a_valid_cross_reference_table(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repository"
    paper = root / "paper"
    build = root / "build/paper"
    paper.mkdir(parents=True)
    for document in reference_compile.DOCUMENTS:
        (paper / document).write_text("document", encoding="utf-8")

    monkeypatch.setattr(reference_compile, "ROOT", root)
    monkeypatch.setattr(reference_compile, "PAPER_DIR", paper)
    monkeypatch.setattr(reference_compile, "BUILD_DIR", build)
    monkeypatch.setattr(semantic_compile, "PAPER_DIR", paper)
    monkeypatch.setattr(semantic_compile, "BUILD_DIR", build)

    def fake_run(command, **kwargs):
        stem = Path(command[-1]).stem
        pdf = build / f"{stem}.pdf"
        payload = b"%PDF-1.7\nnot a structurally valid PDF\n%%EOF\n"
        pdf.write_bytes(payload)
        (build / f"{stem}.log").write_text(
            "This is pdfTeX, Version 3.141592653 (TeX Live nonreference)\n"
            f"Output written on {pdf} (1 page, {len(payload)} bytes).\n",
            encoding="utf-8",
        )
        if stem == "supplement":
            (build / "supplement.aux").write_text("labels", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(semantic_compile.subprocess, "run", fake_run)

    try:
        semantic_compile.compile_paper_semantic()
    except reference_compile.PaperCompileError as error:
        assert "invalid PDF structure" in str(error)
    else:
        raise AssertionError("semantic compile accepted a structurally invalid PDF")


def test_semantic_compiler_is_directly_executable_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/compile_paper_semantic.py", "--help"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout
    assert "strict semantic gates on a nonreference TeX host" in completed.stdout
