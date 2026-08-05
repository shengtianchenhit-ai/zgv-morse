from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_documents_the_executable_reproduction_contract() -> None:
    text = README.read_text(encoding="utf-8")
    required = (
        "uv sync --python 3.12.13 --frozen --all-extras",
        "uv run python scripts/reproduce_all.py --profile smoke --skip-paper",
        "uv run python scripts/reproduce_all.py --profile full --skip-paper",
        "uv run python scripts/export_manuscript_values.py",
        "uv run python scripts/compile_paper.py",
        "uv run python scripts/compile_paper_semantic.py",
        "data/generated/",
        "data/provenance_manifest.json",
        "figures/main/",
        "figures/supplementary/",
        "data/source_data/",
        "build/paper/main.pdf",
        "build/paper/supplement.pdf",
    )
    for phrase in required:
        assert phrase in text


def test_readme_marks_profile_mutation_and_pdf_reproduction_scope() -> None:
    text = README.read_text(encoding="utf-8")
    lowered = " ".join(text.lower().split())

    assert "overwrites the canonical" in lowered
    assert "smoke profile is not manuscript evidence" in lowered
    assert "byte-identical" in lowered
    assert "reference environment" in lowered
    assert "semantic" in lowered
    assert "macos 26.4.1" in lowered
    assert "arm64" in lowered
    assert "tex live 2026" in lowered
