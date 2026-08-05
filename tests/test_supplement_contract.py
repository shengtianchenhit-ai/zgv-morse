from __future__ import annotations

from collections import Counter
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MAIN = PAPER / "main.tex"
METHODS = PAPER / "sections/07_methods.tex"
STATEMENTS = PAPER / "sections/09_reproducibility_statements.tex"
PYTHON_VERSION = ROOT / ".python-version"
SUPPLEMENT = PAPER / "supplement.tex"
SUPPLEMENT_SECTIONS = tuple(
    PAPER / f"supplement/{name}.tex"
    for name in (
        "01_exact_elasticity",
        "02_perturbation_proofs",
        "03_asymptotics",
        "04_numerical_methods",
        "05_convergence_and_robustness",
    )
)
DERIVATIONS = tuple(sorted((ROOT / "docs/derivations").glob("*.tex")))
SUPPLEMENTARY_FIGURES = tuple(
    f"figure_s{number:02d}_{suffix}"
    for number, suffix in enumerate(
        (
            "polynomial_two_element",
            "quadrature_phase",
            "mode_tracking",
            "fd_convergence",
            "source_window_sensitivity",
            "silicon_stress_test",
        ),
        start=1,
    )
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tex_sources() -> tuple[Path, ...]:
    return (
        MAIN,
        SUPPLEMENT,
        METHODS,
        STATEMENTS,
        *SUPPLEMENT_SECTIONS,
        *DERIVATIONS,
        *tuple(sorted((PAPER / "sections").glob("*.tex"))),
    )


def _commands(text: str, command: str) -> list[str]:
    return re.findall(rf"\\{command}\{{([^}}]+)\}}", text)


def test_complete_supplement_file_set_exists():
    required = (PYTHON_VERSION, METHODS, STATEMENTS, SUPPLEMENT, *SUPPLEMENT_SECTIONS)
    assert all(path.is_file() for path in required)
    assert _read(PYTHON_VERSION).strip() == "3.12.13"


def test_supplement_inputs_each_canonical_derivation_once_without_copying_it():
    supplement_tree = "\n".join(_read(path) for path in (SUPPLEMENT, *SUPPLEMENT_SECTIONS))
    for derivation in DERIVATIONS:
        relative = f"../docs/derivations/{derivation.name}"
        assert supplement_tree.count(rf"\input{{{relative}}}") == 1
        distinctive = next(
            line.strip()
            for line in _read(derivation).splitlines()
            if len(line.strip()) > 70 and not line.lstrip().startswith("%")
        )
        assert distinctive not in supplement_tree


def test_main_imports_supplement_labels_and_reproducibility_statements():
    text = _read(MAIN)
    assert r"\usepackage{xr-hyper}" in text
    assert r"\externaldocument{../build/paper/supplement}[supplement.pdf]" in text
    assert text.count(r"\input{sections/09_reproducibility_statements.tex}") == 1


def test_methods_are_executable_detail_not_a_heading_stub():
    text = _read(METHODS)
    required_phrases = (
        "governing equations and nondimensionalization",
        "Rayleigh--Lamb",
        "Legendre--Gauss--Lobatto",
        "traction-free",
        "mode tracking",
        "eigengap",
        "generalized-eigenvalue sensitivity",
        "differentiated-mode",
        "critical-point search",
        "annular",
        "endpoint-free trapezoidal",
        "composite Simpson",
        "fit masks",
        "phase-discrepancy",
        "config/reference.yaml",
        "uv.lock",
        "uv sync --python 3.12.13 --frozen --all-extras",
        "uv run python scripts/reproduce_all.py",
        "--profile full --skip-paper",
        "supplement.tex",
        "main.tex",
        "Python 3.12.13",
        "NumPy 2.5.1",
        "SciPy 1.18.0",
    )
    lowered = re.sub(r"\s+", " ", text.lower())
    for phrase in required_phrases:
        assert phrase.lower() in lowered, f"Methods omit {phrase}"
    assert len(re.findall(r"\b\w+\b", re.sub(r"\\[A-Za-z]+", " ", text))) >= 900


def test_data_and_code_statements_are_truthful_and_deposition_neutral():
    text = _read(STATEMENTS)
    lowered = re.sub(r"\s+", " ", text.lower())
    assert "data availability" in lowered
    assert "code availability" in lowered
    assert "accompanying source tree" in lowered
    assert "machine-readable source data" in lowered
    assert "sha-256" in lowered
    assert "external accession" in lowered
    assert "pre-deposition stage" in lowered
    assert "release-wide checksum manifest is available" in lowered
    assert not re.search(r"\bauthor\b", lowered)
    assert not re.search(r"\baffiliation\b", lowered)
    assert not re.search(r"\bfunding\b", lowered)
    assert not re.search(r"10\.5281/zenodo|doi:\s*(?:tbd|todo|xxx)", lowered)


def test_supplement_contains_both_generated_tables_and_six_figure_sources():
    text = "\n".join(_read(path) for path in (SUPPLEMENT, *SUPPLEMENT_SECTIONS))
    for table in ("table_s01_convergence.tex", "table_s02_parameters.tex"):
        assert text.count(table) == 1
    for stem in SUPPLEMENTARY_FIGURES:
        assert text.count(f"../figures/supplementary/{stem}.pdf") == 1
    assert set(_commands(text, "label")) >= {
        "tab:supplement-convergence",
        "tab:supplement-parameters",
        *(f"fig:supplement-{number:02d}" for number in range(1, 7)),
    }


def test_all_labels_are_defined_exactly_once_across_the_two_documents():
    counts: Counter[str] = Counter()
    seen_paths: set[Path] = set()
    for path in _tex_sources():
        if path in seen_paths:
            continue
        seen_paths.add(path)
        counts.update(_commands(_read(path), "label"))
    duplicates = {label: count for label, count in counts.items() if count != 1}
    assert not duplicates


def test_every_internal_reference_resolves_in_main_or_supplement_source():
    labels: set[str] = set()
    references: set[str] = set()
    seen_paths: set[Path] = set()
    for path in _tex_sources():
        if path in seen_paths:
            continue
        seen_paths.add(path)
        text = _read(path)
        labels.update(_commands(text, "label"))
        references.update(_commands(text, "ref"))
        references.update(_commands(text, "eqref"))
    missing = sorted(references - labels)
    assert not missing


def test_supplement_preamble_defines_theorem_and_reproducible_paths():
    text = _read(SUPPLEMENT)
    assert r"\usepackage{amsthm}" in text
    assert r"\usepackage{mathrsfs}" in text
    assert r"\newtheorem{theorem}{Theorem}[section]" in text
    assert all(
        text.count(rf"\input{{supplement/{path.stem}.tex}}") == 1
        for path in SUPPLEMENT_SECTIONS
    )
    assert r"\input{generated/results_macros.tex}" in text
    assert r"\bibliography{references}" not in text
