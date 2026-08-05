from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper/main.tex"
SUPPLEMENT = ROOT / "paper/supplement.tex"
README = ROOT / "README.md"
ABSTRACT = ROOT / "paper/sections/00_abstract.tex"
CAPTIONS = ROOT / "paper/figure_captions.tex"
SPECTRAL_DERIVATION = ROOT / "docs/derivations/04_spectral_numerics.tex"
FIGURE_CONTRACTS = ROOT / "docs/figures/figure_contracts.md"
MAIN_FILES = {
    "FigureOneCaption": ROOT / "paper/sections/02_isotropic_ring.tex",
    "FigureTwoCaption": ROOT / "paper/sections/02_isotropic_ring.tex",
    "FigureThreeCaption": ROOT / "paper/sections/03_morse_unfolding.tex",
    "FigureFourCaption": ROOT / "paper/sections/03_morse_unfolding.tex",
    "FigureFiveCaption": ROOT / "paper/sections/03_morse_unfolding.tex",
    "FigureSixCaption": ROOT / "paper/sections/04_temporal_crossover.tex",
}
SUPPLEMENT_FILE = ROOT / "paper/supplement/05_convergence_and_robustness.tex"
SUPPLEMENT_MACROS = tuple(f"SupplementaryFigure{word}Caption" for word in (
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
))
PANEL_COUNTS = {
    "FigureOneCaption": 3,
    "FigureTwoCaption": 4,
    "FigureThreeCaption": 4,
    "FigureFourCaption": 4,
    "FigureFiveCaption": 4,
    "FigureSixCaption": 6,
    "SupplementaryFigureOneCaption": 3,
    "SupplementaryFigureTwoCaption": 3,
    "SupplementaryFigureThreeCaption": 3,
    "SupplementaryFigureFourCaption": 3,
    "SupplementaryFigureFiveCaption": 2,
    "SupplementaryFigureSixCaption": 3,
}
SOURCE_DIRECTORIES = {
    **{f"Figure{word}Caption": f"data/source_data/figure_{number:02d}" for number, word in enumerate(
        ("One", "Two", "Three", "Four", "Five", "Six"), start=1
    )},
    **{name: "data/source_data/supplementary" for name in SUPPLEMENT_MACROS},
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _macro_body(text: str, name: str) -> str:
    token = rf"\newcommand{{\{name}}}{{"
    assert text.count(token) == 1, f"expected one definition of {name}"
    start = text.index(token) + len(token)
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start:index]
    raise AssertionError(f"unterminated caption macro {name}")


def _plain_words(text: str) -> list[str]:
    text = re.sub(r"%.*", " ", text)
    text = re.sub(r"\\(?:begin|end)\{[^{}]+\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    return re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|[A-Za-z]+_[A-Za-z0-9]+", text)


def test_abstract_and_caption_registry_exist():
    assert ABSTRACT.is_file()
    assert CAPTIONS.is_file()


def test_abstract_is_last_draft_evidence_bounded_and_compact():
    text = _read(ABSTRACT)
    searchable = re.sub(r"\s+", " ", text)
    assert text.count(r"\begin{abstract}") == 1
    assert text.count(r"\end{abstract}") == 1
    words = _plain_words(text)
    assert 150 <= len(words) <= 210, len(words)
    required_macros = {"MorseMinimumCount", "MorseSaddleCount", "MaxPhaseError"}
    assert required_macros <= set(re.findall(r"\\([A-Za-z]+)", text))
    assert r"t^{-1/2}" in text
    assert r"J_0" in text
    assert r"t^{-1}" in text
    assert "shifted carrier" in searchable.lower()
    assert r"\Omega_0+\varepsilon V_0" in text
    assert "noncommuting" in searchable.lower()
    assert "inter-resolution phase-discrepancy estimator" in searchable.lower()
    assert "continuum error bound" in searchable.lower()
    assert "without fitted alignment" in searchable.lower()
    assert "dated literature audit" in searchable.lower()
    assert "eight resolved, refinement-stable roots" in searchable.lower()
    assert r"\MaxPhaseError\,\mathrm{rad}" in text
    assert re.search(
        r"(?:selected|tracked).{0,80}branch|branch.{0,80}(?:selected|tracked)",
        searchable,
        re.I,
    )
    assert re.search(r"local annulus|registered annulus", searchable, re.I)
    assert re.search(r"linear|lossless|infinite plate", searchable, re.I)
    banned = (
        "first-ever",
        "for the first time",
        "unprecedented",
        "universal",
        "experimentally",
        "observed experimentally",
        "topological phase transition",
    )
    assert not any(term in text.lower() for term in banned)
    without_macros = re.sub(r"\\[A-Za-z]+", "", text)
    assert not re.search(r"(?<![A-Za-z0-9_])[-+]?\d+\.\d+", without_macros)


def test_title_foregrounds_the_coefficient_and_phase_resolved_bridge() -> None:
    title = re.search(r"\\title\{([^{}]+)\}", _read(MAIN))
    assert title is not None
    paper_title = title.group(1)
    lowered = paper_title.lower()
    assert "coefficient- and phase-resolved" in lowered
    assert "noncommuting limits" in lowered
    assert "lamb-wave zgv" in lowered

    supplement = re.sub(r"\s+", " ", _read(SUPPLEMENT))
    readme = re.sub(r"\s+", " ", _read(README).replace("*", ""))
    assert paper_title in supplement
    assert paper_title in readme


def test_all_twelve_caption_macros_are_defined_and_used_exactly_once():
    registry = _read(CAPTIONS)
    all_macros = (*MAIN_FILES, *SUPPLEMENT_MACROS)
    assert len(all_macros) == 12
    for name in all_macros:
        _macro_body(registry, name)
        usage = rf"\caption{{\{name}}}"
        corpus = "\n".join(
            _read(path)
            for path in {*MAIN_FILES.values(), SUPPLEMENT_FILE}
        )
        assert corpus.count(usage) == 1, f"caption macro usage mismatch: {name}"


def test_captions_are_panel_complete_self_contained_and_source_anchored():
    registry = _read(CAPTIONS)
    for name, count in PANEL_COUNTS.items():
        body = _macro_body(registry, name)
        searchable = re.sub(r"\s+", " ", body)
        assert len(_plain_words(body)) >= 65, f"caption too short: {name}"
        for index in range(count):
            letter = chr(ord("a") + index)
            assert rf"\textbf{{{letter}}}" in body, f"{name} omits panel {letter}"
        assert "Source data" in searchable
        # Captions must print the repository path; TeX escapes its underscores.
        assert SOURCE_DIRECTORIES[name] in searchable.replace(r"\_", "_")
        assert re.search(r"full-wave|exact|perturb|computed|diagnostic", searchable, re.I)
        assert re.search(r"solid|dashed|dotted|marker|symbol|colour|color|curve|line", searchable, re.I)


def test_main_captions_define_model_level_normalization_and_error_semantics():
    registry = _read(CAPTIONS)
    joined = re.sub(
        r"\s+", " ", "\n".join(_macro_body(registry, name) for name in MAIN_FILES)
    )
    for phrase in (
        "exact Rayleigh--Lamb",
        "full-wave",
        "perturbation",
        "normalized",
        "absolute complex error",
        "without fitted alignment",
        "declared",
    ):
        assert phrase.lower() in joined.lower()

    figure_six = re.sub(r"\s+", " ", _macro_body(registry, "FigureSixCaption"))
    figure_two = re.sub(r"\s+", " ", _macro_body(registry, "FigureTwoCaption"))
    assert "squared-displacement proxy" in figure_two.lower()
    assert re.search(r"neither.{0,80}strain-energy", figure_two, re.I)
    assert "uniform transition" in figure_six.lower()
    assert "fixed-anisotropy" in figure_six.lower()
    assert "early fit" in figure_six.lower()
    assert "late fit" in figure_six.lower()
    assert "cancellation" in figure_six.lower()
    assert "shifted carrier" in figure_six.lower()
    assert r"\Omega_0+\varepsilon V_0" in figure_six
    assert all(token in figure_six for token in ("1500", "10200", "0.30"))
    assert "two numerical asymptotic slices" in figure_six.lower()
    assert "not one numerical time trajectory" in figure_six.lower()
    assert "proved growing" in figure_six.lower()
    assert "theory-centred consistency diagnostic" in figure_six.lower()
    assert "without fitted alignment" in figure_six.lower()
    assert len(_plain_words(figure_six)) <= 285

    figure_one = re.sub(r"\s+", " ", _macro_body(registry, "FigureOneCaption"))
    figure_four = re.sub(r"\s+", " ", _macro_body(registry, "FigureFourCaption"))
    assert "mechanism overview" in figure_one.lower()
    assert "does not carry the numerical realization claim" in figure_one.lower()
    assert "full-wave numerical realization" in figure_four.lower()
    assert "eight resolved roots" in figure_four.lower()


def test_current_delivery_never_labels_squared_displacement_as_energy_density():
    for path in (SPECTRAL_DERIVATION, FIGURE_CONTRACTS):
        text = _read(path)
        assert not re.search(r"(?:strain-)?energy density", text, re.I), path
        assert "squared-displacement proxy" in text, path


def test_supplementary_captions_state_thresholds_and_scope_boundaries():
    registry = _read(CAPTIONS)
    joined = re.sub(
        r"\s+", " ",
        "\n".join(_macro_body(registry, name) for name in SUPPLEMENT_MACROS),
    )
    for phrase in (
        "phase-discrepancy",
        "do not bound the unknown continuum error",
        "relative eigengap",
        "finite-difference",
        "source independence",
        "finite-anisotropy",
        "weak-anisotropy theorem",
    ):
        assert phrase.lower() in joined.lower()
