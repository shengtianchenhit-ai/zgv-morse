from __future__ import annotations

from collections import Counter
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "paper/sections"
DERIVATION_DIR = ROOT / "docs/derivations"

SECTION_CONTRACT = {
    "02_isotropic_ring.tex": (
        "Exact Rayleigh--Lamb reference",
        "Morse--Bott geometry",
        "Independent spectral verification",
    ),
    "03_morse_unfolding.tex": (
        "Full-elastic angular sensitivities",
        "Normal form and splitting theorem",
        "Eight resolved critical points",
        "Scaling and sign checks",
    ),
    "04_temporal_crossover.tex": (
        "Joint-limit Bessel law",
        "Overlap with isolated stationary points",
        "Fixed-anisotropy Morse endpoint",
        "Frequency separation and modulation rate",
    ),
    "05_numerical_verification.tex": (
        "Convergence and resolution checks",
        "Response and phase accuracy",
        "Source and window robustness",
        "Silicon stress test",
    ),
}

CLAIM_SECTION = {
    "C1": "02_isotropic_ring.tex",
    "C2": "03_morse_unfolding.tex",
    "C3": "03_morse_unfolding.tex",
    "C4": "03_morse_unfolding.tex",
    "C5": "03_morse_unfolding.tex",
    "C6": "04_temporal_crossover.tex",
    "C7": "04_temporal_crossover.tex",
}

FIGURE_CONTRACT = {
    "fig:geometry-mechanism": (
        "02_isotropic_ring.tex",
        "../figures/main/figure_01_geometry_mechanism.pdf",
    ),
    "fig:isotropic-zgv": (
        "02_isotropic_ring.tex",
        "../figures/main/figure_02_isotropic_zgv.pdf",
    ),
    "fig:angular-sensitivity": (
        "03_morse_unfolding.tex",
        "../figures/main/figure_03_angular_sensitivity.pdf",
    ),
    "fig:morse-points": (
        "03_morse_unfolding.tex",
        "../figures/main/figure_04_morse_points.pdf",
    ),
    "fig:perturbation-scaling": (
        "03_morse_unfolding.tex",
        "../figures/main/figure_05_perturbation_scaling.pdf",
    ),
    "fig:decay-crossover": (
        "04_temporal_crossover.tex",
        "../figures/main/figure_06_decay_crossover.pdf",
    ),
}

REQUIRED_THEORY_REFS = {
    "02_isotropic_ring.tex": {
        "thm:morse-bott-ring",
        "eq:zgv-ring-hessian",
        "eq:morse-bott-kernel-and-normal-form",
    },
    "03_morse_unfolding.tex": {
        "eq:general-frequency-sensitivity",
        "eq:complete-radial-frequency-sensitivity",
        "eq:cubic-vfour",
        "thm:cubic-morse-splitting",
        "eq:cubic-eight-critical-points",
        "eq:critical-point-uncertainty",
        "eq:annular-index-certificate",
        "eq:radial-shift",
        "eq:cubic-minimum-saddle-separation",
    },
    "04_temporal_crossover.tex": {
        "thm:uniform-bessel-crossover",
        "thm:growing-bessel-morse-overlap",
        "eq:uniform-response",
        "thm:fixed-anisotropy-decay",
        "eq:morse-stationary-sum",
        "eq:critical-frequency-separation",
        "eq:signed-modulation-rate",
    },
    "05_numerical_verification.tex": {
        "eq:normalized-eigen-residual",
        "eq:eigengap-rejection",
        "eq:annular-index-certificate",
        "eq:polar-green-quadrature",
        "eq:phase-error-certificate",
    },
}

GENERATED_VALUE_MACROS = {
    "ZGVKappa",
    "ZGVOmega",
    "ZGVCurvature",
    "VFour",
    "SplittingSlope",
    "RemainderSlope",
    "EarlyDecaySlope",
    "LateDecaySlope",
    "MaxPhaseError",
    "MorseMinimumCount",
    "MorseSaddleCount",
}


def _read_results() -> dict[str, str]:
    return {
        name: (RESULTS_DIR / name).read_text(encoding="utf-8")
        for name in SECTION_CONTRACT
    }


def _without_comments(text: str) -> str:
    return "\n".join(line.split("%", maxsplit=1)[0] for line in text.splitlines())


def _prose_words(text: str) -> list[str]:
    text = re.sub(r"\\(?:label|ref|eqref|autoref|cref|Cref|cite\w*)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:includegraphics)(?:\[[^]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", text)
    text = re.sub(r"\$[^$]*\$|\\\[[\s\S]*?\\\]", " ", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    return re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text)


def _subsection_bodies(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"\\subsection\{([^{}]+)\}", text))
    return [
        (
            match.group(1),
            text[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None],
        )
        for index, match in enumerate(matches)
    ]


def _first_prose_paragraph(body: str) -> str:
    uncommented = _without_comments(body).strip()
    for paragraph in re.split(r"\n\s*\n", uncommented):
        paragraph = paragraph.strip()
        if paragraph and not paragraph.startswith((r"\begin", r"\[", "$$")):
            return paragraph
    return ""


def _references(text: str, prefix: str) -> set[str]:
    labels: set[str] = set()
    for command in re.finditer(r"\\(?:ref|eqref|autoref|cref|Cref)\{([^{}]+)\}", text):
        labels.update(item.strip() for item in command.group(1).split(","))
    return {label for label in labels if label.startswith(prefix)}


def test_results_have_fixed_subsections_and_substantive_bounded_opening_paragraphs():
    boundary_language = re.compile(
        r"\b(?:within|restricted|registered|local|under|provided|only|fixed|joint-limit|"
        r"does not|not a|rather than|tracked)\b",
        re.IGNORECASE,
    )
    for name, expected_titles in SECTION_CONTRACT.items():
        text = (RESULTS_DIR / name).read_text(encoding="utf-8")
        bodies = _subsection_bodies(text)
        assert tuple(title for title, _body in bodies) == expected_titles
        assert len(_prose_words(_without_comments(text))) >= 250, f"{name} is still a skeleton"
        for title, body in bodies:
            assert len(_prose_words(_without_comments(body))) >= 60, (
                f"{name}: subsection {title!r} needs substantive Results prose"
            )
            first = _first_prose_paragraph(body)
            count = len(_prose_words(first))
            assert 30 <= count <= 180, (
                f"{name}: first paragraph of {title!r} must contain 30--180 prose words"
            )
            assert boundary_language.search(first), (
                f"{name}: first paragraph of {title!r} must state a scope boundary"
            )


def test_c1_through_c7_markers_occur_once_in_their_evidence_sections():
    texts = _read_results()
    occurrences: list[tuple[str, str]] = []
    marker = re.compile(r"(?m)^\s*%\s*Claim\s+(C[1-7])\s*$")
    for name, text in texts.items():
        occurrences.extend((claim, name) for claim in marker.findall(text))
    counts = Counter(claim for claim, _name in occurrences)
    assert counts == Counter({f"C{index}": 1 for index in range(1, 8)})
    assert all(CLAIM_SECTION[claim] == name for claim, name in occurrences)


def test_main_figures_are_defined_once_at_fixed_paths_and_referenced_in_results():
    texts = _read_results()
    corpus = "\n".join(texts.values())
    figure_blocks = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", corpus, re.DOTALL)
    assert len(figure_blocks) == len(FIGURE_CONTRACT)

    for label, (owner, relative_path) in FIGURE_CONTRACT.items():
        label_token = rf"\label{{{label}}}"
        assert corpus.count(label_token) == 1, f"{label} must be defined exactly once"
        block = next((item for item in figure_blocks if label_token in item), None)
        assert block is not None, f"{label} must label a figure environment"
        assert r"\includegraphics" in block and f"{{{relative_path}}}" in block
        assert label_token in texts[owner], f"{label} belongs in {owner}"
        assert (ROOT / "paper" / relative_path).resolve().is_file()
        assert len(_references(corpus, "fig:")) and label in _references(corpus, "fig:")

    for name, text in texts.items():
        assert _references(text, "fig:"), f"{name} must cite at least one main figure"


def test_results_cite_the_registered_theory_and_every_internal_reference_resolves():
    texts = _read_results()
    derivations = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(DERIVATION_DIR.glob("0[1-4]_*.tex"))
    )
    corpus = derivations + "\n" + "\n".join(texts.values())
    defined = set(re.findall(r"\\label\{([^{}]+)\}", corpus))

    for name, required in REQUIRED_THEORY_REFS.items():
        theory_refs = {
            label
            for prefix in ("eq:", "thm:", "alg:", "rem:")
            for label in _references(texts[name], prefix)
        }
        assert required <= theory_refs, f"{name} misses registered theory references"
        assert theory_refs, f"{name} must cite at least one theorem or equation"

    results_refs = {
        label
        for prefix in ("eq:", "thm:", "alg:", "rem:", "fig:")
        for label in _references("\n".join(texts.values()), prefix)
    }
    assert results_refs <= defined, f"undefined Results references: {sorted(results_refs - defined)}"


def test_results_use_generated_macros_for_every_reported_computed_quantity():
    corpus = _without_comments("\n".join(_read_results().values()))
    for macro in GENERATED_VALUE_MACROS:
        assert rf"\{macro}" in corpus, f"Results must report artifact-backed macro {macro}"
    assert r"\newcommand" not in corpus

    scrubbed = re.sub(
        r"\\(?:label|ref|eqref|autoref|cref|Cref|cite\w*|includegraphics)"
        r"(?:\[[^]]*\])?\{[^{}]*\}",
        " ",
        corpus,
    )
    decimal = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d+|\.\d+|\d+[eE][+-]?\d+)")
    assert not decimal.findall(scrubbed), "computed decimals must come from generated macros"

    integers = re.findall(r"(?<![A-Za-z0-9_])\d+(?![A-Za-z0-9_])", scrubbed)
    nonstructural = [token for token in integers if token not in {"0", "1", "2", "4", "8"}]
    assert not nonstructural, (
        "nonstructural numbers and preregistered thresholds require artifact-backed macros: "
        f"{nonstructural}"
    )


def test_results_avoid_overclaiming_and_preserve_required_semantic_boundaries():
    corpus = _without_comments("\n".join(_read_results().values()))
    lowered = corpus.lower()
    banned = (
        "spectral line",
        "beat frequency",
        "first-ever",
        "first ever",
        "for the first time",
        "we are the first",
        "topological phase transition",
        "analytic signal",
        "universal",
    )
    assert not [phrase for phrase in banned if phrase in lowered]
    assert not re.search(r"\bpriority\b", lowered)
    assert not re.search(
        r"(?:silicon.{0,100}(?:proof|proves?|validat(?:e[sd]?|ion)|establish(?:es|ed)?)|"
        r"(?:proof|proves?|validat(?:e[sd]?|ion)|establish(?:es|ed)?).{0,100}silicon)",
        lowered,
        re.DOTALL,
    )

    required_semantics = {
        "declared local annulus": r"declared local annulus",
        "tracked branch": r"tracked(?: full(?: three-dimensional)?)? (?:spectral )?branch",
        "joint versus fixed limits": (
            r"joint-limit[\s\S]{0,240}fixed-anisotropy|"
            r"fixed-anisotropy[\s\S]{0,240}joint-limit"
        ),
        "positive-frequency normalization": r"undoubled positive-frequency (?:branch )?response",
        "separation/modulation factor two": (
            r"(?:separation[\s\S]{0,160}twice[\s\S]{0,160}modulation|"
            r"modulation[\s\S]{0,160}twice[\s\S]{0,160}separation)"
        ),
        "no response fitting": r"without fitted alignment of amplitude, phase, frequency, (?:or )?time",
    }
    for boundary, pattern in required_semantics.items():
        assert re.search(pattern, lowered), f"Results omit the {boundary} boundary"


def test_reviewed_dimensionless_and_asymptotic_boundaries_are_explicit():
    isotropic = (RESULTS_DIR / "02_isotropic_ring.tex").read_text(encoding="utf-8")
    temporal = (RESULTS_DIR / "04_temporal_crossover.tex").read_text(encoding="utf-8")
    green_derivation = (DERIVATION_DIR / "03_green_function_asymptotics.tex").read_text(
        encoding="utf-8"
    )
    isotropic_compact = "".join(isotropic.split())
    temporal_compact = "".join(temporal.split())

    assert r"\boldsymbol\kappa=h\mathbfk^{\mathrm{phys}}" in isotropic_compact
    assert "notusedasasignedparitytest" in isotropic_compact.lower()
    assert r"t:=t_{\mathrm{phys}}c_T/h" in temporal_compact
    assert (
        r"O(\lvert\varepsilonV_4\rvert^{-1/2}t^{-1})" in temporal_compact
    )
    assert "afixedsmallnonzero" in temporal_compact.lower()
    assert r"G_+" not in green_derivation
    assert r"G^{+}" in green_derivation and r"G^{+}" in temporal


def test_exact_cardinality_is_reserved_for_the_sufficiently_small_theorem() -> None:
    splitting = (RESULTS_DIR / "03_morse_unfolding.tex").read_text(encoding="utf-8")
    claim_matrix = (ROOT / "docs/manuscript/claim_evidence_matrix.csv").read_text(
        encoding="utf-8"
    )

    theorem_body, finite_body = splitting.split(r"\subsection{Eight resolved critical points}")
    assert "sufficiently small" in theorem_body.lower()
    assert "exact local count" in theorem_body.lower()
    assert "eight resolved roots" in finite_body.lower()
    finite_compact = re.sub(r"\s+", " ", finite_body).lower()
    assert "cannot exclude an unresolved opposite-index pair" in finite_compact
    assert "resolved exactly" not in finite_body.lower()
    assert "certified local" not in finite_body.lower()
    assert "eight resolved, refinement-stable roots" in claim_matrix.lower()
    assert "sufficiently small nonzero epsilon" in claim_matrix.lower()
    assert "realizes exactly" not in claim_matrix.lower()
    assert "certified computed" not in claim_matrix.lower()


def test_phase_discrepancy_is_not_presented_as_a_continuum_error_bound() -> None:
    prose = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            RESULTS_DIR / "00_abstract.tex",
            RESULTS_DIR / "05_numerical_verification.tex",
            RESULTS_DIR / "06_discussion.tex",
            RESULTS_DIR / "07_methods.tex",
            RESULTS_DIR / "08_conclusion.tex",
            DERIVATION_DIR / "04_spectral_numerics.tex",
        ]
    )
    lowered = prose.lower()
    assert "accumulated inter-resolution phase-discrepancy estimator" in lowered
    assert "does not bound the unknown continuum" in lowered
    assert "phase-certified" not in lowered
    assert "phase certificate" not in lowered
    assert "phase-error certificate" not in lowered
    assert "accumulated-phase bound" not in lowered
