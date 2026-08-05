"""Contracts for the manuscript Introduction and Conclusion."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper/main.tex"
SECTIONS = ROOT / "paper/sections"
INTRODUCTION = SECTIONS / "01_introduction.tex"
CONCLUSION = SECTIONS / "08_conclusion.tex"
REFERENCES = ROOT / "paper/references.bib"
CITATION_AUDIT = ROOT / "docs/literature/citation_audit.json"
GENERATED_MACROS = ROOT / "paper/generated/results_macros.tex"
PRIOR_SECTION_NAMES = (
    "00_abstract.tex",
    "01_introduction.tex",
    "02_isotropic_ring.tex",
    "03_morse_unfolding.tex",
    "04_temporal_crossover.tex",
    "05_numerical_verification.tex",
    "06_discussion.tex",
    "07_methods.tex",
)


def _without_comments(text: str) -> str:
    return "\n".join(line.split("%", maxsplit=1)[0] for line in text.splitlines())


def _plain(text: str) -> str:
    text = _without_comments(text)
    text = re.sub(r"\$(?:[^$]|\\\$)*\$|\\\[[\s\S]*?\\\]", " ", text)
    text = re.sub(
        r"\\(?:cite\w*|ref|eqref|autoref|cref|Cref)(?:\[[^]]*\]){0,2}\{[^{}]*\}",
        " ",
        text,
    )
    text = re.sub(r"\\[A-Za-z]+\*?", " ", text)
    text = text.replace("{", " ").replace("}", " ").replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", _plain(text)))


def _prose_paragraphs(path: Path) -> list[str]:
    text = _without_comments(path.read_text(encoding="utf-8"))
    text = re.sub(r"(?m)^\s*\\section\{[^{}]+\}\s*$", "", text)
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if _word_count(paragraph)
    ]


def _citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    pattern = re.compile(r"\\cite\w*\*?(?:\[[^]]*\]){0,2}\{([^{}]+)\}")
    for match in pattern.finditer(_without_comments(text)):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def _internal_references(text: str) -> set[str]:
    references: set[str] = set()
    pattern = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref)\{([^{}]+)\}")
    for match in pattern.finditer(_without_comments(text)):
        references.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return references


def _prior_report(text: str, concept: str) -> bool:
    prior = (
        r"(?:previously|already)\s+(?:been\s+)?(?:reported|observed|identified|established|known)"
        r"|prior\s+(?:work|studies)\s+(?:reported|observed|identified|established)"
        r"|(?:has|have|had)\s+been\s+(?:reported|observed|identified|established)"
    )
    return bool(
        re.search(rf"(?:{concept}).{{0,140}}(?:{prior})", text, re.IGNORECASE)
        or re.search(rf"(?:{prior}).{{0,140}}(?:{concept})", text, re.IGNORECASE)
    )


def test_introduction_has_four_substantive_paragraphs_in_the_registered_order():
    text = INTRODUCTION.read_text(encoding="utf-8")
    paragraphs = _prose_paragraphs(INTRODUCTION)

    assert r"\subsection" not in _without_comments(text)
    assert len(paragraphs) == 4, "Introduction must contain exactly four prose paragraphs"
    assert all(_word_count(paragraph) >= 45 for paragraph in paragraphs)

    importance, established, unresolved, contribution = map(_plain, paragraphs)
    assert re.search(r"(?:zero[- ]group[- ]velocity|\bZGV\b)", importance, re.IGNORECASE)
    assert re.search(r"locali[sz](?:ation|ed|es|ing)", importance, re.IGNORECASE)
    assert re.search(r"\bdecay\b", importance, re.IGNORECASE)
    assert re.search(r"\b(?:important|importance|enables?|relevant|central)\b", importance, re.I)

    assert re.search(r"\bestablished\b|\bknown\b|\bprior\b", established, re.IGNORECASE)
    assert re.search(r"\bisotropic\b", established, re.IGNORECASE)
    assert re.search(r"\banisotrop", established, re.IGNORECASE)

    assert re.search(r"\b(?:unresolved|missing|unclear|open)\b", unresolved, re.IGNORECASE)
    assert re.search(r"\b(?:geometry|geometric|Morse--?Bott|critical manifold)\b", unresolved, re.I)
    assert re.search(r"\b(?:asymptotic|temporal|decay|crossover)\b", unresolved, re.I)
    assert re.search(r"\b(?:connect|connection|link|bridge)\w*\b", unresolved, re.I)

    assert re.search(r"\b(?:we|this (?:work|paper|study))\b", contribution, re.IGNORECASE)
    assert "declared local annulus" in contribution.lower()
    assert "weak-anisotropy" in contribution.lower()

    evidence = (
        r"(?:full[- ]elastic(?:ity)?|generalized[- ]eigenvalue).{0,90}sensitiv",
        r"(?:critical[- ]point|Morse).{0,90}(?:geometry|search|check|index)",
        r"(?:uniform.{0,70}Bessel|Bessel.{0,70}asymptotic|uniform.{0,70}asymptotic)",
        r"(?:validat|phase[- ]discrepancy).{0,90}(?:spectral|comput)|"
        r"(?:spectral|comput).{0,90}(?:validat|phase[- ]discrepancy)",
    )
    positions = [re.search(pattern, contribution, re.IGNORECASE) for pattern in evidence]
    assert all(match is not None for match in positions), "paragraph four must state the evidence chain"
    assert [match.start() for match in positions if match] == sorted(
        match.start() for match in positions if match
    )


def test_introduction_marks_the_anisotropic_points_and_eight_point_pattern_as_prior_work():
    paragraphs = _prose_paragraphs(INTRODUCTION)
    assert len(paragraphs) >= 2, "Introduction needs its established-knowledge paragraph"
    established = _plain(paragraphs[1])
    isolated_points = r"isolated\s+anisotropic\s+(?:zero[- ]group[- ]velocity|ZGV)\s+points"
    alternating_set = r"four[- ]minim(?:um|a).{0,100}four[- ]saddles?"

    assert _prior_report(established, isolated_points), (
        "Introduction must say explicitly that isolated anisotropic ZGV points were reported before"
    )
    assert _prior_report(established, alternating_set), (
        "Introduction must say explicitly that the four-minimum/four-saddle pattern was reported before"
    )


def test_high_risk_decay_and_interference_claims_remain_bound_to_prior_work():
    paragraphs = _prose_paragraphs(INTRODUCTION)
    assert len(paragraphs) >= 2
    isotropic_raw, anisotropic_raw = paragraphs[:2]
    anisotropic = _plain(anisotropic_raw)

    assert re.search(
        r"established.{0,100}\\\(t\^\{-1/2\}\\\)\s+decay",
        isotropic_raw,
        re.IGNORECASE | re.DOTALL,
    )
    assert {"prada2008power", "laurent2014temporal"} <= _citation_keys(isotropic_raw)
    assert _prior_report(anisotropic, r"(?:interference|beating)")
    assert "kiefer2023beating" in _citation_keys(anisotropic_raw)


def test_introduction_cites_only_task28_verified_bibliography_entries():
    text = INTRODUCTION.read_text(encoding="utf-8")
    paragraphs = _prose_paragraphs(INTRODUCTION)
    cited = _citation_keys(text)
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", REFERENCES.read_text(encoding="utf-8")))
    audit = json.loads(CITATION_AUDIT.read_text(encoding="utf-8"))
    verified = {
        entry["key"]
        for entry in audit["entries"]
        if entry.get("status") == "core_metadata_verified"
    }

    assert cited
    assert cited <= bib_keys, f"unknown Introduction citation keys: {sorted(cited - bib_keys)}"
    assert cited <= verified, f"unverified Introduction citation keys: {sorted(cited - verified)}"
    assert all(_citation_keys(paragraph) for paragraph in paragraphs[:3])


def test_framing_avoids_overclaiming_manual_numbers_and_experimental_implication():
    introduction = INTRODUCTION.read_text(encoding="utf-8")
    conclusion = CONCLUSION.read_text(encoding="utf-8")
    corpus = _without_comments(introduction + "\n" + conclusion)
    lowered = _plain(corpus).lower()

    banned = (
        "first-ever",
        "first ever",
        "for the first time",
        "we are the first",
        "priority",
        "novel",
        "novelty",
        "discovery",
        "discovered",
        "spectral line",
        "beat frequency",
    )
    assert not [phrase for phrase in banned if phrase in lowered]

    for sentence in re.split(r"(?<=[.!?])\s+", lowered):
        if "universal" not in sentence:
            continue
        assert re.search(
            r"\b(?:within|under|restricted|qualified|controlled|weak-anisotropy|registered|"
            r"local|source|weight)\b",
            sentence,
        ), "universal must be qualified in the same sentence"

    experimental_implication = (
        r"\bwe\s+(?:experimentally\s+)?(?:measure[sd]?|recorded|detected)\b",
        r"\bour\s+(?:experiment|experiments|measurement|measurements|specimen|sample)\b",
        r"\b(?:experimental|measured)\s+(?:data|results|validation|verification)\s+"
        r"(?:in|from)\s+(?:this|the present|our)\s+(?:work|study)\b",
    )
    assert not [pattern for pattern in experimental_implication if re.search(pattern, lowered)]

    assert r"\newcommand" not in corpus
    scrubbed = re.sub(
        r"\\(?:cite\w*|ref|eqref|autoref|cref|Cref)(?:\[[^]]*\]){0,2}\{[^{}]*\}",
        " ",
        corpus,
    )
    decimal = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d+|\.\d+|\d+[eE][+-]?\d+)")
    assert not decimal.findall(scrubbed), "computed values must use generated macros"


def test_conclusion_follows_evidence_implication_boundary_structure():
    paragraphs = _prose_paragraphs(CONCLUSION)
    assert 2 <= len(paragraphs) <= 4, "Conclusion must contain two to four prose paragraphs"
    assert all(_word_count(paragraph) >= 35 for paragraph in paragraphs)

    plain = [_plain(paragraph) for paragraph in paragraphs]
    pre_boundary = " ".join(plain[:-1])
    first = plain[0]
    boundary = plain[-1]

    contribution = re.search(
        r"\b(?:we|this (?:work|paper|analysis))\s+"
        r"(?:connect|derive|establish|show|formulate|demonstrate)\w*\b",
        first,
        re.IGNORECASE,
    )
    evidence = re.search(
        r"\b(?:full[- ]elastic(?:ity)?|coefficient[- ]resolved|critical[- ]point|Bessel|"
        r"resolved critical|spectral validation)\b",
        pre_boundary,
        re.IGNORECASE,
    )
    implication = re.search(
        r"\b(?:therefore|thus|consequently|this (?:connection|result|framework) "
        r"(?:shows|explains|provides|makes))\b",
        pre_boundary,
        re.IGNORECASE,
    )
    assert contribution is not None
    assert evidence is not None
    assert implication is not None
    assert contribution.start() < evidence.start() < implication.start()

    lowered_boundary = boundary.lower()
    assert re.search(r"\b(?:validity|scope|restricted|confined|applies only)\b", lowered_boundary)
    assert "declared local annulus" in lowered_boundary
    assert "weak-anisotropy" in lowered_boundary
    assert re.search(r"\bsimple(?:\s+\w+){0,2}\s+branch\b", lowered_boundary)
    assert "lossless infinite plate" in lowered_boundary
    assert "computation-only" in lowered_boundary
    assert re.search(
        r"nonlinear amplitude equations?.{0,100}(?:a |the )?separate (?:problem|paper|study)",
        lowered_boundary,
    )


def test_conclusion_introduces_no_new_citation_display_result_macro_or_label():
    conclusion = CONCLUSION.read_text(encoding="utf-8")
    prior = "\n".join(
        (SECTIONS / name).read_text(encoding="utf-8") for name in PRIOR_SECTION_NAMES
    )

    assert _citation_keys(conclusion) <= _citation_keys(prior)
    assert _internal_references(conclusion) <= (
        _internal_references(prior) | set(re.findall(r"\\label\{([^{}]+)\}", prior))
    )
    assert not re.search(
        r"\\begin\{(?:figure|table|equation|align|gather|multline|theorem|lemma|proposition)\*?\}",
        _without_comments(conclusion),
    )
    assert r"\label{" not in _without_comments(conclusion)
    assert r"\[" not in _without_comments(conclusion)
    assert "$$" not in _without_comments(conclusion)

    generated_names = set(
        re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}", GENERATED_MACROS.read_text(), re.MULTILINE)
    )
    conclusion_macros = {name for name in generated_names if rf"\{name}" in conclusion}
    prior_macros = {name for name in generated_names if rf"\{name}" in prior}
    assert conclusion_macros <= prior_macros, (
        f"Conclusion introduces result macros absent from the body: "
        f"{sorted(conclusion_macros - prior_macros)}"
    )
