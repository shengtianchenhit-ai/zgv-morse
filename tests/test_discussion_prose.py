"""Contracts for a bounded, evidence-linked manuscript Discussion."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DISCUSSION = ROOT / "paper/sections/06_discussion.tex"
REFERENCES = ROOT / "paper/references.bib"
CITATION_AUDIT = ROOT / "docs/literature/citation_audit.json"
GENERATED_MACROS = ROOT / "paper/generated/results_macros.tex"
SHARED_MACROS = ROOT / "paper/shared_macros.tex"
EXPECTED_SUBSECTIONS = (
    "Geometry and noncommuting limits",
    "Relation to prior work",
    "Scope and boundary statements",
    "Nonlinear amplitude equations are a separate problem",
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


def _subsection_bodies(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"\\subsection\{([^{}]+)\}", _without_comments(text)))
    clean = _without_comments(text)
    return [
        (
            match.group(1),
            clean[
                match.end() : matches[index + 1].start()
                if index + 1 < len(matches)
                else None
            ],
        )
        for index, match in enumerate(matches)
    ]


def _first_prose_paragraph(body: str) -> str:
    for paragraph in re.split(r"\n\s*\n", body.strip()):
        if _word_count(paragraph) and not paragraph.lstrip().startswith(r"\begin"):
            return paragraph.strip()
    return ""


def _citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    pattern = re.compile(r"\\cite\w*\*?(?:\[[^]]*\]){0,2}\{([^{}]+)\}")
    for match in pattern.finditer(_without_comments(text)):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def _internal_references(text: str) -> set[str]:
    labels: set[str] = set()
    pattern = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref)\{([^{}]+)\}")
    for match in pattern.finditer(_without_comments(text)):
        labels.update(item.strip() for item in match.group(1).split(",") if item.strip())
    return labels


def _prior_report(text: str, concept: str) -> bool:
    prior = (
        r"(?:previously|already)\s+(?:been\s+)?(?:reported|observed|identified|known)"
        r"|(?:established|known)\s+(?:work|literature|result|pattern)"
        r"|prior\s+(?:work|studies|literature).{0,40}"
        r"(?:reported|observed|identified|established|known)"
    )
    return bool(
        re.search(rf"(?:{concept}).{{0,160}}(?:{prior})", text, re.IGNORECASE)
        or re.search(rf"(?:{prior}).{{0,160}}(?:{concept})", text, re.IGNORECASE)
    )


def test_discussion_keeps_four_substantive_subsections_with_bounded_openings():
    text = DISCUSSION.read_text(encoding="utf-8")
    bodies = _subsection_bodies(text)
    assert tuple(title for title, _body in bodies) == EXPECTED_SUBSECTIONS

    boundary = re.compile(
        r"\b(?:within|under|restricted|registered|local|only|provided|fixed|joint-limit|"
        r"does not|rather than|subject to|in contrast)\b",
        re.IGNORECASE,
    )
    for title, body in bodies:
        assert _word_count(body) >= 100, f"{title!r} needs at least 100 prose words"
        opening = _first_prose_paragraph(body)
        assert 30 <= _word_count(opening) <= 180, (
            f"{title!r} needs a bounded 30--180-word opening paragraph"
        )
        assert boundary.search(_plain(opening)), f"{title!r} must open with a scope boundary"


def test_discussion_connects_geometry_and_the_two_temporal_limits():
    text = DISCUSSION.read_text(encoding="utf-8")
    plain = _plain(text)
    assert re.search(r"Morse--?Bott.{0,160}(?:Morse|four minima|four saddles)", plain, re.I)
    assert re.search(r"noncommut(?:e|ing|ativity)", plain, re.IGNORECASE)
    assert re.search(
        r"joint-limit.{0,220}fixed-anisotropy|fixed-anisotropy.{0,220}joint-limit",
        plain,
        re.IGNORECASE,
    )
    assert re.search(r"uniform.{0,80}Bessel|Bessel.{0,80}uniform", plain, re.IGNORECASE)
    assert re.search(r"exact[- ]Morse|Cartesian Morse", plain, re.IGNORECASE)
    assert re.search(r"complement(?:ary|s|ed)|overlap|matched", plain, re.IGNORECASE)

    required = {
        "thm:morse-bott-ring",
        "thm:cubic-morse-splitting",
        "eq:noncommuting-green-limits",
        "thm:uniform-bessel-crossover",
        "thm:fixed-anisotropy-decay",
    }
    assert required <= _internal_references(text)


def test_discussion_positions_known_results_and_the_relevant_literatures():
    text = DISCUSSION.read_text(encoding="utf-8")
    plain = _plain(text)
    assert _prior_report(plain, r"isolated\s+anisotropic\s+(?:ZGV|zero[- ]group[- ]velocity)\s+points")
    assert _prior_report(plain, r"four\s+minim(?:a|um).{0,80}four\s+saddles?")
    assert _prior_report(plain, r"(?:beating|interference)\s+(?:pattern|response|phenomenon)s?")
    assert re.search(r"stationary[- ]phase", plain, re.IGNORECASE)
    assert re.search(r"broken[- ]symmetry|symmetry[- ]breaking", plain, re.IGNORECASE)

    cited = _citation_keys(text)
    assert {"prada2009anisotropy", "kiefer2023beating"} <= cited
    assert cited & {"velichko2007excitation", "chapuis2010focusing", "karmazin2013caustics"}
    assert cited & {"creagh1996broken", "brack1999uniform", "brack2009closed"}


def test_discussion_covers_competing_explanations_and_scope_limits():
    text = DISCUSSION.read_text(encoding="utf-8")
    plain = _plain(text)
    searchable = plain + " " + _without_comments(text)
    required = {
        "source and angular weight": r"source.{0,120}angular weight|angular weight.{0,120}source",
        "branch contamination and eigengap": (
            r"branch contamination.{0,120}eigengap|eigengap.{0,120}branch contamination"
        ),
        "higher harmonics": r"higher harmonic",
        "vanishing V4": r"vanishing.{0,40}V|V_?\{?4\}?.{0,20}(?:=\s*0|vanishes|zero)",
        "B and the controlled remainder": (
            r"(?:vanishing|zero).{0,40}\bB\b|\bB\b.{0,80}remainder|B\s*=\s*0"
        ),
        "late-time window": r"late[- ]time.{0,80}(?:window|resolution)|late[- ]window",
        "damping": r"\bdamping\b|\bloss(?:es|y)?\b",
        "finite boundaries": r"finite (?:lateral )?boundar|boundary reflection",
        "layers": r"\blayers?\b|\blayered\b",
        "defects": r"\bdefects?\b",
    }
    for name, pattern in required.items():
        assert re.search(pattern, searchable, re.IGNORECASE), f"Discussion omits {name}"

    assert re.search(r"silicon.{0,100}(?:stress test|robustness check)", plain, re.IGNORECASE)
    assert re.search(
        r"(?:no|without)\s+(?:laboratory\s+)?experiments?|"
        r"does not (?:include|provide|report|use) (?:an? )?experiment",
        plain,
        re.IGNORECASE,
    )


def test_discussion_maps_theorem_level_and_instance_level_generality() -> None:
    plain = _plain(DISCUSSION.read_text(encoding="utf-8"))
    assert re.search(r"at theorem level", plain, re.I)
    assert re.search(r"at instance level", plain, re.I)
    assert re.search(
        r"theorem level.{0,260}(?:simple|gapped).{0,260}(?:nonzero|fourfold)",
        plain,
        re.I,
    )
    assert re.search(
        r"instance level.{0,260}(?:one|selected).{0,260}(?:branch|cubic)",
        plain,
        re.I,
    )
    assert re.search(
        r"pure leading fourfold phase.{0,260}matched asymptotic regimes",
        plain,
        re.I,
    )


def test_nonlinear_amplitude_equations_remain_an_explicit_separate_problem():
    text = DISCUSSION.read_text(encoding="utf-8")
    body = dict(_subsection_bodies(text))[EXPECTED_SUBSECTIONS[-1]]
    plain = _plain(body)
    assert re.search(
        r"nonlinear amplitude equations?.{0,100}(?:a |the )?separate (?:problem|paper|study)",
        plain,
        re.IGNORECASE,
    )
    assert re.search(r"\b(?:equation|equations)\b", plain, re.IGNORECASE)
    assert re.search(r"\b(?:coefficient|coefficients)\b", plain, re.IGNORECASE)
    assert re.search(r"\b(?:do|does|are|is|will) not\b|\bneither\b", plain, re.IGNORECASE)
    assert not re.search(r"\$|\\\[|\\begin\{(?:equation|align|gather|multline)", body)
    assert not re.search(
        r"\bwe\s+(?:derive|obtain|calculate|predict|propose|establish)\w*.{0,100}"
        r"(?:nonlinear|amplitude equation|coefficient)",
        plain,
        re.IGNORECASE,
    )


def test_discussion_uses_only_registered_evidence_and_avoids_overclaiming():
    text = DISCUSSION.read_text(encoding="utf-8")
    clean = _without_comments(text)
    plain = _plain(clean)
    lowered = plain.lower()

    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", REFERENCES.read_text()))
    audit = json.loads(CITATION_AUDIT.read_text(encoding="utf-8"))
    verified = {
        entry["key"]
        for entry in audit["entries"]
        if entry.get("status") == "core_metadata_verified"
    }
    cited = _citation_keys(text)
    assert cited and cited <= bib_keys
    assert cited <= verified, f"unverified Discussion citations: {sorted(cited - verified)}"

    evidence_corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            *sorted((ROOT / "docs/derivations").glob("0[1-4]_*.tex")),
            *sorted((ROOT / "paper/sections").glob("0[0-5]_*.tex")),
        ]
    )
    defined = set(re.findall(r"\\label\{([^{}]+)\}", evidence_corpus))
    assert _internal_references(text) <= defined

    generated = set(
        re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}", GENERATED_MACROS.read_text(), re.M)
    )
    shared = set(
        re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}", SHARED_MACROS.read_text(), re.M)
    )
    capitalized_commands = set(re.findall(r"\\([A-Z][A-Za-z]+)\b", clean))
    assert capitalized_commands <= generated | shared | {"Cref"}

    banned = (
        "first-ever",
        "first ever",
        "for the first time",
        "priority",
        "discovery",
        "discovered",
        "topological phase transition",
        "spectral line",
        "beat frequency",
    )
    assert not [phrase for phrase in banned if phrase in lowered]
    for sentence in re.split(r"(?<=[.!?])\s+", lowered):
        if "universal" in sentence:
            assert re.search(
                r"\b(?:within|under|restricted|qualified|controlled|registered|local|source|weight)\b",
                sentence,
            ), "universal must be qualified in the same sentence"
    assert not re.search(
        r"silicon.{0,100}(?:validat|prov|establish|confirm)|"
        r"(?:validat|prov|establish|confirm).{0,100}silicon",
        lowered,
    )

    assert r"\label{" not in clean and r"\newcommand" not in clean
    assert not re.search(
        r"\\begin\{(?:figure|table|equation|align|gather|multline|theorem|lemma|proposition)\*?\}",
        clean,
    )
    assert not re.search(r"(?m)^\s*%\s*Claim\s+C[1-7]\s*$", text)
    scrubbed = re.sub(
        r"\\(?:cite\w*|ref|eqref|autoref|cref|Cref)(?:\[[^]]*\]){0,2}\{[^{}]*\}",
        " ",
        clean,
    )
    decimal = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d+|\.\d+|\d+[eE][+-]?\d+)")
    assert not decimal.findall(scrubbed), "computed decimals must use generated macros"
