"""Tests for the DOI-backed bibliography audit."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import urllib.error

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.verify_bibliography as verify_bibliography  # noqa: E402


VALID_BIB = r"""
@article{prada2005laser,
  author = {Claire Prada and Oluwaseyi Balogun and Todd W. Murray},
  title = {Laser-based ultrasonic generation and detection of zero-group velocity Lamb waves in thin plates},
  journal = {Applied Physics Letters},
  year = {2005},
  doi = {10.1063/1.2128063}
}
"""

FIXED_CLOCK_VALUE = datetime(
    2026,
    7,
    11,
    11,
    4,
    5,
    tzinfo=timezone(timedelta(hours=8)),
)


def _write_bib(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "references.bib"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_fixture_passes_offline_validation(tmp_path: Path) -> None:
    bib = _write_bib(tmp_path, VALID_BIB)

    audit = verify_bibliography.audit_bibliography(bib)

    assert audit["summary"] == {
        "entries": 1,
        "mismatch": 0,
        "duplicate_doi": 0,
        "invalid_doi": 0,
        "manual_needed": 0,
        "missing_required": 0,
    }
    assert audit["entries"][0]["doi"] == "10.1063/1.2128063"
    assert audit["entries"][0]["status"] == "offline_validated"


def test_metadata_normalization_handles_crossref_html_and_dash_variants() -> None:
    crossref = "Uniform trace formulae for<b><i>SU</i></b>(2) and Morse–Bott splitting"
    bibtex = "Uniform trace formulae for {SU}(2) and {Morse--Bott} splitting"

    assert verify_bibliography.normalize_metadata_text(crossref) == (
        verify_bibliography.normalize_metadata_text(bibtex)
    )


def test_metadata_normalization_equates_html_and_latex_ampersands() -> None:
    assert verify_bibliography.normalize_metadata_text(
        "Algebraic &amp; Geometric Topology"
    ) == verify_bibliography.normalize_metadata_text(r"Algebraic \& Geometric Topology")


def test_metadata_normalization_preserves_semantic_word_boundaries() -> None:
    assert verify_bibliography.normalize_metadata_text("A BC") != (
        verify_bibliography.normalize_metadata_text("AB C")
    )


def test_author_comparison_handles_particles_and_latex_accents_but_keeps_order() -> None:
    assert verify_bibliography._author_comparison(
        ["Ben Amor", 'Gr{\\"u}nsteidl', "Ozorio de Almeida"],
        ["Amor", "Grünsteidl", "Almeida"],
    )["match"]
    assert not verify_bibliography._author_comparison(
        ["Ben Amor", "Almeida"],
        ["Almeida", "Amor"],
    )["match"]


def test_duplicate_normalized_doi_is_rejected(tmp_path: Path) -> None:
    duplicate = VALID_BIB + VALID_BIB.replace("prada2005laser", "prada2005duplicate").replace(
        "10.1063/1.2128063", "https://doi.org/10.1063/1.2128063"
    )
    bib = _write_bib(tmp_path, duplicate)

    audit = verify_bibliography.audit_bibliography(bib)

    assert audit["summary"]["duplicate_doi"] == 1
    assert audit["duplicates"] == [
        {
            "doi": "10.1063/1.2128063",
            "keys": ["prada2005duplicate", "prada2005laser"],
        }
    ]
    assert not verify_bibliography.audit_passed(audit)


@pytest.mark.parametrize("field", ["author", "title", "journal", "year", "doi"])
def test_every_required_field_is_rejected_when_missing(tmp_path: Path, field: str) -> None:
    line_prefix = f"  {field} ="
    incomplete = "\n".join(
        line for line in VALID_BIB.splitlines() if not line.startswith(line_prefix)
    )
    bib = _write_bib(tmp_path, incomplete)

    audit = verify_bibliography.audit_bibliography(bib)

    assert audit["summary"]["missing_required"] == 1
    assert audit["entries"][0]["missing_required"] == [field]
    assert audit["entries"][0]["status"] == "missing_required"
    assert not verify_bibliography.audit_passed(audit)


@pytest.mark.parametrize(
    "bad_doi",
    [
        "10.123/short-prefix",
        "10.1063/no whitespace allowed",
        "https://example.org/10.1063/1.2128063",
        "https://doi.org/",
        "10.1063/1.2128063?query=not-part-of-a-doi",
        "10.1063/1.2128063.",
    ],
)
def test_malformed_doi_is_rejected_without_online_lookup(
    tmp_path: Path,
    bad_doi: str,
) -> None:
    bib = _write_bib(tmp_path, VALID_BIB.replace("10.1063/1.2128063", bad_doi))

    def should_not_fetch(_doi: str) -> dict[str, object]:
        raise AssertionError("invalid DOI must never be sent to Crossref")

    audit = verify_bibliography.audit_bibliography(
        bib,
        online=True,
        fetcher=should_not_fetch,
    )

    assert audit["summary"]["invalid_doi"] == 1
    assert audit["summary"]["manual_needed"] == 0
    assert audit["entries"][0]["status"] == "invalid_doi"
    assert not verify_bibliography.audit_passed(audit)


def test_missing_field_and_invalid_doi_are_counted_independently(tmp_path: Path) -> None:
    malformed = VALID_BIB.replace(
        "  title = {Laser-based ultrasonic generation and detection of zero-group velocity Lamb waves in thin plates},\n",
        "",
    ).replace("10.1063/1.2128063", "not-a-doi")
    bib = _write_bib(tmp_path, malformed)

    audit = verify_bibliography.audit_bibliography(bib)

    assert audit["summary"]["missing_required"] == 1
    assert audit["summary"]["invalid_doi"] == 1
    assert audit["entries"][0]["missing_required"] == ["title"]
    assert audit["entries"][0]["invalid_doi"] is True


def test_online_title_mismatch_is_recorded_with_mocked_crossref(
    tmp_path: Path,
) -> None:
    bib = _write_bib(tmp_path, VALID_BIB)
    audit_path = tmp_path / "citation_audit.json"

    def fake_crossref(doi: str) -> dict[str, object]:
        assert doi == "10.1063/1.2128063"
        return {
            "DOI": doi,
            "title": ["A different title"],
            "container-title": ["Applied Physics Letters"],
            "published": {"date-parts": [[2005, 11, 7]]},
            "author": [
                {"given": "Claire", "family": "Prada"},
                {"given": "Oluwaseyi", "family": "Balogun"},
                {"given": "Todd W.", "family": "Murray"},
            ],
        }

    audit = verify_bibliography.audit_bibliography(
        bib,
        online=True,
        audit_path=audit_path,
        fetcher=fake_crossref,
        clock=lambda: FIXED_CLOCK_VALUE,
    )

    assert audit["summary"]["mismatch"] == 1
    comparison = audit["entries"][0]["comparison"]
    assert comparison["title"] == {
        "local": "Laser-based ultrasonic generation and detection of zero-group velocity Lamb waves in thin plates",
        "crossref": "A different title",
        "match": False,
    }
    assert comparison["year"]["match"] is True
    assert comparison["container"]["match"] is True
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit
    assert not verify_bibliography.audit_passed(audit)


def test_non_object_crossref_response_is_manual_needed(tmp_path: Path) -> None:
    bib = _write_bib(tmp_path, VALID_BIB)

    audit = verify_bibliography.audit_bibliography(
        bib,
        online=True,
        fetcher=lambda _doi: [],  # type: ignore[arg-type,return-value]
    )

    assert audit["summary"]["manual_needed"] == 1
    assert audit["entries"][0]["status"] == "manual_needed"
    assert audit["entries"][0]["online_error"] == (
        "CrossrefLookupError: Crossref fetcher returned a non-object response"
    )
    assert not verify_bibliography.audit_passed(audit)


def test_crossref_transport_error_is_wrapped_with_doi_context(monkeypatch) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(verify_bibliography.urllib.request, "urlopen", fail_urlopen)

    with pytest.raises(
        verify_bibliography.CrossrefLookupError,
        match=r"Crossref lookup failed for 10\.1063/1\.2128063: offline",
    ):
        verify_bibliography.fetch_crossref_metadata("10.1063/1.2128063")


def test_crossref_transport_retries_transient_url_error(monkeypatch) -> None:
    attempts = 0
    payload = {
        "message": {
            "title": ["Recovered metadata"],
            "container-title": ["Journal"],
            "published": {"date-parts": [[2026]]},
            "author": [{"family": "Example"}],
        }
    }

    def transient_urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise urllib.error.URLError("transient TLS close")
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(verify_bibliography.urllib.request, "urlopen", transient_urlopen)

    assert verify_bibliography.fetch_crossref_metadata("10.1063/1.2128063") == payload[
        "message"
    ]
    assert attempts == 3


def test_crossref_transport_retries_transient_timeout(monkeypatch) -> None:
    attempts = 0
    payload = {"message": {"title": ["Recovered after timeout"]}}

    def transient_urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("read timed out")
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(verify_bibliography.urllib.request, "urlopen", transient_urlopen)

    assert verify_bibliography.fetch_crossref_metadata("10.1063/1.2128063") == payload[
        "message"
    ]
    assert attempts == 2


def test_generated_timestamp_uses_injected_clock_and_is_canonical_utc(
    tmp_path: Path,
) -> None:
    bib = _write_bib(tmp_path, VALID_BIB)

    audit = verify_bibliography.audit_bibliography(
        bib,
        clock=lambda: FIXED_CLOCK_VALUE,
    )

    assert audit["generated_at"] == "2026-07-11T03:04:05+00:00"


def test_naive_injected_clock_is_rejected(tmp_path: Path) -> None:
    bib = _write_bib(tmp_path, VALID_BIB)

    with pytest.raises(ValueError, match="clock must return a timezone-aware datetime"):
        verify_bibliography.audit_bibliography(
            bib,
            clock=lambda: datetime(2026, 7, 11, 3, 4, 5),
        )


def test_fixed_inputs_produce_byte_identical_audits(tmp_path: Path) -> None:
    bib = _write_bib(tmp_path, VALID_BIB)
    audit_path = tmp_path / "citation_audit.json"

    first = verify_bibliography.audit_bibliography(
        bib,
        audit_path=audit_path,
        clock=lambda: FIXED_CLOCK_VALUE,
    )
    first_bytes = audit_path.read_bytes()
    second = verify_bibliography.audit_bibliography(
        bib,
        audit_path=audit_path,
        clock=lambda: FIXED_CLOCK_VALUE,
    )

    assert second == first
    assert audit_path.read_bytes() == first_bytes


def test_audit_write_is_atomic_when_replace_fails(tmp_path: Path, monkeypatch) -> None:
    audit_path = tmp_path / "citation_audit.json"
    audit_path.write_text("sentinel\n", encoding="utf-8")

    def fail_replace(_source: Path, _target: Path) -> Path:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        verify_bibliography._write_audit(audit_path, {"schema_version": 1})

    assert audit_path.read_text(encoding="utf-8") == "sentinel\n"
    assert list(tmp_path.glob(".citation_audit.json.*.tmp")) == []


def test_committed_online_audit_has_zero_failure_counters() -> None:
    audit = json.loads(
        (PROJECT_ROOT / "docs/literature/citation_audit.json").read_text(encoding="utf-8")
    )

    assert audit["online"] is True
    assert audit["summary"]["entries"] == 39
    assert verify_bibliography.audit_passed(audit)
    assert verify_bibliography.audit_matches_bibliography(
        audit, PROJECT_ROOT / "paper/references.bib"
    )
    assert {entry["status"] for entry in audit["entries"]} == {"core_metadata_verified"}


def test_audit_binding_detects_bibtex_changed_after_generation(tmp_path: Path) -> None:
    bib = _write_bib(tmp_path, VALID_BIB)
    audit = verify_bibliography.audit_bibliography(bib)
    assert verify_bibliography.audit_matches_bibliography(audit, bib)

    bib.write_text(VALID_BIB.replace("Prada", "Changed"), encoding="utf-8")

    assert not verify_bibliography.audit_matches_bibliography(audit, bib)


def test_registered_queries_and_five_claim_boundaries_are_committed() -> None:
    protocol = (PROJECT_ROOT / "docs/literature/search_protocol.md").read_text(encoding="utf-8")
    matrix = (PROJECT_ROOT / "docs/literature/novelty_matrix.md").read_text(encoding="utf-8")
    queries = (
        '"zero group velocity" Lamb wave anisotropic plate',
        '("ZGV" OR "zero group velocity") anisotropy critical point stationary phase',
        "Morse-Bott stationary phase critical manifold wave decay",
        "Lamb wave van Hove singularity anisotropic plate",
        "Bessel crossover stationary ring weak anisotropy",
        '"zero group velocity" cubic silicon plate',
    )
    claims = (
        "Morse--Bott identification of an isotropic ZGV ring",
        "Symmetry-enforced alternating Morse splitting",
        "Elastic closure of `V4` and `B`",
        "Uniform Bessel transition law",
        "Fixed-anisotropy exact-Morse `t^-1` asymptotics",
    )

    assert all(query in protocol for query in queries)
    assert all(matrix.count(claim) == 1 for claim in claims)
    assert "general Bessel identity" in matrix
    assert "39 retained records" in matrix


def test_machine_search_trace_has_registered_queries_and_complete_citation_pages() -> None:
    with (PROJECT_ROOT / "docs/literature/search_candidates.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    with (PROJECT_ROOT / "docs/literature/exclusions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        exclusions = list(csv.DictReader(handle))

    query_rows = [row for row in rows if row["record_type"] == "query"]
    assert {row["query_id"] for row in query_rows} == {f"Q{index}" for index in range(1, 7)}
    assert all("stable total unavailable" in row["returned_scope"] for row in query_rows)

    graph_2023 = [row for row in rows if row["query_id"] == "CG2023"]
    graph_2025 = [row for row in rows if row["query_id"] == "CG2025"]
    assert len(graph_2023) == 14
    assert len(graph_2025) == 4
    assert len({row["identifier"] for row in graph_2023}) == 14
    assert len({row["identifier"] for row in graph_2025}) == 4
    assert exclusions
    assert all(row["reason"] for row in exclusions)
