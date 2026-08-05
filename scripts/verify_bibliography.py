#!/usr/bin/env python3
"""Validate DOI-backed BibTeX locally and, optionally, against Crossref."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import html
import json
import os
import re
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pybtex.database import parse_file


REQUIRED_FIELDS = ("title", "journal", "year", "doi")
DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_OPEN_BOUNDARY_RE = re.compile(r"(?<=\w)(?:<[^/>][^>]*>)+(?=\w)")
WHITESPACE_RE = re.compile(r"\s+")
DASH_RE = re.compile(r"(?:--|[\u2010-\u2015\u2212])")
LATEX_ACCENT_RE = re.compile(r"\\[\"'`^~=.uvHckbr]\{?([A-Za-z])\}?")
AUTHOR_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


class CrossrefLookupError(RuntimeError):
    """A stable, DOI-scoped failure raised by the Crossref transport layer."""


def normalize_doi(value: str) -> str:
    """Return a comparison-safe DOI while preserving its semantic suffix."""

    normalized = value.strip()
    normalized = DOI_PREFIX_RE.sub("", normalized)
    return normalized.strip().casefold()


def is_valid_doi(value: str) -> bool:
    """Return whether a normalized value satisfies the Crossref DOI grammar."""

    return DOI_RE.fullmatch(value) is not None and not value.endswith(".")


def normalize_metadata_text(value: str) -> str:
    """Normalize harmless BibTeX/Crossref representation differences."""

    value = html.unescape(value)
    value = value.replace(r"\&", "&")
    value = HTML_OPEN_BOUNDARY_RE.sub(" ", value)
    value = HTML_TAG_RE.sub("", value)
    value = value.replace("{", "").replace("}", "")
    value = unicodedata.normalize("NFKC", value)
    value = DASH_RE.sub("-", value)
    return WHITESPACE_RE.sub(" ", value).strip().casefold()


def _local_author_families(entry: Any) -> list[str]:
    families: list[str] = []
    for person in entry.persons.get("author", []):
        parts = [*person.prelast_names, *person.last_names, *person.lineage_names]
        families.append(" ".join(parts))
    return families


def _crossref_author_families(message: Mapping[str, Any]) -> list[str] | None:
    authors = message.get("author")
    if not isinstance(authors, list) or not authors:
        return None
    families: list[str] = []
    for author in authors:
        if not isinstance(author, Mapping) or not author.get("family"):
            return None
        families.append(str(author["family"]))
    return families


def _author_comparison(local: list[str], crossref: list[str]) -> dict[str, Any]:
    def canonical_family(value: str) -> str:
        value = LATEX_ACCENT_RE.sub(r"\1", value)
        value = "".join(
            character
            for character in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(character)
        )
        tokens = AUTHOR_TOKEN_RE.findall(normalize_metadata_text(value))
        return tokens[-1] if tokens else ""

    return {
        "local": local,
        "crossref": crossref,
        # Crossref is inconsistent about storing surname particles such as
        # ``Ben`` or ``Ozorio de`` in the family field.  Compare the ordered
        # terminal family tokens after accent normalization, while retaining
        # the complete raw sequences for human inspection.
        "match": tuple(map(canonical_family, local)) == tuple(map(canonical_family, crossref)),
    }


def bibliography_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest binding an audit to its BibTeX bytes."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _crossref_scalar(message: Mapping[str, Any], key: str) -> str | None:
    value = message.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    return str(value)


def _crossref_year(message: Mapping[str, Any]) -> str | None:
    for key in ("published", "published-print", "published-online", "issued"):
        block = message.get(key)
        if not isinstance(block, dict):
            continue
        parts = block.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            return str(parts[0][0])
    return None


def fetch_crossref_metadata(doi: str) -> dict[str, Any]:
    """Resolve one DOI through Crossref's public Works API."""

    encoded = urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(
        f"https://api.crossref.org/works/{encoded}",
        headers={
            "Accept": "application/json",
            "User-Agent": "zgv-morse-bibliography-audit/0.1",
        },
    )
    payload: Any = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            raise CrossrefLookupError(
                f"Crossref HTTP {exc.code} for {doi}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            if attempt == 2:
                raise CrossrefLookupError(
                    f"Crossref lookup failed for {doi}: {exc.reason}"
                ) from exc
        except OSError as exc:
            if attempt == 2:
                raise CrossrefLookupError(
                    f"Crossref response failed for {doi}: {type(exc).__name__}: {exc}"
                ) from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CrossrefLookupError(
                f"Crossref response failed for {doi}: {type(exc).__name__}: {exc}"
            ) from exc
    if not isinstance(payload, dict):
        raise CrossrefLookupError(f"Crossref response for {doi} is not an object")
    message = payload.get("message")
    if not isinstance(message, dict):
        raise CrossrefLookupError(f"Crossref response for {doi} has no object-valued 'message'")
    return message


def _comparison(local: str, crossref: str) -> dict[str, Any]:
    return {
        "local": local,
        "crossref": crossref,
        "match": normalize_metadata_text(local) == normalize_metadata_text(crossref),
    }


def _write_audit(path: Path, audit: dict[str, Any]) -> None:
    """Atomically replace ``path`` with deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _generated_at(clock: Callable[[], datetime] | None) -> str:
    moment = datetime.now(timezone.utc) if clock is None else clock()
    if not isinstance(moment, datetime) or moment.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return moment.astimezone(timezone.utc).isoformat()


def audit_bibliography(
    bib_path: str | Path,
    *,
    online: bool = False,
    audit_path: str | Path | None = None,
    fetcher: Callable[[str], Mapping[str, Any]] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Audit a BibTeX database and return a JSON-serializable report."""

    bib_path = Path(bib_path)
    bib_sha256 = bibliography_sha256(bib_path)
    bibliography = parse_file(str(bib_path), bib_format="bibtex")
    fetch = fetcher or fetch_crossref_metadata

    rows: list[dict[str, Any]] = []
    doi_to_keys: dict[str, list[str]] = defaultdict(list)
    invalid_doi = 0
    missing_required = 0

    for key in sorted(bibliography.entries):
        entry = bibliography.entries[key]
        missing = [field for field in REQUIRED_FIELDS if not entry.fields.get(field, "").strip()]
        if not entry.persons.get("author"):
            missing.insert(0, "author")

        raw_doi = entry.fields.get("doi", "")
        doi = normalize_doi(raw_doi) if raw_doi else ""
        doi_is_invalid = bool(raw_doi.strip()) and not is_valid_doi(doi)
        if doi and not doi_is_invalid:
            doi_to_keys[doi].append(key)

        local = {
            "title": entry.fields.get("title", "").strip(),
            "year": entry.fields.get("year", "").strip(),
            "container": entry.fields.get("journal", "").strip(),
            "author_families": _local_author_families(entry),
        }
        row: dict[str, Any] = {
            "key": key,
            "doi": doi,
            "local": local,
            "invalid_doi": doi_is_invalid,
            "missing_required": missing,
            "status": (
                "missing_required"
                if missing
                else "invalid_doi"
                if doi_is_invalid
                else "offline_validated"
            ),
        }
        if missing:
            missing_required += 1
        if doi_is_invalid:
            invalid_doi += 1
        rows.append(row)

    duplicates = [
        {"doi": doi, "keys": sorted(keys)}
        for doi, keys in sorted(doi_to_keys.items())
        if len(keys) > 1
    ]
    duplicate_dois = {item["doi"] for item in duplicates}
    for row in rows:
        if row["doi"] in duplicate_dois and row["status"] != "missing_required":
            row["status"] = "duplicate_doi"

    mismatch = 0
    manual_needed = 0
    if online:
        for row in rows:
            if row["missing_required"] or row["invalid_doi"]:
                continue
            try:
                message = fetch(row["doi"])
                if not isinstance(message, Mapping):
                    raise CrossrefLookupError("Crossref fetcher returned a non-object response")
                remote = {
                    "title": _crossref_scalar(message, "title"),
                    "year": _crossref_year(message),
                    "container": _crossref_scalar(message, "container-title"),
                    "author_families": _crossref_author_families(message),
                }
                if any(value is None for value in remote.values()):
                    missing_remote = sorted(
                        field for field, value in remote.items() if value is None
                    )
                    row["crossref"] = remote
                    row["status"] = "manual_needed"
                    row["online_error"] = "Crossref metadata missing: " + ", ".join(missing_remote)
                    manual_needed += 1
                    continue

                comparison = {
                    field: _comparison(row["local"][field], str(remote[field]))
                    for field in ("title", "year", "container")
                }
                comparison["author_families"] = _author_comparison(
                    row["local"]["author_families"],
                    remote["author_families"],
                )
                row["crossref"] = remote
                row["comparison"] = comparison
                if all(result["match"] for result in comparison.values()):
                    row["status"] = (
                        "duplicate_doi"
                        if row["doi"] in duplicate_dois
                        else "core_metadata_verified"
                    )
                else:
                    row["status"] = "mismatch"
                    mismatch += 1
            except (
                CrossrefLookupError,
                OSError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                row["status"] = "manual_needed"
                row["online_error"] = f"{type(exc).__name__}: {exc}"
                manual_needed += 1

    audit: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _generated_at(clock),
        "bib": str(bib_path),
        "bib_sha256": bib_sha256,
        "online": online,
        "summary": {
            "entries": len(rows),
            "mismatch": mismatch,
            "duplicate_doi": len(duplicates),
            "invalid_doi": invalid_doi,
            "manual_needed": manual_needed,
            "missing_required": missing_required,
        },
        "duplicates": duplicates,
        "entries": rows,
    }
    if audit_path is not None:
        _write_audit(Path(audit_path), audit)
    return audit


def audit_passed(audit: dict[str, Any]) -> bool:
    """Return whether counters and entry-level evidence are internally consistent."""

    try:
        summary = audit["summary"]
        entries = audit["entries"]
        duplicates = audit["duplicates"]
        digest = audit["bib_sha256"]
        online = audit["online"]
        counters_are_zero = all(
            summary[field] == 0
            for field in (
                "mismatch",
                "duplicate_doi",
                "invalid_doi",
                "manual_needed",
                "missing_required",
            )
        )
        if not counters_are_zero or summary["entries"] != len(entries) or duplicates:
            return False
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            return False
        dois = [row["doi"] for row in entries]
        if len(dois) != len(set(dois)):
            return False
        expected_status = "core_metadata_verified" if online else "offline_validated"
        if any(row["status"] != expected_status for row in entries):
            return False
        if online and any(
            not all(result["match"] for result in row["comparison"].values()) for row in entries
        ):
            return False
        return True
    except (KeyError, TypeError):
        return False


def audit_matches_bibliography(audit: Mapping[str, Any], bib_path: str | Path) -> bool:
    """Return whether an audit is cryptographically and structurally bound to ``bib_path``."""

    path = Path(bib_path)
    try:
        if audit.get("bib_sha256") != bibliography_sha256(path):
            return False
        bibliography = parse_file(str(path), bib_format="bibtex")
        audited = {
            (str(row["key"]), normalize_doi(str(row["doi"]))) for row in audit.get("entries", [])
        }
        current = {
            (key, normalize_doi(entry.fields.get("doi", "")))
            for key, entry in bibliography.entries.items()
        }
        return audited == current
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bib", type=Path, required=True, help="BibTeX file to validate")
    parser.add_argument(
        "--online",
        action="store_true",
        help="resolve every DOI with Crossref and compare title/year/container",
    )
    parser.add_argument("--audit", type=Path, help="write the JSON audit to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        audit = audit_bibliography(
            args.bib,
            online=args.online,
            audit_path=args.audit,
        )
    except Exception as exc:  # pragma: no cover - argparse-facing parse/I/O guard
        print(f"bibliography_error={type(exc).__name__}: {exc}")
        return 2

    summary = audit["summary"]
    print(
        f"entries={summary['entries']} "
        f"mismatch={summary['mismatch']} "
        f"duplicate_doi={summary['duplicate_doi']} "
        f"invalid_doi={summary['invalid_doi']} "
        f"manual_needed={summary['manual_needed']} "
        f"missing_required={summary['missing_required']}"
    )
    return 0 if audit_passed(audit) else 1


if __name__ == "__main__":
    raise SystemExit(main())
