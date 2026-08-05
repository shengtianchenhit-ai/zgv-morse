"""Validate the manuscript claim-to-evidence matrix as a hard drafting gate."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
import re
import subprocess
import sys

from zgv_morse.artifact_schema import SCHEMAS, validate_artifact


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/manuscript/claim_evidence_matrix.csv"
DERIVATIONS = tuple(sorted((ROOT / "docs/derivations").glob("0[1-4]_*.tex")))

HEADER = (
    "claim_id",
    "claim",
    "scope_boundary",
    "theory_labels",
    "artifact_keys",
    "metadata_paths",
    "test_nodes",
    "figure_ids",
    "literature_keys",
    "status",
)
MULTIVALUE_COLUMNS = (
    "theory_labels",
    "artifact_keys",
    "metadata_paths",
    "test_nodes",
    "figure_ids",
    "literature_keys",
)
FIGURE_MODULES = {
    "figure_01_geometry_mechanism": "src/zgv_morse/figures/figure01_geometry.py",
    "figure_02_isotropic_zgv": "src/zgv_morse/figures/figure02_isotropic.py",
    "figure_03_angular_sensitivity": "src/zgv_morse/figures/figure03_sensitivity.py",
    "figure_04_morse_points": "src/zgv_morse/figures/figure04_morse.py",
    "figure_05_perturbation_scaling": "src/zgv_morse/figures/figure05_scaling.py",
    "figure_06_decay_crossover": "src/zgv_morse/figures/figure06_crossover.py",
}


class EvidenceError(ValueError):
    """Raised when a registered claim has a broken or unsupported evidence edge."""


def _items(row: dict[str, str], column: str) -> tuple[str, ...]:
    value = row[column]
    if value != value.strip() or not value:
        raise EvidenceError(f"{row['claim_id']}.{column} is empty or has outer whitespace")
    if "," in value:
        raise EvidenceError(f"{row['claim_id']}.{column} must use semicolons")
    items = tuple(value.split(";"))
    if any(not item or item != item.strip() for item in items):
        raise EvidenceError(f"{row['claim_id']}.{column} has an empty or padded item")
    if len(items) != len(set(items)):
        raise EvidenceError(f"{row['claim_id']}.{column} has a duplicate item")
    return items


def _read_rows() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != HEADER:
            raise EvidenceError("claim matrix header does not match the registered schema")
        rows = list(reader)
    if len(rows) != 7 or [row["claim_id"] for row in rows] != [f"C{i}" for i in range(1, 8)]:
        raise EvidenceError("claim matrix must contain ordered rows C1 through C7 exactly once")
    for row in rows:
        if any(not row[column].strip() for column in HEADER):
            raise EvidenceError(f"{row['claim_id']} contains an empty field")
        if row["status"] not in {"supported", "unsupported"}:
            raise EvidenceError(f"{row['claim_id']} has an invalid status")
        for column in MULTIVALUE_COLUMNS:
            _items(row, column)
    return rows


def _bibtex_keys() -> set[str]:
    text = (ROOT / "paper/references.bib").read_text(encoding="utf-8")
    return set(re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", text))


def _run_registered_tests(nodes: tuple[str, ...]) -> None:
    """Require every unique evidence test to pass, not merely to collect."""

    if not nodes:
        raise EvidenceError("the claim matrix contains no registered evidence tests")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *nodes],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        diagnostic = (result.stdout + result.stderr).strip()
        if len(diagnostic) > 4000:
            diagnostic = diagnostic[-4000:]
        raise EvidenceError(f"registered evidence tests failed:\n{diagnostic}")


def validate_matrix(*, require_supported: bool = False) -> list[dict[str, str]]:
    """Validate every graph edge from C1--C7 to its registered evidence."""

    rows = _read_rows()
    if require_supported:
        unsupported = [row["claim_id"] for row in rows if row["status"] != "supported"]
        if unsupported:
            raise EvidenceError(f"unsupported claims: {', '.join(unsupported)}")

    if len(DERIVATIONS) != 4:
        raise EvidenceError("exactly four canonical derivation files are required")
    derivation_corpus = "\n".join(path.read_text(encoding="utf-8") for path in DERIVATIONS)
    bibtex_keys = _bibtex_keys()
    artifact_cache: dict[str, dict[str, object]] = {}
    test_nodes: set[str] = set()

    for row in rows:
        for label in _items(row, "theory_labels"):
            if derivation_corpus.count(rf"\label{{{label}}}") != 1:
                raise EvidenceError(f"{row['claim_id']} theory label is absent or ambiguous: {label}")

        referenced_artifacts: set[str] = set()
        for reference in _items(row, "artifact_keys"):
            artifact, separator, key = reference.partition(".")
            if not separator or artifact not in SCHEMAS or key not in SCHEMAS[artifact]:
                raise EvidenceError(f"{row['claim_id']} has an unknown artifact key: {reference}")
            referenced_artifacts.add(artifact)
            if artifact not in artifact_cache:
                path = ROOT / "data/generated" / f"{artifact}.npz"
                arrays, _metadata = validate_artifact(path, path.with_suffix(".json"))
                artifact_cache[artifact] = arrays
            if key not in artifact_cache[artifact]:
                raise EvidenceError(f"{row['claim_id']} artifact array is missing: {reference}")

        metadata_artifacts: set[str] = set()
        for relative in _items(row, "metadata_paths"):
            path = ROOT / relative
            if not path.is_file() or path.suffix != ".json":
                raise EvidenceError(f"{row['claim_id']} metadata path is missing: {relative}")
            if path.parent != ROOT / "data/generated" or path.stem not in SCHEMAS:
                raise EvidenceError(f"{row['claim_id']} metadata path is not canonical: {relative}")
            metadata_artifacts.add(path.stem)
        if metadata_artifacts != referenced_artifacts:
            raise EvidenceError(
                f"{row['claim_id']} metadata sidecars do not match its artifact references"
            )

        for node in _items(row, "test_nodes"):
            relative, separator, function = node.partition("::")
            path = ROOT / relative
            if not separator or not function.startswith("test_") or not path.is_file():
                raise EvidenceError(f"{row['claim_id']} has an invalid test node: {node}")
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            functions = {
                item.name
                for item in ast.walk(tree)
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if function not in functions:
                raise EvidenceError(f"{row['claim_id']} test function is missing: {node}")
            test_nodes.add(node)

        for figure_id in _items(row, "figure_ids"):
            relative = FIGURE_MODULES.get(figure_id)
            if relative is None:
                raise EvidenceError(f"{row['claim_id']} has an unknown figure: {figure_id}")
            source = (ROOT / relative).read_text(encoding="utf-8")
            if f'output_dir / "{figure_id}"' not in source:
                raise EvidenceError(f"{row['claim_id']} figure output stem is missing: {figure_id}")

        for key in _items(row, "literature_keys"):
            if key not in bibtex_keys:
                raise EvidenceError(f"{row['claim_id']} BibTeX key is missing: {key}")

    _run_registered_tests(tuple(sorted(test_nodes)))
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-supported",
        action="store_true",
        help="fail if any C1--C7 row is not explicitly marked supported",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        rows = validate_matrix(require_supported=arguments.require_supported)
    except (EvidenceError, OSError, SyntaxError, ValueError) as error:
        print(f"claim-evidence gate failed: {error}", file=sys.stderr)
        return 1
    print(f"claim-evidence gate passed: {len(rows)} supported claim rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
