#!/usr/bin/env python3
"""Build and verify a deterministic, strict-allowlist paper release bundle."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Final
import unicodedata
import uuid


ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_clean_reproduction import (  # noqa: E402
    ReproductionError,
    ScientificClosure,
    discover_scientific_closure,
    git_subprocess_environment,
    measured_readme_bytes,
    validate_reproduction_report,
)
from scripts.verify_bibliography import (  # noqa: E402
    audit_matches_bibliography,
    audit_passed,
)


BUNDLE_NAME: Final = "zgv-morse-paper"
CHECKSUM_NAME: Final = "SHA256SUMS"
ARCHIVE_NAME: Final = f"{BUNDLE_NAME}.tar.gz"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_VERIFIER_UPDATED_PATHS: Final = frozenset(
    {"README.md", "data/reproduction_report.json"}
)
_FIGURE_FORMATS: Final = ("pdf", "png", "svg", "tiff")
_MAIN_FIGURE_STEMS: Final = (
    "figure_01_geometry_mechanism",
    "figure_02_isotropic_zgv",
    "figure_03_angular_sensitivity",
    "figure_04_morse_points",
    "figure_05_perturbation_scaling",
    "figure_06_decay_crossover",
)
_SUPPLEMENTARY_FIGURE_STEMS: Final = (
    "figure_s01_polynomial_two_element",
    "figure_s02_quadrature_phase",
    "figure_s03_mode_tracking",
    "figure_s04_fd_convergence",
    "figure_s05_source_window_sensitivity",
    "figure_s06_silicon_stress_test",
)
_EXPECTED_MAIN_FIGURE_OUTPUTS: Final = tuple(
    sorted(
        f"figures/main/{stem}.{kind}"
        for stem in _MAIN_FIGURE_STEMS
        for kind in _FIGURE_FORMATS
    )
)
_EXPECTED_SUPPLEMENTARY_FIGURE_OUTPUTS: Final = tuple(
    sorted(
        f"figures/supplementary/{stem}.{kind}"
        for stem in _SUPPLEMENTARY_FIGURE_STEMS
        for kind in _FIGURE_FORMATS
    )
)
_FIXED_FILES: Final = (
    ".gitignore",
    ".python-version",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "config/reference.yaml",
    "data/reproduction_report.json",
)
_REGISTERED_SOURCE_FILES: Final = (
    "src/zgv_morse/__init__.py",
    "src/zgv_morse/artifact_schema.py",
    "src/zgv_morse/artifacts.py",
    "src/zgv_morse/asymptotics.py",
    "src/zgv_morse/config.py",
    "src/zgv_morse/critical_points.py",
    "src/zgv_morse/dispersion.py",
    "src/zgv_morse/elasticity.py",
    "src/zgv_morse/figures/__init__.py",
    "src/zgv_morse/figures/common.py",
    "src/zgv_morse/figures/figure01_geometry.py",
    "src/zgv_morse/figures/figure02_isotropic.py",
    "src/zgv_morse/figures/figure03_sensitivity.py",
    "src/zgv_morse/figures/figure04_morse.py",
    "src/zgv_morse/figures/figure05_scaling.py",
    "src/zgv_morse/figures/figure06_crossover.py",
    "src/zgv_morse/figures/supplementary.py",
    "src/zgv_morse/gll.py",
    "src/zgv_morse/green_response.py",
    "src/zgv_morse/mode_tracking.py",
    "src/zgv_morse/perturbation.py",
    "src/zgv_morse/provenance.py",
    "src/zgv_morse/rayleigh_lamb.py",
    "src/zgv_morse/spectral_plate.py",
    "src/zgv_morse/validation.py",
    "src/zgv_morse/workflows/__init__.py",
    "src/zgv_morse/workflows/__main__.py",
    "src/zgv_morse/workflows/common.py",
    "src/zgv_morse/workflows/convergence.py",
    "src/zgv_morse/workflows/critical_points.py",
    "src/zgv_morse/workflows/green.py",
    "src/zgv_morse/workflows/isotropic.py",
    "src/zgv_morse/workflows/scaling.py",
    "src/zgv_morse/workflows/sensitivity.py",
    "src/zgv_morse/workflows/silicon.py",
    "src/zgv_morse/zgv.py",
)
_REGISTERED_SCRIPT_FILES: Final = tuple(
    f"scripts/{name}.py"
    for name in (
        "__init__",
        "build_release",
        "check_analytic_identities",
        "check_claim_evidence",
        "check_latex_log",
        "compile_paper",
        "compile_paper_semantic",
        "export_manuscript_values",
        "export_supplement_tables",
        "make_figure_01",
        "make_figure_02",
        "make_figure_03",
        "make_figure_04",
        "make_figure_05",
        "make_figure_06",
        "make_supplementary_figures",
        "qa_figures",
        "reproduce_all",
        "validate_isotropic",
        "verify_bibliography",
        "verify_clean_reproduction",
    )
)
_REGISTERED_TEST_FILES: Final = (
    "tests/figures/test_common.py",
    "tests/figures/test_figure01.py",
    "tests/figures/test_figure02.py",
    "tests/figures/test_figure03.py",
    "tests/figures/test_figure04.py",
    "tests/figures/test_figure05.py",
    "tests/figures/test_figure06.py",
    "tests/figures/test_figure_qa.py",
    "tests/figures/test_supplementary.py",
    *tuple(
        f"tests/test_{name}.py"
        for name in (
            "abstract_and_captions",
            "artifact_schema",
            "artifacts",
            "asymptotics",
            "bibliography",
            "ci_contract",
            "config",
            "derivation_contract",
            "derivation_identities",
            "discussion_prose",
            "elasticity",
            "framing_prose",
            "gll",
            "green_response",
            "isotropic_validation",
            "latex_log",
            "manuscript_evidence",
            "mode_tracking",
            "morse_splitting",
            "provenance",
            "rayleigh_lamb",
            "readme_contract",
            "reference_workflow",
            "release_bundle",
            "reproduce_all",
            "results_prose",
            "semantic_compile",
            "sensitivity",
            "spectral_plate",
            "supplement_contract",
            "validation_metrics",
            "workflow_dependencies",
            "workflow_provenance",
            "zgv",
        )
    ),
)
_REGISTERED_PAPER_FILES: Final = (
    "paper/figure_captions.tex",
    "paper/generated/results_macros.tex",
    "paper/generated/table_s01_convergence.tex",
    "paper/generated/table_s02_parameters.tex",
    "paper/main.tex",
    "paper/references.bib",
    "paper/sections/00_abstract.tex",
    "paper/sections/01_introduction.tex",
    "paper/sections/02_isotropic_ring.tex",
    "paper/sections/03_morse_unfolding.tex",
    "paper/sections/04_temporal_crossover.tex",
    "paper/sections/05_numerical_verification.tex",
    "paper/sections/06_discussion.tex",
    "paper/sections/07_methods.tex",
    "paper/sections/08_conclusion.tex",
    "paper/sections/09_reproducibility_statements.tex",
    "paper/shared_macros.tex",
    "paper/supplement.tex",
    "paper/supplement/01_exact_elasticity.tex",
    "paper/supplement/02_perturbation_proofs.tex",
    "paper/supplement/03_asymptotics.tex",
    "paper/supplement/04_numerical_methods.tex",
    "paper/supplement/05_convergence_and_robustness.tex",
)
_REGISTERED_DOC_FILES: Final = (
    "docs/derivations/01_isotropic_rayleigh_lamb.tex",
    "docs/derivations/02_anisotropic_morse_unfolding.tex",
    "docs/derivations/03_green_function_asymptotics.tex",
    "docs/derivations/04_spectral_numerics.tex",
    "docs/figures/figure_contracts.md",
    "docs/figures/qa_report.md",
    "docs/literature/citation_audit.json",
    "docs/literature/exclusions.csv",
    "docs/literature/novelty_matrix.md",
    "docs/literature/search_candidates.csv",
    "docs/literature/search_log.md",
    "docs/literature/search_protocol.md",
    "docs/literature/seed_dois.txt",
    "docs/manuscript/claim_evidence_matrix.csv",
    "docs/manuscript/terminology_ledger.md",
    "docs/manuscript/title_candidates.md",
    "docs/reviews/mathematical_review.md",
    "docs/reviews/numerical_reproducibility_review.md",
    "docs/reviews/prr_style_review.md",
    "docs/reviews/resolution_matrix.csv",
)
_REGISTERED_WORKFLOW_FILES: Final = (
    ".github/workflows/repro-full.yml",
    ".github/workflows/repro-smoke.yml",
)
_REGISTERED_TREE_GROUPS: Final = (
    (".github/workflows", frozenset({".yaml", ".yml"}), _REGISTERED_WORKFLOW_FILES),
    ("src", frozenset({".py"}), _REGISTERED_SOURCE_FILES),
    ("scripts", frozenset({".py"}), _REGISTERED_SCRIPT_FILES),
    ("tests", frozenset({".py"}), _REGISTERED_TEST_FILES),
    ("paper", frozenset({".bib", ".tex"}), _REGISTERED_PAPER_FILES),
    (
        "docs/derivations",
        frozenset({".tex"}),
        tuple(path for path in _REGISTERED_DOC_FILES if path.startswith("docs/derivations/")),
    ),
    (
        "docs/figures",
        frozenset({".md"}),
        tuple(path for path in _REGISTERED_DOC_FILES if path.startswith("docs/figures/")),
    ),
    (
        "docs/literature",
        frozenset({".csv", ".json", ".md", ".txt"}),
        tuple(path for path in _REGISTERED_DOC_FILES if path.startswith("docs/literature/")),
    ),
    (
        "docs/manuscript",
        frozenset({".csv", ".md"}),
        tuple(path for path in _REGISTERED_DOC_FILES if path.startswith("docs/manuscript/")),
    ),
    (
        "docs/reviews",
        frozenset({".csv", ".md"}),
        tuple(path for path in _REGISTERED_DOC_FILES if path.startswith("docs/reviews/")),
    ),
)


class ReleaseValidationError(RuntimeError):
    """Raised when an input or produced release violates the safety contract."""


@dataclass(frozen=True, slots=True)
class ReleaseInventory:
    """The validated strict allowlist copied into a release."""

    artifact_pairs: tuple[tuple[str, str], ...]
    source_csvs: tuple[str, ...]
    figure_outputs: tuple[str, ...]
    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Paths produced by :func:`build_release`."""

    bundle_dir: Path
    checksums: Path
    archive: Path
    inventory: ReleaseInventory


def _safe_relative_path(raw: object, field: str = "release path") -> str:
    if type(raw) is not str or not raw:
        raise ReleaseValidationError(f"{field} must be a nonempty relative path")
    if (
        "\\" in raw
        or re.match(r"^[A-Za-z]:", raw) is not None
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise ReleaseValidationError(f"{field} contains an unsafe path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or pure.as_posix() != raw or any(
        component in {"", ".", ".."} for component in pure.parts
    ):
        raise ReleaseValidationError(f"{field} is not a canonical safe relative path")
    return raw


def validate_safe_relative_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Reject unsafe, duplicate, casefold, and Unicode-normalization collisions."""

    if isinstance(paths, (str, bytes)):
        raise TypeError("paths must be an iterable of path strings")
    result: list[str] = []
    normalized: dict[str, str] = {}
    normalized_prefixes: dict[str, str] = {}
    for index, raw in enumerate(paths):
        value = _safe_relative_path(raw, f"release path {index}")
        key = unicodedata.normalize("NFKC", value).casefold()
        previous = normalized.get(key)
        if previous is not None:
            if previous == value:
                raise ReleaseValidationError(f"duplicate release path: {value}")
            raise ReleaseValidationError(
                "release path collision after casefold/Unicode normalization: "
                f"{previous!r}, {value!r}"
            )
        normalized[key] = value
        parts = PurePosixPath(value).parts
        for length in range(1, len(parts) + 1):
            prefix = PurePosixPath(*parts[:length]).as_posix()
            prefix_key = unicodedata.normalize("NFKC", prefix).casefold()
            previous_prefix = normalized_prefixes.get(prefix_key)
            if previous_prefix is not None and previous_prefix != prefix:
                raise ReleaseValidationError(
                    "release path-prefix collision after casefold/Unicode normalization: "
                    f"{previous_prefix!r}, {prefix!r}"
                )
            normalized_prefixes[prefix_key] = prefix
        result.append(value)
    return tuple(sorted(result))


def _regular_file(root: Path, relative: str) -> Path:
    relative = _safe_relative_path(relative)
    current = root
    mode = root.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ReleaseValidationError("release source root must be a real directory")
    for component in PurePosixPath(relative).parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise ReleaseValidationError(f"required release file is missing: {relative}") from error
        if stat.S_ISLNK(mode):
            raise ReleaseValidationError(f"release file traverses a symbolic link: {relative}")
    if not stat.S_ISREG(mode):
        raise ReleaseValidationError(f"release path is not a regular file: {relative}")
    return current


def _scan_tree(root: Path, directory: str) -> tuple[str, ...]:
    """List regular files below a real directory, rejecting every unsafe entry."""

    directory = _safe_relative_path(directory, "release directory")
    base = root / directory
    try:
        base_mode = base.lstat().st_mode
    except OSError as error:
        raise ReleaseValidationError(f"required release directory is missing: {directory}") from error
    if stat.S_ISLNK(base_mode) or not stat.S_ISDIR(base_mode):
        raise ReleaseValidationError(f"release directory is unsafe: {directory}")
    files: list[str] = []
    for parent, directories, names in os.walk(base, topdown=True, followlinks=False):
        parent_path = Path(parent)
        for name in sorted(directories):
            path = parent_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                relative = path.relative_to(root).as_posix()
                raise ReleaseValidationError(f"release tree contains a symbolic link/special file: {relative}")
        for name in sorted(names):
            path = parent_path / name
            mode = path.lstat().st_mode
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(mode):
                raise ReleaseValidationError(f"release tree contains a symbolic link: {relative}")
            if not stat.S_ISREG(mode):
                raise ReleaseValidationError(f"release tree contains a special file: {relative}")
            files.append(relative)
    return tuple(sorted(files))


def _collect_suffixes(root: Path, directory: str, suffixes: frozenset[str]) -> tuple[str, ...]:
    return tuple(path for path in _scan_tree(root, directory) if Path(path).suffix in suffixes)


def _require_exact_tree(
    root: Path,
    directory: str,
    expected: Iterable[str],
    description: str,
    *,
    suffixes: frozenset[str] | None = None,
) -> None:
    actual = set(_scan_tree(root, directory))
    if suffixes is not None:
        actual = {path for path in actual if Path(path).suffix in suffixes}
    registered = set(expected)
    missing = sorted(registered.difference(actual))
    extra = sorted(actual.difference(registered))
    if missing:
        raise ReleaseValidationError(f"missing registered {description}: {missing}")
    if extra:
        qualifier = "unregistered source CSV" if description == "source CSV" else f"unregistered {description}"
        raise ReleaseValidationError(f"{qualifier}: {extra}")


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ReleaseValidationError(f"{description} is missing or malformed: {error}") from error
    if type(payload) is not dict:
        raise ReleaseValidationError(f"{description} root must be an object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text(root: Path, *arguments: str) -> str:
    command = ("git", *arguments)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=git_subprocess_environment(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="strict",
        )
    except (subprocess.CalledProcessError, OSError, UnicodeError) as error:
        raise ReleaseValidationError(
            f"release source must be a Git worktree; {' '.join(command)} failed: {error}"
        ) from error
    return completed.stdout


def _git_bytes(root: Path, *arguments: str) -> bytes:
    command = ("git", *arguments)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=git_subprocess_environment(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, OSError) as error:
        raise ReleaseValidationError(
            f"release source must be a Git worktree; {' '.join(command)} failed: {error}"
        ) from error
    return completed.stdout


def _validate_git_release_binding(
    root: Path,
    inventory: ReleaseInventory,
) -> tuple[str, dict[str, tuple[str, str]]]:
    """Bind the report and packaged bytes to the source worktree's exact HEAD."""

    top_text = _git_text(root, "rev-parse", "--show-toplevel").strip()
    try:
        top = Path(top_text).resolve(strict=True)
    except OSError as error:
        raise ReleaseValidationError(f"Git worktree top level is invalid: {error}") from error
    if top != root:
        raise ReleaseValidationError("release source root must be the Git worktree top level")
    head = _git_text(root, "rev-parse", "--verify", "HEAD").strip()
    if _GIT_COMMIT.fullmatch(head) is None:
        raise ReleaseValidationError("Git HEAD must be a full SHA-1 commit identifier")

    report = _read_json_object(root / "data/reproduction_report.json", "reproduction report")
    if report.get("head_commit") != head:
        raise ReleaseValidationError(
            "reproduction report head_commit does not match Git HEAD"
        )

    protected = tuple(
        path for path in inventory.paths if path not in _VERIFIER_UPDATED_PATHS
    )
    tree_output = _git_text(
        root,
        "--literal-pathspecs",
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        head,
        "--",
        *protected,
    )
    head_entries: dict[str, tuple[str, str]] = {}
    for record in tree_output.split("\0"):
        if not record:
            continue
        try:
            metadata, relative = record.split("\t", 1)
            mode, kind, object_id = metadata.split()
        except ValueError as error:
            raise ReleaseValidationError("Git HEAD tree output is malformed") from error
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ReleaseValidationError(
                f"registered release input is not a regular Git blob: {relative}"
            )
        if relative in head_entries:
            raise ReleaseValidationError(f"duplicate Git HEAD release path: {relative}")
        head_entries[relative] = (mode, object_id)
    missing_from_head = sorted(set(protected).difference(head_entries))
    if missing_from_head:
        raise ReleaseValidationError(
            f"registered release inputs are not tracked at Git HEAD: {missing_from_head}"
        )

    index_output = _git_text(
        root,
        "--literal-pathspecs",
        "ls-files",
        "--stage",
        "-z",
        "--",
        *protected,
    )
    index_entries: dict[str, tuple[str, str]] = {}
    for record in index_output.split("\0"):
        if not record:
            continue
        try:
            metadata, relative = record.split("\t", 1)
            mode, object_id, stage = metadata.split()
        except ValueError as error:
            raise ReleaseValidationError("Git index output is malformed") from error
        if stage != "0" or relative in index_entries:
            raise ReleaseValidationError(
                f"registered release input has an unresolved Git index state: {relative}"
            )
        index_entries[relative] = (mode, object_id)
    if index_entries != head_entries:
        changed = sorted(
            path
            for path in set(protected).union(index_entries)
            if index_entries.get(path) != head_entries.get(path)
        )
        raise ReleaseValidationError(
            f"registered release inputs differ from Git HEAD in the index: {changed}"
        )

    working_output = _git_text(
        root,
        "hash-object",
        "--no-filters",
        "--",
        *protected,
    )
    working_ids = working_output.splitlines()
    if len(working_ids) != len(protected):
        raise ReleaseValidationError("Git could not hash every registered release input")
    changed_worktree: list[str] = []
    for relative, working_id in zip(protected, working_ids, strict=True):
        head_mode, head_id = head_entries[relative]
        live_mode = _regular_file(root, relative).stat().st_mode
        live_executable = bool(live_mode & 0o111)
        head_executable = head_mode == "100755"
        if working_id != head_id or live_executable != head_executable:
            changed_worktree.append(relative)
    if changed_worktree:
        raise ReleaseValidationError(
            f"registered release inputs differ from Git HEAD in the worktree: "
            f"{changed_worktree}"
        )

    committed_readme = _git_bytes(root, "show", f"{head}:README.md")
    try:
        expected_readme = measured_readme_bytes(committed_readme, report)
        actual_readme = _regular_file(root, "README.md").read_bytes()
    except (OSError, ReproductionError) as error:
        raise ReleaseValidationError(f"cannot validate verifier-published README: {error}") from error
    if not hmac.compare_digest(actual_readme, expected_readme):
        raise ReleaseValidationError(
            "README does not match the verifier-published form of Git HEAD"
        )
    return head, head_entries


def _validate_staged_release_binding(
    git_root: Path,
    staged_root: Path,
    inventory: ReleaseInventory,
    expected_head: str,
    head_entries: dict[str, tuple[str, str]],
    *,
    checksums_written: bool = False,
) -> None:
    """Bind the completed staging tree to the previously validated Git state."""

    current_head = _git_text(git_root, "rev-parse", "--verify", "HEAD").strip()
    if current_head != expected_head:
        raise ReleaseValidationError("Git HEAD changed during release construction")
    prefix = f"{staged_root.name}/"
    staged_paths = {
        path.removeprefix(prefix)
        for path in _scan_tree(staged_root.parent, staged_root.name)
    }
    expected_paths = set(inventory.paths)
    if checksums_written:
        expected_paths.add(CHECKSUM_NAME)
    if staged_paths != expected_paths:
        missing = sorted(expected_paths.difference(staged_paths))
        extra = sorted(staged_paths.difference(expected_paths))
        raise ReleaseValidationError(
            f"staged release tree differs from the validated allowlist: "
            f"missing={missing}, extra={extra}"
        )

    report_path = staged_root / "data/reproduction_report.json"
    try:
        report = validate_reproduction_report(report_path, staged_root)
    except ReproductionError as error:
        raise ReleaseValidationError(str(error)) from error
    if report["head_commit"] != expected_head:
        raise ReleaseValidationError(
            "staged reproduction report head_commit does not match captured Git HEAD"
        )

    protected = tuple(
        path for path in inventory.paths if path not in _VERIFIER_UPDATED_PATHS
    )
    staged_files = tuple(str(_regular_file(staged_root, path)) for path in protected)
    staged_output = _git_text(
        git_root,
        "hash-object",
        "--no-filters",
        "--",
        *staged_files,
    )
    staged_ids = staged_output.splitlines()
    if len(staged_ids) != len(protected):
        raise ReleaseValidationError("Git could not hash every staged release input")
    changed = [
        relative
        for relative, staged_id in zip(protected, staged_ids, strict=True)
        if staged_id != head_entries[relative][1]
    ]
    if changed:
        raise ReleaseValidationError(
            f"staged registered release inputs differ from Git HEAD: {changed}"
        )

    committed_readme = _git_bytes(git_root, "show", f"{expected_head}:README.md")
    try:
        expected_readme = measured_readme_bytes(committed_readme, report)
        actual_readme = _regular_file(staged_root, "README.md").read_bytes()
    except (OSError, ReproductionError) as error:
        raise ReleaseValidationError(
            f"cannot validate staged verifier-published README: {error}"
        ) from error
    if not hmac.compare_digest(actual_readme, expected_readme):
        raise ReleaseValidationError(
            "staged README does not match the verifier-published form of Git HEAD"
        )


def _validate_citation_audit(root: Path) -> None:
    audit_path = _regular_file(root, "docs/literature/citation_audit.json")
    bib_path = _regular_file(root, "paper/references.bib")
    audit = _read_json_object(audit_path, "citation audit")
    if not audit_passed(audit):
        raise ReleaseValidationError("citation audit did not pass its internal validation")
    if not audit_matches_bibliography(audit, bib_path):
        raise ReleaseValidationError("citation audit does not exactly cover the bibliography")
    if audit.get("online") is not True:
        raise ReleaseValidationError("citation audit must be verified online")
    if audit.get("bib") != "paper/references.bib":
        raise ReleaseValidationError("citation audit BibTeX path is invalid")
    claimed = audit.get("bib_sha256")
    if type(claimed) is not str or _SHA256.fullmatch(claimed) is None:
        raise ReleaseValidationError("citation audit BibTeX hash is malformed")
    if not hmac.compare_digest(claimed, _sha256_file(bib_path)):
        raise ReleaseValidationError("citation audit BibTeX hash mismatch")
    summary = audit.get("summary")
    error_fields = (
        "duplicate_doi",
        "invalid_doi",
        "manual_needed",
        "mismatch",
        "missing_required",
    )
    if type(summary) is not dict or set(summary) != {"entries", *error_fields}:
        raise ReleaseValidationError("citation audit summary is invalid")
    for field in error_fields:
        if type(summary.get(field)) is not int or summary[field] != 0:
            raise ReleaseValidationError(f"citation audit {field} count must be zero")
    if audit.get("duplicates") != []:
        raise ReleaseValidationError("citation audit duplicate records must be empty")
    entries = audit.get("entries")
    if (
        type(summary["entries"]) is not int
        or summary["entries"] <= 0
        or type(entries) is not list
        or len(entries) != summary["entries"]
    ):
        raise ReleaseValidationError("citation audit entry count is invalid")
    for entry in entries:
        if type(entry) is not dict:
            raise ReleaseValidationError("citation audit entry must be an object")
        if (
            entry.get("status") != "core_metadata_verified"
            or entry.get("invalid_doi") is not False
            or entry.get("missing_required") != []
        ):
            raise ReleaseValidationError("citation audit contains an unverified entry")
        comparison = entry.get("comparison")
        if type(comparison) is not dict or set(comparison) != {
            "author_families",
            "container",
            "title",
            "year",
        } or any(
            type(value) is not dict or value.get("match") is not True
            for value in comparison.values()
        ):
            raise ReleaseValidationError("citation audit contains a metadata mismatch")


def _validate_scientific_tree(root: Path, closure: ScientificClosure) -> None:
    artifact_files = {path for pair in closure.artifact_pairs for path in pair}
    artifact_files.update(closure.isotropic_validation)
    _require_exact_tree(root, "data/generated", artifact_files, "generated scientific file")
    _require_exact_tree(root, "data/source_data", closure.source_csvs, "source CSV")
    expected_figure_outputs = set(_EXPECTED_MAIN_FIGURE_OUTPUTS).union(
        _EXPECTED_SUPPLEMENTARY_FIGURE_OUTPUTS
    )
    registered_figure_outputs = set(closure.figure_outputs)
    if registered_figure_outputs != expected_figure_outputs:
        missing = sorted(expected_figure_outputs.difference(registered_figure_outputs))
        unexpected = sorted(registered_figure_outputs.difference(expected_figure_outputs))
        raise ReleaseValidationError(
            f"registered figure output layout mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    _require_exact_tree(
        root,
        "figures/main",
        _EXPECTED_MAIN_FIGURE_OUTPUTS,
        "main figure output",
    )
    _require_exact_tree(
        root,
        "figures/supplementary",
        _EXPECTED_SUPPLEMENTARY_FIGURE_OUTPUTS,
        "supplementary figure output",
    )
    _require_exact_tree(root, "paper/generated", closure.generated_tex, "generated TeX file")
    _require_exact_tree(
        root,
        "build/paper",
        closure.paper_pdfs,
        "compiled paper PDF",
        suffixes=frozenset({".pdf"}),
    )


def _derive_allowlist(root: Path, closure: ScientificClosure) -> tuple[str, ...]:
    paths: list[str] = [
        *_FIXED_FILES,
        *closure.paths,
        *_REGISTERED_WORKFLOW_FILES,
        *_REGISTERED_SOURCE_FILES,
        *_REGISTERED_SCRIPT_FILES,
        *_REGISTERED_TEST_FILES,
        *_REGISTERED_PAPER_FILES,
        *_REGISTERED_DOC_FILES,
    ]
    # Generated TeX is both scientific closure and manuscript source.  Collapse
    # that intentional exact overlap while retaining differently spelled paths
    # so casefold/Unicode collisions are still rejected below.
    return validate_safe_relative_paths(dict.fromkeys(paths))


def validate_release_inputs(root: Path) -> ReleaseInventory:
    """Validate all registered inputs and return the exact release allowlist."""

    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    if root.is_symlink():
        raise ReleaseValidationError("release source root must not be a symbolic link")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise ReleaseValidationError(f"release source root is missing: {error}") from error
    try:
        closure = discover_scientific_closure(root)
    except ReproductionError as error:
        raise ReleaseValidationError(str(error)) from error
    _validate_scientific_tree(root, closure)
    for directory, suffixes, registered in _REGISTERED_TREE_GROUPS:
        _require_exact_tree(
            root,
            directory,
            registered,
            f"allowlist file below {directory}",
            suffixes=suffixes,
        )
    _validate_citation_audit(root)
    try:
        validate_reproduction_report(root / "data/reproduction_report.json", root)
    except ReproductionError as error:
        raise ReleaseValidationError(str(error)) from error
    paths = _derive_allowlist(root, closure)
    for relative in paths:
        _regular_file(root, relative)
    if any(path == "release" or path.startswith("release/") for path in paths):
        raise ReleaseValidationError("release output must never recursively include release/")
    return ReleaseInventory(
        artifact_pairs=closure.artifact_pairs,
        source_csvs=closure.source_csvs,
        figure_outputs=closure.figure_outputs,
        paths=paths,
    )


def _bundle_files(bundle_dir: Path, *, include_checksums: bool) -> tuple[str, ...]:
    files = _scan_tree(bundle_dir.parent, bundle_dir.name)
    prefix = f"{bundle_dir.name}/"
    relative = tuple(path.removeprefix(prefix) for path in files)
    if include_checksums:
        return relative
    return tuple(path for path in relative if path != CHECKSUM_NAME)


def write_checksums(bundle_dir: Path) -> Path:
    """Write a POSIX-sorted GNU-style SHA-256 manifest, excluding itself."""

    if not isinstance(bundle_dir, Path):
        raise TypeError("bundle_dir must be a pathlib.Path")
    bundle_dir = bundle_dir.resolve(strict=True)
    entries: list[str] = []
    for relative in _bundle_files(bundle_dir, include_checksums=False):
        path = _regular_file(bundle_dir, relative)
        entries.append(f"{_sha256_file(path)}  {relative}")
    content = ("\n".join(entries) + "\n").encode("ascii")
    target = bundle_dir / CHECKSUM_NAME
    if target.is_symlink():
        raise ReleaseValidationError("SHA256SUMS target must not be a symbolic link")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=bundle_dir,
        prefix=f".{CHECKSUM_NAME}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    verify_checksums(bundle_dir)
    return target


def verify_checksums(bundle_dir: Path) -> None:
    """Recompute and strictly verify a complete ``SHA256SUMS`` file."""

    if not isinstance(bundle_dir, Path):
        raise TypeError("bundle_dir must be a pathlib.Path")
    bundle_dir = bundle_dir.resolve(strict=True)
    checksum_path = _regular_file(bundle_dir, CHECKSUM_NAME)
    try:
        content = checksum_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ReleaseValidationError(f"cannot read SHA256SUMS: {error}") from error
    if not content.endswith("\n"):
        raise ReleaseValidationError("SHA256SUMS must end with a newline")
    lines = content.splitlines()
    records: list[tuple[str, str]] = []
    for line in lines:
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ReleaseValidationError(f"malformed SHA256SUMS line: {line!r}")
        digest, relative = match.groups()
        relative = _safe_relative_path(relative, "checksum path")
        if relative == CHECKSUM_NAME:
            raise ReleaseValidationError("SHA256SUMS must exclude itself")
        records.append((digest, relative))
    paths = [relative for _, relative in records]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ReleaseValidationError("SHA256SUMS paths must be unique and POSIX-sorted")
    expected = set(_bundle_files(bundle_dir, include_checksums=False))
    if set(paths) != expected:
        missing = sorted(expected.difference(paths))
        extra = sorted(set(paths).difference(expected))
        raise ReleaseValidationError(
            f"SHA256SUMS file closure mismatch: missing={missing}, extra={extra}"
        )
    for claimed, relative in records:
        actual = _sha256_file(_regular_file(bundle_dir, relative))
        if not hmac.compare_digest(claimed, actual):
            raise ReleaseValidationError(f"checksum mismatch for {relative}")


def _tar_entries(bundle_dir: Path) -> tuple[tuple[str, Path, bool], ...]:
    entries: list[tuple[str, Path, bool]] = [(BUNDLE_NAME, bundle_dir, True)]
    for parent, directories, files in os.walk(bundle_dir, topdown=True, followlinks=False):
        parent_path = Path(parent)
        for name in directories:
            path = parent_path / name
            relative = path.relative_to(bundle_dir).as_posix()
            entries.append((f"{BUNDLE_NAME}/{relative}", path, True))
        for name in files:
            path = parent_path / name
            relative = path.relative_to(bundle_dir).as_posix()
            entries.append((f"{BUNDLE_NAME}/{relative}", path, False))
    entries.sort(key=lambda item: item[0])
    for name, path, is_directory in entries:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ReleaseValidationError(f"cannot archive symbolic link: {name}")
        expected = stat.S_ISDIR(mode) if is_directory else stat.S_ISREG(mode)
        if not expected:
            raise ReleaseValidationError(f"cannot archive special file: {name}")
    return tuple(entries)


def _tar_info(name: str, path: Path, is_directory: bool) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if is_directory else 0o644
    if is_directory:
        info.type = tarfile.DIRTYPE
        info.size = 0
    else:
        info.type = tarfile.REGTYPE
        info.size = path.stat().st_size
    return info


def create_tarball(bundle_dir: Path, archive: Path) -> Path:
    """Create a deterministic gzip-compressed USTAR archive with fixed metadata."""

    if not isinstance(bundle_dir, Path) or not isinstance(archive, Path):
        raise TypeError("bundle_dir and archive must be pathlib.Path values")
    bundle_dir = bundle_dir.resolve(strict=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.is_symlink():
        raise ReleaseValidationError("archive target must not be a symbolic link")
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                for name, path, is_directory in _tar_entries(bundle_dir):
                    info = _tar_info(name, path, is_directory)
                    if is_directory:
                        tar.addfile(info)
                    else:
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
        raw.flush()
        os.fsync(raw.fileno())
    archive.chmod(0o644)
    return archive


def _hash_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def verify_tarball(archive: Path, bundle_dir: Path) -> None:
    """Safely reopen an archive and compare every member without extracting it."""

    if not isinstance(archive, Path) or not isinstance(bundle_dir, Path):
        raise TypeError("archive and bundle_dir must be pathlib.Path values")
    if archive.is_symlink() or not archive.is_file():
        raise ReleaseValidationError("release archive is missing or unsafe")
    try:
        with archive.open("rb") as handle:
            gzip_header = handle.read(10)
    except OSError as error:
        raise ReleaseValidationError(f"cannot read release archive metadata: {error}") from error
    if gzip_header != bytes.fromhex("1f8b08000000000002ff"):
        raise ReleaseValidationError("release gzip metadata is not fixed")
    bundle_dir = bundle_dir.resolve(strict=True)
    expected_entries = _tar_entries(bundle_dir)
    expected_names = [entry[0] for entry in expected_entries]
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            names = [member.name for member in members]
            if names != expected_names or len(names) != len(set(names)):
                raise ReleaseValidationError("tar member closure/order mismatch")
            for member, (name, source, is_directory) in zip(
                members, expected_entries, strict=True
            ):
                safe = _safe_relative_path(name, "tar member path")
                if PurePosixPath(safe).parts[0] != BUNDLE_NAME:
                    raise ReleaseValidationError("tar member escapes the bundle root")
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                ):
                    raise ReleaseValidationError(f"tar member metadata is not fixed: {name}")
                if is_directory:
                    if not member.isdir() or member.mode != 0o755 or member.size != 0:
                        raise ReleaseValidationError(f"unsafe tar directory member: {name}")
                    continue
                if not member.isfile() or member.mode != 0o644:
                    raise ReleaseValidationError(f"unsafe tar file member: {name}")
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ReleaseValidationError(f"cannot read tar file member: {name}")
                with extracted:
                    archived_hash = _hash_stream(extracted)
                if member.size != source.stat().st_size or not hmac.compare_digest(
                    archived_hash, _sha256_file(source)
                ):
                    raise ReleaseValidationError(f"tar member content mismatch: {name}")
    except (tarfile.TarError, OSError) as error:
        raise ReleaseValidationError(f"release archive is malformed: {error}") from error

    descriptor, canonical_name = tempfile.mkstemp(
        prefix=f".{ARCHIVE_NAME}.canonical-",
        suffix=".tmp",
    )
    os.close(descriptor)
    canonical = Path(canonical_name)
    try:
        create_tarball(bundle_dir, canonical)
        same_size = archive.stat().st_size == canonical.stat().st_size
        same_hash = hmac.compare_digest(_sha256_file(archive), _sha256_file(canonical))
        if not same_size or not same_hash:
            raise ReleaseValidationError(
                "release archive is not the canonical deterministic byte stream"
            )
    except OSError as error:
        raise ReleaseValidationError(
            f"cannot verify canonical release archive bytes: {error}"
        ) from error
    finally:
        canonical.unlink(missing_ok=True)


def _copy_regular_file(source_root: Path, destination_root: Path, relative: str) -> None:
    source = _regular_file(source_root, relative)
    destination = destination_root.joinpath(*PurePosixPath(relative).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    destination.chmod(0o644)


def _replace_outputs_atomically(
    staging_bundle: Path,
    staging_archive: Path,
    bundle: Path,
    archive: Path,
    verify_installed: Callable[[], None],
) -> None:
    token = uuid.uuid4().hex
    bundle_backup = bundle.parent / f".{bundle.name}.backup-{token}"
    archive_backup = archive.parent / f".{archive.name}.backup-{token}"
    bundle_backed_up = False
    archive_backed_up = False
    bundle_installed = False
    archive_installed = False
    committed = False
    try:
        if os.path.lexists(bundle):
            if bundle.is_symlink() or not bundle.is_dir():
                raise ReleaseValidationError("existing bundle output is unsafe")
            os.replace(bundle, bundle_backup)
            bundle_backed_up = True
        if os.path.lexists(archive):
            if archive.is_symlink() or not archive.is_file():
                raise ReleaseValidationError("existing archive output is unsafe")
            os.replace(archive, archive_backup)
            archive_backed_up = True
        os.replace(staging_bundle, bundle)
        bundle_installed = True
        os.replace(staging_archive, archive)
        archive_installed = True
        verify_installed()
        committed = True
    except BaseException:
        if archive_installed:
            archive.unlink(missing_ok=True)
        if bundle_installed:
            shutil.rmtree(bundle)
        if archive_backed_up:
            os.replace(archive_backup, archive)
        if bundle_backed_up:
            os.replace(bundle_backup, bundle)
        raise
    finally:
        if committed:
            if bundle_backup.exists():
                shutil.rmtree(bundle_backup)
            archive_backup.unlink(missing_ok=True)


def build_release(root: Path = ROOT, output_root: Path | None = None) -> BuildResult:
    """Validate, copy, checksum, archive, and reverify the release allowlist."""

    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    if root.is_symlink():
        raise ReleaseValidationError("release source root must not be a symbolic link")
    source_root = root.resolve(strict=True)
    release_root = source_root / "release" if output_root is None else output_root
    if not isinstance(release_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    resolved_output = release_root.resolve(strict=False)
    default_output = source_root / "release"
    try:
        resolved_output.relative_to(source_root)
    except ValueError:
        pass
    else:
        if resolved_output != default_output:
            raise ReleaseValidationError(
                "release output nested in the source tree must be the excluded release/ directory"
            )
    inventory = validate_release_inputs(source_root)
    captured_head, head_entries = _validate_git_release_binding(source_root, inventory)
    release_root.mkdir(parents=True, exist_ok=True)
    if release_root.is_symlink() or not release_root.is_dir():
        raise ReleaseValidationError("release output root is unsafe")
    release_root = release_root.resolve(strict=True)
    token = uuid.uuid4().hex
    staging_bundle = release_root / f".{BUNDLE_NAME}.staging-{token}"
    staging_archive = release_root / f".{ARCHIVE_NAME}.staging-{token}"
    bundle = release_root / BUNDLE_NAME
    archive = release_root / ARCHIVE_NAME
    try:
        staging_bundle.mkdir()
        for relative in inventory.paths:
            _copy_regular_file(source_root, staging_bundle, relative)
        current_head, current_entries = _validate_git_release_binding(
            source_root, inventory
        )
        if current_head != captured_head or current_entries != head_entries:
            raise ReleaseValidationError(
                "Git HEAD tree changed during release construction"
            )
        _validate_staged_release_binding(
            source_root,
            staging_bundle,
            inventory,
            captured_head,
            head_entries,
        )
        write_checksums(staging_bundle)
        verify_checksums(staging_bundle)
        create_tarball(staging_bundle, staging_archive)
        verify_tarball(staging_archive, staging_bundle)
        _validate_staged_release_binding(
            source_root,
            staging_bundle,
            inventory,
            captured_head,
            head_entries,
            checksums_written=True,
        )
        verify_checksums(staging_bundle)
        verify_tarball(staging_archive, staging_bundle)
        _replace_outputs_atomically(
            staging_bundle,
            staging_archive,
            bundle,
            archive,
            lambda: (verify_checksums(bundle), verify_tarball(archive, bundle)),
        )
    finally:
        if staging_bundle.exists():
            shutil.rmtree(staging_bundle, ignore_errors=True)
        staging_archive.unlink(missing_ok=True)
    return BuildResult(
        bundle_dir=bundle,
        checksums=bundle / CHECKSUM_NAME,
        archive=archive,
        inventory=inventory,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the CLI and build a deterministic release."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args(argv)
    result = build_release(arguments.root, arguments.output_root)
    print(f"release bundle: {result.bundle_dir}")
    print(f"release archive: {result.archive}")


if __name__ == "__main__":
    main()
