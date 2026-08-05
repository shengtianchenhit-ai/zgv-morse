#!/usr/bin/env python3
"""Verify a cold and retained-state full reproduction in a detached worktree."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any, Final
import unicodedata

from zgv_morse.provenance import validate_manifest


ROOT: Final = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION: Final = 1
README_RESULTS_BEGIN: Final = "<!-- BEGIN VERIFIED REPRODUCTION RESULTS -->"
README_RESULTS_END: Final = "<!-- END VERIFIED REPRODUCTION RESULTS -->"
WORKFLOW_STAGES: Final = (
    "isotropic",
    "sensitivity",
    "critical_points",
    "scaling",
    "green",
    "convergence",
    "silicon",
)
ISOTROPIC_VALIDATION_FILES: Final = (
    "data/generated/isotropic_validation.json",
    "data/generated/isotropic_convergence.csv",
)
GENERATED_TEX_FILES: Final = (
    "paper/generated/results_macros.tex",
    "paper/generated/table_s01_convergence.tex",
    "paper/generated/table_s02_parameters.tex",
)
PAPER_PDFS: Final = (
    "build/paper/main.pdf",
    "build/paper/supplement.pdf",
)
GENERATED_DIRECTORIES: Final = (
    "data/generated",
    "data/source_data",
    "figures/main",
    "figures/supplementary",
    "paper/generated",
    "build/paper",
)
CACHE_DIRECTORIES: Final = (
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cache",
)
CACHE_FILES: Final = (".coverage",)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_VERSION_NAMES: Final = frozenset(
    {
        "PyYAML",
        "bibtex",
        "latexmk",
        "matplotlib",
        "mpmath",
        "numpy",
        "pdflatex",
        "pypdf",
        "scipy",
        "sympy",
    }
)
_SUBPROCESS_BASE_ENVIRONMENT: Final = frozenset(
    {
        "ALL_PROXY",
        "COMSPEC",
        "CURL_CA_BUNDLE",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LD_LIBRARY_PATH",
        "NO_PROXY",
        "PATH",
        "PATHEXT",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)
_GIT_BASE_ENVIRONMENT: Final = frozenset(
    {
        "COMSPEC",
        "HOME",
        "PATH",
        "PATHEXT",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
    }
)
_SAFE_GIT_ENVIRONMENT: Final = {
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_COUNT": "0",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_TERMINAL_PROMPT": "0",
}


class ReproductionError(RuntimeError):
    """Raised when clean-room reproduction cannot be proven."""


@dataclass(frozen=True, slots=True)
class ScientificClosure:
    """The complete registered set of generated scientific deliverables."""

    artifact_pairs: tuple[tuple[str, str], ...]
    isotropic_validation: tuple[str, ...]
    manifest: str
    source_csvs: tuple[str, ...]
    figure_outputs: tuple[str, ...]
    generated_tex: tuple[str, ...]
    paper_pdfs: tuple[str, ...]
    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FileDigest:
    """A path-bound digest used in the aggregate scientific hash."""

    path: str
    bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ScientificSnapshot:
    """A deterministic digest of every file in :class:`ScientificClosure`."""

    files: tuple[FileDigest, ...]
    file_count: int
    total_bytes: int
    aggregate_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "aggregate_sha256": self.aggregate_sha256,
            "file_count": self.file_count,
            "files": {record.path: record.as_dict() for record in self.files},
            "total_bytes": self.total_bytes,
        }


Command = tuple[str, ...]
CommandRunner = Callable[[Command, Path, Mapping[str, str]], None]
_TOOL_VERSION_PREFIXES: Final[dict[Command, str]] = {
    ("bibtex", "--version"): "BibTeX ",
    ("latexmk", "-version"): "Latexmk,",
    ("pdflatex", "--version"): "pdfTeX ",
    ("uv", "--version"): "uv ",
}


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ReproductionError(f"reproduction report is not JSON serializable: {error}") from error
    return (text + "\n").encode("utf-8")


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
        raise ReproductionError(f"{description} is missing or malformed: {error}") from error
    if type(payload) is not dict:
        raise ReproductionError(f"{description} root must be an object")
    return payload


def _safe_relative_path(raw: object, field: str) -> str:
    if type(raw) is not str or not raw:
        raise ReproductionError(f"{field} must be a nonempty relative path")
    if (
        "\\" in raw
        or re.match(r"^[A-Za-z]:", raw) is not None
        or any(ord(character) < 32 or ord(character) == 127 for character in raw)
    ):
        raise ReproductionError(f"{field} contains an unsafe path")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.as_posix() != raw or any(
        component in {"", ".", ".."} for component in path.parts
    ):
        raise ReproductionError(f"{field} is not a canonical safe relative path")
    return raw


def _validate_path_set(paths: Sequence[str]) -> tuple[str, ...]:
    normalized: dict[str, str] = {}
    normalized_prefixes: dict[str, str] = {}
    safe: list[str] = []
    for index, raw in enumerate(paths):
        value = _safe_relative_path(raw, f"scientific path {index}")
        collision_key = unicodedata.normalize("NFKC", value).casefold()
        previous = normalized.get(collision_key)
        if previous is not None and previous != value:
            raise ReproductionError(
                f"scientific path collision after case/Unicode normalization: "
                f"{previous!r}, {value!r}"
            )
        if previous == value:
            raise ReproductionError(f"duplicate scientific path: {value}")
        normalized[collision_key] = value
        parts = PurePosixPath(value).parts
        for length in range(1, len(parts) + 1):
            prefix = PurePosixPath(*parts[:length]).as_posix()
            prefix_key = unicodedata.normalize("NFKC", prefix).casefold()
            previous_prefix = normalized_prefixes.get(prefix_key)
            if previous_prefix is not None and previous_prefix != prefix:
                raise ReproductionError(
                    "scientific path-prefix collision after case/Unicode normalization: "
                    f"{previous_prefix!r}, {prefix!r}"
                )
            normalized_prefixes[prefix_key] = prefix
        safe.append(value)
    return tuple(sorted(safe))


def _regular_file(root: Path, relative: str) -> Path:
    """Return a registered file without following any symbolic-link component."""

    relative = _safe_relative_path(relative, "scientific file")
    current = root
    for component in PurePosixPath(relative).parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise ReproductionError(f"scientific file is missing: {relative}") from error
        if stat.S_ISLNK(mode):
            raise ReproductionError(f"scientific file traverses a symbolic link: {relative}")
    if not stat.S_ISREG(mode):
        raise ReproductionError(f"scientific path is not a regular file: {relative}")
    return current


def discover_scientific_closure(root: Path) -> ScientificClosure:
    """Validate the manifest and derive the exact 117-file scientific closure."""

    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    root = root.resolve(strict=True)
    manifest_path = root / "data/provenance_manifest.json"
    try:
        manifest = validate_manifest(manifest_path, require_figures=True)
    except Exception as error:
        raise ReproductionError(f"scientific provenance manifest is invalid: {error}") from error

    artifacts: list[tuple[str, str]] = []
    for name in sorted(manifest["artifacts"]):
        record = manifest["artifacts"][name]
        artifacts.append((f"data/{record['path']}", f"data/{record['sidecar']}"))

    source_csvs = sorted(
        {
            str(record["path"])
            for figure in manifest["figures"].values()
            for record in figure["source_data"]
        }
    )
    figure_outputs = sorted(
        {
            str(record["path"])
            for figure in manifest["figures"].values()
            for record in figure["outputs"].values()
        }
    )
    artifact_paths = [path for pair in artifacts for path in pair]
    paths = _validate_path_set(
        (
            *artifact_paths,
            *ISOTROPIC_VALIDATION_FILES,
            "data/provenance_manifest.json",
            *source_csvs,
            *figure_outputs,
            *GENERATED_TEX_FILES,
            *PAPER_PDFS,
        )
    )
    closure = ScientificClosure(
        artifact_pairs=tuple(artifacts),
        isotropic_validation=tuple(ISOTROPIC_VALIDATION_FILES),
        manifest="data/provenance_manifest.json",
        source_csvs=tuple(source_csvs),
        figure_outputs=tuple(figure_outputs),
        generated_tex=tuple(GENERATED_TEX_FILES),
        paper_pdfs=tuple(PAPER_PDFS),
        paths=paths,
    )
    expected_counts = {
        "artifact pairs": (len(closure.artifact_pairs), 7),
        "source CSVs": (len(closure.source_csvs), 47),
        "figure outputs": (len(closure.figure_outputs), 48),
        "generated TeX files": (len(closure.generated_tex), 3),
        "paper PDFs": (len(closure.paper_pdfs), 2),
        "scientific files": (len(closure.paths), 117),
    }
    wrong = [f"{name}={actual}, expected {expected}" for name, (actual, expected) in expected_counts.items() if actual != expected]
    if wrong:
        raise ReproductionError("scientific closure count mismatch: " + "; ".join(wrong))
    for relative in closure.paths:
        _regular_file(root, relative)
    return closure


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_scientific_closure(root: Path) -> ScientificSnapshot:
    """Hash every scientific file and bind the path, size, and digest together."""

    closure = discover_scientific_closure(root)
    records: list[FileDigest] = []
    aggregate = hashlib.sha256()
    for relative in closure.paths:
        path = _regular_file(root.resolve(strict=True), relative)
        size = path.stat().st_size
        digest = _sha256_file(path)
        record = FileDigest(relative, size, digest)
        records.append(record)
        encoded = relative.encode("utf-8")
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
        aggregate.update(size.to_bytes(16, "big"))
        aggregate.update(bytes.fromhex(digest))
    return ScientificSnapshot(
        files=tuple(records),
        file_count=len(records),
        total_bytes=sum(record.bytes for record in records),
        aggregate_sha256=aggregate.hexdigest(),
    )


def cold_reproduction_commands() -> tuple[Command, ...]:
    """Return the explicit cold sequence; no omnibus warm-cache shortcut is used."""

    commands: list[Command] = [("uv", "sync", "--frozen", "--all-extras")]
    commands.extend(
        (
            "uv",
            "run",
            "python",
            "-m",
            "zgv_morse.workflows",
            "--stage",
            stage,
            "--profile",
            "full",
        )
        for stage in WORKFLOW_STAGES
    )
    commands.append(("uv", "run", "python", "scripts/validate_isotropic.py"))
    commands.extend(
        ("uv", "run", "python", f"scripts/make_figure_{number:02d}.py")
        for number in range(1, 7)
    )
    commands.extend(
        (
            ("uv", "run", "python", "scripts/make_supplementary_figures.py"),
            ("uv", "run", "python", "scripts/export_supplement_tables.py"),
            ("uv", "run", "python", "scripts/qa_figures.py", "--strict"),
            ("uv", "run", "python", "scripts/export_manuscript_values.py"),
            ("uv", "run", "python", "scripts/compile_paper.py"),
            ("uv", "run", "pytest", "-q"),
            (
                "uv",
                "run",
                "python",
                "scripts/check_claim_evidence.py",
                "--require-supported",
            ),
        )
    )
    return tuple(commands)


def retained_state_command() -> Command:
    """Return the retained-state rebuild, which still recomputes all stages."""

    return (
        "uv",
        "run",
        "python",
        "scripts/reproduce_all.py",
        "--profile",
        "full",
    )


def deterministic_environment(
    workspace: Path,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Create an isolated, deterministic single-thread subprocess environment."""

    if not isinstance(workspace, Path):
        raise TypeError("workspace must be a pathlib.Path")
    source = os.environ if base is None else base
    if not isinstance(source, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in source.items()
    ):
        raise TypeError("base must map strings to strings")
    environment = {
        key: value for key, value in source.items() if key in _SUBPROCESS_BASE_ENVIRONMENT
    }
    cache_root = workspace.parent / "isolated-cache"
    environment.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(workspace / ".venv"),
            "UV_CACHE_DIR": str(cache_root / "uv"),
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "MPLBACKEND": "Agg",
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
            "LC_ALL": "C",
            "LANG": "C",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        }
    )
    return environment


def git_subprocess_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a minimal environment for local Git provenance operations."""

    source = os.environ if base is None else base
    if not isinstance(source, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in source.items()
    ):
        raise TypeError("base must map strings to strings")
    environment = {
        key: value for key, value in source.items() if key in _GIT_BASE_ENVIRONMENT
    }
    environment.update({"LANG": "C", "LC_ALL": "C"})
    environment.update(_SAFE_GIT_ENVIRONMENT)
    return environment


def _remove_path(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ReproductionError(f"refusing to delete outside temporary checkout: {path}") from error
    if not os.path.lexists(path):
        return
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or stat.S_ISREG(mode):
        path.unlink()
    elif stat.S_ISDIR(mode):
        shutil.rmtree(path)
    else:
        raise ReproductionError(f"refusing to delete special file: {path}")


def delete_generated_outputs(root: Path, relative_paths: Sequence[str]) -> tuple[str, ...]:
    """Delete only registered generated outputs and local caches in a temp checkout."""

    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    root = root.resolve(strict=True)
    safe = _validate_path_set(tuple(relative_paths))
    present = tuple(relative for relative in safe if os.path.lexists(root / relative))

    # Whole registered generated directories are removed so stale logs and caches
    # cannot influence the cold run.  No source directory is on this allowlist.
    for relative in GENERATED_DIRECTORIES:
        _remove_path(root / relative, root)
    for relative in safe:
        _remove_path(root / relative, root)
    for relative in CACHE_DIRECTORIES:
        _remove_path(root / relative, root)
    for relative in CACHE_FILES:
        _remove_path(root / relative, root)
    for cache in sorted(root.glob(".coverage.*")):
        _remove_path(cache, root)
    for cache in sorted(root.rglob("__pycache__"), reverse=True):
        _remove_path(cache, root)
    for cache in sorted(root.rglob("*.py[co]")):
        _remove_path(cache, root)
    return present


def generated_output_paths(closure: ScientificClosure) -> tuple[str, ...]:
    """Return the closure subset that a cold checkout must regenerate."""

    if not isinstance(closure, ScientificClosure):
        raise TypeError("closure must be a ScientificClosure")
    return closure.paths


def _run_command(command: Command, cwd: Path, environment: Mapping[str, str]) -> None:
    try:
        subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment),
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise ReproductionError(
            f"command failed with exit code {error.returncode}: {' '.join(command)}"
        ) from error
    except OSError as error:
        raise ReproductionError(f"cannot execute {' '.join(command)}: {error}") from error


def _run_commands(
    commands: Sequence[Command],
    workspace: Path,
    environment: Mapping[str, str],
    runner: CommandRunner,
) -> float:
    started = time.perf_counter()
    for command in commands:
        runner(command, workspace, environment)
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ReproductionError("invalid reproduction wall time")
    return elapsed


def _git_output(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            env=git_subprocess_environment(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="strict",
        )
    except (subprocess.CalledProcessError, OSError, UnicodeError) as error:
        raise ReproductionError(f"git {' '.join(arguments)} failed: {error}") from error
    return completed.stdout


def require_clean_committed_baseline(root: Path) -> tuple[str, ScientificSnapshot]:
    """Require that HEAD and the current tracked scientific baseline do not diverge."""

    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    root = root.resolve(strict=True)
    top = Path(_git_output(root, "rev-parse", "--show-toplevel").strip()).resolve(strict=True)
    if top != root:
        raise ReproductionError(f"root is not the Git worktree top level: {root}")
    head = _git_output(root, "rev-parse", "--verify", "HEAD").strip()
    if _COMMIT.fullmatch(head) is None:
        raise ReproductionError("HEAD is not a full SHA-1 commit identifier")
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ReproductionError("working tree is not clean; commit or remove all changes first")

    closure = discover_scientific_closure(root)
    tracked_output = _git_output(root, "ls-files", "-z", "--", *closure.paths)
    tracked = {item for item in tracked_output.split("\0") if item}
    missing = sorted(set(closure.paths).difference(tracked))
    if missing:
        raise ReproductionError(f"scientific baseline files are not committed: {missing}")
    return head, snapshot_scientific_closure(root)


@contextmanager
def detached_worktree(root: Path, head_commit: str) -> Iterator[Path]:
    """Yield a detached temporary checkout and remove it robustly on every exit."""

    root = root.resolve(strict=True)
    if _COMMIT.fullmatch(head_commit) is None:
        raise ReproductionError("head_commit must be a full SHA-1 identifier")
    temporary_root = Path(tempfile.mkdtemp(prefix="zgv-morse-reproduction-"))
    checkout = temporary_root / "checkout"
    active_error: BaseException | None = None
    try:
        try:
            subprocess.run(
                ("git", "worktree", "add", "--detach", str(checkout), head_commit),
                cwd=root,
                env=git_subprocess_environment(),
                check=True,
            )
        except (subprocess.CalledProcessError, OSError) as error:
            raise ReproductionError(f"cannot create detached temporary worktree: {error}") from error
        yield checkout
    except BaseException as error:
        active_error = error

    cleanup_error: ReproductionError | None = None
    try:
        _cleanup_detached_worktree(root, temporary_root, checkout)
    except ReproductionError as error:
        cleanup_error = error
    if cleanup_error is not None:
        if active_error is not None:
            raise ReproductionError(
                f"detached worktree cleanup could not be proven after "
                f"{type(active_error).__name__}: {active_error}; {cleanup_error}"
            ) from active_error
        raise cleanup_error
    if active_error is not None:
        raise active_error.with_traceback(active_error.__traceback__)


def _cleanup_git_worktree_command(
    root: Path,
    arguments: Command,
    diagnostics: list[str],
) -> subprocess.CompletedProcess[str] | None:
    command = ("git", "worktree", *arguments)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=git_subprocess_environment(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        diagnostics.append(f"{' '.join(command)} could not run: {error}")
        return None
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        diagnostics.append(
            f"{' '.join(command)} exited {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    return completed


def _registered_worktree_paths(
    root: Path,
    diagnostics: list[str],
) -> tuple[Path, ...] | None:
    completed = _cleanup_git_worktree_command(
        root,
        ("list", "--porcelain", "-z"),
        diagnostics,
    )
    if completed is None or completed.returncode != 0:
        return None
    paths: list[Path] = []
    for field in (completed.stdout or "").split("\0"):
        if field.startswith("worktree "):
            paths.append(Path(field.removeprefix("worktree ")).resolve(strict=False))
    return tuple(paths)


def _cleanup_detached_worktree(root: Path, temporary_root: Path, checkout: Path) -> None:
    """Remove checkout files and metadata, or fail with complete diagnostics."""

    diagnostics: list[str] = []
    _cleanup_git_worktree_command(
        root,
        ("remove", "--force", str(checkout)),
        diagnostics,
    )
    _cleanup_git_worktree_command(root, ("prune", "--expire", "now"), diagnostics)
    if os.path.lexists(temporary_root):
        try:
            shutil.rmtree(temporary_root)
        except OSError as error:
            diagnostics.append(f"cannot delete temporary worktree files: {error}")

    # Deleting the checkout can turn a failed remove into stale metadata.  This
    # final prune must therefore occur after the filesystem fallback.
    _cleanup_git_worktree_command(root, ("prune", "--expire", "now"), diagnostics)
    registered = _registered_worktree_paths(root, diagnostics)
    files_removed = not os.path.lexists(temporary_root)
    checkout_key = checkout.resolve(strict=False)
    metadata_removed = registered is not None and checkout_key not in registered
    if files_removed and metadata_removed:
        return
    if not files_removed:
        diagnostics.append(f"temporary worktree path still exists: {temporary_root}")
    if registered is None:
        diagnostics.append("could not inspect registered Git worktrees")
    elif not metadata_removed:
        diagnostics.append(f"temporary checkout remains registered: {checkout_key}")
    raise ReproductionError(
        "detached worktree cleanup could not be proven: " + "; ".join(diagnostics)
    )


def run_in_temporary_worktree(
    workspace_factory: Callable[[], Any],
    verifier: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Run a verifier inside an injected context manager (unit-test seam)."""

    with workspace_factory() as workspace:
        return verifier(workspace)


def _snapshot_equal(reference: ScientificSnapshot, candidate: ScientificSnapshot, label: str) -> None:
    if reference == candidate:
        return
    reference_files = {record.path: record for record in reference.files}
    candidate_files = {record.path: record for record in candidate.files}
    changed = sorted(
        path
        for path in set(reference_files) | set(candidate_files)
        if reference_files.get(path) != candidate_files.get(path)
    )
    raise ReproductionError(
        f"{label} scientific hash divergence in {len(changed)} file(s): {changed[:10]}"
    )


def _tool_version(command: Command, cwd: Path, environment: Mapping[str, str]) -> str:
    expected_prefix = _TOOL_VERSION_PREFIXES.get(command)
    if expected_prefix is None:
        raise ReproductionError(
            f"tool version command has no declared banner prefix: {' '.join(command)}"
        )
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        raise ReproductionError(f"cannot record tool version for {' '.join(command)}: {error}") from error
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith(expected_prefix):
            return line
    raise ReproductionError(
        f"tool version command did not produce the expected {expected_prefix!r} "
        f"version banner: {' '.join(command)}"
    )


def collect_environment(workspace: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    """Record architecture, rendering backend, interpreter, and dependency versions."""

    version_program = (
        "import importlib.metadata as m, json, matplotlib, platform; "
        "names=('numpy','scipy','sympy','matplotlib','mpmath','PyYAML','pypdf'); "
        "print(json.dumps({'architecture': platform.machine(), "
        "'backend': matplotlib.get_backend(), "
        "'operating_system': platform.platform(), "
        "'python': platform.python_version(), "
        "'versions': {name: m.version(name) for name in names}}, sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            ("uv", "run", "python", "-c", version_program),
            cwd=workspace,
            env=dict(environment),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="strict",
            timeout=60,
        )
        python_payload = json.loads(completed.stdout)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise ReproductionError(f"cannot record reproduction environment: {error}") from error
    if type(python_payload) is not dict:
        raise ReproductionError("Python environment probe did not return an object")
    probe: dict[str, str] = {}
    for field in ("architecture", "backend", "operating_system", "python"):
        value = python_payload.get(field)
        if type(value) is not str or not value.strip():
            raise ReproductionError(
                f"Python environment probe field {field} must be a nonempty string"
            )
        probe[field] = value
    raw_versions = python_payload.get("versions")
    if type(raw_versions) is not dict:
        raise ReproductionError("Python environment probe versions must be an object")
    versions = dict(raw_versions)
    versions.update(
        {
            "bibtex": _tool_version(("bibtex", "--version"), workspace, environment),
            "latexmk": _tool_version(("latexmk", "-version"), workspace, environment),
            "pdflatex": _tool_version(("pdflatex", "--version"), workspace, environment),
        }
    )
    return {
        "architecture": probe["architecture"],
        "backend": probe["backend"],
        "operating_system": probe["operating_system"],
        "python": probe["python"],
        "uv": _tool_version(("uv", "--version"), workspace, environment),
        "versions": versions,
    }


def make_reproduction_report(
    *,
    head_commit: str,
    baseline: ScientificSnapshot,
    cold_run: ScientificSnapshot,
    retained_state_run: ScientificSnapshot,
    cold_wall_seconds: float,
    retained_wall_seconds: float,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical success-only report payload."""

    if _COMMIT.fullmatch(head_commit) is None:
        raise ReproductionError("head_commit must be a full SHA-1 identifier")
    for label, value in (
        ("cold_wall_seconds", cold_wall_seconds),
        ("retained_wall_seconds", retained_wall_seconds),
    ):
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ReproductionError(f"{label} must be a finite nonnegative number")
    _snapshot_equal(baseline, cold_run, "cold run")
    _snapshot_equal(baseline, retained_state_run, "retained-state run")
    return {
        "baseline": baseline.as_dict(),
        "cold_run": {
            "commands": [list(command) for command in cold_reproduction_commands()],
            "scientific_snapshot": cold_run.as_dict(),
            "wall_seconds": float(cold_wall_seconds),
        },
        "environment": dict(environment),
        "head_commit": head_commit,
        "persistent_scientific_cache": False,
        "profile": "full",
        "retained_state_run": {
            "commands": [list(retained_state_command())],
            "scientific_snapshot": retained_state_run.as_dict(),
            "wall_seconds": float(retained_wall_seconds),
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified",
    }


def _validate_snapshot_payload(
    payload: object,
    expected: ScientificSnapshot,
    label: str,
) -> dict[str, Any]:
    keys = {"aggregate_sha256", "file_count", "files", "total_bytes"}
    if type(payload) is not dict or set(payload) != keys:
        raise ReproductionError(f"{label} scientific snapshot schema is invalid")
    if type(payload["file_count"]) is not int or payload["file_count"] != expected.file_count:
        raise ReproductionError(f"{label} scientific file count mismatch")
    if type(payload["total_bytes"]) is not int or payload["total_bytes"] != expected.total_bytes:
        raise ReproductionError(f"{label} scientific hash byte-count mismatch")
    aggregate = payload["aggregate_sha256"]
    if type(aggregate) is not str or _SHA256.fullmatch(aggregate) is None:
        raise ReproductionError(f"{label} aggregate scientific hash is malformed")
    if aggregate != expected.aggregate_sha256:
        raise ReproductionError(f"{label} aggregate scientific hash mismatch")
    files = payload["files"]
    expected_files = expected.as_dict()["files"]
    if type(files) is not dict or files != expected_files:
        raise ReproductionError(f"{label} per-file scientific hash mismatch")
    return payload


def validate_reproduction_report(path: Path, root: Path) -> dict[str, Any]:
    """Validate report schema and bind every reported hash to current files."""

    if not isinstance(path, Path) or not isinstance(root, Path):
        raise TypeError("path and root must be pathlib.Path values")
    if path.is_symlink():
        raise ReproductionError("reproduction report must not be a symbolic link")
    payload = _read_json_object(path, "reproduction report")
    keys = {
        "baseline",
        "cold_run",
        "environment",
        "head_commit",
        "persistent_scientific_cache",
        "profile",
        "retained_state_run",
        "schema_version",
        "status",
    }
    if set(payload) != keys:
        raise ReproductionError("reproduction report root schema is invalid")
    if payload["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ReproductionError("reproduction report schema_version is unsupported")
    if payload["status"] != "verified":
        raise ReproductionError("reproduction report status must be verified")
    if payload["profile"] != "full":
        raise ReproductionError("reproduction report profile must be full")
    if payload["persistent_scientific_cache"] is not False:
        raise ReproductionError("persistent_scientific_cache must be false")
    if type(payload["head_commit"]) is not str or _COMMIT.fullmatch(payload["head_commit"]) is None:
        raise ReproductionError("reproduction report head_commit is malformed")

    current = snapshot_scientific_closure(root)
    _validate_snapshot_payload(payload["baseline"], current, "baseline")
    expected_commands: tuple[tuple[str, tuple[Command, ...]], ...] = (
        ("cold_run", cold_reproduction_commands()),
        ("retained_state_run", (retained_state_command(),)),
    )
    for label, commands in expected_commands:
        run = payload[label]
        if type(run) is not dict or set(run) != {
            "commands",
            "scientific_snapshot",
            "wall_seconds",
        }:
            raise ReproductionError(f"reproduction report {label} schema is invalid")
        expected_command_lists = [list(command) for command in commands]
        if run["commands"] != expected_command_lists:
            raise ReproductionError(f"reproduction report {label} commands are invalid")
        wall = run["wall_seconds"]
        if type(wall) not in {int, float} or not math.isfinite(wall) or wall < 0:
            raise ReproductionError(f"reproduction report {label} wall_seconds is invalid")
        _validate_snapshot_payload(run["scientific_snapshot"], current, label)

    environment = payload["environment"]
    environment_keys = {
        "architecture",
        "backend",
        "operating_system",
        "python",
        "uv",
        "versions",
    }
    if type(environment) is not dict or set(environment) != environment_keys:
        raise ReproductionError("reproduction report environment schema is invalid")
    for key in environment_keys.difference({"versions"}):
        if type(environment[key]) is not str or not environment[key]:
            raise ReproductionError(f"reproduction report environment.{key} is invalid")
    versions = environment["versions"]
    if (
        type(versions) is not dict
        or set(versions) != _REQUIRED_VERSION_NAMES
        or any(
            type(key) is not str or type(value) is not str or not value
            for key, value in versions.items()
        )
    ):
        raise ReproductionError("reproduction report environment.versions is invalid")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def atomic_write_report(path: Path, report: Mapping[str, Any]) -> Path:
    """Atomically write canonical JSON without risking an existing report."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    content = _canonical_json(report)
    if path.is_symlink():
        raise ReproductionError("reproduction report target must not be a symbolic link")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def measured_readme_bytes(content: bytes, report: Mapping[str, Any]) -> bytes:
    """Return the verifier-published README bytes for baseline ``content``."""

    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    try:
        text = content.decode("utf-8")
    except UnicodeError as error:
        raise ReproductionError(f"README is malformed: {error}") from error
    if text.count(README_RESULTS_BEGIN) != 1 or text.count(README_RESULTS_END) != 1:
        raise ReproductionError("README must contain exactly one measured-results sentinel block")
    begin = text.index(README_RESULTS_BEGIN)
    end = text.index(README_RESULTS_END, begin)
    if end <= begin:
        raise ReproductionError("README measured-results sentinel order is invalid")

    baseline = report["baseline"]
    cold = report["cold_run"]
    retained = report["retained_state_run"]
    block = "\n".join(
        (
            README_RESULTS_BEGIN,
            "Verified by the clean-room reproducer using measured values:",
            "",
            f"- Scientific closure: `{baseline['file_count']}` files, "
            f"`{baseline['total_bytes']}` bytes.",
            f"- Aggregate SHA-256: `{baseline['aggregate_sha256']}`.",
            f"- Cold-run wall time: `{cold['wall_seconds']:.6f}` seconds.",
            "- Retained-state wall time: "
            f"`{retained['wall_seconds']:.6f}` seconds.",
            "- Persistent scientific stage cache: `false`; all scientific "
            "stages were recomputed.",
            README_RESULTS_END,
        )
    )
    updated = text[:begin] + block + text[end + len(README_RESULTS_END) :]
    return updated.encode("utf-8")


def _measured_readme_bytes(readme_path: Path, report: Mapping[str, Any]) -> bytes:
    try:
        content = readme_path.read_bytes()
    except OSError as error:
        raise ReproductionError(f"README is missing or malformed: {error}") from error
    return measured_readme_bytes(content, report)


def _stage_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _replace_report_and_readme(
    staged_report: Path,
    staged_readme: Path,
    report_path: Path,
    readme_path: Path,
) -> None:
    token = os.urandom(8).hex()
    report_backup = report_path.parent / f".{report_path.name}.backup-{token}"
    readme_backup = readme_path.parent / f".{readme_path.name}.backup-{token}"
    had_report = os.path.lexists(report_path)
    report_installed = False
    readme_installed = False
    committed = False
    try:
        if had_report:
            os.replace(report_path, report_backup)
        os.replace(readme_path, readme_backup)
        os.replace(staged_report, report_path)
        report_installed = True
        os.replace(staged_readme, readme_path)
        readme_installed = True
        _fsync_directory(report_path.parent)
        if readme_path.parent != report_path.parent:
            _fsync_directory(readme_path.parent)
        committed = True
    except BaseException:
        if readme_installed:
            readme_path.unlink(missing_ok=True)
        if report_installed:
            report_path.unlink(missing_ok=True)
        if readme_backup.exists():
            os.replace(readme_backup, readme_path)
        if had_report and report_backup.exists():
            os.replace(report_backup, report_path)
        raise
    finally:
        if committed:
            report_backup.unlink(missing_ok=True)
            readme_backup.unlink(missing_ok=True)


def publish_reproduction_results(
    root: Path,
    report_path: Path,
    readme_path: Path,
    report: Mapping[str, Any],
) -> None:
    """Validate and transactionally publish the report and measured README block."""

    if not all(isinstance(path, Path) for path in (root, report_path, readme_path)):
        raise TypeError("root, report_path, and readme_path must be pathlib.Path values")
    root = root.resolve(strict=True)
    expected_report = root / "data/reproduction_report.json"
    expected_readme = root / "README.md"
    if report_path.resolve(strict=False) != expected_report or readme_path.resolve(
        strict=False
    ) != expected_readme:
        raise ReproductionError("reproduction results must use canonical repository paths")
    if report_path.is_symlink() or readme_path.is_symlink() or not readme_path.is_file():
        raise ReproductionError("reproduction result targets are missing or unsafe")

    staged_report = _stage_bytes(report_path, _canonical_json(report))
    staged_readme: Path | None = None
    try:
        # Validate the exact bytes to be installed before either live file changes.
        validated = validate_reproduction_report(staged_report, root)
        staged_readme = _stage_bytes(
            readme_path,
            _measured_readme_bytes(readme_path, validated),
        )
        _replace_report_and_readme(
            staged_report,
            staged_readme,
            report_path,
            readme_path,
        )
    finally:
        staged_report.unlink(missing_ok=True)
        if staged_readme is not None:
            staged_readme.unlink(missing_ok=True)


def write_report_after_success(
    path: Path,
    verifier: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Call ``verifier`` first, so failures leave the prior report untouched."""

    report = verifier()
    atomic_write_report(path, report)
    return report


def verify_clean_reproduction(
    root: Path = ROOT,
    *,
    report_path: Path | None = None,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    """Run both isolated reproductions and atomically publish a success report."""

    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    root = root.resolve(strict=True)
    destination = root / "data/reproduction_report.json" if report_path is None else report_path
    if not isinstance(destination, Path):
        raise TypeError("report_path must be a pathlib.Path")
    canonical_report = root / "data/reproduction_report.json"
    if destination.is_symlink() or destination.resolve(strict=False) != canonical_report:
        raise ReproductionError("report_path must be canonical data/reproduction_report.json")

    head_commit, baseline = require_clean_committed_baseline(root)

    def perform() -> dict[str, Any]:
        with detached_worktree(root, head_commit) as workspace:
            environment = deterministic_environment(workspace)
            closure = discover_scientific_closure(root)
            delete_generated_outputs(workspace, generated_output_paths(closure))
            cold_wall = _run_commands(
                cold_reproduction_commands(), workspace, environment, runner
            )
            cold_snapshot = snapshot_scientific_closure(workspace)
            _snapshot_equal(baseline, cold_snapshot, "cold run")

            retained_wall = _run_commands(
                (retained_state_command(),), workspace, environment, runner
            )
            retained_snapshot = snapshot_scientific_closure(workspace)
            _snapshot_equal(baseline, retained_snapshot, "retained-state run")
            environment_report = collect_environment(workspace, environment)
            return make_reproduction_report(
                head_commit=head_commit,
                baseline=baseline,
                cold_run=cold_snapshot,
                retained_state_run=retained_snapshot,
                cold_wall_seconds=cold_wall,
                retained_wall_seconds=retained_wall,
                environment=environment_report,
            )

    report = perform()
    try:
        current_head, current_baseline = require_clean_committed_baseline(root)
    except ReproductionError as error:
        raise ReproductionError(
            f"original checkout changed during clean reproduction: {error}"
        ) from error
    if current_head != head_commit:
        raise ReproductionError(
            "original checkout HEAD changed during clean reproduction"
        )
    if current_baseline != baseline:
        raise ReproductionError(
            "original checkout scientific baseline changed during clean reproduction"
        )
    publish_reproduction_results(root, destination, root / "README.md", report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the CLI and run the clean-room verification."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args(argv)
    report = verify_clean_reproduction(arguments.root, report_path=arguments.report)
    baseline = report["baseline"]
    print(
        "clean reproduction verified: "
        f"{baseline['file_count']} files, {baseline['total_bytes']} bytes, "
        f"sha256={baseline['aggregate_sha256']}"
    )


if __name__ == "__main__":
    main()
