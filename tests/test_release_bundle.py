from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Any, Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_release  # noqa: E402
from scripts import verify_clean_reproduction as reproduction  # noqa: E402


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, copy_function=os.link)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)


def _copy_release_source(destination: Path) -> None:
    """Make a cheap, independently removable release-input fixture."""

    destination.mkdir()
    for relative in (
        ".github/workflows",
        ".gitignore",
        ".python-version",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "config/reference.yaml",
        "data/provenance_manifest.json",
        "data/generated",
        "data/source_data",
        "figures/main",
        "figures/supplementary",
        "paper",
        "src",
        "scripts",
        "tests",
        "docs/derivations",
        "docs/figures",
        "docs/literature",
        "docs/manuscript",
        "docs/reviews",
        "build/paper/main.pdf",
        "build/paper/supplement.pdf",
    ):
        _copy_tree(ROOT / relative, destination / relative)

    _git(destination, "init", "-q")
    _git(destination, "config", "user.name", "Release Test")
    _git(destination, "config", "user.email", "release@example.invalid")
    _git(destination, "add", ".")
    _git(
        destination,
        "add",
        "-f",
        "build/paper/main.pdf",
        "build/paper/supplement.pdf",
    )
    _git(destination, "commit", "-qm", "committed release baseline")
    head_commit = _git(destination, "rev-parse", "HEAD").stdout.strip()
    snapshot = reproduction.snapshot_scientific_closure(destination)
    report = reproduction.make_reproduction_report(
        head_commit=head_commit,
        baseline=snapshot,
        cold_run=snapshot,
        retained_state_run=snapshot,
        cold_wall_seconds=1.0,
        retained_wall_seconds=0.5,
        environment={
            "architecture": "test",
            "backend": "Agg",
            "operating_system": "test",
            "python": "3.12.13",
            "uv": "0.0.0-test",
            "versions": {
                "PyYAML": "test",
                "bibtex": "test",
                "latexmk": "test",
                "matplotlib": "test",
                "mpmath": "test",
                "numpy": "test",
                "pdflatex": "test",
                "pypdf": "test",
                "scipy": "test",
                "sympy": "test",
            },
        },
    )
    reproduction.publish_reproduction_results(
        destination,
        destination / "data/reproduction_report.json",
        destination / "README.md",
        report,
    )


@pytest.fixture(scope="module")
def release_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = tmp_path_factory.mktemp("release-source") / "repository"
    _copy_release_source(source)
    return source


def _clone_fixture(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination, copy_function=os.link)
    return destination


def _replace_text(path: Path, text: str) -> None:
    path.unlink()
    path.write_text(text, encoding="utf-8")


def _directory_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _make_git_repository(root: Path) -> str:
    root.mkdir()
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.name", "Release Test"),
        ("git", "config", "user.email", "release@example.invalid"),
        ("git", "add", "tracked.txt"),
        ("git", "commit", "-qm", "baseline"),
    ):
        subprocess.run(command, cwd=root, check=True)
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    ).stdout.strip()


def _malicious_git_environment(repository: Path) -> dict[str, str]:
    git_dir = repository / ".git"
    return {
        "GIT_DIR": str(git_dir),
        "GIT_WORK_TREE": str(repository),
        "GIT_COMMON_DIR": str(git_dir),
        "GIT_COMMONDIR": str(git_dir),
        "GIT_IMPLICIT_WORK_TREE": "1",
        "GIT_INDEX_FILE": str(git_dir / "index"),
        "GIT_OBJECT_DIRECTORY": str(git_dir / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(git_dir / "objects"),
        "GIT_QUARANTINE_PATH": str(git_dir / "objects"),
        "GIT_NAMESPACE": "malicious",
        "GIT_SHALLOW_FILE": str(git_dir / "malicious-shallow"),
        "GIT_GRAFT_FILE": str(git_dir / "malicious-grafts"),
        "GIT_REPLACE_REF_BASE": "refs/malicious-replacements/",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.worktree",
        "GIT_CONFIG_VALUE_0": str(repository),
        "GIT_CONFIG_PARAMETERS": "'core.worktree=malicious'",
        "GIT_CONFIG": str(repository / "malicious-repository-config"),
        "GIT_CONFIG_GLOBAL": str(repository / "malicious-global-config"),
        "GIT_CONFIG_SYSTEM": str(repository / "malicious-system-config"),
        "GIT_CONFIG_NOSYSTEM": "0",
        "GIT_EXEC_PATH": str(repository / "malicious-git-exec"),
        "GIT_ATTR_SOURCE": "malicious-tree",
        "GIT_ATTR_GLOBAL": str(repository / "malicious-global-attributes"),
        "GIT_ATTR_SYSTEM": str(repository / "malicious-system-attributes"),
        "GIT_ATTR_NOSYSTEM": "0",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_GLOB_PATHSPECS": "1",
        "GIT_NOGLOB_PATHSPECS": "1",
        "GIT_ICASE_PATHSPECS": "1",
        "GIT_CEILING_DIRECTORIES": str(repository),
        "GIT_DISCOVERY_ACROSS_FILESYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "0",
        "GIT_TERMINAL_PROMPT": "1",
    }


_EXPECTED_SAFE_GIT_ENVIRONMENT = {
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_COUNT": "0",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_TERMINAL_PROMPT": "0",
}


def _assert_sanitized_git_environments(
    environments: list[dict[str, str] | None],
) -> None:
    assert environments
    assert all(environment is not None for environment in environments)
    for environment in environments:
        assert environment is not None
        git_environment = {
            key: value
            for key, value in environment.items()
            if key.upper().startswith("GIT_")
        }
        assert git_environment == _EXPECTED_SAFE_GIT_ENVIRONMENT
        assert environment["PATH"] == os.environ["PATH"]
        assert environment["LC_ALL"] == "C"
        assert environment["LANG"] == "C"
        assert "DYLD_LIBRARY_PATH" not in environment
        assert "LD_LIBRARY_PATH" not in environment
        assert "UNRELATED_GIT_TEST_SECRET" not in environment


def _prepare_clean_verifier_source(
    release_source: Path,
    destination: Path,
) -> tuple[Path, bytes, bytes, dict[str, Any]]:
    source = _clone_fixture(release_source, destination)
    report_path = source / "data/reproduction_report.json"
    readme_path = source / "README.md"
    environment = json.loads(report_path.read_text(encoding="utf-8"))["environment"]
    prior_report = b"prior reproduction report sentinel\n"
    prior_readme = readme_path.read_bytes() + b"\nPrior README sentinel.\n"
    _replace_text(report_path, prior_report.decode("utf-8"))
    _replace_text(readme_path, prior_readme.decode("utf-8"))
    _git(source, "add", "-f", "README.md", "data/reproduction_report.json")
    _git(source, "commit", "-qm", "prior published reproduction results")
    assert _git(source, "status", "--porcelain=v1").stdout == ""
    return source, prior_report, prior_readme, environment


def _install_fast_clean_verifier_seams(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    workspace: Path,
    environment_report: dict[str, Any],
) -> None:
    @contextmanager
    def copied_checkout(_root: Path, _head_commit: str) -> Iterator[Path]:
        shutil.copytree(source, workspace, ignore=shutil.ignore_patterns(".git"))
        yield workspace

    def retain_generated_outputs(
        _root: Path,
        _relative_paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        return ()

    monkeypatch.setattr(reproduction, "detached_worktree", copied_checkout)
    monkeypatch.setattr(
        reproduction,
        "delete_generated_outputs",
        retain_generated_outputs,
    )
    monkeypatch.setattr(
        reproduction,
        "collect_environment",
        lambda _workspace, _environment: dict(environment_report),
    )


def test_scientific_closure_has_registered_dynamic_category_counts() -> None:
    closure = reproduction.discover_scientific_closure(ROOT)

    assert len(closure.artifact_pairs) == 7
    assert len(closure.source_csvs) == 47
    assert len(closure.figure_outputs) == 48
    assert len(closure.generated_tex) == 3
    assert len(closure.paper_pdfs) == 2
    assert len(closure.paths) == 117
    assert len(set(closure.paths)) == len(closure.paths)


def test_readme_has_exact_release_commands_and_measurement_sentinel() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    shell_blocks = re.findall(r"```sh\n(.*?)```", text, flags=re.DOTALL)

    assert shell_blocks == [
        "uv sync --frozen --all-extras\n"
        "uv run python scripts/reproduce_all.py --profile smoke\n"
        "uv run python scripts/reproduce_all.py --profile full\n"
    ]
    assert text.count(reproduction.README_RESULTS_BEGIN) == 1
    assert text.count(reproduction.README_RESULTS_END) == 1
    measured_block = text.split(reproduction.README_RESULTS_BEGIN, 1)[1].split(
        reproduction.README_RESULTS_END,
        1,
    )[0]
    if "Pending the release-wide clean reproduction" not in measured_block:
        assert "Verified by the clean-room reproducer using measured values" in measured_block
        assert re.search(r"Scientific closure: `117` files, `\d+` bytes", measured_block)
        assert re.search(r"Aggregate SHA-256: `[0-9a-f]{64}`", measured_block)
        assert re.search(r"Cold-run wall time: `\d+\.\d{6}` seconds", measured_block)
        assert re.search(r"Retained-state wall time: `\d+\.\d{6}` seconds", measured_block)


def test_release_inventory_closes_over_manifest_and_required_content(
    release_source: Path,
) -> None:
    inventory = build_release.validate_release_inputs(release_source)

    assert len(inventory.artifact_pairs) == 7
    assert len(inventory.source_csvs) == 47
    assert len(inventory.figure_outputs) == 48
    assert inventory.paths == tuple(sorted(inventory.paths))
    assert "data/reproduction_report.json" in inventory.paths
    assert "data/provenance_manifest.json" in inventory.paths
    assert "build/paper/main.pdf" in inventory.paths
    assert "build/paper/supplement.pdf" in inventory.paths
    assert ".gitignore" in inventory.paths
    assert ".python-version" in inventory.paths
    assert set(path for path in inventory.paths if path.startswith(".github/workflows/")) == {
        ".github/workflows/repro-full.yml",
        ".github/workflows/repro-smoke.yml",
    }
    assert not any(path == "release" or path.startswith("release/") for path in inventory.paths)


def test_two_release_builds_are_byte_deterministic(
    release_source: Path,
    tmp_path: Path,
) -> None:
    first = build_release.build_release(release_source, tmp_path / "first")
    second = build_release.build_release(release_source, tmp_path / "second")

    assert _directory_digest(first.bundle_dir) == _directory_digest(second.bundle_dir)
    assert first.archive.read_bytes() == second.archive.read_bytes()
    assert first.archive.read_bytes()[4:8] == b"\0\0\0\0"


def test_release_builder_requires_a_git_worktree(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    shutil.rmtree(source / ".git")

    with pytest.raises(build_release.ReleaseValidationError, match="Git worktree"):
        build_release.build_release(source, tmp_path / "release")


def test_release_builder_rejects_report_head_not_matching_git_head(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    report_path = source / "data/reproduction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["head_commit"] = "0" * 40
    reproduction.atomic_write_report(report_path, report)

    with pytest.raises(build_release.ReleaseValidationError, match="head_commit.*Git HEAD"):
        build_release.build_release(source, tmp_path / "release")


def test_release_builder_rejects_report_retained_after_new_code_commit(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    script = source / "scripts/build_release.py"
    _replace_text(
        script,
        script.read_text(encoding="utf-8") + "\n# committed code change\n",
    )
    _git(source, "add", "scripts/build_release.py")
    _git(source, "commit", "-qm", "change release code")

    with pytest.raises(build_release.ReleaseValidationError, match="head_commit.*Git HEAD"):
        build_release.build_release(source, tmp_path / "release")


def test_release_builder_disables_replacement_object_bypass(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    victim = "scripts/build_release.py"
    original = (source / victim).read_text(encoding="utf-8")
    actual_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    _replace_text(source / victim, original + "\n# replacement-object code drift\n")
    _git(source, "add", victim)
    _git(source, "commit", "-qm", "replacement commit")
    replacement = _git(source, "rev-parse", "HEAD").stdout.strip()
    _git(source, "replace", actual_head, replacement)
    _git(source, "reset", "--hard", actual_head)

    report_path = source / "data/reproduction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["head_commit"] = actual_head
    reproduction.publish_reproduction_results(
        source,
        report_path,
        source / "README.md",
        report,
    )

    assert _git(source, "rev-parse", "HEAD").stdout.strip() == actual_head
    assert "replacement-object code drift" in _git(
        source,
        "show",
        f"{actual_head}:{victim}",
    ).stdout
    assert "replacement-object code drift" not in _git(
        source,
        "--no-replace-objects",
        "show",
        f"{actual_head}:{victim}",
    ).stdout

    with pytest.raises(build_release.ReleaseValidationError, match="registered.*Git HEAD"):
        build_release.build_release(source, tmp_path / "release")


def test_release_builder_ignores_replacement_ref_and_packages_actual_head(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    victim = "scripts/build_release.py"
    original = (source / victim).read_bytes()
    actual_head = _git(source, "rev-parse", "HEAD").stdout.strip()
    target = source / victim
    target.unlink()
    target.write_bytes(original + b"\n# replacement-only drift\n")
    _git(source, "add", victim)
    _git(source, "commit", "-qm", "replacement-only commit")
    replacement = _git(source, "rev-parse", "HEAD").stdout.strip()
    _git(source, "replace", actual_head, replacement)
    _git(source, "--no-replace-objects", "reset", "--hard", actual_head)
    report_path = source / "data/reproduction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["head_commit"] = actual_head
    reproduction.publish_reproduction_results(
        source,
        report_path,
        source / "README.md",
        report,
    )

    assert "replacement-only drift" in _git(
        source,
        "show",
        f"{actual_head}:{victim}",
    ).stdout

    result = build_release.build_release(source, tmp_path / "release")

    assert (result.bundle_dir / victim).read_bytes() == original


def test_release_git_runner_sanitizes_repository_object_index_and_config_environment(
    release_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    attacker = tmp_path / "attacker"
    _make_git_repository(attacker)
    malicious = _malicious_git_environment(attacker)
    for key, value in malicious.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/malicious/dyld")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/malicious/ld")
    monkeypatch.setenv("UNRELATED_GIT_TEST_SECRET", "must-not-reach-git")
    real_run = subprocess.run
    environments: list[dict[str, str] | None] = []
    execution_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }

    def record_environment(
        command: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        if command[0] == "git":
            supplied = kwargs.get("env")
            environments.append(None if supplied is None else dict(supplied))
            kwargs["env"] = execution_environment
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(build_release.subprocess, "run", record_environment)

    build_release.build_release(source, tmp_path / "release")

    _assert_sanitized_git_environments(environments)


def test_release_builder_ignores_inherited_git_repository_redirection(
    release_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    attacker = tmp_path / "attacker"
    _make_git_repository(attacker)
    for key, value in _malicious_git_environment(attacker).items():
        monkeypatch.setenv(key, value)

    result = build_release.build_release(source, tmp_path / "release")

    assert (result.bundle_dir / "scripts/build_release.py").read_bytes() == (
        source / "scripts/build_release.py"
    ).read_bytes()


@pytest.mark.parametrize("staged", (False, True), ids=("unstaged", "staged"))
def test_release_builder_rejects_modified_registered_code(
    release_source: Path,
    tmp_path: Path,
    staged: bool,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    script = source / "scripts/build_release.py"
    _replace_text(
        script,
        script.read_text(encoding="utf-8") + "\n# uncommitted code change\n",
    )
    if staged:
        _git(source, "add", "scripts/build_release.py")

    with pytest.raises(build_release.ReleaseValidationError, match="registered.*Git HEAD"):
        build_release.build_release(source, tmp_path / "release")


def test_release_builder_rejects_registered_input_untracked_at_head(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    relative = "scripts/build_release.py"
    script = source / relative
    content = script.read_bytes()
    _git(source, "rm", relative)
    _git(source, "commit", "-qm", "remove registered release input")
    script.write_bytes(content)
    report_path = source / "data/reproduction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["head_commit"] = _git(source, "rev-parse", "HEAD").stdout.strip()
    reproduction.publish_reproduction_results(
        source,
        report_path,
        source / "README.md",
        report,
    )

    with pytest.raises(build_release.ReleaseValidationError, match="not tracked at Git HEAD"):
        build_release.build_release(source, tmp_path / "release")


def test_release_builder_rejects_registered_bytes_changed_only_during_copy(
    release_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    victim = "scripts/build_release.py"
    original = (source / victim).read_bytes()
    real_copy = build_release._copy_regular_file

    def copy_mutated_then_restore(
        source_root: Path,
        destination_root: Path,
        relative: str,
    ) -> None:
        if relative != victim:
            real_copy(source_root, destination_root, relative)
            return
        target = source_root / relative
        _replace_text(target, original.decode("utf-8") + "\n# copy-time mutation\n")
        try:
            real_copy(source_root, destination_root, relative)
        finally:
            _replace_text(target, original.decode("utf-8"))

    monkeypatch.setattr(build_release, "_copy_regular_file", copy_mutated_then_restore)

    with pytest.raises(build_release.ReleaseValidationError, match="staged.*Git HEAD"):
        build_release.build_release(source, tmp_path / "release")

    assert (source / victim).read_bytes() == original


def test_release_builder_rejects_registered_bytes_changed_during_checksums(
    release_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    victim = "scripts/build_release.py"
    real_write_checksums = build_release.write_checksums

    def mutate_staging_then_write(bundle_dir: Path) -> Path:
        target = bundle_dir / victim
        target.write_bytes(target.read_bytes() + b"\n# checksum-time mutation\n")
        return real_write_checksums(bundle_dir)

    monkeypatch.setattr(build_release, "write_checksums", mutate_staging_then_write)

    with pytest.raises(build_release.ReleaseValidationError, match="staged.*Git HEAD"):
        build_release.build_release(source, tmp_path / "release")


def test_release_builder_rejects_readme_not_published_by_verifier(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    readme = source / "README.md"
    _replace_text(
        readme,
        readme.read_text(encoding="utf-8") + "\nUnverified README change.\n",
    )

    with pytest.raises(build_release.ReleaseValidationError, match="README.*verifier"):
        build_release.build_release(source, tmp_path / "release")


def test_head_bound_report_and_measured_readme_build_relocatable_release(
    release_source: Path,
    tmp_path: Path,
) -> None:
    report = reproduction.validate_reproduction_report(
        release_source / "data/reproduction_report.json",
        release_source,
    )
    assert report["head_commit"] == _git(release_source, "rev-parse", "HEAD").stdout.strip()
    assert "Pending the release-wide clean reproduction" not in (
        release_source / "README.md"
    ).read_text(encoding="utf-8")
    assert _git(
        release_source,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout.splitlines() == [" M README.md", "?? data/reproduction_report.json"]

    result = build_release.build_release(release_source, tmp_path / "release")

    assert not (result.bundle_dir / ".git").exists()
    relocated = reproduction.validate_reproduction_report(
        result.bundle_dir / "data/reproduction_report.json",
        result.bundle_dir,
    )
    assert relocated["head_commit"] == report["head_commit"]


def test_existing_default_release_outputs_are_ignored_by_git_binding(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")

    first = build_release.build_release(source)
    first_archive = first.archive.read_bytes()
    second = build_release.build_release(source)

    assert second.archive.read_bytes() == first_archive


def test_post_install_verification_failure_restores_previous_release(
    release_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    output = tmp_path / "release"
    previous = build_release.build_release(source, output)
    old_readme = (previous.bundle_dir / "README.md").read_bytes()
    old_archive = previous.archive.read_bytes()
    report_path = source / "data/reproduction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["cold_run"]["wall_seconds"] += 1.0
    reproduction.publish_reproduction_results(
        source,
        report_path,
        source / "README.md",
        report,
    )
    real_verify_tarball = build_release.verify_tarball

    def fail_installed_archive(archive: Path, bundle_dir: Path) -> None:
        if archive.name == build_release.ARCHIVE_NAME:
            raise build_release.ReleaseValidationError("injected post-install failure")
        real_verify_tarball(archive, bundle_dir)

    monkeypatch.setattr(build_release, "verify_tarball", fail_installed_archive)

    with pytest.raises(
        build_release.ReleaseValidationError,
        match="injected post-install failure",
    ):
        build_release.build_release(source, output)

    assert (previous.bundle_dir / "README.md").read_bytes() == old_readme
    assert previous.archive.read_bytes() == old_archive
    build_release.verify_checksums(previous.bundle_dir)
    real_verify_tarball(previous.archive, previous.bundle_dir)
    assert not list(output.glob(f".{build_release.BUNDLE_NAME}.backup-*"))
    assert not list(output.glob(f".{build_release.ARCHIVE_NAME}.backup-*"))
    assert not list(output.glob("*.staging-*"))
    assert not list(output.glob(".*.staging-*"))


def test_checksum_manifest_is_sorted_complete_and_reverified(
    release_source: Path,
    tmp_path: Path,
) -> None:
    result = build_release.build_release(release_source, tmp_path / "release")
    lines = result.checksums.read_text(encoding="ascii").splitlines()

    assert lines == sorted(lines, key=lambda line: line[66:])
    assert all(len(line) > 66 and line[64:66] == "  " for line in lines)
    assert all(len(line[:64]) == 64 and int(line[:64], 16) >= 0 for line in lines)
    checksummed = {line[66:] for line in lines}
    expected = {
        path.relative_to(result.bundle_dir).as_posix()
        for path in result.bundle_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert checksummed == expected
    build_release.verify_checksums(result.bundle_dir)

    victim = result.bundle_dir / min(checksummed)
    victim.write_bytes(victim.read_bytes() + b"damage")
    with pytest.raises(build_release.ReleaseValidationError, match="checksum mismatch"):
        build_release.verify_checksums(result.bundle_dir)


def test_checksum_verifier_rejects_malformed_duplicate_and_incomplete_manifests(
    release_source: Path,
    tmp_path: Path,
) -> None:
    result = build_release.build_release(release_source, tmp_path / "release")
    original = result.checksums.read_text(encoding="ascii")
    lines = original.splitlines()
    damaged = (
        "not-a-checksum\n",
        original + lines[0] + "\n",
        "\n".join(lines[:-1]) + "\n",
    )
    for content in damaged:
        result.checksums.write_text(content, encoding="ascii")
        with pytest.raises(build_release.ReleaseValidationError):
            build_release.verify_checksums(result.bundle_dir)
    result.checksums.write_text(original, encoding="ascii")
    build_release.verify_checksums(result.bundle_dir)


def test_tar_members_are_sorted_safe_regular_content(
    release_source: Path,
    tmp_path: Path,
) -> None:
    result = build_release.build_release(release_source, tmp_path / "release")
    build_release.verify_tarball(result.archive, result.bundle_dir)

    with tarfile.open(result.archive, "r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == sorted(member.name for member in members)
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute()
            assert ".." not in path.parts
            assert path.parts[0] == build_release.BUNDLE_NAME
            assert member.isfile() or member.isdir()
            assert not member.issym() and not member.islnk()
            assert member.uid == 0 and member.gid == 0 and member.mtime == 0
            assert member.uname == "" and member.gname == ""


def test_tar_verifier_rejects_nonfixed_member_and_gzip_metadata(
    release_source: Path,
    tmp_path: Path,
) -> None:
    result = build_release.build_release(release_source, tmp_path / "release")
    damaged = tmp_path / "damaged.tar.gz"
    with tarfile.open(result.archive, "r:gz") as source, tarfile.open(
        damaged,
        "w:gz",
    ) as target:
        for member in source.getmembers():
            member.uname = "not-fixed"
            handle = source.extractfile(member) if member.isfile() else None
            try:
                target.addfile(member, handle)
            finally:
                if handle is not None:
                    handle.close()

    with pytest.raises(build_release.ReleaseValidationError, match="metadata"):
        build_release.verify_tarball(damaged, result.bundle_dir)


@pytest.mark.parametrize(
    "tail",
    (
        b"JUNK",
        bytes.fromhex("1f8b08"),
        gzip.compress(b"second gzip member", compresslevel=9, mtime=0),
    ),
    ids=("junk", "partial-gzip-header", "second-gzip-member"),
)
def test_tar_verifier_rejects_noncanonical_trailing_bytes(
    release_source: Path,
    tmp_path: Path,
    tail: bytes,
) -> None:
    result = build_release.build_release(release_source, tmp_path / "release")
    damaged = tmp_path / "damaged.tar.gz"
    shutil.copyfile(result.archive, damaged)
    with damaged.open("ab") as handle:
        handle.write(tail)

    with pytest.raises(build_release.ReleaseValidationError, match="canonical"):
        build_release.verify_tarball(damaged, result.bundle_dir)


@pytest.mark.parametrize(
    "missing",
    (
        "uv.lock",
        "data/generated/isotropic_zgv.npz",
        "data/source_data/figure_01/panel_a_ring.csv",
        "figures/main/figure_01_geometry_mechanism.svg",
        "build/paper/main.pdf",
    ),
)
def test_release_validation_rejects_missing_required_file(
    release_source: Path,
    tmp_path: Path,
    missing: str,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    (source / missing).unlink()

    with pytest.raises(build_release.ReleaseValidationError, match="missing"):
        build_release.validate_release_inputs(source)


def test_release_validation_rejects_unregistered_source_csv(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    rogue = source / "data/source_data/rogue.csv"
    rogue.write_text("not,registered\n", encoding="utf-8")

    with pytest.raises(build_release.ReleaseValidationError, match="unregistered.*CSV"):
        build_release.validate_release_inputs(source)


def test_release_validation_rejects_registered_figure_output_outside_fixed_layout(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    closure = reproduction.discover_scientific_closure(source)
    original = "figures/main/figure_01_geometry_mechanism.svg"
    relocated = "figures/relocated/figure_01_geometry_mechanism.svg"
    destination = source / relocated
    destination.parent.mkdir()
    (source / original).rename(destination)
    malformed = replace(
        closure,
        figure_outputs=tuple(
            sorted(relocated if path == original else path for path in closure.figure_outputs)
        ),
        paths=tuple(sorted(relocated if path == original else path for path in closure.paths)),
    )

    with pytest.raises(build_release.ReleaseValidationError, match="figure.*layout"):
        build_release._validate_scientific_tree(source, malformed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"status": "failed"}, "status"),
        ({"profile": "smoke"}, "profile"),
        ({"persistent_scientific_cache": True}, "persistent_scientific_cache"),
    ),
)
def test_release_validation_rejects_damaged_reproduction_report(
    release_source: Path,
    tmp_path: Path,
    mutation: dict[str, Any],
    message: str,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    report_path = source / "data/reproduction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(mutation)
    _replace_text(report_path, json.dumps(report, sort_keys=True))

    with pytest.raises(build_release.ReleaseValidationError, match=message):
        build_release.validate_release_inputs(source)


def test_release_validation_rejects_scientific_hash_drift(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    target = source / "paper/generated/results_macros.tex"
    original = target.read_bytes()
    target.unlink()
    target.write_bytes(original + b"% drift\n")

    with pytest.raises(build_release.ReleaseValidationError, match="scientific.*hash"):
        build_release.validate_release_inputs(source)


def test_release_validation_rejects_unknown_nonzero_citation_error_count(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    audit_path = source / "docs/literature/citation_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["summary"]["network_error"] = 1
    _replace_text(audit_path, json.dumps(audit, sort_keys=True))

    with pytest.raises(build_release.ReleaseValidationError, match="summary|error"):
        build_release.validate_release_inputs(source)


@pytest.mark.parametrize("damage", ("truncated", "duplicated", "empty-comparison"))
def test_release_validation_requires_audit_to_cover_the_bibliography_exactly(
    release_source: Path,
    tmp_path: Path,
    damage: str,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    audit_path = source / "docs/literature/citation_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if damage == "truncated":
        audit["entries"].pop()
        audit["summary"]["entries"] -= 1
    elif damage == "duplicated":
        audit["entries"][-1] = audit["entries"][0]
    else:
        audit["entries"][0]["comparison"] = {}
    _replace_text(audit_path, json.dumps(audit, sort_keys=True))

    with pytest.raises(build_release.ReleaseValidationError, match="citation audit"):
        build_release.validate_release_inputs(source)


@pytest.mark.parametrize(
    "rogue",
    (
        "scripts/private_secret.py",
        "docs/reviews/confidential.md",
    ),
)
def test_release_validation_rejects_unregistered_allowlist_files(
    release_source: Path,
    tmp_path: Path,
    rogue: str,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    path = source / rogue
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("must not be packaged\n", encoding="utf-8")

    with pytest.raises(build_release.ReleaseValidationError, match="unregistered"):
        build_release.validate_release_inputs(source)


def test_release_build_rejects_custom_output_nested_in_source_tree(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")

    with pytest.raises(build_release.ReleaseValidationError, match="output.*source"):
        build_release.build_release(source, source / "scripts/release-output")


def test_release_validation_rejects_symlink(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    readme = source / "README.md"
    readme.unlink()
    readme.symlink_to("pyproject.toml")

    with pytest.raises(build_release.ReleaseValidationError, match="symbolic link"):
        build_release.validate_release_inputs(source)


def test_release_validation_rejects_symlinked_source_root(
    release_source: Path,
    tmp_path: Path,
) -> None:
    alias = tmp_path / "source-alias"
    alias.symlink_to(release_source, target_is_directory=True)

    with pytest.raises(build_release.ReleaseValidationError, match="source root.*symbolic"):
        build_release.validate_release_inputs(alias)


def test_release_validation_rejects_special_files(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    os.mkfifo(source / "scripts/rogue.py")

    with pytest.raises(build_release.ReleaseValidationError, match="special file"):
        build_release.validate_release_inputs(source)


@pytest.mark.parametrize(
    "unsafe",
    (
        "/absolute.txt",
        "../escape.txt",
        "a/../../escape.txt",
        "back\\slash.txt",
        "control\nname.txt",
        "./noncanonical.txt",
        "C:/windows-absolute.txt",
    ),
)
def test_registered_paths_reject_unsafe_spellings(unsafe: str) -> None:
    with pytest.raises(build_release.ReleaseValidationError, match="unsafe|canonical"):
        build_release.validate_safe_relative_paths((unsafe,))


@pytest.mark.parametrize(
    "colliding",
    (
        ("docs/Readme.txt", "docs/readme.TXT"),
        ("docs/caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt", "docs/cafe\N{COMBINING ACUTE ACCENT}.txt"),
        ("docs/\N{FULLWIDTH LATIN CAPITAL LETTER A}.txt", "docs/A.txt"),
        ("scripts/Foo/a.py", "scripts/foo/b.py"),
        (
            "scripts/caf\N{LATIN SMALL LETTER E WITH ACUTE}/a.py",
            "scripts/cafe\N{COMBINING ACUTE ACCENT}/b.py",
        ),
    ),
)
def test_registered_paths_reject_casefold_and_unicode_collisions(
    colliding: tuple[str, str],
) -> None:
    with pytest.raises(build_release.ReleaseValidationError, match="collision"):
        build_release.validate_safe_relative_paths(colliding)


def test_reproduction_report_schema_binds_all_three_snapshots(
    release_source: Path,
) -> None:
    path = release_source / "data/reproduction_report.json"
    report = reproduction.validate_reproduction_report(path, release_source)

    assert report["status"] == "verified"
    assert report["profile"] == "full"
    assert report["persistent_scientific_cache"] is False
    for key in ("baseline", "cold_run", "retained_state_run"):
        snapshot = report[key]["scientific_snapshot"] if key != "baseline" else report[key]
        assert snapshot["file_count"] == 117
        assert snapshot["aggregate_sha256"] == report["baseline"]["aggregate_sha256"]


def test_reproduction_report_requires_recorded_dependency_versions(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    report_path = source / "data/reproduction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["environment"]["versions"] = {}
    reproduction.atomic_write_report(report_path, report)

    with pytest.raises(reproduction.ReproductionError, match="versions"):
        reproduction.validate_reproduction_report(report_path, source)


def test_successful_publish_updates_report_and_readme_with_measured_values(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    report_path = source / "data/reproduction_report.json"
    readme_path = source / "README.md"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["cold_run"]["wall_seconds"] = 12.345678
    report["retained_state_run"]["wall_seconds"] = 4.25

    reproduction.publish_reproduction_results(source, report_path, readme_path, report)

    validated = reproduction.validate_reproduction_report(report_path, source)
    text = readme_path.read_text(encoding="utf-8")
    baseline = validated["baseline"]
    assert "Pending the release-wide clean reproduction" not in text
    assert f"`{baseline['file_count']}` files" in text
    assert f"`{baseline['total_bytes']}` bytes" in text
    assert f"`{baseline['aggregate_sha256']}`" in text
    assert "`12.345678` seconds" in text
    assert "`4.250000` seconds" in text
    assert text.count(reproduction.README_RESULTS_BEGIN) == 1
    assert text.count(reproduction.README_RESULTS_END) == 1


def test_invalid_report_publish_leaves_prior_report_and_readme_unchanged(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    report_path = source / "data/reproduction_report.json"
    readme_path = source / "README.md"
    old_report = report_path.read_bytes()
    old_readme = readme_path.read_bytes()
    invalid = json.loads(old_report)
    invalid["status"] = "failed"

    with pytest.raises(reproduction.ReproductionError, match="status"):
        reproduction.publish_reproduction_results(
            source,
            report_path,
            readme_path,
            invalid,
        )

    assert report_path.read_bytes() == old_report
    assert readme_path.read_bytes() == old_readme


def test_pair_publication_rolls_back_if_second_install_fails(
    release_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")
    report_path = source / "data/reproduction_report.json"
    readme_path = source / "README.md"
    old_report = report_path.read_bytes()
    old_readme = readme_path.read_bytes()
    report = json.loads(old_report)
    report["cold_run"]["wall_seconds"] = 2.0
    real_replace = os.replace
    calls = 0

    def fail_readme_install(source_path: os.PathLike[str], target_path: os.PathLike[str]) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("simulated README install failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(reproduction.os, "replace", fail_readme_install)

    with pytest.raises(OSError, match="simulated README install failure"):
        reproduction.publish_reproduction_results(source, report_path, readme_path, report)

    assert report_path.read_bytes() == old_report
    assert readme_path.read_bytes() == old_readme


def test_clean_baseline_accepts_all_117_scientific_files_committed(
    release_source: Path,
    tmp_path: Path,
) -> None:
    source = _clone_fixture(release_source, tmp_path / "source")

    def git(*arguments: str) -> None:
        subprocess.run(("git", *arguments), cwd=source, check=True)

    git("init", "-q")
    git("config", "user.name", "Release Test")
    git("config", "user.email", "release@example.invalid")
    git("add", ".")
    git("add", "-f", "build/paper/main.pdf", "build/paper/supplement.pdf")
    git("commit", "-qm", "test baseline")

    head, snapshot = reproduction.require_clean_committed_baseline(source)

    assert len(head) == 40
    assert snapshot.file_count == 117


def test_clean_verifier_git_runner_sanitizes_every_worktree_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    head = _make_git_repository(repository)
    attacker = tmp_path / "attacker"
    _make_git_repository(attacker)
    for key, value in _malicious_git_environment(attacker).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/malicious/dyld")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/malicious/ld")
    monkeypatch.setenv("UNRELATED_GIT_TEST_SECRET", "must-not-reach-git")
    real_run = subprocess.run
    environments: list[dict[str, str] | None] = []
    commands: list[tuple[str, ...]] = []
    execution_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }

    def record_environment(
        command: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        if command[0] == "git":
            commands.append(tuple(command))
            supplied = kwargs.get("env")
            environments.append(None if supplied is None else dict(supplied))
            kwargs["env"] = execution_environment
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(reproduction.subprocess, "run", record_environment)

    assert reproduction._git_output(repository, "rev-parse", "HEAD").strip() == head
    with reproduction.detached_worktree(repository, head) as checkout:
        assert (checkout / "tracked.txt").read_text(encoding="utf-8") == "tracked\n"

    assert ("git", "worktree", "add", "--detach") == commands[1][:4]
    assert any(command[:3] == ("git", "worktree", "remove") for command in commands)
    assert any(command[:3] == ("git", "worktree", "prune") for command in commands)
    assert any(command[:4] == ("git", "worktree", "list", "--porcelain") for command in commands)
    _assert_sanitized_git_environments(environments)


def test_clean_verifier_ignores_inherited_git_repository_redirection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    head = _make_git_repository(repository)
    attacker = tmp_path / "attacker"
    _make_git_repository(attacker)
    (attacker / "tracked.txt").write_text("attacker\n", encoding="utf-8")
    _git(attacker, "add", "tracked.txt")
    _git(attacker, "commit", "-qm", "attacker state")
    attacker_head = _git(attacker, "rev-parse", "HEAD").stdout.strip()
    assert attacker_head != head
    for key, value in _malicious_git_environment(attacker).items():
        monkeypatch.setenv(key, value)

    observed = reproduction._git_output(repository, "rev-parse", "HEAD").strip()

    assert observed == head
    with reproduction.detached_worktree(repository, head) as checkout:
        assert (checkout / "tracked.txt").read_text(encoding="utf-8") == "tracked\n"


def test_cold_commands_have_required_deletion_independent_order() -> None:
    commands = reproduction.cold_reproduction_commands()
    strings = [" ".join(command) for command in commands]

    assert strings[0] == "uv sync --frozen --all-extras"
    stage_commands = strings[1:8]
    assert [command.split("--stage ", 1)[1].split()[0] for command in stage_commands] == [
        "isotropic",
        "sensitivity",
        "critical_points",
        "scaling",
        "green",
        "convergence",
        "silicon",
    ]
    assert strings[8] == "uv run python scripts/validate_isotropic.py"
    assert strings[9:15] == [
        f"uv run python scripts/make_figure_{number:02d}.py" for number in range(1, 7)
    ]
    assert strings[-7:] == [
        "uv run python scripts/make_supplementary_figures.py",
        "uv run python scripts/export_supplement_tables.py",
        "uv run python scripts/qa_figures.py --strict",
        "uv run python scripts/export_manuscript_values.py",
        "uv run python scripts/compile_paper.py",
        "uv run pytest -q",
        "uv run python scripts/check_claim_evidence.py --require-supported",
    ]


def test_retained_state_command_recomputes_the_full_profile() -> None:
    assert reproduction.retained_state_command() == (
        "uv",
        "run",
        "python",
        "scripts/reproduce_all.py",
        "--profile",
        "full",
    )


def test_reproduction_environment_isolates_project_and_runtime_caches(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "checkout"
    environment = reproduction.deterministic_environment(
        workspace,
        {
            "PATH": "/usr/bin",
            "PYTHONHOME": "/contaminated/python",
            "PYTHONPATH": "/contaminated/modules",
            "UV_PROJECT_ENVIRONMENT": "/contaminated/venv",
            "VIRTUAL_ENV": "/contaminated/active",
        },
    )

    assert environment["PATH"] == "/usr/bin"
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(workspace / ".venv")
    assert environment["UV_CACHE_DIR"] == str(tmp_path / "isolated-cache/uv")
    assert environment["MPLCONFIGDIR"] == str(tmp_path / "isolated-cache/matplotlib")
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert "VIRTUAL_ENV" not in environment
    assert environment["OMP_NUM_THREADS"] == "1"


def test_reproduction_environment_drops_behavior_overrides_and_unlisted_values(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "checkout"
    environment = reproduction.deterministic_environment(
        workspace,
        {
            "PATH": "/trusted/bin",
            "HOME": "/trusted/home",
            "TMPDIR": "/trusted/tmp",
            "LD_LIBRARY_PATH": "/trusted/lib",
            "PYTEST_ADDOPTS": "-m not slow",
            "PYTEST_PLUGINS": "attacker.plugin",
            "COVERAGE_PROCESS_START": "/attacker/coverage.ini",
            "PYTHONWARNINGS": "ignore",
            "PYTHONBREAKPOINT": "attacker.breakpoint",
            "UNRELATED_SECRET": "must-not-leak",
        },
    )

    assert environment["PATH"] == "/trusted/bin"
    assert environment["HOME"] == "/trusted/home"
    assert environment["TMPDIR"] == "/trusted/tmp"
    assert environment["LD_LIBRARY_PATH"] == "/trusted/lib"
    for variable in (
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "COVERAGE_PROCESS_START",
        "PYTHONWARNINGS",
        "PYTHONBREAKPOINT",
        "UNRELATED_SECRET",
    ):
        assert variable not in environment


def test_tool_version_skips_warning_before_declared_version_banner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = "Use of uninitialized value $version_num in latexmk line 1300."
    banner = "Latexmk, John Collins, 9 March 2026. Version 4.88"

    def version_output(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ("latexmk", "-version"),
            0,
            stdout=f"{warning}\n{banner}\n",
            stderr="",
        )

    monkeypatch.setattr(reproduction.subprocess, "run", version_output)

    assert reproduction._tool_version(
        ("latexmk", "-version"),
        tmp_path,
        {"PATH": "/trusted/bin"},
    ) == banner


def test_tool_version_rejects_output_without_declared_version_banner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def warning_only(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ("latexmk", "-version"),
            0,
            stdout="Use of uninitialized value $version_num in latexmk line 1300.\n",
            stderr="",
        )

    monkeypatch.setattr(reproduction.subprocess, "run", warning_only)

    with pytest.raises(reproduction.ReproductionError, match="Latexmk,.*banner"):
        reproduction._tool_version(
            ("latexmk", "-version"),
            tmp_path,
            {"PATH": "/trusted/bin"},
        )


def _environment_probe_runner(
    payload: dict[str, Any],
) -> Any:
    real_run = subprocess.run

    def run(
        command: Any,
        *args: Any,
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        if tuple(command[:4]) == ("uv", "run", "python", "-c"):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(payload),
                stderr="",
            )
        version_outputs = {
            "bibtex": "BibTeX 0.99d (TeX Live 2026)\n",
            "latexmk": (
                "Use of uninitialized value $version_num in latexmk line 1300.\n"
                "Latexmk, John Collins, 9 March 2026. Version 4.88\n"
            ),
            "pdflatex": "pdfTeX 3.141592653-2.6-1.40.28 (TeX Live 2026)\n",
            "uv": "uv 0.8.11 (test build)\n",
        }
        if command[0] in version_outputs:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=version_outputs[command[0]],
                stderr="",
            )
        return real_run(command, *args, **_kwargs)

    return run


def test_environment_report_uses_child_probe_architecture_and_operating_system(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = {
        "architecture": "child-architecture",
        "backend": "Agg",
        "operating_system": "child-operating-system",
        "python": "3.12-child",
        "versions": {
            name: "child-version"
            for name in ("numpy", "scipy", "sympy", "matplotlib", "mpmath", "PyYAML", "pypdf")
        },
    }
    monkeypatch.setattr(
        reproduction.subprocess,
        "run",
        _environment_probe_runner(child),
    )

    environment = reproduction.collect_environment(tmp_path, {"PATH": "/trusted/bin"})

    assert environment["architecture"] == "child-architecture"
    assert environment["operating_system"] == "child-operating-system"
    assert environment["uv"] == "uv 0.8.11 (test build)"
    assert environment["versions"]["bibtex"] == "BibTeX 0.99d (TeX Live 2026)"
    assert environment["versions"]["latexmk"] == (
        "Latexmk, John Collins, 9 March 2026. Version 4.88"
    )
    assert environment["versions"]["pdflatex"] == (
        "pdfTeX 3.141592653-2.6-1.40.28 (TeX Live 2026)"
    )
    assert all(
        "uninitialized value" not in version
        for version in (
            environment["uv"],
            environment["versions"]["bibtex"],
            environment["versions"]["latexmk"],
            environment["versions"]["pdflatex"],
        )
    )


@pytest.mark.parametrize("field", ("architecture", "operating_system"))
def test_environment_report_rejects_empty_child_platform_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    child = {
        "architecture": "child-architecture",
        "backend": "Agg",
        "operating_system": "child-operating-system",
        "python": "3.12-child",
        "versions": {
            name: "child-version"
            for name in ("numpy", "scipy", "sympy", "matplotlib", "mpmath", "PyYAML", "pypdf")
        },
    }
    child[field] = ""
    monkeypatch.setattr(
        reproduction.subprocess,
        "run",
        _environment_probe_runner(child),
    )

    with pytest.raises(reproduction.ReproductionError, match=field):
        reproduction.collect_environment(tmp_path, {"PATH": "/trusted/bin"})


def test_generated_output_cleanup_removes_registered_outputs_but_not_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    generated = (
        "data/generated/a.npz",
        "figures/main/a.svg",
        "paper/generated/a.tex",
        "build/paper/main.pdf",
    )
    source = root / "src/package.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    for relative in generated:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated\n", encoding="utf-8")
    cache = root / ".pytest_cache/state"
    cache.parent.mkdir(parents=True)
    cache.write_text("cache\n", encoding="utf-8")
    coverage = root / ".coverage"
    coverage.write_text("cache\n", encoding="utf-8")

    removed = reproduction.delete_generated_outputs(root, generated)

    assert removed == tuple(sorted(generated))
    assert source.is_file()
    assert not any((root / relative).exists() for relative in generated)
    assert not (root / ".pytest_cache").exists()
    assert not coverage.exists()


def test_failed_verification_never_replaces_existing_report(tmp_path: Path) -> None:
    report_path = tmp_path / "data/reproduction_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("sentinel\n", encoding="utf-8")

    def fail() -> dict[str, Any]:
        raise reproduction.ReproductionError("cold command failed")

    with pytest.raises(reproduction.ReproductionError, match="cold command failed"):
        reproduction.write_report_after_success(report_path, fail)
    assert report_path.read_text(encoding="utf-8") == "sentinel\n"


def test_verifier_refuses_noncanonical_report_destination_before_work(
    release_source: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(reproduction.ReproductionError, match="canonical"):
        reproduction.verify_clean_reproduction(
            release_source,
            report_path=tmp_path / "outside-report.json",
        )


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("committed-code", "original checkout HEAD changed"),
        ("uncommitted-code", "original checkout changed.*working tree"),
        ("scientific", "original checkout scientific baseline changed"),
    ),
)
def test_clean_verifier_refuses_to_publish_after_original_checkout_drift(
    release_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    source, prior_report, prior_readme, environment_report = (
        _prepare_clean_verifier_source(release_source, tmp_path / "source")
    )
    _install_fast_clean_verifier_seams(
        monkeypatch,
        source,
        tmp_path / "detached-checkout",
        environment_report,
    )
    mutated = False

    def mutate_original_checkout(
        _command: tuple[str, ...],
        _workspace: Path,
        _environment: dict[str, str],
    ) -> None:
        nonlocal mutated
        if mutated:
            return
        mutated = True
        if drift in {"committed-code", "uncommitted-code"}:
            relative = "scripts/build_release.py"
            target = source / relative
            _replace_text(
                target,
                target.read_text(encoding="utf-8") + "\n# during-run code drift\n",
            )
            if drift == "committed-code":
                _git(source, "add", relative)
                _git(source, "commit", "-qm", "advance original checkout")
            return
        relative = "paper/generated/results_macros.tex"
        target = source / relative
        _replace_text(
            target,
            target.read_text(encoding="utf-8") + "\n% hidden scientific drift\n",
        )
        _git(source, "update-index", "--assume-unchanged", relative)
        assert _git(source, "status", "--porcelain=v1").stdout == ""

    with pytest.raises(reproduction.ReproductionError, match=message):
        reproduction.verify_clean_reproduction(
            source,
            runner=mutate_original_checkout,
        )

    assert mutated
    assert (source / "data/reproduction_report.json").read_bytes() == prior_report
    assert (source / "README.md").read_bytes() == prior_readme


def test_clean_verifier_publishes_when_original_checkout_is_unchanged(
    release_source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, prior_report, prior_readme, environment_report = (
        _prepare_clean_verifier_source(release_source, tmp_path / "source")
    )
    _install_fast_clean_verifier_seams(
        monkeypatch,
        source,
        tmp_path / "detached-checkout",
        environment_report,
    )
    head = _git(source, "rev-parse", "HEAD").stdout.strip()

    def no_op_runner(
        _command: tuple[str, ...],
        _workspace: Path,
        _environment: dict[str, str],
    ) -> None:
        return None

    report = reproduction.verify_clean_reproduction(source, runner=no_op_runner)

    assert report["head_commit"] == head
    assert (source / "data/reproduction_report.json").read_bytes() != prior_report
    assert (source / "README.md").read_bytes() != prior_readme
    assert reproduction.validate_reproduction_report(
        source / "data/reproduction_report.json",
        source,
    ) == report


def test_temporary_worktree_cleanup_runs_after_failure(tmp_path: Path) -> None:
    events: list[str] = []

    @contextmanager
    def workspace() -> Iterator[Path]:
        events.append("create")
        try:
            yield tmp_path / "checkout"
        finally:
            events.append("cleanup")

    def fail(_: Path) -> dict[str, Any]:
        events.append("verify")
        raise reproduction.ReproductionError("failure")

    with pytest.raises(reproduction.ReproductionError, match="failure"):
        reproduction.run_in_temporary_worktree(workspace, fail)
    assert events == ["create", "verify", "cleanup"]


def test_detached_worktree_recovers_from_remove_failure_and_prunes_after_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    head = _make_git_repository(repository)
    real_run = subprocess.run
    real_rmtree = shutil.rmtree
    events: list[str] = []

    def fail_remove(
        command: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        if command[:3] == ("git", "worktree", "remove"):
            events.append("remove")
            return subprocess.CompletedProcess(command, 1)
        if command[:3] == ("git", "worktree", "prune"):
            events.append("prune")
        return real_run(command, *args, **kwargs)

    def record_rmtree(path: os.PathLike[str], *args: Any, **kwargs: Any) -> None:
        events.append("rmtree")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(reproduction.subprocess, "run", fail_remove)
    monkeypatch.setattr(reproduction.shutil, "rmtree", record_rmtree)
    checkout: Path
    with reproduction.detached_worktree(repository, head) as checkout:
        assert checkout.is_dir()

    listed = real_run(
        ("git", "worktree", "list", "--porcelain"),
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    ).stdout
    deletion_index = events.index("rmtree")
    assert "prune" in events[deletion_index + 1 :]
    assert not checkout.exists()
    assert str(checkout) not in listed


def test_detached_worktree_cleans_partial_add_failure_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    head = _make_git_repository(repository)
    real_run = subprocess.run
    checkout_holder: list[Path] = []

    def fail_after_partial_add(
        command: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        if command[:3] == ("git", "worktree", "add"):
            real_run(command, *args, **kwargs)
            checkout_holder.append(Path(command[4]))
            raise subprocess.CalledProcessError(1, command)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(reproduction.subprocess, "run", fail_after_partial_add)

    with pytest.raises(reproduction.ReproductionError, match="cannot create"):
        with reproduction.detached_worktree(repository, head):
            pytest.fail("partial add must not yield a checkout")

    checkout = checkout_holder[0]
    listed = real_run(
        ("git", "worktree", "list", "--porcelain"),
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        encoding="utf-8",
    ).stdout
    assert not checkout.exists()
    assert str(checkout) not in listed


def test_detached_worktree_reports_cleanup_failure_with_active_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    head = _make_git_repository(repository)
    real_run = subprocess.run
    real_rmtree = shutil.rmtree
    checkout_holder: list[Path] = []

    def fail_cleanup_commands(
        command: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[Any]:
        if command[:3] == ("git", "worktree", "add"):
            checkout_holder.append(Path(command[4]))
            return real_run(command, *args, **kwargs)
        if command[:3] in {
            ("git", "worktree", "remove"),
            ("git", "worktree", "prune"),
        }:
            return subprocess.CompletedProcess(command, 1)
        return real_run(command, *args, **kwargs)

    def fail_rmtree(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected rmtree failure")

    try:
        monkeypatch.setattr(reproduction.subprocess, "run", fail_cleanup_commands)
        monkeypatch.setattr(reproduction.shutil, "rmtree", fail_rmtree)
        with pytest.raises(
            reproduction.ReproductionError,
            match="cleanup.*active reproduction failure",
        ) as caught:
            with reproduction.detached_worktree(repository, head):
                raise ValueError("active reproduction failure")
        assert isinstance(caught.value.__cause__, ValueError)
    finally:
        monkeypatch.undo()
        if checkout_holder:
            real_run(
                ("git", "worktree", "remove", "--force", str(checkout_holder[0])),
                cwd=repository,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            real_rmtree(checkout_holder[0].parent, ignore_errors=True)
        real_run(
            ("git", "worktree", "prune", "--expire", "now"),
            cwd=repository,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
