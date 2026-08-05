from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"

DETERMINISTIC_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
}


def _load_workflow(name: str) -> dict:
    path = WORKFLOW_DIR / name
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    # PyYAML follows YAML 1.1 and otherwise parses an unquoted `on` key as True.
    assert "on" in workflow
    assert True not in workflow
    return workflow


def _job(workflow: dict, name: str) -> dict:
    jobs = workflow["jobs"]
    assert set(jobs) == {name}
    return jobs[name]


def test_ci_triggers_are_explicit_and_yaml_1_1_safe() -> None:
    smoke = _load_workflow("repro-smoke.yml")
    full = _load_workflow("repro-full.yml")

    assert set(smoke["on"]) == {"push", "pull_request"}
    assert full["on"] == {
        "workflow_dispatch": None,
        "schedule": [{"cron": "17 3 1 * *"}],
    }


def test_ci_pins_determinism_and_limits_privileges_and_runtime() -> None:
    expected = {
        "repro-smoke.yml": ("smoke", 30),
        "repro-full.yml": ("full", 360),
    }

    for filename, (job_name, timeout) in expected.items():
        workflow = _load_workflow(filename)
        expected_environment = dict(DETERMINISTIC_ENV)
        if filename == "repro-full.yml":
            expected_environment["PDF_REPRODUCTION_SCOPE"] = "semantic-nonreference"
        assert workflow["env"] == expected_environment
        assert workflow["permissions"] == {"contents": "read"}
        assert workflow["concurrency"] == {
            "group": "${{ github.workflow }}-${{ github.ref }}",
            "cancel-in-progress": True,
        }

        job = _job(workflow, job_name)
        assert job["runs-on"] == "ubuntu-latest"
        assert job["timeout-minutes"] == timeout
        assert "permissions" not in job


def test_ci_uses_frozen_dependencies_and_exact_reproduction_commands() -> None:
    expected = {
        "repro-smoke.yml": (
            "smoke",
            "uv run python scripts/reproduce_all.py --profile smoke --skip-paper",
        ),
        "repro-full.yml": (
            "full",
            "uv run python scripts/reproduce_all.py --profile full --skip-paper",
        ),
    }

    for filename, (job_name, reproduction_command) in expected.items():
        workflow = _load_workflow(filename)
        steps = _job(workflow, job_name)["steps"]
        actions = [step["uses"] for step in steps if "uses" in step]
        commands = [step["run"] for step in steps if "run" in step]

        assert actions[:2] == ["actions/checkout@v4", "astral-sh/setup-uv@v5"]
        assert commands.count("uv sync --frozen --all-extras") == 1
        reproduction_commands = [
            command
            for command in commands
            if command.startswith("uv run python scripts/reproduce_all.py")
        ]
        assert reproduction_commands == [reproduction_command]

    full = _load_workflow("repro-full.yml")
    full_commands = [step["run"] for step in _job(full, "full")["steps"] if "run" in step]
    assert "sudo apt-get update" in full_commands
    assert (
        "sudo apt-get install -y --no-install-recommends cm-super latexmk "
        "texlive-fonts-recommended texlive-latex-extra texlive-science" in full_commands
    )


def test_full_ci_scopes_byte_identity_to_the_reference_environment() -> None:
    workflow = _load_workflow("repro-full.yml")
    commands = [step["run"] for step in _job(workflow, "full")["steps"] if "run" in step]

    assert "uv run python scripts/export_manuscript_values.py" in commands
    assert "uv run python scripts/compile_paper_semantic.py" in commands
    assert "latexmk -version\npdflatex --version\nbibtex --version\n" in commands
    assert not any(command.endswith("scripts/compile_paper.py") for command in commands)
    assert workflow["env"]["PDF_REPRODUCTION_SCOPE"] == "semantic-nonreference"


def test_full_ci_uploads_the_complete_reproduction_bundle() -> None:
    workflow = _load_workflow("repro-full.yml")
    steps = _job(workflow, "full")["steps"]
    uploads = [step for step in steps if step.get("uses") == "actions/upload-artifact@v4"]

    assert len(uploads) == 1
    upload = uploads[0]["with"]
    assert upload["name"] == "first-paper-reproduction"
    assert upload["if-no-files-found"] == "error"
    assert upload["path"].splitlines() == [
        "data/generated",
        "data/provenance_manifest.json",
        "figures",
        "build/paper",
    ]
