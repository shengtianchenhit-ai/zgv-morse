"""Run the deterministic first-paper reproduction pipeline in isolated processes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys

from zgv_morse.artifact_schema import validate_artifact
from zgv_morse.provenance import validate_manifest


ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "isotropic",
    "sensitivity",
    "critical_points",
    "scaling",
    "green",
    "convergence",
    "silicon",
)
ARTIFACTS = {
    "isotropic": "isotropic_zgv.npz",
    "sensitivity": "angular_sensitivity.npz",
    "critical_points": "critical_points.npz",
    "scaling": "perturbation_scaling.npz",
    "green": "green_crossover.npz",
    "convergence": "convergence.npz",
    "silicon": "silicon_stress_test.npz",
}


def _require_profile(profile: str) -> None:
    if not isinstance(profile, str):
        raise TypeError("profile must be a string")
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'")


def _require_bool(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")


def build_commands(profile: str, include_figures: bool = True) -> list[list[str]]:
    """Build the deterministic test, workflow, and optional figure commands."""

    _require_profile(profile)
    _require_bool(include_figures, "include_figures")

    tests = [sys.executable, "-m", "pytest", "-q", "tests"]
    if profile == "smoke":
        tests.extend(["-m", "not slow"])
    commands = [tests]
    commands.extend(
        [
            sys.executable,
            "-m",
            "zgv_morse.workflows",
            "--stage",
            stage,
            "--profile",
            profile,
        ]
        for stage in STAGES
    )
    if include_figures:
        commands.append([sys.executable, "scripts/validate_isotropic.py"])
        commands.extend(
            [sys.executable, f"scripts/make_figure_{number:02d}.py"]
            for number in range(1, 7)
        )
        commands.extend(
            (
                [sys.executable, "scripts/make_supplementary_figures.py"],
                [sys.executable, "scripts/export_supplement_tables.py"],
                [sys.executable, "scripts/qa_figures.py", "--strict"],
            )
        )
    return commands


def _run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def run(profile: str, include_figures: bool, include_paper: bool) -> None:
    """Run all requested phases, validating each scientific output immediately."""

    _require_profile(profile)
    _require_bool(include_figures, "include_figures")
    _require_bool(include_paper, "include_paper")
    commands = build_commands(profile, include_figures)

    _run_command(commands[0])
    for stage, command in zip(STAGES, commands[1 : 1 + len(STAGES)], strict=True):
        _run_command(command)
        artifact = ROOT / "data/generated" / ARTIFACTS[stage]
        validate_artifact(artifact, artifact.with_suffix(".json"))

    validate_manifest(ROOT / "data/provenance_manifest.json")
    for command in commands[1 + len(STAGES) :]:
        _run_command(command)

    if include_paper:
        _run_command([sys.executable, "scripts/export_manuscript_values.py"])
        _run_command([sys.executable, "scripts/compile_paper.py"])


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the public command-line interface and run the selected profile."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-paper", action="store_true")
    arguments = parser.parse_args(argv)
    run(arguments.profile, not arguments.skip_figures, not arguments.skip_paper)


if __name__ == "__main__":
    main()
