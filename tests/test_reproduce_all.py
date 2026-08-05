from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.reproduce_all as reproduce_all  # noqa: E402
from scripts.reproduce_all import (  # noqa: E402
    ARTIFACTS,
    ROOT,
    STAGES,
    build_commands,
    main,
    run,
)


EXPECTED_STAGES = (
    "isotropic",
    "sensitivity",
    "critical_points",
    "scaling",
    "green",
    "convergence",
    "silicon",
)
EXPECTED_ARTIFACTS = {
    "isotropic": "isotropic_zgv.npz",
    "sensitivity": "angular_sensitivity.npz",
    "critical_points": "critical_points.npz",
    "scaling": "perturbation_scaling.npz",
    "green": "green_crossover.npz",
    "convergence": "convergence.npz",
    "silicon": "silicon_stress_test.npz",
}
EXPECTED_FIGURE_COMMANDS = [
    [sys.executable, "scripts/validate_isotropic.py"],
    *[
        [sys.executable, f"scripts/make_figure_{number:02d}.py"]
        for number in range(1, 7)
    ],
    [sys.executable, "scripts/make_supplementary_figures.py"],
    [sys.executable, "scripts/export_supplement_tables.py"],
    [sys.executable, "scripts/qa_figures.py", "--strict"],
]


def test_stage_and_artifact_order_is_stable() -> None:
    assert STAGES == EXPECTED_STAGES
    assert ARTIFACTS == EXPECTED_ARTIFACTS
    assert tuple(ARTIFACTS) == STAGES


def test_smoke_commands_are_deterministic_and_figures_are_optional() -> None:
    commands = build_commands("smoke", include_figures=False)

    assert commands[0] == [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests",
        "-m",
        "not slow",
    ]
    assert commands[1:] == [
        [
            sys.executable,
            "-m",
            "zgv_morse.workflows",
            "--stage",
            stage,
            "--profile",
            "smoke",
        ]
        for stage in STAGES
    ]

    with_figures = build_commands("smoke", include_figures=True)
    assert with_figures[: len(commands)] == commands
    assert with_figures[len(commands) :] == EXPECTED_FIGURE_COMMANDS


def test_ci_smoke_runs_the_complete_figure_pipeline() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/repro-smoke.yml").read_text(
        encoding="utf-8"
    )

    expected = (
        "uv run python scripts/reproduce_all.py --profile smoke --skip-paper"
    )
    assert f"- run: {expected}" in workflow
    assert "--skip-figures" not in workflow


def test_full_commands_do_not_filter_slow_tests() -> None:
    commands = build_commands("full", include_figures=False)

    assert commands[0] == [sys.executable, "-m", "pytest", "-q", "tests"]
    assert all(command[-1] == "full" for command in commands[1:])


@pytest.mark.parametrize("profile", ["fast", "", True, 1, None])
def test_build_commands_rejects_invalid_profiles(profile: object) -> None:
    expected_error = TypeError if not isinstance(profile, str) else ValueError
    with pytest.raises(expected_error):
        build_commands(profile, include_figures=False)  # type: ignore[arg-type]


@pytest.mark.parametrize("include_figures", [0, 1, "yes", None])
def test_build_commands_rejects_non_boolean_figure_flags(include_figures: object) -> None:
    with pytest.raises(TypeError, match="include_figures"):
        build_commands("smoke", include_figures=include_figures)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("include_figures", "include_paper"),
    [(False, 0), (False, 1), (False, "yes"), (None, False)],
)
def test_run_rejects_non_boolean_phase_flags(
    include_figures: object,
    include_paper: object,
) -> None:
    with pytest.raises(TypeError):
        run(
            "smoke",
            include_figures=include_figures,  # type: ignore[arg-type]
            include_paper=include_paper,  # type: ignore[arg-type]
        )


def test_run_isolates_subprocesses_and_validates_each_stage_before_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []

    def fake_subprocess_run(command: list[str], *, cwd: Path, check: bool) -> None:
        events.append(("command", tuple(command), cwd, check))

    def fake_validate_artifact(npz_path: Path, sidecar_path: Path) -> None:
        events.append(("artifact", npz_path, sidecar_path))

    def fake_validate_manifest(path: Path) -> None:
        events.append(("manifest", path))

    monkeypatch.setattr(reproduce_all.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(reproduce_all, "validate_artifact", fake_validate_artifact)
    monkeypatch.setattr(reproduce_all, "validate_manifest", fake_validate_manifest)

    run("smoke", include_figures=True, include_paper=True)

    expected: list[tuple[object, ...]] = [
        (
            "command",
            (sys.executable, "-m", "pytest", "-q", "tests", "-m", "not slow"),
            ROOT,
            True,
        )
    ]
    for stage in STAGES:
        expected.append(
            (
                "command",
                (
                    sys.executable,
                    "-m",
                    "zgv_morse.workflows",
                    "--stage",
                    stage,
                    "--profile",
                    "smoke",
                ),
                ROOT,
                True,
            )
        )
        artifact = ROOT / "data/generated" / ARTIFACTS[stage]
        expected.append(("artifact", artifact, artifact.with_suffix(".json")))
    expected.append(("manifest", ROOT / "data/provenance_manifest.json"))
    expected.extend(
        ("command", tuple(command), ROOT, True)
        for command in build_commands("smoke", include_figures=True)[1 + len(STAGES) :]
    )
    expected.extend(
        [
            (
                "command",
                (sys.executable, "scripts/export_manuscript_values.py"),
                ROOT,
                True,
            ),
            (
                "command",
                (sys.executable, "scripts/compile_paper.py"),
                ROOT,
                True,
            ),
        ]
    )
    assert events == expected


def test_run_can_skip_both_optional_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    manifests: list[Path] = []

    monkeypatch.setattr(
        reproduce_all.subprocess,
        "run",
        lambda command, *, cwd, check: commands.append(command),
    )
    monkeypatch.setattr(reproduce_all, "validate_artifact", lambda *_: None)
    monkeypatch.setattr(reproduce_all, "validate_manifest", manifests.append)

    run("full", include_figures=False, include_paper=False)

    assert commands == build_commands("full", include_figures=False)
    assert manifests == [ROOT / "data/provenance_manifest.json"]


def test_main_maps_skip_options_to_boolean_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool, bool]] = []
    monkeypatch.setattr(
        reproduce_all,
        "run",
        lambda profile, include_figures, include_paper: calls.append(
            (profile, include_figures, include_paper)
        ),
    )

    main(["--profile", "smoke", "--skip-figures", "--skip-paper"])
    main(["--profile", "full"])

    assert calls == [("smoke", False, False), ("full", True, True)]
