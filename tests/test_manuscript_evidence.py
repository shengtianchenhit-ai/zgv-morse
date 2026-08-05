from __future__ import annotations

import ast
import csv
from pathlib import Path
import re
import runpy
import subprocess
import sys

import numpy as np
import pytest

from zgv_morse.artifact_schema import validate_artifact


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/manuscript/claim_evidence_matrix.csv"
EXPORTER = ROOT / "scripts/export_manuscript_values.py"
CHECKER = ROOT / "scripts/check_claim_evidence.py"
GENERATED_MACROS = ROOT / "paper/generated/results_macros.tex"

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

VALUE_MAP = {
    "ZGVKappa": ("isotropic_zgv", "kappa0"),
    "ZGVOmega": ("isotropic_zgv", "omega0"),
    "ZGVCurvature": ("isotropic_zgv", "curvature_a"),
    "VFour": ("angular_sensitivity", "V4"),
    "SplittingSlope": ("perturbation_scaling", "slope_splitting"),
    "RemainderSlope": ("perturbation_scaling", "slope_remainder"),
    "EarlyDecaySlope": ("green_crossover", "slope_early"),
    "LateDecaySlope": ("green_crossover", "slope_late"),
    "MaxPhaseError": ("green_crossover", "phase_error"),
}

FIXED_COUNT_MACROS = {
    "MorseMinimumCount": "4",
    "MorseSaddleCount": "4",
}

FIGURE_MODULES = {
    "figure_01_geometry_mechanism": "src/zgv_morse/figures/figure01_geometry.py",
    "figure_02_isotropic_zgv": "src/zgv_morse/figures/figure02_isotropic.py",
    "figure_03_angular_sensitivity": "src/zgv_morse/figures/figure03_sensitivity.py",
    "figure_04_morse_points": "src/zgv_morse/figures/figure04_morse.py",
    "figure_05_perturbation_scaling": "src/zgv_morse/figures/figure05_scaling.py",
    "figure_06_decay_crossover": "src/zgv_morse/figures/figure06_crossover.py",
}

RESULTS_FILES = tuple(
    ROOT / "paper/sections" / name
    for name in (
        "02_isotropic_ring.tex",
        "03_morse_unfolding.tex",
        "04_temporal_crossover.tex",
        "05_numerical_verification.tex",
    )
)

DERIVATION_FILES = tuple(sorted((ROOT / "docs/derivations").glob("0[1-4]_*.tex")))
MULTIVALUE_COLUMNS = (
    "theory_labels",
    "artifact_keys",
    "metadata_paths",
    "test_nodes",
    "figure_ids",
    "literature_keys",
)


def _read_matrix() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == HEADER
        return list(reader)


def _items(row: dict[str, str], column: str) -> tuple[str, ...]:
    value = row[column]
    assert value == value.strip()
    assert "," not in value, f"{row['claim_id']}.{column} must use semicolons"
    items = tuple(value.split(";"))
    assert items and all(item and item == item.strip() for item in items)
    assert len(items) == len(set(items)), f"duplicate {row['claim_id']}.{column} entry"
    return items


def _latex_number(value: float) -> str:
    rendered = format(float(value), ".6g")
    if "e" not in rendered:
        return rendered
    mantissa, exponent = rendered.split("e", maxsplit=1)
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def _parse_macros(text: str) -> dict[str, str]:
    pattern = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.+)\}$")
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = pattern.fullmatch(line)
        assert match is not None, f"malformed generated macro line: {line!r}"
        name, value = match.groups()
        assert name not in parsed
        parsed[name] = value
    return parsed


@pytest.fixture(scope="module")
def claims() -> list[dict[str, str]]:
    return _read_matrix()


@pytest.fixture(scope="module")
def generated_macro_text() -> str:
    assert EXPORTER.is_file(), f"missing exporter: {EXPORTER.relative_to(ROOT)}"
    command = [sys.executable, str(EXPORTER)]
    first = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    first_bytes = GENERATED_MACROS.read_bytes()
    second = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert second.returncode == 0, second.stdout + second.stderr
    second_bytes = GENERATED_MACROS.read_bytes()
    assert first_bytes == second_bytes, "macro export must be byte deterministic"
    assert second_bytes.endswith(b"\n")
    return second_bytes.decode("utf-8")


def test_claim_matrix_has_exact_schema_and_supported_c1_through_c7(claims):
    assert len(claims) == 7
    assert [row["claim_id"] for row in claims] == [f"C{index}" for index in range(1, 8)]
    assert all(row["status"] == "supported" for row in claims)
    for row in claims:
        assert all(row[column].strip() for column in HEADER)
        for column in MULTIVALUE_COLUMNS:
            _items(row, column)


def test_claim_artifact_arrays_exist_and_each_pair_validates(claims):
    validated: dict[str, dict[str, np.ndarray]] = {}
    for row in claims:
        for reference in _items(row, "artifact_keys"):
            artifact, separator, array_key = reference.partition(".")
            assert separator and artifact and array_key, f"invalid artifact key: {reference}"
            if artifact not in validated:
                npz_path = ROOT / "data/generated" / f"{artifact}.npz"
                json_path = npz_path.with_suffix(".json")
                assert npz_path.is_file() and json_path.is_file()
                arrays, _metadata = validate_artifact(npz_path, json_path)
                validated[artifact] = arrays
            assert array_key in validated[artifact], f"missing registered array: {reference}"


def test_metadata_paths_exist_and_test_nodes_collect(claims):
    nodes: list[str] = []
    for row in claims:
        for relative in _items(row, "metadata_paths"):
            path = ROOT / relative
            assert path.is_file(), f"missing metadata path: {relative}"
        for node in _items(row, "test_nodes"):
            relative, separator, function = node.partition("::")
            assert separator and function.startswith("test_") and "::" not in function
            path = ROOT / relative
            assert path.is_file(), f"missing test module: {relative}"
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            functions = {
                item.name
                for item in ast.walk(tree)
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert function in functions, f"missing test function: {node}"
            nodes.append(node)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *nodes],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_figure_ids_name_real_modules_and_output_stems(claims):
    for row in claims:
        for figure_id in _items(row, "figure_ids"):
            assert figure_id in FIGURE_MODULES, f"unregistered main figure: {figure_id}"
            module = ROOT / FIGURE_MODULES[figure_id]
            assert module.is_file()
            source = module.read_text(encoding="utf-8")
            assert f'output_dir / "{figure_id}"' in source


def test_every_registered_theory_label_occurs_once_in_four_derivations(claims):
    assert len(DERIVATION_FILES) == 4
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in DERIVATION_FILES)
    for row in claims:
        for label in _items(row, "theory_labels"):
            token = rf"\label{{{label}}}"
            assert corpus.count(token) == 1, f"theory label must occur exactly once: {label}"


def test_results_sections_contain_no_bare_computed_decimal_literals():
    decimal = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d+|\.\d+|\d+[eE][+-]?\d+)")
    structural = re.compile(
        r"\\(?:label|ref|eqref|autoref|cref|Cref|cite[a-zA-Z]*)\{[^{}]*\}"
    )
    for path in RESULTS_FILES:
        assert path.is_file()
        uncommented = "\n".join(line.split("%", maxsplit=1)[0] for line in path.read_text().splitlines())
        prose = structural.sub("", uncommented)
        matches = [match.group(0) for match in decimal.finditer(prose)]
        assert not matches, (
            f"{path.relative_to(ROOT)} contains bare computed decimals {matches}; "
            "export artifact-backed values as generated macros"
        )


def test_exporter_declares_exact_artifact_mapping_and_uses_validator():
    assert EXPORTER.is_file(), f"missing exporter: {EXPORTER.relative_to(ROOT)}"
    namespace = runpy.run_path(str(EXPORTER))
    assert namespace.get("VALUES") == VALUE_MAP
    tree = ast.parse(EXPORTER.read_text(encoding="utf-8"), filename=str(EXPORTER))
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "validate_artifact" in call_names


def test_exported_macros_are_deterministic_six_digit_artifact_values(generated_macro_text):
    macros = _parse_macros(generated_macro_text)
    assert set(macros) == set(VALUE_MAP) | set(FIXED_COUNT_MACROS)

    cache: dict[str, dict[str, np.ndarray]] = {}
    for macro, (artifact, array_key) in VALUE_MAP.items():
        if artifact not in cache:
            path = ROOT / "data/generated" / f"{artifact}.npz"
            cache[artifact], _metadata = validate_artifact(path, path.with_suffix(".json"))
        value = np.asarray(cache[artifact][array_key])
        if macro == "MaxPhaseError":
            expected = float(np.max(value))
        else:
            assert value.size == 1, f"{macro} must scalarize a one-value artifact array"
            expected = float(value.reshape(-1)[0])
        assert macros[macro] == _latex_number(expected)

    assert {name: macros[name] for name in FIXED_COUNT_MACROS} == FIXED_COUNT_MACROS


def test_critical_artifact_contains_four_minima_and_four_saddles():
    path = ROOT / "data/generated/critical_points.npz"
    arrays, _metadata = validate_artifact(path, path.with_suffix(".json"))
    kinds = np.asarray(arrays["kind"])
    assert np.count_nonzero(kinds == "minimum") == 4
    assert np.count_nonzero(kinds == "saddle") == 4
    assert kinds.size == 8


def test_hard_evidence_checker_accepts_supported_matrix():
    assert CHECKER.is_file(), f"missing checker: {CHECKER.relative_to(ROOT)}"
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--require-supported"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hard_evidence_checker_rejects_a_registered_test_failure(monkeypatch):
    namespace = runpy.run_path(str(CHECKER))
    evidence_error = namespace["EvidenceError"]
    run_registered_tests = namespace["_run_registered_tests"]

    def fail_registered_node(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["pytest"],
            returncode=1,
            stdout="registered failure",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fail_registered_node)
    with pytest.raises(evidence_error, match="registered evidence tests failed"):
        run_registered_tests(("tests/test_example.py::test_fails",))


def test_main_assembly_orders_macros_sections_and_bibliography():
    main = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    required = [
        r"\input{shared_macros.tex}",
        r"\input{generated/results_macros.tex}",
        *(rf"\input{{sections/{index:02d}_{name}.tex}}" for index, name in enumerate(
            (
                "abstract",
                "introduction",
                "isotropic_ring",
                "morse_unfolding",
                "temporal_crossover",
                "numerical_verification",
                "discussion",
                "methods",
                "conclusion",
            )
        )),
        r"\bibliography{references}",
    ]
    positions = [main.find(token) for token in required]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
    assert main.index(r"\usepackage[numbers,sort&compress]{natbib}") < main.index(
        r"\usepackage[hidelinks]{hyperref}"
    )
