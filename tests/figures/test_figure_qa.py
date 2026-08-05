from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys
from types import ModuleType
from typing import Any

from PIL import Image
import pytest

from zgv_morse.artifacts import sha256_file
from zgv_morse.provenance import ArtifactValidationError, FIGURE_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _load_qa_module() -> ModuleType:
    path = ROOT / "scripts/qa_figures.py"
    specification = importlib.util.spec_from_file_location("test_qa_figures", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import figure QA module from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


qa_figures = _load_qa_module()
FILE_RECORD_KEYS = {"path", "sha256"}
FIGURE_RECORD_KEYS = {
    "script",
    "script_sha256",
    "inputs",
    "source_data",
    "outputs",
    "theory_refs",
}
FORMATS = {"svg", "pdf", "png", "tiff"}
EXPECTED_THEORY_REFS = {
    "figure_01_geometry_mechanism": (
        "thm:morse-bott-ring",
        "thm:cubic-morse-splitting",
        "thm:uniform-bessel-crossover",
        "thm:fixed-anisotropy-decay",
    ),
    "figure_02_isotropic_zgv": ("thm:morse-bott-ring",),
    "figure_03_angular_sensitivity": (
        "thm:anisotropic-normal-form",
        "thm:cubic-morse-splitting",
    ),
    "figure_04_morse_points": ("thm:cubic-morse-splitting",),
    "figure_05_perturbation_scaling": (
        "thm:anisotropic-normal-form",
        "thm:cubic-morse-splitting",
    ),
    "figure_06_decay_crossover": (
        "thm:uniform-bessel-crossover",
        "thm:fixed-anisotropy-decay",
    ),
    "figure_s01_polynomial_two_element": ("thm:morse-bott-ring",),
    "figure_s02_quadrature_phase": (
        "thm:uniform-bessel-crossover",
        "thm:fixed-anisotropy-decay",
    ),
    "figure_s03_mode_tracking": (
        "thm:morse-bott-ring",
        "thm:anisotropic-normal-form",
    ),
    "figure_s04_fd_convergence": (
        "thm:anisotropic-normal-form",
        "thm:cubic-morse-splitting",
    ),
    "figure_s05_source_window_sensitivity": ("thm:uniform-bessel-crossover",),
    "figure_s06_silicon_stress_test": ("thm:cubic-morse-splitting",),
}


@pytest.fixture(scope="module")
def canonical_records() -> dict[str, dict[str, Any]]:
    return qa_figures.build_figure_records(ROOT)


def _copy_record_closure(
    project_root: Path,
    name: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    paths = [record["script"]]
    paths.extend(item["path"] for item in record["inputs"])
    paths.extend(item["path"] for item in record["source_data"])
    paths.extend(item["path"] for item in record["outputs"].values())
    for relative in paths:
        destination = project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return {name: deepcopy(record)}


def _minimal_contact_records(project_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for index, contract in enumerate(qa_figures.CONTRACTS):
        relative = f"figures/contact/{contract.stem}.png"
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (32, 20),
            color=((37 * index) % 255, (79 * index) % 255, (113 * index) % 255),
        ).save(path, format="PNG")
        records[contract.stem] = {
            "outputs": {"png": {"path": relative, "sha256": sha256_file(path)}}
        }
    return records


def test_declared_records_form_an_exact_valid_twelve_figure_closure(
    canonical_records: dict[str, dict[str, Any]],
) -> None:
    assert len(qa_figures.CONTRACTS) == 12
    assert {contract.stem for contract in qa_figures.CONTRACTS} == FIGURE_NAMES
    assert set(canonical_records) == FIGURE_NAMES
    assert {
        contract.stem: contract.theory_refs for contract in qa_figures.CONTRACTS
    } == EXPECTED_THEORY_REFS

    for name, record in canonical_records.items():
        assert set(record) == FIGURE_RECORD_KEYS
        assert record["script"].endswith(".py")
        assert len(record["script_sha256"]) == 64
        assert record["inputs"]
        assert record["source_data"]
        assert record["theory_refs"]
        assert tuple(record["theory_refs"]) == EXPECTED_THEORY_REFS[name]
        assert set(record["outputs"]) == FORMATS
        for item in [
            *record["inputs"],
            *record["source_data"],
            *record["outputs"].values(),
        ]:
            assert set(item) == FILE_RECORD_KEYS
            assert len(item["sha256"]) == 64
        assert all(item["path"].endswith(".csv") for item in record["source_data"])
        assert all(record["outputs"][kind]["path"].endswith(f"{name}.{kind}") for kind in FORMATS)

    all_sources = {
        item["path"]
        for record in canonical_records.values()
        for item in record["source_data"]
    }
    all_outputs = {
        item["path"]
        for record in canonical_records.values()
        for item in record["outputs"].values()
    }
    assert len(all_sources) == 47
    assert len(all_outputs) == 48
    assert all("pyproject.toml" in {item["path"] for item in record["inputs"]} for record in canonical_records.values())
    assert all("uv.lock" in {item["path"] for item in record["inputs"]} for record in canonical_records.values())
    for name in EXPECTED_THEORY_REFS:
        inputs = {item["path"] for item in canonical_records[name]["inputs"]}
        if name.startswith("figure_s"):
            assert "scripts/validate_isotropic.py" in inputs

    qa_figures.validate_figure_records_content(ROOT, canonical_records)


def test_content_validator_rejects_a_tampered_source_csv(
    tmp_path: Path,
    canonical_records: dict[str, dict[str, Any]],
) -> None:
    name = "figure_05_perturbation_scaling"
    records = _copy_record_closure(tmp_path, name, canonical_records[name])
    qa_figures.validate_figure_records_content(
        tmp_path,
        records,
        require_complete=False,
    )

    source = tmp_path / records[name]["source_data"][0]["path"]
    source.write_bytes(source.read_bytes() + b"\n0,0,tampered\n")

    with pytest.raises(ArtifactValidationError, match="source_data checksum mismatch"):
        qa_figures.validate_figure_records_content(
            tmp_path,
            records,
            require_complete=False,
        )


def test_contact_sheet_is_byte_deterministic(tmp_path: Path) -> None:
    records = _minimal_contact_records(tmp_path)
    first = qa_figures.make_contact_sheet(tmp_path, records, tmp_path / "first.png")
    second = qa_figures.make_contact_sheet(tmp_path, records, tmp_path / "second.png")

    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.size == (2400, 2400)


def test_qa_report_is_byte_deterministic_and_has_no_blocked_item(
    tmp_path: Path,
) -> None:
    records = _minimal_contact_records(tmp_path)
    first = qa_figures.write_qa_report(tmp_path / "first.md", records)
    second = qa_figures.write_qa_report(tmp_path / "second.md", records)
    text = first.read_text(encoding="utf-8")

    assert first.read_bytes() == second.read_bytes()
    assert sum(line.startswith("| figure_") for line in text.splitlines()) == 12
    assert "reviewed PNG SHA-256" in text
    assert "grayscale ambiguity" in text
    assert "contour readability" in text
    assert "## Blocked items\n\nNone.\n" in text
    assert "| blocked |" not in text
    qa_figures.validate_qa_report(first, records, require_hash_match=True)

    changed = deepcopy(records)
    changed[qa_figures.CONTRACTS[0].stem]["outputs"]["png"]["sha256"] = "0" * 64
    with pytest.raises(ArtifactValidationError, match="changed since"):
        qa_figures.validate_qa_report(first, changed, require_hash_match=True)
    qa_figures.validate_qa_report(first, changed, require_hash_match=False)


def test_export_parsers_reject_empty_or_truncated_shells(tmp_path: Path) -> None:
    contract = next(
        item for item in qa_figures.CONTRACTS if item.stem == "figure_05_perturbation_scaling"
    )
    fake_pdf = tmp_path / "empty.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    with pytest.raises(ArtifactValidationError, match="PDF"):
        qa_figures._validate_pdf(fake_pdf, contract)

    fake_svg = tmp_path / "empty.svg"
    labels = "".join(f"<text>{label}</text>" for label in contract.panels)
    fake_svg.write_text(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="518.740157pt" '
            'height="334.488189pt" viewBox="0 0 518.740157 334.488189">'
            f"<path d=\"M 0 0\"/>{labels}</svg>"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactValidationError, match="little editable text"):
        qa_figures._validate_svg(fake_svg, contract)

    source_png = ROOT / "figures/main/figure_05_perturbation_scaling.png"
    truncated_png = tmp_path / "truncated.png"
    content = source_png.read_bytes()
    truncated_png.write_bytes(content[: len(content) // 2])
    expected_pixels = (
        int(contract.width_mm / 25.4 * 600),
        int(contract.height_mm / 25.4 * 600),
    )
    with pytest.raises(ArtifactValidationError, match="malformed|truncated"):
        qa_figures._validate_raster(truncated_png, "png", expected_pixels)

    edge_touching = tmp_path / "edge-touching.png"
    with Image.new("RGB", expected_pixels, "white") as image:
        image.putpixel((expected_pixels[0] - 1, expected_pixels[1] // 2), (0, 0, 0))
        image.save(edge_touching, format="PNG", dpi=(600, 600))
    with pytest.raises(ArtifactValidationError, match="print-safe margin"):
        qa_figures._validate_raster(edge_touching, "png", expected_pixels)


def test_strict_content_failure_does_not_publish_a_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "data/provenance_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"figures": {}}\n', encoding="utf-8")
    before = manifest.read_bytes()
    published = False

    monkeypatch.setattr(qa_figures, "_validate_current_artifact_context", lambda _root: "full")
    monkeypatch.setattr(qa_figures, "build_figure_records", lambda _root: {})

    def reject_content(_root: Path, _records: dict[str, Any]) -> None:
        raise ArtifactValidationError("semantic export failure")

    def publish(*_args: object, **_kwargs: object) -> None:
        nonlocal published
        published = True

    monkeypatch.setattr(qa_figures, "validate_figure_records_content", reject_content)
    monkeypatch.setattr(qa_figures, "replace_figure_records", publish)

    with pytest.raises(ArtifactValidationError, match="semantic export failure"):
        qa_figures.run_strict(tmp_path)

    assert not published
    assert manifest.read_bytes() == before


def test_current_artifact_context_rejects_a_stale_repository_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qa_figures,
        "_scientific_context",
        lambda _config, _root: {
            "config_hash": "0" * 64,
            "source_hash": "0" * 64,
            "code_hash": "0" * 64,
            "uv_lock_hash": "0" * 64,
        },
    )

    with pytest.raises(ArtifactValidationError, match="does not match"):
        qa_figures._validate_current_artifact_context(ROOT)
