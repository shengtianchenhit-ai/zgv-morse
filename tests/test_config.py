from pathlib import Path

import pytest
import yaml

from zgv_morse.config import load_reference_config


REFERENCE_CONFIG = Path(__file__).parents[1] / "config" / "reference.yaml"
EXPECTED_EPSILON_VALUES = (
    -0.08,
    -0.04,
    -0.02,
    -0.01,
    -0.005,
    -0.0025,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.04,
    0.08,
)
NUMERIC_SCALAR_FIELDS = (
    "h",
    "rho",
    "lambda",
    "mu",
    "delta",
    "source_radius_over_h",
    "window_sigma_over_k0",
    "annulus_fraction",
    "eigen_residual_tolerance",
    "isotropic_match_tolerance",
    "curvature_match_tolerance",
    "sensitivity_match_tolerance",
    "phase_error_tolerance",
)
POSITIVE_SCALAR_FIELDS = (
    "h",
    "rho",
    "mu",
    "delta",
    "source_radius_over_h",
    "window_sigma_over_k0",
    "annulus_fraction",
)
TOLERANCE_FIELDS = (
    "eigen_residual_tolerance",
    "isotropic_match_tolerance",
    "curvature_match_tolerance",
    "sensitivity_match_tolerance",
    "phase_error_tolerance",
)
SEQUENCE_FIELDS = (
    "epsilon_values",
    "window_sensitivity",
    "zgv_search_kappa",
    "zgv_search_omega",
)
PAIR_FIELDS = (
    "window_sensitivity",
    "zgv_search_kappa",
    "zgv_search_omega",
)


def _reference_data() -> dict[str, object]:
    data = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _write_yaml(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "reference.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _write_variant(tmp_path: Path, field: str, value: object) -> Path:
    data = _reference_data()
    data[field] = value
    return _write_yaml(tmp_path, data)


def test_reference_config_loads_approved_values() -> None:
    config = load_reference_config(REFERENCE_CONFIG)

    assert config.schema_version == 1
    assert config.h == 1.0
    assert config.rho == 1.0
    assert config.lam == 2.0
    assert config.mu == 1.0
    assert config.delta == 1.0
    assert config.total_thickness == 2.0
    assert config.c_t == 1.0
    assert config.c_l == 2.0
    assert config.poisson_ratio == 1.0 / 3.0
    assert config.epsilon_values == EXPECTED_EPSILON_VALUES
    assert config.source_radius_over_h == pytest.approx(0.5)
    assert config.window_sigma_over_k0 == 0.15
    assert config.window_sensitivity == pytest.approx((0.10, 0.20))
    assert config.annulus_fraction == 0.15
    assert config.zgv_search_kappa == pytest.approx((0.20, 1.40))
    assert config.zgv_search_omega == pytest.approx((2.40, 3.60))
    assert config.eigen_residual_tolerance == pytest.approx(1e-10)
    assert config.isotropic_match_tolerance == pytest.approx(1e-7)
    assert config.curvature_match_tolerance == pytest.approx(1e-4)
    assert config.sensitivity_match_tolerance == pytest.approx(1e-4)
    assert config.phase_error_tolerance == pytest.approx(5e-2)


@pytest.mark.parametrize("document", [[], "not a mapping", 1], ids=["list", "string", "integer"])
def test_loader_rejects_non_mapping_yaml_root(tmp_path: Path, document: object) -> None:
    path = _write_yaml(tmp_path, document)

    with pytest.raises(ValueError, match="mapping"):
        load_reference_config(path)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool"),
        pytest.param(1.0, id="float"),
        pytest.param(2, id="wrong-integer"),
    ],
)
def test_loader_rejects_invalid_schema_version(tmp_path: Path, value: object) -> None:
    path = _write_variant(tmp_path, "schema_version", value)

    with pytest.raises(ValueError, match="schema_version"):
        load_reference_config(path)


@pytest.mark.parametrize("field", NUMERIC_SCALAR_FIELDS)
@pytest.mark.parametrize("value", [pytest.param(True, id="bool"), pytest.param("1.0", id="string")])
def test_loader_rejects_non_numeric_scalar_types(tmp_path: Path, field: str, value: object) -> None:
    path = _write_variant(tmp_path, field, value)

    with pytest.raises(ValueError, match=field):
        load_reference_config(path)


@pytest.mark.parametrize("field", SEQUENCE_FIELDS)
@pytest.mark.parametrize(
    "value",
    [pytest.param("0.1, 0.2", id="string"), pytest.param({"low": 0.1}, id="mapping")],
)
def test_loader_rejects_non_list_sequence_fields(tmp_path: Path, field: str, value: object) -> None:
    path = _write_variant(tmp_path, field, value)

    with pytest.raises(ValueError, match=field):
        load_reference_config(path)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([], id="empty"),
        pytest.param([-0.1, 0.0, 0.1], id="zero"),
        pytest.param([-0.1, -0.2], id="unsorted"),
        pytest.param([-0.1, float("inf")], id="nonfinite"),
    ],
)
def test_loader_rejects_invalid_epsilon_values(tmp_path: Path, value: object) -> None:
    path = _write_variant(tmp_path, "epsilon_values", value)

    with pytest.raises(ValueError, match="epsilon_values"):
        load_reference_config(path)


@pytest.mark.parametrize("field", PAIR_FIELDS)
@pytest.mark.parametrize(
    "value",
    [pytest.param([0.1], id="short"), pytest.param([0.1, 0.2, 0.3], id="long")],
)
def test_loader_rejects_wrong_pair_lengths(tmp_path: Path, field: str, value: object) -> None:
    path = _write_variant(tmp_path, field, value)

    with pytest.raises(ValueError, match=field):
        load_reference_config(path)


@pytest.mark.parametrize("field", ["zgv_search_kappa", "zgv_search_omega"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param([1.4, 0.2], id="reversed"),
        pytest.param([0.2, float("inf")], id="nonfinite"),
    ],
)
def test_loader_rejects_invalid_search_bounds(tmp_path: Path, field: str, value: object) -> None:
    path = _write_variant(tmp_path, field, value)

    with pytest.raises(ValueError, match=field):
        load_reference_config(path)


@pytest.mark.parametrize(
    "value",
    [pytest.param([0.1, 0.0], id="nonpositive"), pytest.param([0.1, float("nan")], id="nonfinite")],
)
def test_loader_rejects_invalid_window_sensitivity(tmp_path: Path, value: object) -> None:
    path = _write_variant(tmp_path, "window_sensitivity", value)

    with pytest.raises(ValueError, match="window_sensitivity"):
        load_reference_config(path)


@pytest.mark.parametrize("field", NUMERIC_SCALAR_FIELDS)
def test_loader_rejects_nonfinite_scalars(tmp_path: Path, field: str) -> None:
    path = _write_variant(tmp_path, field, float("inf"))

    with pytest.raises(ValueError, match=field):
        load_reference_config(path)


@pytest.mark.parametrize("field", POSITIVE_SCALAR_FIELDS)
def test_loader_rejects_nonpositive_required_scalars(tmp_path: Path, field: str) -> None:
    path = _write_variant(tmp_path, field, 0.0)

    with pytest.raises(ValueError, match=field):
        load_reference_config(path)


@pytest.mark.parametrize("field", TOLERANCE_FIELDS)
@pytest.mark.parametrize(
    "value", [pytest.param(0.0, id="nonpositive"), pytest.param(float("nan"), id="nonfinite")]
)
def test_loader_rejects_invalid_tolerances(tmp_path: Path, field: str, value: object) -> None:
    path = _write_variant(tmp_path, field, value)

    with pytest.raises(ValueError, match=field):
        load_reference_config(path)
