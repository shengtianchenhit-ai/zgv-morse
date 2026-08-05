from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import mpmath as mp
import pytest

from zgv_morse.config import ReferenceConfig, load_reference_config
from zgv_morse.rayleigh_lamb import det_symmetric_mp
from zgv_morse.zgv import ZGVPoint, _roots_in_window, find_s1_zgv


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = ROOT / "config" / "reference.yaml"


@pytest.fixture(scope="module")
def cfg() -> ReferenceConfig:
    return load_reference_config(REFERENCE_CONFIG)


def _manual_mp_determinant(
    k: mp.mpf,
    omega: mp.mpf,
    c_l: mp.mpf,
    c_t: mp.mpf,
    h: mp.mpf,
) -> mp.mpf | mp.mpc:
    """Independent determinant evaluation through the two traction columns."""

    p = mp.sqrt((omega / c_l) ** 2 - k**2)
    s = mp.sqrt((omega / c_t) ** 2 - k**2)
    sin_s_over_s = h if s == 0 else mp.sin(s * h) / s
    shear_factor = s**2 - k**2
    first_column = (-2j * k * p * mp.sin(p * h), -shear_factor * mp.cos(p * h))
    second_column = (shear_factor * sin_s_over_s, 2j * k * mp.cos(s * h))
    return first_column[0] * second_column[1] - second_column[0] * first_column[1]


@pytest.mark.parametrize(
    ("k_text", "omega_text"),
    [
        pytest.param("0.8", "2.86", id="fully-propagating"),
        pytest.param("2.0", "2.5", id="mixed"),
        pytest.param("3.0", "0.5", id="fully-evanescent"),
        pytest.param("1.0", "1.0", id="s-cutoff"),
        pytest.param("1.0", "2.0", id="p-cutoff"),
    ],
)
def test_mp_determinant_matches_independent_high_precision_formula(
    k_text: str,
    omega_text: str,
) -> None:
    with mp.workdps(110):
        k = mp.mpf(k_text)
        omega = mp.mpf(omega_text)
        c_l = mp.mpf("2")
        c_t = mp.mpf("1")
        h = mp.mpf("1")
        expected = _manual_mp_determinant(k, omega, c_l, c_t, h)
        actual = det_symmetric_mp(k, omega, c_l, c_t, h)

        assert mp.almosteq(actual, expected, rel_eps=mp.mpf("1e-100"), abs_eps=mp.mpf("1e-100"))


def test_mp_determinant_preserves_caller_precision_and_context() -> None:
    with mp.workdps(90):
        dps_before = mp.mp.dps
        k = mp.mpf("0.8042173193715180865063397354")
        value = det_symmetric_mp(k, mp.mpf("2.851758774960090066"), 2, 1, 1)
        expected = _manual_mp_determinant(
            k, mp.mpf("2.851758774960090066"), mp.mpf(2), mp.mpf(1), mp.mpf(1)
        )

        assert mp.mp.dps == dps_before
        assert mp.almosteq(value, expected, rel_eps=mp.mpf("1e-80"), abs_eps=mp.mpf("1e-80"))


@pytest.mark.parametrize("field", ["k", "omega", "c_l", "c_t", "h"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="boolean"),
        pytest.param("1.0", id="string"),
        pytest.param(1.0 + 0.0j, id="complex"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_mp_determinant_rejects_nonreal_or_nonfinite_inputs(field: str, value: object) -> None:
    arguments: dict[str, object] = {"k": 0.8, "omega": 2.86, "c_l": 2.0, "c_t": 1.0, "h": 1.0}
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        det_symmetric_mp(**arguments)


@pytest.mark.parametrize("field", ["c_l", "c_t", "h"])
@pytest.mark.parametrize("value", [0.0, -1.0], ids=["zero", "negative"])
def test_mp_determinant_rejects_nonpositive_material_parameters(
    field: str,
    value: float,
) -> None:
    arguments = {"k": 0.8, "omega": 2.86, "c_l": 2.0, "c_t": 1.0, "h": 1.0}
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        det_symmetric_mp(**arguments)


def test_reference_s1_zgv_matches_independent_high_precision_values(
    cfg: ReferenceConfig,
) -> None:
    point = find_s1_zgv(cfg)

    assert point.kappa0 == pytest.approx(0.8042173193715180865, rel=2e-8, abs=2e-9)
    assert point.omega0 == pytest.approx(2.851758774960090066, rel=2e-8, abs=2e-9)
    assert point.curvature_a == pytest.approx(1.196862725073930122, rel=2e-5)
    assert abs(point.det_residual) < 1e-10
    assert abs(point.group_velocity) < 1e-8
    assert point.curvature_a > 0.0
    assert point.branch_index == 1


def test_reference_solution_is_the_lowest_local_minimum_in_the_window(
    cfg: ReferenceConfig,
) -> None:
    point = find_s1_zgv(cfg)
    roots = _roots_in_window(point.kappa0, cfg)
    left = _roots_in_window(point.kappa0 - 1.0e-3, cfg)[0]
    right = _roots_in_window(point.kappa0 + 1.0e-3, cfg)[0]

    assert roots == sorted(roots)
    assert roots[0] == pytest.approx(point.omega0, rel=0.0, abs=2.0e-10)
    assert left > point.omega0
    assert right > point.omega0


def test_refinement_is_deterministic_at_sixty_and_eighty_digits(
    cfg: ReferenceConfig,
) -> None:
    at_sixty = find_s1_zgv(cfg, dps=60)
    at_eighty = find_s1_zgv(cfg, dps=80)

    assert at_sixty.kappa0 == at_eighty.kappa0
    assert at_sixty.omega0 == at_eighty.omega0
    assert at_sixty.curvature_a == at_eighty.curvature_a
    assert at_sixty.branch_index == at_eighty.branch_index == 1
    assert max(at_sixty.det_residual, at_eighty.det_residual) < 1.0e-50
    assert max(abs(at_sixty.group_velocity), abs(at_eighty.group_velocity)) < 1.0e-50


def test_zgv_point_is_frozen() -> None:
    point = ZGVPoint(0.8, 2.85, 1.2, 0.0, 0.0)

    with pytest.raises(FrozenInstanceError):
        point.omega0 = 3.0  # type: ignore[misc]


@pytest.mark.parametrize("dps", [True, 60.0, "60", None], ids=["bool", "float", "str", "none"])
def test_find_s1_zgv_requires_a_strict_integer_dps(
    cfg: ReferenceConfig,
    dps: object,
) -> None:
    with pytest.raises(TypeError, match="dps"):
        find_s1_zgv(cfg, dps=dps)  # type: ignore[arg-type]


@pytest.mark.parametrize("dps", [0, 49])
def test_find_s1_zgv_requires_at_least_fifty_digits(
    cfg: ReferenceConfig,
    dps: int,
) -> None:
    with pytest.raises(ValueError, match="dps"):
        find_s1_zgv(cfg, dps=dps)


def test_fifty_digit_request_uses_the_sixty_digit_refinement_floor(
    cfg: ReferenceConfig,
) -> None:
    assert find_s1_zgv(cfg, dps=50) == find_s1_zgv(cfg, dps=60)


def test_find_s1_zgv_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="cfg"):
        find_s1_zgv(object())  # type: ignore[arg-type]


def test_find_s1_zgv_validates_manually_constructed_config(cfg: ReferenceConfig) -> None:
    invalid = replace(cfg, zgv_search_kappa=(1.4, 0.2))

    with pytest.raises(ValueError, match="zgv_search_kappa"):
        find_s1_zgv(invalid)


def test_find_s1_zgv_reports_a_configured_window_without_roots(cfg: ReferenceConfig) -> None:
    no_root_cfg = replace(
        cfg,
        zgv_search_kappa=(0.75, 0.85),
        zgv_search_omega=(0.01, 0.02),
    )

    with pytest.raises(RuntimeError, match="no symmetric root"):
        find_s1_zgv(no_root_cfg)
