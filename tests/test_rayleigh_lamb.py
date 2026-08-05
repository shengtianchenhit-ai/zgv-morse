from __future__ import annotations

import warnings
from collections.abc import Callable

import mpmath as mp
import numpy as np
import pytest

import zgv_morse.rayleigh_lamb as rayleigh_lamb
from zgv_morse.rayleigh_lamb import (
    det_symmetric,
    det_symmetric_real,
    sin_wavenumber_over_wavenumber,
    thickness_wavenumbers,
    traction_matrix_symmetric,
)


WaveOperation = Callable[[object, object, object, object], object]
WAVE_OPERATIONS: tuple[object, ...] = (
    pytest.param(thickness_wavenumbers, id="wavenumbers"),
    pytest.param(
        lambda k, omega, c_l, c_t: traction_matrix_symmetric(k, omega, c_l, c_t, h=1.0),
        id="traction-matrix",
    ),
    pytest.param(
        lambda k, omega, c_l, c_t: det_symmetric(k, omega, c_l, c_t, h=1.0),
        id="determinant",
    ),
    pytest.param(
        lambda k, omega, c_l, c_t: det_symmetric_real(k, omega, c_l, c_t, h=1.0),
        id="real-determinant",
    ),
)


def _high_precision_determinant(
    k: float,
    omega: float,
    c_l: float,
    c_t: float,
    h: float,
    decimal_precision: int = 120,
) -> complex:
    with mp.workdps(decimal_precision):
        k_mp, omega_mp, c_l_mp, c_t_mp, h_mp = (
            mp.mpf(str(value)) for value in (k, omega, c_l, c_t, h)
        )
        p = mp.sqrt((omega_mp / c_l_mp) ** 2 - k_mp**2)
        s = mp.sqrt((omega_mp / c_t_mp) ** 2 - k_mp**2)
        sine_ratio = h_mp if s == 0 else mp.sin(s * h_mp) / s
        value = (s**2 - k_mp**2) ** 2 * sine_ratio * mp.cos(p * h_mp) + 4 * k_mp**2 * p * mp.cos(
            s * h_mp
        ) * mp.sin(p * h_mp)
        return complex(value)


def _high_precision_sine_ratio(wavenumber: complex, h: float) -> complex:
    with mp.workdps(100):
        wavenumber_mp = mp.mpc(str(wavenumber.real), str(wavenumber.imag))
        h_mp = mp.mpf(str(h))
        return complex(mp.sin(wavenumber_mp * h_mp) / wavenumber_mp)


def _invoke_with_wave_argument(
    operation: WaveOperation,
    field: str,
    value: object,
) -> object:
    parameters: dict[str, object] = {
        "k": 0.8,
        "omega": 2.86,
        "c_l": 2.0,
        "c_t": 1.0,
    }
    parameters[field] = value
    return operation(
        parameters["k"],
        parameters["omega"],
        parameters["c_l"],
        parameters["c_t"],
    )


def test_desingularized_determinant_is_finite_at_s_zero() -> None:
    k = 1.0
    omega = 1.0

    _, s = thickness_wavenumbers(k, omega, c_l=2.0, c_t=1.0)
    value = det_symmetric(k, omega, c_l=2.0, c_t=1.0, h=1.0)

    assert abs(s) < 1.0e-14
    assert np.isfinite(value)


@pytest.mark.parametrize(
    ("k", "omega"),
    [(0.3, 2.9), (0.8, 2.86), (1.2, 3.0), (2.0, 2.5)],
)
def test_closed_form_matches_scaled_traction_matrix(k: float, omega: float) -> None:
    direct = det_symmetric(k, omega, c_l=2.0, c_t=1.0, h=1.0)
    matrix = traction_matrix_symmetric(k, omega, c_l=2.0, c_t=1.0, h=1.0)

    assert np.linalg.det(matrix) == pytest.approx(direct, rel=2.0e-12, abs=2.0e-12)


def test_traction_matrix_has_documented_entries() -> None:
    k = 0.8
    omega = 2.86
    h = 1.3
    p, s = thickness_wavenumbers(k, omega, c_l=2.0, c_t=1.0)
    expected = np.array(
        [
            [
                -2.0j * k * p * np.sin(p * h),
                (s**2 - k**2) * sin_wavenumber_over_wavenumber(s, h),
            ],
            [
                -(s**2 - k**2) * np.cos(p * h),
                2.0j * k * np.cos(s * h),
            ],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_array_equal(
        traction_matrix_symmetric(k, omega, c_l=2.0, c_t=1.0, h=h),
        expected,
    )


def test_known_k_zero_symmetric_cutoff_is_a_root() -> None:
    value = det_symmetric_real(0.0, np.pi, c_l=2.0, c_t=1.0, h=1.0)

    assert abs(value) < 1.0e-10


@pytest.mark.parametrize(
    ("component", "cutoff"),
    [pytest.param(0, 2.0, id="p-cutoff"), pytest.param(1, 1.0, id="s-cutoff")],
)
def test_thickness_wavenumbers_use_principal_complex_branch(
    component: int,
    cutoff: float,
) -> None:
    below = np.nextafter(cutoff, -np.inf)
    above = np.nextafter(cutoff, np.inf)

    below_value = thickness_wavenumbers(1.0, below, c_l=2.0, c_t=1.0)[component]
    at_value = thickness_wavenumbers(1.0, cutoff, c_l=2.0, c_t=1.0)[component]
    above_value = thickness_wavenumbers(1.0, above, c_l=2.0, c_t=1.0)[component]

    assert below_value.real == 0.0
    assert below_value.imag > 0.0
    assert at_value == 0.0j
    assert above_value.real > 0.0
    assert above_value.imag == 0.0


@pytest.mark.parametrize(
    "wavenumber",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(1.0e-12, id="positive-real"),
        pytest.param(-1.0e-12, id="negative-real"),
        pytest.param(1.0e-12j, id="positive-imaginary"),
        pytest.param(-1.0e-12j, id="negative-imaginary"),
        pytest.param(1.0e-12 + 2.0e-12j, id="complex"),
    ],
)
def test_sine_ratio_is_continuous_at_zero(wavenumber: complex) -> None:
    result = sin_wavenumber_over_wavenumber(wavenumber, h=1.7)

    assert np.isfinite(result)
    assert result == pytest.approx(1.7 + 0.0j, rel=0.0, abs=2.0e-15)


@pytest.mark.parametrize(
    "wavenumber",
    [
        pytest.param(0.3, id="real"),
        pytest.param(0.3j, id="imaginary"),
        pytest.param(0.2 + 0.4j, id="complex"),
    ],
)
def test_sine_ratio_matches_analytic_expression_away_from_zero(
    wavenumber: complex,
) -> None:
    h = 1.7

    result = sin_wavenumber_over_wavenumber(wavenumber, h)

    assert result == pytest.approx(np.sin(wavenumber * h) / wavenumber, rel=2.0e-15)


def test_public_values_are_complex128_compatible_and_finite() -> None:
    inputs = tuple(np.float64(value) for value in (0.8, 2.86, 2.0, 1.0))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p, s = thickness_wavenumbers(*inputs)
        ratio = sin_wavenumber_over_wavenumber(s, np.float64(1.0))
        matrix = traction_matrix_symmetric(*inputs, np.float64(1.0))
        determinant = det_symmetric(*inputs, np.float64(1.0))
        real_determinant = det_symmetric_real(*inputs, np.float64(1.0))

    assert isinstance(p, np.complex128)
    assert isinstance(s, np.complex128)
    assert isinstance(ratio, np.complex128)
    assert matrix.dtype == np.complex128
    assert isinstance(determinant, np.complex128)
    assert isinstance(real_determinant, float)
    assert np.isfinite(matrix).all()
    assert np.isfinite(determinant)
    assert np.isfinite(real_determinant)


@pytest.mark.parametrize("operation", WAVE_OPERATIONS)
@pytest.mark.parametrize("field", ["k", "omega", "c_l", "c_t"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="infinity"),
        pytest.param(1.0 + 0.0j, id="complex"),
        pytest.param(True, id="boolean"),
        pytest.param("1.0", id="string"),
        pytest.param(10**1000, id="out-of-range-integer"),
    ],
)
def test_wave_apis_reject_nonfinite_or_nonreal_scalar_inputs(
    operation: WaveOperation,
    field: str,
    value: object,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=field):
            _invoke_with_wave_argument(operation, field, value)


@pytest.mark.parametrize("operation", WAVE_OPERATIONS)
@pytest.mark.parametrize("field", ["c_l", "c_t"])
@pytest.mark.parametrize("value", [0.0, -1.0], ids=["zero", "negative"])
def test_wave_apis_reject_nonpositive_speeds(
    operation: WaveOperation,
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        _invoke_with_wave_argument(operation, field, value)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda h: sin_wavenumber_over_wavenumber(0.2j, h), id="sine-ratio"),
        pytest.param(
            lambda h: traction_matrix_symmetric(0.8, 2.86, 2.0, 1.0, h),
            id="traction-matrix",
        ),
        pytest.param(
            lambda h: det_symmetric(0.8, 2.86, 2.0, 1.0, h),
            id="determinant",
        ),
        pytest.param(
            lambda h: det_symmetric_real(0.8, 2.86, 2.0, 1.0, h),
            id="real-determinant",
        ),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="infinity"),
        pytest.param(1.0 + 0.0j, id="complex"),
        pytest.param(True, id="boolean"),
        pytest.param("1.0", id="string"),
        pytest.param(10**1000, id="out-of-range-integer"),
    ],
)
def test_h_apis_reject_invalid_thickness(
    operation: Callable[[object], object],
    value: object,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="h"):
            operation(value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="infinity"),
        pytest.param(complex(0.0, np.nan), id="complex-nan"),
        pytest.param(complex(0.0, np.inf), id="complex-infinity"),
        pytest.param(True, id="boolean"),
        pytest.param("1.0", id="string"),
        pytest.param(10**1000, id="out-of-range-integer"),
    ],
)
def test_sine_ratio_rejects_invalid_wavenumbers(value: object) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="wavenumber"):
            sin_wavenumber_over_wavenumber(value, h=1.0)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="infinity"),
        pytest.param(1.0 + 0.0j, id="complex"),
        pytest.param(True, id="boolean"),
        pytest.param("1.0", id="string"),
        pytest.param(10**1000, id="out-of-range-integer"),
    ],
)
def test_real_determinant_rejects_invalid_imaginary_tolerance(value: object) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="imag_tolerance"):
            det_symmetric_real(0.8, 2.86, 2.0, 1.0, 1.0, imag_tolerance=value)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda: thickness_wavenumbers(1.0e308, 1.0, 2.0, 1.0),
            id="wavenumbers",
        ),
        pytest.param(
            lambda: sin_wavenumber_over_wavenumber(1000.0j, 1.0),
            id="sine-ratio",
        ),
        pytest.param(
            lambda: traction_matrix_symmetric(1.0e308, 1.0, 2.0, 1.0, 1.0),
            id="traction-matrix",
        ),
        pytest.param(
            lambda: det_symmetric(1.0e308, 1.0, 2.0, 1.0, 1.0),
            id="determinant",
        ),
        pytest.param(
            lambda: det_symmetric_real(1.0e308, 1.0, 2.0, 1.0, 1.0),
            id="real-determinant",
        ),
    ],
)
def test_nonfinite_computed_results_raise_without_warnings(
    operation: Callable[[], object],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FloatingPointError, match="non-finite"):
            operation()


def test_real_determinant_rejects_significant_imaginary_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rayleigh_lamb,
        "det_symmetric",
        lambda *args, **kwargs: np.complex128(2.0 + 1.0e-6j),
    )

    with pytest.raises(FloatingPointError, match="imaginary leakage"):
        det_symmetric_real(0.8, 2.86, 2.0, 1.0, 1.0, imag_tolerance=1.0e-10)


def test_real_determinant_accepts_roundoff_scale_imaginary_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rayleigh_lamb,
        "det_symmetric",
        lambda *args, **kwargs: np.complex128(3.0 + 2.0e-10j),
    )

    assert det_symmetric_real(
        0.8,
        2.86,
        2.0,
        1.0,
        1.0,
        imag_tolerance=1.0e-10,
    ) == pytest.approx(3.0)


def test_real_determinant_rejects_nonfinite_patched_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rayleigh_lamb,
        "det_symmetric",
        lambda *args, **kwargs: np.complex128(np.inf),
    )

    with pytest.raises(FloatingPointError, match="non-finite"):
        det_symmetric_real(0.8, 2.86, 2.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("k", "h"),
    [pytest.param(15.0, 1.0, id="moderate-depth"), pytest.param(10.0, 10.0, id="deep")],
)
def test_fully_evanescent_quasistatic_determinant_is_exactly_zero(
    k: float,
    h: float,
) -> None:
    assert det_symmetric(k, 0.0, c_l=2.0, c_t=1.0, h=h) == 0.0j


@pytest.mark.parametrize(
    ("k", "omega", "h"),
    [
        pytest.param(15.0, 1.0e-3, 1.0, id="low-frequency"),
        pytest.param(10.0, 0.1, 10.0, id="deep-evanescence"),
    ],
)
def test_fully_evanescent_determinant_matches_high_precision_reference(
    k: float,
    omega: float,
    h: float,
) -> None:
    expected = _high_precision_determinant(k, omega, 2.0, 1.0, h)

    assert det_symmetric(k, omega, 2.0, 1.0, h) == pytest.approx(
        expected,
        rel=5.0e-13,
        abs=1.0e-12,
    )


@pytest.mark.parametrize(
    ("k", "omega", "h"),
    [
        pytest.param(3.0, 0.5, 1.0, id="fully-evanescent"),
        pytest.param(2.0, 2.5, 1.0, id="mixed"),
        pytest.param(0.8, 2.86, 1.0, id="fully-propagating"),
    ],
)
def test_symmetric_determinant_is_even_in_signed_k_and_omega(
    k: float,
    omega: float,
    h: float,
) -> None:
    expected = det_symmetric(k, omega, 2.0, 1.0, h)

    assert det_symmetric(-k, omega, 2.0, 1.0, h) == pytest.approx(
        expected,
        rel=2.0e-14,
        abs=2.0e-14,
    )
    assert det_symmetric(k, -omega, 2.0, 1.0, h) == pytest.approx(
        expected,
        rel=2.0e-14,
        abs=2.0e-14,
    )


@pytest.mark.parametrize(
    "cutoff",
    [pytest.param(1.0, id="s-cutoff"), pytest.param(2.0, id="p-cutoff")],
)
def test_symmetric_determinant_has_two_sided_cutoff_limit(cutoff: float) -> None:
    step = 1.0e-8
    frequencies = (cutoff - step, cutoff, cutoff + step)
    values = [det_symmetric(1.0, omega, 2.0, 1.0, 1.0) for omega in frequencies]
    references = [_high_precision_determinant(1.0, omega, 2.0, 1.0, 1.0) for omega in frequencies]

    for value, reference in zip(values, references, strict=True):
        assert value == pytest.approx(reference, rel=2.0e-13, abs=2.0e-13)
    assert abs(values[0] - values[1]) < 1.0e-6
    assert abs(values[2] - values[1]) < 1.0e-6


@pytest.mark.parametrize(
    ("k", "omega", "h"),
    [
        pytest.param(15.0, 1.0e-3, 1.0, id="fully-evanescent"),
        pytest.param(2.0, 2.5, 1.0, id="mixed"),
        pytest.param(0.8, 2.86, 1.0, id="fully-propagating"),
        pytest.param(1.0, 1.0, 1.0, id="s-cutoff"),
        pytest.param(1.0, 2.0, 1.0, id="p-cutoff"),
    ],
)
def test_physical_determinant_is_real_in_every_wavenumber_regime(
    k: float,
    omega: float,
    h: float,
) -> None:
    value = det_symmetric(k, omega, 2.0, 1.0, h)

    assert np.isfinite(value)
    assert abs(value.imag) <= 2.0e-14 * max(1.0, abs(value.real))
    assert det_symmetric_real(k, omega, 2.0, 1.0, h) == pytest.approx(value.real)


@pytest.mark.parametrize(
    "direction",
    [
        pytest.param(1.0 + 0.0j, id="real"),
        pytest.param(1.0j, id="imaginary"),
        pytest.param((1.0 + 1.0j) / np.sqrt(2.0), id="complex"),
    ],
)
@pytest.mark.parametrize(
    "multiplier",
    [pytest.param(1.0 - 1.0e-8, id="below"), pytest.param(1.0 + 1.0e-8, id="above")],
)
def test_sine_ratio_is_accurate_on_both_sides_of_taylor_switch(
    direction: complex,
    multiplier: float,
) -> None:
    h = 1.7
    wavenumber = direction * (1.0e-4 * multiplier / h)
    expected = _high_precision_sine_ratio(wavenumber, h)

    assert abs(wavenumber * h) == pytest.approx(1.0e-4 * multiplier, rel=2.0e-15)
    assert sin_wavenumber_over_wavenumber(wavenumber, h) == pytest.approx(
        expected,
        rel=2.0e-15,
        abs=2.0e-15,
    )


@pytest.mark.parametrize(
    ("k", "omega"),
    [
        pytest.param(0.8042173193715181, 2.8517587749600901, id="zgv"),
        pytest.param(0.8043173193715181, 2.8517587749600901, id="radial-offset"),
        pytest.param(0.8041173193715181, 2.8519587749600902, id="mixed-offset"),
    ],
)
def test_zgv_neighborhood_determinant_matches_high_precision_reference(
    k: float,
    omega: float,
) -> None:
    expected = _high_precision_determinant(k, omega, 2.0, 1.0, 1.0)

    assert det_symmetric(k, omega, 2.0, 1.0, 1.0) == pytest.approx(
        expected,
        rel=3.0e-13,
        abs=7.0e-15,
    )


@pytest.mark.parametrize(
    "omega",
    [
        pytest.param(1.0e-160, id="subnormal-squares"),
        pytest.param(1.0e-162, id="zero-squares"),
        pytest.param(1.0e-200, id="deep-underflow"),
        pytest.param(np.nextafter(0.0, 1.0), id="ratio-to-zero"),
    ],
)
def test_underresolved_evanescent_frequency_scale_raises_instead_of_false_root(
    omega: float,
) -> None:
    high_precision_value = _high_precision_determinant(
        50.0,
        omega,
        2.0,
        1.0,
        10.0,
        decimal_precision=700,
    )
    assert high_precision_value != 0.0j

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FloatingPointError, match="frequency scale"):
            det_symmetric(50.0, omega, 2.0, 1.0, 10.0)
