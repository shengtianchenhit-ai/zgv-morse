from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import j0

from zgv_morse.validation import (
    BesselCollapseError,
    PowerLawFit,
    bessel_collapse_error,
    fit_power_law,
    operational_crossover_time,
    rms_envelope,
)


def test_registered_power_law_and_crossover_metrics() -> None:
    time = np.geomspace(10.0, 1.0e4, 1000)
    envelope = 3.0 * time**-0.5
    fit = fit_power_law(time, envelope, np.ones(time.shape, dtype=np.bool_))

    assert_allclose(fit.slope, -0.5, atol=1.0e-12)
    assert_allclose(fit.intercept, np.log(3.0), atol=1.0e-12)
    assert fit.sample_count == time.size
    assert fit.time_start == time[0]
    assert fit.time_stop == time[-1]
    crossover = operational_crossover_time(0.02, 0.5)
    rate = abs(0.02 * 0.5)
    assert crossover * rate > 0.0
    assert j0(crossover * rate) == pytest.approx(0.9, abs=2.0e-14)
    assert np.all(np.isfinite(rms_envelope(envelope.astype(complex), 31)))


def test_power_law_fit_is_frozen_slotted_and_uses_selected_window() -> None:
    time = np.geomspace(1.0, 100.0, 30)
    envelope = 2.5 * time**-1.25
    mask = np.zeros(time.shape, dtype=np.bool_)
    mask[5:25] = True

    fit = fit_power_law(time, envelope, mask)

    assert isinstance(fit, PowerLawFit)
    assert not hasattr(fit, "__dict__")
    assert fit.sample_count == 20
    assert fit.time_start == time[5]
    assert fit.time_stop == time[24]
    assert fit.slope == pytest.approx(-1.25, abs=2.0e-14)
    assert fit.intercept == pytest.approx(np.log(2.5), abs=2.0e-14)
    with pytest.raises(FrozenInstanceError):
        fit.slope = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("time", "envelope", "mask", "match"),
    [
        (np.arange(12.0) + 1.0, np.ones(11), np.ones(12, dtype=bool), "shape"),
        (np.ones((3, 4)), np.ones(12), np.ones(12, dtype=bool), "one-dimensional"),
        (
            np.arange(12.0) + 1.0,
            np.full(12, np.nan),
            np.ones(12, dtype=bool),
            "finite",
        ),
        (np.arange(12.0) + 1.0, np.ones(12), np.ones(12, dtype=int), "boolean"),
        (
            np.arange(12.0) + 1.0,
            np.ones(12),
            np.array([True] * 9 + [False] * 3),
            "at least ten",
        ),
        (
            np.arange(12.0),
            np.ones(12),
            np.ones(12, dtype=bool),
            "selected time",
        ),
        (
            np.arange(12.0) + 1.0,
            np.array([0.0] + [1.0] * 11),
            np.ones(12, dtype=bool),
            "selected envelope",
        ),
        (
            np.array([1.0, 3.0, 2.0, *np.arange(4.0, 13.0)]),
            np.ones(12),
            np.ones(12, dtype=bool),
            "strictly increasing",
        ),
    ],
)
def test_power_law_fit_rejects_malformed_or_unregistered_samples(
    time: object,
    envelope: object,
    mask: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        fit_power_law(time, envelope, mask)


def test_rms_envelope_is_overflow_resistant_for_large_finite_signal() -> None:
    phase = np.linspace(0.0, 4.0 * np.pi, 101)
    signal = 1.0e300 * np.exp(1j * phase)

    envelope = rms_envelope(signal, 31)

    assert envelope.shape == signal.shape
    assert np.isfinite(envelope).all()
    assert_allclose(envelope, 1.0e300, rtol=2.0e-15)
    assert not np.shares_memory(envelope, signal)
    assert_allclose(rms_envelope(np.zeros(11, dtype=complex), 3), 0.0)


@pytest.mark.parametrize(
    ("signal", "window", "match"),
    [
        (np.ones(9), 2, "at least three"),
        (np.ones(9), 4, "odd"),
        (np.ones(9), True, "built-in integer"),
        (np.ones(9), np.int64(3), "built-in integer"),
        (np.ones(3), 5, "signal length"),
        (np.ones((3, 3)), 3, "one-dimensional"),
        (np.array([1.0, np.nan, 2.0]), 3, "finite"),
    ],
)
def test_rms_envelope_rejects_invalid_signal_or_window(
    signal: object,
    window: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        rms_envelope(signal, window)  # type: ignore[arg-type]


def test_operational_crossover_is_sign_invariant_and_strictly_finite() -> None:
    positive = operational_crossover_time(0.02, 0.5)
    assert operational_crossover_time(-0.02, 0.5) == pytest.approx(positive)
    assert operational_crossover_time(0.02, -0.5) == pytest.approx(positive)

    invalid = (
        (0.0, 1.0),
        (1.0, 0.0),
        (True, 1.0),
        (np.inf, 1.0),
        (1.0e308, 1.0e308),
        (np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)),
        (np.nextafter(0.0, 1.0), 1.0),
    )
    for epsilon, coefficient in invalid:
        with pytest.raises((TypeError, ValueError)):
            operational_crossover_time(epsilon, coefficient)


def test_bessel_collapse_error_is_typed_and_overflow_resistant() -> None:
    target = np.array([1.0e300, -1.0e300, 0.01, 0.0])
    scaled = np.array([0.0, 0.0, 0.02, 0.0], dtype=np.complex128)

    error = bessel_collapse_error(scaled, target, zero_threshold=0.05)

    assert isinstance(error, BesselCollapseError)
    assert not hasattr(error, "__dict__")
    assert error.max_absolute == pytest.approx(1.0e300)
    assert error.rms_absolute == pytest.approx(np.sqrt(0.5) * 1.0e300)
    assert error.max_relative_away_from_zeros == pytest.approx(1.0)
    assert error.away_zero_sample_count == 2
    with pytest.raises(FrozenInstanceError):
        error.max_absolute = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("scaled", "target", "threshold", "match"),
    [
        (np.ones(4), np.ones(3), 0.05, "shape"),
        (np.ones((2, 2)), np.ones((2, 2)), 0.05, "one-dimensional"),
        (np.array([1.0, np.nan]), np.ones(2), 0.05, "finite"),
        (np.ones(3), np.ones(3), 0.0, "positive"),
        (np.ones(3), np.ones(3), True, "real scalar"),
        (np.zeros(3), np.zeros(3), 0.05, "away from zeros"),
    ],
)
def test_bessel_collapse_error_rejects_invalid_inputs(
    scaled: object,
    target: object,
    threshold: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        bessel_collapse_error(scaled, target, threshold)  # type: ignore[arg-type]

