"""Regular symmetric Rayleigh--Lamb determinant for an isotropic plate."""

from __future__ import annotations

import math
from numbers import Complex, Real

import numpy as np
from numpy.typing import NDArray


def _finite_real_scalar(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _positive_real_scalar(name: str, value: object) -> float:
    scalar = _finite_real_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _finite_complex_scalar(name: str, value: object) -> np.complex128:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Complex):
        raise ValueError(f"{name} must be a finite complex scalar")
    try:
        scalar = np.complex128(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite complex scalar") from error
    if not math.isfinite(float(scalar.real)) or not math.isfinite(float(scalar.imag)):
        raise ValueError(f"{name} must be a finite complex scalar")
    return scalar


def _finite_complex_result(name: str, value: object) -> np.complex128:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Complex):
        raise FloatingPointError(f"{name} produced a non-finite result")
    try:
        result = np.complex128(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FloatingPointError(f"{name} produced a non-finite result") from error
    if not math.isfinite(float(result.real)) or not math.isfinite(float(result.imag)):
        raise FloatingPointError(f"{name} produced a non-finite result")
    return result


def _wave_parameters(
    k: object,
    omega: object,
    c_l: object,
    c_t: object,
) -> tuple[float, float, float, float]:
    return (
        _finite_real_scalar("k", k),
        _finite_real_scalar("omega", omega),
        _positive_real_scalar("c_l", c_l),
        _positive_real_scalar("c_t", c_t),
    )


def _principal_wavenumber(k: float, omega: float, speed: float) -> np.complex128:
    with np.errstate(all="ignore"):
        k_value = np.float64(k)
        ratio = np.float64(omega) / np.float64(speed)
        radicand = ratio * ratio - k_value * k_value
    if not np.isfinite(ratio) or not np.isfinite(radicand):
        raise FloatingPointError("thickness wavenumber calculation produced a non-finite result")
    return _finite_complex_result(
        "thickness wavenumber",
        np.sqrt(np.complex128(radicand)),
    )


def _thickness_wavenumbers_from_values(
    k: float,
    omega: float,
    c_l: float,
    c_t: float,
) -> tuple[np.complex128, np.complex128]:
    return (
        _principal_wavenumber(k, omega, c_l),
        _principal_wavenumber(k, omega, c_t),
    )


def thickness_wavenumbers(
    k: float,
    omega: float,
    c_l: float,
    c_t: float,
) -> tuple[np.complex128, np.complex128]:
    """Return the principal longitudinal and shear thickness wavenumbers."""

    values = _wave_parameters(k, omega, c_l, c_t)
    return _thickness_wavenumbers_from_values(*values)


def _sin_wavenumber_over_wavenumber(
    wavenumber: np.complex128,
    h: float,
) -> np.complex128:
    with np.errstate(all="ignore"):
        z = wavenumber * np.float64(h)
    if not np.isfinite(z):
        raise FloatingPointError("sine ratio produced a non-finite result")

    with np.errstate(all="ignore"):
        if abs(z) < 1.0e-4:
            z_squared = z * z
            series = 1.0 + z_squared * (-1.0 / 6.0 + z_squared * (1.0 / 120.0 - z_squared / 5040.0))
            result = np.float64(h) * series
        else:
            result = np.sin(z) / wavenumber
    return _finite_complex_result("sine ratio", result)


def sin_wavenumber_over_wavenumber(
    wavenumber: complex,
    h: float,
) -> np.complex128:
    """Return ``sin(wavenumber*h)/wavenumber``, continued analytically at zero."""

    value = _finite_complex_scalar("wavenumber", wavenumber)
    thickness = _positive_real_scalar("h", h)
    return _sin_wavenumber_over_wavenumber(value, thickness)


def traction_matrix_symmetric(
    k: float,
    omega: float,
    c_l: float,
    c_t: float,
    h: float,
) -> NDArray[np.complex128]:
    """Return the regular two-by-two symmetric-mode traction matrix.

    The second amplitude column is rescaled by ``1/s`` relative to the
    physical-amplitude traction matrix.  This changes that column's units and
    normalization, but makes its ``sin(s*h)/s`` entry entire at ``s=0``.  The
    matrix determinant is therefore the corresponding desingularized
    Rayleigh--Lamb determinant.
    """

    k_value, omega_value, c_l_value, c_t_value = _wave_parameters(k, omega, c_l, c_t)
    thickness = _positive_real_scalar("h", h)
    p, s = _thickness_wavenumbers_from_values(k_value, omega_value, c_l_value, c_t_value)
    sin_s_over_s = _sin_wavenumber_over_wavenumber(s, thickness)

    with np.errstate(all="ignore"):
        k_scalar = np.float64(k_value)
        k_squared = k_scalar * k_scalar
        shear_factor = s * s - k_squared
        p_argument = p * np.float64(thickness)
        s_argument = s * np.float64(thickness)
        matrix = np.array(
            [
                [
                    -2.0j * k_scalar * p * np.sin(p_argument),
                    shear_factor * sin_s_over_s,
                ],
                [
                    -shear_factor * np.cos(p_argument),
                    2.0j * k_scalar * np.cos(s_argument),
                ],
            ],
            dtype=np.complex128,
        )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("traction matrix produced a non-finite result")
    return matrix


def _det_symmetric_fully_evanescent(
    k: float,
    omega: float,
    c_l: float,
    c_t: float,
    h: float,
) -> np.complex128:
    """Evaluate the determinant after combining cancelling hyperbolic terms.

    Nonzero frequency ratios and their squares must remain normal ``float64``
    values.  Subnormal or underflowed scales are rejected because the
    compensated coefficients can otherwise collapse to a false zero.
    """

    with np.errstate(all="ignore"):
        k_scalar = np.float64(k)
        omega_scalar = np.float64(omega)
        k_squared = k_scalar * k_scalar
        longitudinal_ratio = omega_scalar / np.float64(c_l)
        shear_ratio = omega_scalar / np.float64(c_t)
        longitudinal_squared = longitudinal_ratio * longitudinal_ratio
        shear_squared = shear_ratio * shear_ratio

        normal_minimum = np.finfo(np.float64).tiny
        frequency_scales = (
            ("longitudinal", longitudinal_ratio, longitudinal_squared),
            ("shear", shear_ratio, shear_squared),
        )
        for name, ratio, squared_ratio in frequency_scales:
            if (
                ratio == 0.0
                or abs(ratio) < normal_minimum
                or squared_ratio == 0.0
                or squared_ratio < normal_minimum
            ):
                raise FloatingPointError(f"{name} frequency scale is underresolved in float64")

        alpha = np.sqrt(max(0.0, k_squared - longitudinal_squared))
        beta = np.sqrt(max(0.0, k_squared - shear_squared))
        alpha_plus_beta = alpha + beta

        if alpha_plus_beta == 0.0:
            value = k_squared * k_squared * np.float64(h)
        else:
            sine_beta_over_beta = _sin_wavenumber_over_wavenumber(
                np.complex128(1.0j * beta),
                h,
            ).real
            regular_coefficient = (
                shear_squared * shear_squared
                + 4.0 * k_squared * beta * (longitudinal_squared - shear_squared) / alpha_plus_beta
            )
            difference_argument = (
                np.float64(h) * (shear_squared - longitudinal_squared) / alpha_plus_beta
            )
            value = regular_coefficient * sine_beta_over_beta * np.cosh(
                alpha * np.float64(h)
            ) - 4.0 * k_squared * alpha * np.sinh(difference_argument)
    return _finite_complex_result("symmetric determinant", np.complex128(value))


def det_symmetric(
    k: float,
    omega: float,
    c_l: float,
    c_t: float,
    h: float,
) -> np.complex128:
    """Return the desingularized symmetric Rayleigh--Lamb determinant."""

    k_value, omega_value, c_l_value, c_t_value = _wave_parameters(k, omega, c_l, c_t)
    thickness = _positive_real_scalar("h", h)
    if omega_value == 0.0:
        return np.complex128(0.0)

    p, s = _thickness_wavenumbers_from_values(k_value, omega_value, c_l_value, c_t_value)
    if p.real == 0.0 and s.real == 0.0:
        return _det_symmetric_fully_evanescent(
            k_value,
            omega_value,
            c_l_value,
            c_t_value,
            thickness,
        )

    sin_s_over_s = _sin_wavenumber_over_wavenumber(s, thickness)

    with np.errstate(all="ignore"):
        k_scalar = np.float64(k_value)
        k_squared = k_scalar * k_scalar
        shear_factor = s * s - k_squared
        p_argument = p * np.float64(thickness)
        s_argument = s * np.float64(thickness)
        value = shear_factor * shear_factor * sin_s_over_s * np.cos(
            p_argument
        ) + 4.0 * k_squared * p * np.cos(s_argument) * np.sin(p_argument)
    return _finite_complex_result("symmetric determinant", value)


def det_symmetric_real(
    k: float,
    omega: float,
    c_l: float,
    c_t: float,
    h: float,
    imag_tolerance: float = 1.0e-10,
) -> float:
    """Return the determinant's real part after guarding imaginary leakage."""

    tolerance = _positive_real_scalar("imag_tolerance", imag_tolerance)
    value = _finite_complex_result(
        "symmetric determinant",
        det_symmetric(k, omega, c_l, c_t, h),
    )
    scale = max(1.0, abs(float(value.real)))
    if abs(float(value.imag)) > tolerance * scale:
        raise FloatingPointError(f"physical determinant has imaginary leakage {float(value.imag)}")
    return float(value.real)


def det_symmetric_mp(
    k: object,
    omega: object,
    c_l: object,
    c_t: object,
    h: object,
) -> object:
    """Evaluate the desingularized symmetric determinant at caller precision.

    The function deliberately uses the active :mod:`mpmath` context without
    changing it or converting through binary ``float``.  This lets callers
    differentiate the determinant and refine roots at arbitrary precision.
    """

    import mpmath as mp

    def finite_real(name: str, value: object) -> mp.mpf:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real scalar")
        try:
            scalar = mp.mpf(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a finite real scalar") from error
        if not mp.isfinite(scalar):
            raise ValueError(f"{name} must be a finite real scalar")
        return scalar

    k_value = finite_real("k", k)
    omega_value = finite_real("omega", omega)
    c_l_value = finite_real("c_l", c_l)
    c_t_value = finite_real("c_t", c_t)
    h_value = finite_real("h", h)
    for name, value in (("c_l", c_l_value), ("c_t", c_t_value), ("h", h_value)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    p = mp.sqrt((omega_value / c_l_value) ** 2 - k_value**2)
    s = mp.sqrt((omega_value / c_t_value) ** 2 - k_value**2)
    sin_s_over_s = h_value if s == 0 else mp.sin(s * h_value) / s
    shear_factor = s**2 - k_value**2
    result = shear_factor**2 * sin_s_over_s * mp.cos(p * h_value) + 4 * k_value**2 * p * mp.cos(
        s * h_value
    ) * mp.sin(p * h_value)
    if not mp.isfinite(result):
        raise FloatingPointError("symmetric determinant produced a non-finite result")
    return result
