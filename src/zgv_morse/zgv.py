"""Locate the first symmetric zero-group-velocity point of an isotropic plate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import mpmath as mp
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .config import ReferenceConfig
from .rayleigh_lamb import det_symmetric_mp, det_symmetric_real


@dataclass(frozen=True, slots=True)
class ZGVPoint:
    """High-precision location and radial curvature of a symmetric ZGV point."""

    kappa0: float
    omega0: float
    curvature_a: float
    det_residual: float
    group_velocity: float
    branch_index: int = 1


def _validated_config(cfg: object) -> ReferenceConfig:
    if not isinstance(cfg, ReferenceConfig):
        raise TypeError("cfg must be a ReferenceConfig instance")
    cfg.validate()
    return cfg


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _append_distinct_root(roots: list[float], root: float) -> None:
    if not roots or abs(root - roots[-1]) > 1.0e-7:
        roots.append(root)


def _roots_in_window(k: float, cfg: ReferenceConfig, samples: int = 500) -> list[float]:
    """Return sorted, deduplicated symmetric roots in the configured frequency window."""

    config = _validated_config(cfg)
    k_value = _finite_float("k", k)
    if type(samples) is not int:
        raise TypeError("samples must be an integer")
    if samples < 2:
        raise ValueError("samples must be at least 2")

    lower, upper = config.zgv_search_omega
    grid = np.linspace(lower, upper, samples)
    values = np.asarray(
        [det_symmetric_real(k_value, omega, config.c_l, config.c_t, config.h) for omega in grid],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("symmetric-root scan produced a non-finite determinant")

    roots: list[float] = []
    for left, right, f_left, f_right in zip(
        grid[:-1],
        grid[1:],
        values[:-1],
        values[1:],
        strict=True,
    ):
        if f_left == 0.0:
            _append_distinct_root(roots, float(left))
        if f_left != 0.0 and f_right != 0.0 and np.signbit(f_left) != np.signbit(f_right):
            root = brentq(
                lambda omega: det_symmetric_real(
                    k_value,
                    omega,
                    config.c_l,
                    config.c_t,
                    config.h,
                ),
                float(left),
                float(right),
                xtol=1.0e-13,
                rtol=8.0 * np.finfo(np.float64).eps,
            )
            _append_distinct_root(roots, float(root))

    if values[-1] == 0.0:
        _append_distinct_root(roots, float(grid[-1]))
    if not roots:
        raise RuntimeError(f"no symmetric root in configured window at k={k_value}")
    return roots


def _target_branch_frequency(k: float, cfg: ReferenceConfig) -> float:
    """Return the lowest symmetric root in the configured frequency window."""

    return _roots_in_window(k, cfg)[0]


def _real_mp_value(name: str, value: object) -> mp.mpf:
    if not mp.isfinite(value):
        raise RuntimeError(f"{name} is non-finite")
    real = mp.re(value)
    imaginary = mp.im(value)
    scale = max(mp.mpf(1), abs(real))
    if abs(imaginary) > 100 * mp.eps * scale:
        raise RuntimeError(f"{name} is not real")
    return real


def find_s1_zgv(cfg: ReferenceConfig, dps: int = 60) -> ZGVPoint:
    """Find and refine the lowest symmetric-branch ZGV point in ``cfg``."""

    config = _validated_config(cfg)
    if type(dps) is not int:
        raise TypeError("dps must be an integer")
    if dps < 50:
        raise ValueError("dps must be at least 50")

    scan = minimize_scalar(
        lambda kappa: _target_branch_frequency(kappa, config),
        bounds=config.zgv_search_kappa,
        method="bounded",
        options={"xatol": 1.0e-9},
    )
    if not scan.success:
        raise RuntimeError(f"bounded ZGV scan failed: {scan.message}")
    if not math.isfinite(float(scan.x)) or not math.isfinite(float(scan.fun)):
        raise RuntimeError("bounded ZGV scan produced a non-finite seed")

    with mp.workdps(max(60, dps)):
        c_l = mp.mpf(str(config.c_l))
        c_t = mp.mpf(str(config.c_t))
        thickness = mp.mpf(str(config.h))

        def determinant(kappa: mp.mpf, omega: mp.mpf) -> object:
            return det_symmetric_mp(kappa, omega, c_l, c_t, thickness)

        def determinant_k(kappa: mp.mpf, omega: mp.mpf) -> object:
            return mp.diff(lambda argument: determinant(argument, omega), kappa)

        try:
            kappa0_raw, omega0_raw = mp.findroot(
                (determinant, determinant_k),
                (mp.mpf(str(scan.x)), mp.mpf(str(scan.fun))),
            )
        except (ArithmeticError, ValueError) as error:
            raise RuntimeError("high-precision ZGV refinement failed") from error

        kappa0 = _real_mp_value("refined kappa", kappa0_raw)
        omega0 = _real_mp_value("refined omega", omega0_raw)
        lower_kappa, upper_kappa = map(mp.mpf, map(str, config.zgv_search_kappa))
        lower_omega, upper_omega = map(mp.mpf, map(str, config.zgv_search_omega))
        if not lower_kappa < kappa0 < upper_kappa:
            raise RuntimeError("refined ZGV wavenumber is outside the configured interior")
        if not lower_omega <= omega0 <= upper_omega:
            raise RuntimeError("refined ZGV frequency is outside the configured window")

        determinant_value = determinant(kappa0, omega0)
        d_k = _real_mp_value("D_k", determinant_k(kappa0, omega0))
        d_omega = _real_mp_value(
            "D_omega",
            mp.diff(lambda argument: determinant(kappa0, argument), omega0),
        )
        d_kk = _real_mp_value(
            "D_kk",
            mp.diff(lambda argument: determinant(argument, omega0), kappa0, 2),
        )
        if d_omega == 0:
            raise RuntimeError("D_omega vanishes at the refined ZGV point")

        curvature = _real_mp_value("radial curvature", -d_kk / d_omega)
        group_velocity = _real_mp_value("group velocity", -d_k / d_omega)
        determinant_residual = abs(determinant_value)
        if not mp.isfinite(determinant_residual):
            raise RuntimeError("determinant residual is non-finite")
        if determinant_residual > mp.mpf(str(config.eigen_residual_tolerance)):
            raise RuntimeError("refined ZGV determinant residual exceeds configured tolerance")
        if curvature <= 0:
            raise RuntimeError("refined stationary point is not a local minimum")

        kappa0_float = float(kappa0)
        omega0_float = float(omega0)
        curvature_float = float(curvature)
        residual_float = float(determinant_residual)
        group_velocity_float = float(group_velocity)

    roots = _roots_in_window(kappa0_float, config)
    branch_tolerance = max(5.0e-10, config.isotropic_match_tolerance)
    if abs(roots[0] - omega0_float) > branch_tolerance:
        raise RuntimeError("refined point is not on the lowest symmetric root in the window")

    return ZGVPoint(
        kappa0=kappa0_float,
        omega0=omega0_float,
        curvature_a=curvature_float,
        det_residual=residual_float,
        group_velocity=group_velocity_float,
    )
