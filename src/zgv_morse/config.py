"""Typed loading and validation for the reference computation parameters."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
import re

import yaml


class _ReferenceLoader(yaml.SafeLoader):
    """Safe YAML loader with support for plain scientific notation."""


_SCIENTIFIC_FLOAT = re.compile(r"^[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+$")
_ReferenceLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    _SCIENTIFIC_FLOAT,
    list("-+0123456789"),
)


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Immutable parameters for the reference ZGV computation."""

    schema_version: int
    h: float
    rho: float
    lam: float
    mu: float
    delta: float
    epsilon_values: tuple[float, ...]
    source_radius_over_h: float
    window_sigma_over_k0: float
    window_sensitivity: tuple[float, float]
    annulus_fraction: float
    zgv_search_kappa: tuple[float, float]
    zgv_search_omega: tuple[float, float]
    eigen_residual_tolerance: float
    isotropic_match_tolerance: float
    curvature_match_tolerance: float
    sensitivity_match_tolerance: float
    phase_error_tolerance: float

    @property
    def total_thickness(self) -> float:
        """Return the full plate thickness."""

        return 2.0 * self.h

    @property
    def c_t(self) -> float:
        """Return the isotropic transverse wave speed."""

        return sqrt(self.mu / self.rho)

    @property
    def c_l(self) -> float:
        """Return the isotropic longitudinal wave speed."""

        return sqrt((self.lam + 2.0 * self.mu) / self.rho)

    @property
    def poisson_ratio(self) -> float:
        """Return the isotropic Poisson ratio."""

        return self.lam / (2.0 * (self.lam + self.mu))

    def validate(self) -> None:
        """Raise ``ValueError`` when a required reference invariant is violated."""

        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be exactly 1")

        scalar_values = (
            ("h", self.h),
            ("rho", self.rho),
            ("lambda", self.lam),
            ("mu", self.mu),
            ("delta", self.delta),
            ("source_radius_over_h", self.source_radius_over_h),
            ("window_sigma_over_k0", self.window_sigma_over_k0),
            ("annulus_fraction", self.annulus_fraction),
            ("eigen_residual_tolerance", self.eigen_residual_tolerance),
            ("isotropic_match_tolerance", self.isotropic_match_tolerance),
            ("curvature_match_tolerance", self.curvature_match_tolerance),
            ("sensitivity_match_tolerance", self.sensitivity_match_tolerance),
            ("phase_error_tolerance", self.phase_error_tolerance),
        )
        for name, value in scalar_values:
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        positive_values = (
            ("h", self.h),
            ("rho", self.rho),
            ("mu", self.mu),
            ("delta", self.delta),
            ("source_radius_over_h", self.source_radius_over_h),
            ("window_sigma_over_k0", self.window_sigma_over_k0),
            ("annulus_fraction", self.annulus_fraction),
            ("eigen_residual_tolerance", self.eigen_residual_tolerance),
            ("isotropic_match_tolerance", self.isotropic_match_tolerance),
            ("curvature_match_tolerance", self.curvature_match_tolerance),
            ("sensitivity_match_tolerance", self.sensitivity_match_tolerance),
            ("phase_error_tolerance", self.phase_error_tolerance),
        )
        for name, value in positive_values:
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        longitudinal_modulus = self.lam + 2.0 * self.mu
        if not isfinite(longitudinal_modulus) or longitudinal_modulus <= 0.0:
            raise ValueError("lambda + 2 * mu must be positive")

        if not self.epsilon_values:
            raise ValueError("epsilon_values must not be empty")
        if not all(isfinite(epsilon) for epsilon in self.epsilon_values):
            raise ValueError("epsilon_values must contain only finite values")
        if any(epsilon == 0.0 for epsilon in self.epsilon_values):
            raise ValueError("epsilon_values must exclude zero")
        if not all(
            left < right
            for left, right in zip(self.epsilon_values, self.epsilon_values[1:], strict=False)
        ):
            raise ValueError("epsilon_values must be strictly ordered")

        if len(self.window_sensitivity) != 2:
            raise ValueError("window_sensitivity must contain exactly two values")
        if not all(isfinite(value) for value in self.window_sensitivity):
            raise ValueError("window_sensitivity must contain only finite values")
        if not all(value > 0.0 for value in self.window_sensitivity):
            raise ValueError("window_sensitivity values must be positive")

        for field, bounds in (
            ("zgv_search_kappa", self.zgv_search_kappa),
            ("zgv_search_omega", self.zgv_search_omega),
        ):
            if len(bounds) != 2:
                raise ValueError(f"{field} must contain exactly two values")
            if not all(isfinite(value) for value in bounds):
                raise ValueError(f"{field} must contain only finite values")
            if bounds[0] >= bounds[1]:
                raise ValueError(f"{field} lower bound must be less than upper bound")

        if not 0.0 < self.window_sigma_over_k0 < self.annulus_fraction + 0.01:
            raise ValueError(
                "window_sigma_over_k0 must be positive and less than annulus_fraction + 0.01"
            )


def _required_value(data: Mapping[object, object], field: str) -> object:
    """Return a required field with a field-specific error."""

    if field not in data:
        raise ValueError(f"{field} is required")
    return data[field]


def _float_value(value: object, *, field: str) -> float:
    """Convert a real YAML number to ``float`` without string or bool coercion."""

    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _float_field(data: Mapping[object, object], field: str) -> float:
    """Return a required numeric scalar as ``float``."""

    return _float_value(_required_value(data, field), field=field)


def _float_sequence(data: Mapping[object, object], field: str) -> tuple[float, ...]:
    """Convert a required YAML list to an immutable float tuple."""

    values = _required_value(data, field)
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a YAML list")
    return tuple(
        _float_value(value, field=f"{field}[{index}]") for index, value in enumerate(values)
    )


def _float_pair(data: Mapping[object, object], field: str) -> tuple[float, float]:
    """Convert a required two-value YAML list to an immutable float pair."""

    values = _float_sequence(data, field)
    if len(values) != 2:
        raise ValueError(f"{field} must contain exactly two values")
    return values[0], values[1]


def _schema_version(data: Mapping[object, object]) -> int:
    """Return schema version 1 without accepting bool or float equivalents."""

    value = _required_value(data, "schema_version")
    if type(value) is not int:
        raise ValueError("schema_version must be an integer")
    if value != 1:
        raise ValueError("schema_version must be exactly 1")
    return value


def load_reference_config(path: str | Path) -> ReferenceConfig:
    """Load, type, validate, and return a reference YAML configuration."""

    raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_ReferenceLoader)
    if not isinstance(raw, Mapping):
        raise ValueError("reference configuration must be a YAML mapping")

    config = ReferenceConfig(
        schema_version=_schema_version(raw),
        h=_float_field(raw, "h"),
        rho=_float_field(raw, "rho"),
        lam=_float_field(raw, "lambda"),
        mu=_float_field(raw, "mu"),
        delta=_float_field(raw, "delta"),
        epsilon_values=_float_sequence(raw, "epsilon_values"),
        source_radius_over_h=_float_field(raw, "source_radius_over_h"),
        window_sigma_over_k0=_float_field(raw, "window_sigma_over_k0"),
        window_sensitivity=_float_pair(raw, "window_sensitivity"),
        annulus_fraction=_float_field(raw, "annulus_fraction"),
        zgv_search_kappa=_float_pair(raw, "zgv_search_kappa"),
        zgv_search_omega=_float_pair(raw, "zgv_search_omega"),
        eigen_residual_tolerance=_float_field(raw, "eigen_residual_tolerance"),
        isotropic_match_tolerance=_float_field(raw, "isotropic_match_tolerance"),
        curvature_match_tolerance=_float_field(raw, "curvature_match_tolerance"),
        sensitivity_match_tolerance=_float_field(raw, "sensitivity_match_tolerance"),
        phase_error_tolerance=_float_field(raw, "phase_error_tolerance"),
    )
    config.validate()
    return config
