"""Main Figure 5: perturbative scaling and compensated error laws."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
from numpy.typing import NDArray

from .common import (
    FigureSpec,
    MARKERS,
    PALETTE,
    apply_publication_style,
    load_figure_artifact,
    save_publication_figure,
    write_source_csv,
)


_SPEC: Final = FigureSpec(
    "5",
    "the computed unfolding is genuinely perturbative and quantitatively predicted",
    "quantitative grid",
    183.0,
    142.0,
)
_SPLITTING_SLOPE_GATE: Final = 0.1
_RADIAL_SLOPE_GATE: Final = 0.1
_REMAINDER_SLOPE_GATE: Final = 0.2
_RADIAL_LIMIT_RELATIVE_GATE: Final = 0.01
_ROUNDTRIP_RTOL: Final = 2.0e-13
_ROUNDTRIP_ATOL: Final = 2.0e-15


@dataclass(frozen=True, slots=True)
class _ScalingEvidence:
    epsilon: NDArray[np.float64]
    q_min_limit_prediction: NDArray[np.float64]
    q_saddle_limit_prediction: NDArray[np.float64]
    minimum_error_over_epsilon_squared: NDArray[np.float64]
    saddle_error_over_epsilon_squared: NDArray[np.float64]
    negative_minimum_theta: NDArray[np.float64]
    negative_saddle_theta: NDArray[np.float64]
    positive_minimum_theta: NDArray[np.float64]
    positive_saddle_theta: NDArray[np.float64]
    slope_splitting: float
    slope_radial: float
    slope_remainder: float


def _scalar(arrays: dict[str, NDArray[np.generic]], name: str) -> float:
    value = float(arrays[name])
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _aligned_finite_array(
    arrays: dict[str, NDArray[np.generic]],
    name: str,
    size: int,
) -> NDArray[np.float64]:
    value = np.asarray(arrays[name], dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite array aligned with epsilon")
    return value


def _registered_tolerance(metadata: dict[str, object], name: str) -> float:
    tolerances = metadata.get("tolerances")
    if not isinstance(tolerances, dict):
        raise ValueError("registered scaling tolerances are missing")
    value = tolerances.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"registered tolerance {name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"registered tolerance {name} must be finite and nonnegative")
    return result


def _check_compensated_identity(
    stored: NDArray[np.float64],
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
    message: str,
) -> None:
    np.testing.assert_allclose(
        stored,
        numerator / denominator,
        rtol=_ROUNDTRIP_RTOL,
        atol=_ROUNDTRIP_ATOL,
        err_msg=message,
    )


def _radial_limit_evidence(
    epsilon: NDArray[np.float64],
    full_q: NDArray[np.float64],
    predicted_q: NDArray[np.float64],
    stored_compensated_q: NDArray[np.float64],
    role: str,
) -> NDArray[np.float64]:
    _check_compensated_identity(
        stored_compensated_q,
        full_q,
        epsilon,
        f"stored compensated {role} radial shift must equal q/epsilon",
    )
    predicted_limit = predicted_q / epsilon
    reference = float(predicted_limit[0])
    numerical_scale = max(1.0, abs(reference))
    np.testing.assert_allclose(
        predicted_limit,
        np.full(epsilon.size, reference),
        rtol=_ROUNDTRIP_RTOL,
        atol=_ROUNDTRIP_ATOL * numerical_scale,
        err_msg=(
            f"stored radial prediction for {role} must encode one "
            "coefficient-level -B(theta_j)/a limit"
        ),
    )

    coefficient_uncertainty = max(
        float(np.max(np.abs(predicted_limit - reference))),
        64.0 * np.finfo(np.float64).eps * numerical_scale,
    )
    if abs(reference) <= coefficient_uncertainty:
        raise ValueError(
            f"the {role} linear radial coefficient is zero within uncertainty; "
            "a separately derived first nonzero order is required"
        )
    relative_error = abs(float(stored_compensated_q[0]) - reference) / abs(reference)
    if relative_error > _RADIAL_LIMIT_RELATIVE_GATE:
        raise ValueError(
            f"the {role} radial compensated limit does not agree with the stored "
            "-B(theta_j)/a prediction"
        )
    return predicted_limit


def _fourfold_orbit(value: NDArray[np.generic], name: str) -> NDArray[np.float64]:
    theta = np.mod(np.asarray(value, dtype=np.float64), 2.0 * np.pi)
    if theta.shape != (4,) or not np.isfinite(theta).all():
        raise ValueError(f"stored role reversal {name} angles must contain four values")
    ordered = np.sort(theta)
    gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
    np.testing.assert_allclose(
        gaps,
        np.full(4, 0.5 * np.pi),
        rtol=0.0,
        atol=2.0e-12,
        err_msg=f"stored role reversal {name} angles must form a fourfold orbit",
    )
    return ordered


def _validate_evidence(
    scaling: dict[str, NDArray[np.generic]],
    metadata: dict[str, object],
) -> _ScalingEvidence:
    """Fail before source export unless all pre-registered laws are intact."""

    if (
        metadata.get("artifact") != "perturbation_scaling"
        or metadata.get("stage") != "scaling"
        or metadata.get("profile") not in {"smoke", "full"}
    ):
        raise ValueError(
            "Figure 5 requires a registered smoke or full profile scaling artifact"
        )

    epsilon = np.asarray(scaling["epsilon"], dtype=np.float64)
    if (
        epsilon.ndim != 1
        or epsilon.size < 3
        or not np.isfinite(epsilon).all()
        or np.any(epsilon <= 0.0)
        or np.any(np.diff(epsilon) <= 0.0)
    ):
        raise ValueError("epsilon must be a finite, positive, strictly increasing sequence")
    size = epsilon.size
    aligned = {
        name: _aligned_finite_array(scaling, name, size)
        for name in (
            "delta_omega_full",
            "delta_omega_pred",
            "q_min_full",
            "q_min_pred",
            "q_saddle_full",
            "q_saddle_pred",
            "omega_min_error",
            "omega_saddle_error",
            "compensated_splitting",
            "compensated_q_min",
            "compensated_q_saddle",
            "compensated_frequency_error",
        )
    }
    if np.any(aligned["delta_omega_full"] <= 0.0) or np.any(
        aligned["delta_omega_pred"] <= 0.0
    ):
        raise ValueError("minimum-saddle splitting must remain positive")
    if np.any(aligned["omega_min_error"] <= 0.0) or np.any(
        aligned["omega_saddle_error"] <= 0.0
    ):
        raise ValueError("frequency remainders must remain positive")

    slope_splitting = _scalar(scaling, "slope_splitting")
    slope_radial = _scalar(scaling, "slope_radial")
    slope_remainder = _scalar(scaling, "slope_remainder")
    if abs(slope_splitting - 1.0) > _SPLITTING_SLOPE_GATE:
        raise ValueError("stored splitting slope must lie within 0.1 of one")
    if abs(slope_radial - 1.0) > _RADIAL_SLOPE_GATE:
        raise ValueError("stored radial slope must lie within 0.1 of one")
    if abs(slope_remainder - 2.0) > _REMAINDER_SLOPE_GATE:
        raise ValueError("stored frequency remainder slope must lie within 0.2 of two")

    for slope, target, tolerance_name in (
        (slope_splitting, 1.0, "slope_splitting_error"),
        (slope_radial, 1.0, "slope_radial_error"),
        (slope_remainder, 2.0, "slope_remainder_error"),
    ):
        registered = _registered_tolerance(metadata, tolerance_name)
        if not np.isclose(registered, abs(slope - target), rtol=1.0e-12, atol=1.0e-15):
            raise ValueError(f"registered {tolerance_name} does not certify the stored slope")

    _check_compensated_identity(
        aligned["compensated_splitting"],
        aligned["delta_omega_full"],
        epsilon,
        "stored compensated splitting must equal Delta omega/epsilon",
    )
    q_min_limit_prediction = _radial_limit_evidence(
        epsilon,
        aligned["q_min_full"],
        aligned["q_min_pred"],
        aligned["compensated_q_min"],
        "minimum",
    )
    q_saddle_limit_prediction = _radial_limit_evidence(
        epsilon,
        aligned["q_saddle_full"],
        aligned["q_saddle_pred"],
        aligned["compensated_q_saddle"],
        "saddle",
    )

    epsilon_squared = epsilon**2
    minimum_error = aligned["omega_min_error"] / epsilon_squared
    saddle_error = aligned["omega_saddle_error"] / epsilon_squared
    np.testing.assert_allclose(
        aligned["compensated_frequency_error"],
        np.maximum(minimum_error, saddle_error),
        rtol=_ROUNDTRIP_RTOL,
        atol=_ROUNDTRIP_ATOL,
        err_msg=(
            "stored compensated frequency error must equal the maximum of the "
            "minimum and saddle errors divided by epsilon squared"
        ),
    )

    negative_minimum = _fourfold_orbit(
        scaling["role_reversal_theta_min"], "minimum"
    )
    negative_saddle = _fourfold_orbit(
        scaling["role_reversal_theta_saddle"], "saddle"
    )
    combined = np.sort(np.concatenate((negative_minimum, negative_saddle)))
    combined_gaps = np.diff(np.r_[combined, combined[0] + 2.0 * np.pi])
    np.testing.assert_allclose(
        combined_gaps,
        np.full(8, 0.25 * np.pi),
        rtol=0.0,
        atol=2.0e-12,
        err_msg="stored role reversal must alternate minimum and saddle orbits",
    )

    return _ScalingEvidence(
        epsilon=epsilon,
        q_min_limit_prediction=q_min_limit_prediction,
        q_saddle_limit_prediction=q_saddle_limit_prediction,
        minimum_error_over_epsilon_squared=minimum_error,
        saddle_error_over_epsilon_squared=saddle_error,
        negative_minimum_theta=negative_minimum,
        negative_saddle_theta=negative_saddle,
        positive_minimum_theta=negative_saddle,
        positive_saddle_theta=negative_minimum,
        slope_splitting=slope_splitting,
        slope_radial=slope_radial,
        slope_remainder=slope_remainder,
    )


def _write_source_data(
    source_dir: Path,
    scaling: dict[str, NDArray[np.generic]],
    evidence: _ScalingEvidence,
) -> None:
    size = evidence.epsilon.size
    write_source_csv(
        source_dir / "panel_a_splitting.csv",
        {
            "delta_omega_full": scaling["delta_omega_full"],
            "delta_omega_pred": scaling["delta_omega_pred"],
            "epsilon_abs": evidence.epsilon,
            "slope_splitting": np.full(size, evidence.slope_splitting),
        },
    )
    write_source_csv(
        source_dir / "panel_b_compensated_radial_shift.csv",
        {
            "compensated_q_min": scaling["compensated_q_min"],
            "compensated_q_saddle": scaling["compensated_q_saddle"],
            "epsilon_abs": evidence.epsilon,
            "q_min_limit_prediction": evidence.q_min_limit_prediction,
            "q_saddle_limit_prediction": evidence.q_saddle_limit_prediction,
            "slope_radial": np.full(size, evidence.slope_radial),
        },
    )
    write_source_csv(
        source_dir / "panel_c_frequency_remainder.csv",
        {
            "epsilon_abs": evidence.epsilon,
            "minimum_error_over_epsilon_squared": (
                evidence.minimum_error_over_epsilon_squared
            ),
            "saddle_error_over_epsilon_squared": (
                evidence.saddle_error_over_epsilon_squared
            ),
            "slope_remainder": np.full(size, evidence.slope_remainder),
            "stored_max_error_over_epsilon_squared": scaling[
                "compensated_frequency_error"
            ],
        },
    )

    orbit_size = evidence.negative_minimum_theta.size
    epsilon_sign = np.repeat(np.array([-1, -1, 1, 1], dtype=np.int64), orbit_size)
    kind = np.repeat(
        np.array(["minimum", "saddle", "minimum", "saddle"]),
        orbit_size,
    )
    theta = np.concatenate(
        (
            evidence.negative_minimum_theta,
            evidence.negative_saddle_theta,
            evidence.positive_minimum_theta,
            evidence.positive_saddle_theta,
        )
    )
    write_source_csv(
        source_dir / "panel_d_role_reversal.csv",
        {
            "epsilon_sign": epsilon_sign,
            "kind": kind,
            "theta": theta,
            "theta_over_pi": theta / np.pi,
        },
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    """Panel tag, set outside the axes at the bottom left in parentheses."""

    ax.text(
        -0.02,
        -0.16,
        f"({label})",
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        ha="right",
        va="top",
    )


def _set_epsilon_ticks(
    ax: plt.Axes,
    epsilon: NDArray[np.generic],
) -> None:
    """Show four registered abscissae without log-scale minor-label collisions."""

    values = np.asarray(epsilon, dtype=np.float64)
    indices = np.unique(
        np.array([0, values.size // 3, 2 * values.size // 3, values.size - 1])
    )
    ticks = values[indices]
    labels = tuple(format(float(value), ".4g") for value in ticks)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _draw_splitting(
    ax: plt.Axes,
    scaling: dict[str, NDArray[np.generic]],
    evidence: _ScalingEvidence,
) -> None:
    ax.plot(
        evidence.epsilon,
        scaling["delta_omega_pred"],
        color=PALETTE["prediction"],
        linewidth=1.25,
        linestyle=(0, (4, 2)),
        label="first-order prediction",
    )
    ax.scatter(
        evidence.epsilon,
        scaling["delta_omega_full"],
        s=22,
        marker="o",
        facecolor="white",
        edgecolor=PALETTE["anisotropic"],
        linewidth=1.0,
        zorder=4,
        label="full wave",
    )
    ax.text(
        0.05,
        0.93,
        f"stored slope = {evidence.slope_splitting:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=PALETTE["anisotropic"],
        fontsize=8.4,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _set_epsilon_ticks(ax, evidence.epsilon)
    ax.set_xlabel(r"anisotropy magnitude $|\varepsilon|$")
    ax.set_ylabel(r"minimum--saddle splitting $\Delta\Omega_{ms}$")
    ax.legend(loc="lower right", fontsize=8.4)
    ax.grid(which="major", color="#E8E8E8", linewidth=0.45)


def _draw_radial_limits(
    ax: plt.Axes,
    scaling: dict[str, NDArray[np.generic]],
    evidence: _ScalingEvidence,
) -> None:
    ax.plot(
        evidence.epsilon,
        evidence.q_min_limit_prediction,
        color=PALETTE["minimum"],
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label=r"minimum stored $-B(\theta_j)/a$",
    )
    ax.plot(
        evidence.epsilon,
        evidence.q_saddle_limit_prediction,
        color=PALETTE["saddle"],
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        label=r"saddle stored $-B(\theta_j)/a$",
    )
    ax.scatter(
        evidence.epsilon,
        scaling["compensated_q_min"],
        s=23,
        marker=MARKERS["minimum"],
        color=PALETTE["minimum"],
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="minimum full wave",
    )
    ax.scatter(
        evidence.epsilon,
        scaling["compensated_q_saddle"],
        s=23,
        marker=MARKERS["saddle"],
        color=PALETTE["saddle"],
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="saddle full wave",
    )
    ax.text(
        0.05,
        0.07,
        f"stored radial slope = {evidence.slope_radial:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        color=PALETTE["neutral"],
    )
    ax.set_xscale("log")
    _set_epsilon_ticks(ax, evidence.epsilon)
    ax.set_xlabel(r"anisotropy magnitude $|\varepsilon|$")
    ax.set_ylabel(r"compensated radial shift $q/\varepsilon$")
    ax.legend(loc="best", fontsize=8.4, handlelength=1.8)
    ax.grid(which="major", color="#E8E8E8", linewidth=0.45)


def _draw_frequency_remainder(
    ax: plt.Axes,
    evidence: _ScalingEvidence,
) -> None:
    ax.scatter(
        evidence.epsilon,
        evidence.minimum_error_over_epsilon_squared,
        s=23,
        marker=MARKERS["minimum"],
        color=PALETTE["minimum"],
        edgecolor="white",
        linewidth=0.5,
        label="minimum",
        zorder=4,
    )
    ax.plot(
        evidence.epsilon,
        evidence.minimum_error_over_epsilon_squared,
        color=PALETTE["minimum"],
        linewidth=0.8,
        alpha=0.8,
    )
    ax.scatter(
        evidence.epsilon,
        evidence.saddle_error_over_epsilon_squared,
        s=23,
        marker=MARKERS["saddle"],
        color=PALETTE["saddle"],
        edgecolor="white",
        linewidth=0.5,
        label="saddle",
        zorder=4,
    )
    ax.plot(
        evidence.epsilon,
        evidence.saddle_error_over_epsilon_squared,
        color=PALETTE["saddle"],
        linewidth=0.8,
        alpha=0.8,
    )
    ax.text(
        0.05,
        0.93,
        f"stored slope = {evidence.slope_remainder:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color=PALETTE["neutral"],
    )
    ax.set_xscale("log")
    _set_epsilon_ticks(ax, evidence.epsilon)
    ax.set_xlabel(r"anisotropy magnitude $|\varepsilon|$")
    ax.set_ylabel(r"frequency error $/\varepsilon^2$")
    ax.legend(loc="best", fontsize=8.4)
    ax.grid(which="major", color="#E8E8E8", linewidth=0.45)


def _draw_role_reversal(ax: plt.Axes, evidence: _ScalingEvidence) -> None:
    role_rows = (
        (-1.0, evidence.negative_minimum_theta, "minimum"),
        (-1.0, evidence.negative_saddle_theta, "saddle"),
        (1.0, evidence.positive_minimum_theta, "minimum"),
        (1.0, evidence.positive_saddle_theta, "saddle"),
    )
    for sign, theta, kind in role_rows:
        ax.scatter(
            theta / np.pi,
            np.full(theta.size, sign),
            s=28,
            marker=MARKERS[kind],
            color=PALETTE[kind],
            edgecolor="white",
            linewidth=0.5,
            label=kind if sign < 0.0 else None,
            zorder=4,
        )
    ax.set_xlim(-0.08, 2.08)
    ax.set_ylim(-1.55, 1.55)
    ax.set_xticks(np.linspace(0.0, 2.0, 5))
    ax.set_xticklabels(("0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"))
    ax.set_yticks((-1.0, 1.0), labels=(r"$-\varepsilon$", r"$+\varepsilon$"))
    ax.set_xlabel(r"critical-point angle $\theta$")
    ax.set_ylabel("anisotropy sign")
    ax.legend(loc="upper center", ncols=2, fontsize=8.4)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.45)


def build(
    data_dir: Path,
    output_dir: Path,
    source_dir: Path,
) -> dict[str, Path]:
    """Build Figure 5 only from the validated perturbation-scaling artifact."""

    scaling, metadata = load_figure_artifact(data_dir, "perturbation_scaling")
    evidence = _validate_evidence(scaling, metadata)
    _write_source_data(source_dir, scaling, evidence)

    apply_publication_style()
    fig = plt.figure(layout="constrained", facecolor="white")
    axes = fig.subplot_mosaic(
        [["a", "b"], ["c", "d"]],
        width_ratios=(0.92, 1.08),
        height_ratios=(0.95, 1.05),
    )
    _draw_splitting(axes["a"], scaling, evidence)
    _draw_radial_limits(axes["b"], scaling, evidence)
    _draw_frequency_remainder(axes["c"], evidence)
    _draw_role_reversal(axes["d"], evidence)
    for label, ax in axes.items():
        _panel_label(ax, label)
    return save_publication_figure(
        fig,
        output_dir / "figure_05_perturbation_scaling",
        _SPEC,
    )
