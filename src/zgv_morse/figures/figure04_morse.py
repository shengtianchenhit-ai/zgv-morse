"""Main Figure 4: resolved full-wave Morse roots in the declared local annulus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

from zgv_morse.artifacts import sha256_file
from zgv_morse.config import load_reference_config
from zgv_morse.figures.common import (
    FigureSpec,
    MARKERS,
    PALETTE,
    apply_publication_style,
    load_figure_artifact,
    save_publication_figure,
    write_source_csv,
)

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


_SPEC: Final = FigureSpec(
    "4",
    "the full three-dimensional elastic model realizes the Morse theorem",
    "quantitative grid",
    183.0,
    158.0,
)
_REFERENCE_CONFIG_PATH: Final = (
    Path(__file__).resolve().parents[3] / "config" / "reference.yaml"
)
_CONTEXT_FIELDS: Final = (
    "profile",
    "config_hash",
    "source_hash",
    "code_hash",
    "uv_lock_hash",
    "dimensionless_convention",
)


@dataclass(frozen=True, slots=True)
class _Evidence:
    kappa0: float
    omega0: float
    annulus_fraction: float
    inner_radius: float
    outer_radius: float
    order: NDArray[np.int64]
    gradient_uncertainty_bound: float
    positive_boundary_ratio: float
    negative_boundary_ratio: float
    hessian_ratio: float
    hessian_uncertainty_upper_bound: NDArray[np.float64]
    delta_kappa_over_kappa0: NDArray[np.float64]
    delta_omega_over_omega0: NDArray[np.float64]
    cartesian_location_error_over_kappa0: NDArray[np.float64]


def _metric(metadata: dict[str, Any], name: str) -> float:
    tolerances = metadata.get("tolerances")
    assert isinstance(tolerances, dict), "certificate: tolerances must be registered"
    value = tolerances.get(name)
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"certificate: {name} must be numeric"
    )
    result = float(value)
    assert np.isfinite(result) and result >= 0.0, (
        f"certificate: {name} must be finite and nonnegative"
    )
    return result


def _validate_context(
    isotropic_metadata: dict[str, Any],
    points_metadata: dict[str, Any],
) -> float:
    assert isotropic_metadata.get("artifact") == "isotropic_zgv", (
        "scientific context: isotropic metadata identity is invalid"
    )
    assert points_metadata.get("artifact") == "critical_points", (
        "scientific context: critical-point metadata identity is invalid"
    )
    for field in _CONTEXT_FIELDS:
        assert isotropic_metadata.get(field) == points_metadata.get(field), (
            f"scientific context: {field} differs between registered artifacts"
        )

    reference_hash = sha256_file(_REFERENCE_CONFIG_PATH)
    assert isotropic_metadata.get("source_hash") == reference_hash, (
        "scientific context: reference configuration hash is not registered"
    )
    config = load_reference_config(_REFERENCE_CONFIG_PATH)
    return float(config.annulus_fraction)


def _validate_annulus(
    isotropic: dict[str, NDArray[np.generic]],
    points: dict[str, NDArray[np.generic]],
    annulus_fraction: float,
) -> tuple[float, float, float, float]:
    kappa0 = float(isotropic["kappa0"])
    omega0 = float(isotropic["omega0"])
    assert np.isfinite(kappa0) and kappa0 > 0.0, (
        "registered annulus: kappa0 must be finite and positive"
    )
    assert np.isfinite(omega0) and omega0 > 0.0, (
        "registered annulus: omega0 must be finite and positive"
    )
    assert np.isfinite(annulus_fraction) and 0.0 < annulus_fraction < 1.0, (
        "registered annulus: fraction must lie between zero and one"
    )
    half_width = annulus_fraction * kappa0
    inner_radius = kappa0 - half_width
    outer_radius = kappa0 + half_width

    kx_grid = np.asarray(points["kx_grid"], dtype=np.float64)
    ky_grid = np.asarray(points["ky_grid"], dtype=np.float64)
    assert (
        kx_grid.ndim == 1
        and ky_grid.ndim == 1
        and kx_grid.size >= 3
        and ky_grid.size >= 3
        and np.isfinite(kx_grid).all()
        and np.isfinite(ky_grid).all()
        and np.all(np.diff(kx_grid) > 0.0)
        and np.all(np.diff(ky_grid) > 0.0)
    ), "registered annulus: Cartesian grids must be finite and strictly ordered"
    scale = max(outer_radius, 1.0)
    absolute_tolerance = 16.0 * np.finfo(np.float64).eps * scale
    assert np.allclose(kx_grid, -kx_grid[::-1], rtol=0.0, atol=absolute_tolerance), (
        "registered annulus: kx grid must be centered on zero"
    )
    assert np.allclose(ky_grid, -ky_grid[::-1], rtol=0.0, atol=absolute_tolerance), (
        "registered annulus: ky grid must be centered on zero"
    )
    assert np.isclose(kx_grid[-1], outer_radius, rtol=2.0e-15, atol=absolute_tolerance), (
        "registered annulus: kx extent must equal kappa0 plus registered half-width"
    )
    assert np.isclose(ky_grid[-1], outer_radius, rtol=2.0e-15, atol=absolute_tolerance), (
        "registered annulus: ky extent must equal kappa0 plus registered half-width"
    )

    stored_radius = np.asarray(points["kappa"], dtype=np.float64)
    coordinate_radius = np.hypot(points["kx"], points["ky"])
    assert np.allclose(
        stored_radius,
        coordinate_radius,
        rtol=1.0e-12,
        atol=absolute_tolerance,
    ), "registered annulus: stored radii must agree with Cartesian coordinates"
    assert np.all(stored_radius >= inner_radius - absolute_tolerance) and np.all(
        stored_radius <= outer_radius + absolute_tolerance
    ), "registered annulus: every refined point must lie inside the registered annulus"
    return kappa0, omega0, inner_radius, outer_radius


def _validate_morse_certificate(
    points: dict[str, NDArray[np.generic]],
    metadata: dict[str, Any],
) -> tuple[
    NDArray[np.int64],
    float,
    float,
    float,
    float,
    NDArray[np.float64],
]:
    kinds = np.asarray(points["kind"])
    minimum = kinds == "minimum"
    saddle = kinds == "saddle"
    assert np.count_nonzero(minimum) == 4 and np.count_nonzero(saddle) == 4, (
        "count: expected four minima and four saddles"
    )

    theta = np.mod(np.asarray(points["theta"], dtype=np.float64), 2.0 * np.pi)
    order = np.asarray(np.argsort(theta, kind="stable"), dtype=np.int64)
    angular_gaps = np.diff(np.r_[theta[order], theta[order][0] + 2.0 * np.pi])
    assert np.isfinite(theta).all() and np.all(angular_gaps > 0.0), (
        "alternation: critical-point angles must be finite and distinct"
    )
    ordered_kinds = kinds[order]
    assert np.all(ordered_kinds != np.roll(ordered_kinds, 1)), (
        "alternation: minima and saddles must alternate in angular order"
    )

    morse_index = np.asarray(points["morse_index"], dtype=np.int64)
    assert int(np.sum(morse_index)) == 0, "index sum: total gradient index must be zero"
    expected_index = np.where(minimum, 1, -1)
    assert np.array_equal(morse_index, expected_index), (
        "pointwise index: every minimum must have index +1 and every saddle index -1"
    )

    hessian = np.asarray(points["hessian_eigenvalues"], dtype=np.float64)
    assert hessian.shape == (8, 2) and np.isfinite(hessian).all(), (
        "Hessian inertia: eight finite Cartesian eigenvalue pairs are required"
    )
    assert np.all(hessian[:, 0] <= hessian[:, 1]), (
        "Hessian inertia: eigenvalues must be stored in ascending order"
    )
    assert np.all(hessian[minimum] > 0.0) and np.all(
        (hessian[saddle, 0] < 0.0) & (hessian[saddle, 1] > 0.0)
    ), "Hessian inertia: minima must be positive definite and saddles indefinite"
    hessian_ratio = _metric(metadata, "minimum_hessian_to_uncertainty")
    assert hessian_ratio > 10.0, (
        "resolved Hessian: eigenvalues must exceed ten times their uncertainty"
    )
    hessian_uncertainty_upper_bound = np.min(np.abs(hessian), axis=1) / hessian_ratio

    gradient = np.asarray(points["gradient_residual"], dtype=np.float64)
    assert np.isfinite(gradient).all() and np.all(gradient >= 0.0), (
        "finite gradient: residuals must be finite and nonnegative"
    )
    registered_gradient = _metric(metadata, "maximum_gradient_residual")
    assert np.isclose(
        float(np.max(gradient)),
        registered_gradient,
        rtol=1.0e-12,
        atol=0.0,
    ), "bounded gradient: array maximum must match the registered certificate"
    gradient_uncertainty_bound = min(
        _metric(metadata, "positive_maximum_boundary_gradient_uncertainty"),
        _metric(metadata, "negative_maximum_boundary_gradient_uncertainty"),
    )
    assert gradient_uncertainty_bound > 0.0 and (
        registered_gradient <= gradient_uncertainty_bound
    ), "bounded gradient: residuals must lie below an independent uncertainty bound"

    boundary_ratios: dict[str, float] = {}
    for sign in ("positive", "negative"):
        boundary_gradient = _metric(metadata, f"{sign}_minimum_boundary_gradient")
        boundary_uncertainty = _metric(
            metadata,
            f"{sign}_maximum_boundary_gradient_uncertainty",
        )
        assert boundary_uncertainty > 0.0 and (
            boundary_gradient > 10.0 * boundary_uncertainty
        ), f"{sign} boundary: gradient margin must exceed ten times uncertainty"
        boundary_ratios[sign] = boundary_gradient / boundary_uncertainty

    return (
        order,
        gradient_uncertainty_bound,
        boundary_ratios["positive"],
        boundary_ratios["negative"],
        hessian_ratio,
        hessian_uncertainty_upper_bound,
    )


def _validate_evidence(
    isotropic: dict[str, NDArray[np.generic]],
    isotropic_metadata: dict[str, Any],
    points: dict[str, NDArray[np.generic]],
    points_metadata: dict[str, Any],
) -> _Evidence:
    annulus_fraction = _validate_context(isotropic_metadata, points_metadata)
    kappa0, omega0, inner_radius, outer_radius = _validate_annulus(
        isotropic,
        points,
        annulus_fraction,
    )
    (
        order,
        gradient_uncertainty_bound,
        positive_boundary_ratio,
        negative_boundary_ratio,
        hessian_ratio,
        hessian_uncertainty_upper_bound,
    ) = _validate_morse_certificate(points, points_metadata)

    predicted_kappa = np.hypot(points["kx_pred"], points["ky_pred"])
    delta_kappa = (np.asarray(points["kappa"]) - predicted_kappa) / kappa0
    delta_omega = (
        np.asarray(points["omega"]) - np.asarray(points["omega_pred"])
    ) / omega0
    cartesian_error = np.hypot(
        np.asarray(points["kx"]) - np.asarray(points["kx_pred"]),
        np.asarray(points["ky"]) - np.asarray(points["ky_pred"]),
    ) / kappa0
    assert (
        np.isfinite(delta_kappa).all()
        and np.isfinite(delta_omega).all()
        and np.isfinite(cartesian_error).all()
        and np.all(cartesian_error >= 0.0)
    ), "prediction errors must be finite"

    return _Evidence(
        kappa0=kappa0,
        omega0=omega0,
        annulus_fraction=annulus_fraction,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        order=order,
        gradient_uncertainty_bound=gradient_uncertainty_bound,
        positive_boundary_ratio=positive_boundary_ratio,
        negative_boundary_ratio=negative_boundary_ratio,
        hessian_ratio=hessian_ratio,
        hessian_uncertainty_upper_bound=hessian_uncertainty_upper_bound,
        delta_kappa_over_kappa0=np.asarray(delta_kappa, dtype=np.float64),
        delta_omega_over_omega0=np.asarray(delta_omega, dtype=np.float64),
        cartesian_location_error_over_kappa0=np.asarray(
            cartesian_error,
            dtype=np.float64,
        ),
    )


def _circle_arrays(
    count: int,
    evidence: _Evidence,
) -> dict[str, NDArray[np.float64]]:
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=True)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    return {
        "ring_kx": evidence.kappa0 * cosine,
        "ring_ky": evidence.kappa0 * sine,
        "annulus_inner_kx": evidence.inner_radius * cosine,
        "annulus_inner_ky": evidence.inner_radius * sine,
        "annulus_outer_kx": evidence.outer_radius * cosine,
        "annulus_outer_ky": evidence.outer_radius * sine,
    }


def _write_source_data(
    source_dir: Path,
    points: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
    surface_kx: NDArray[np.float64],
    surface_ky: NDArray[np.float64],
    circles: dict[str, NDArray[np.float64]],
) -> None:
    surface_count = points["omega_iso_grid"].size
    common_surface = {
        "annulus_fraction": np.full(surface_count, evidence.annulus_fraction),
        **{name: values for name, values in circles.items() if name.startswith("annulus_")},
        "kx": surface_kx,
        "ky": surface_ky,
    }
    write_source_csv(
        source_dir / "panel_a_isotropic_surface.csv",
        {
            **common_surface,
            "omega_iso": points["omega_iso_grid"],
            "ring_kx": circles["ring_kx"],
            "ring_ky": circles["ring_ky"],
        },
    )
    write_source_csv(
        source_dir / "panel_b_anisotropic_surface.csv",
        {
            **common_surface,
            "omega_aniso": points["omega_aniso_grid"],
        },
    )

    point_count = points["theta"].size
    write_source_csv(
        source_dir / "panel_b_points.csv",
        {
            "artifact_row": np.arange(point_count, dtype=np.int64),
            "gradient_residual": points["gradient_residual"],
            "gradient_residual_over_uncertainty": (
                points["gradient_residual"] / evidence.gradient_uncertainty_bound
            ),
            "gradient_uncertainty_bound": np.full(
                point_count,
                evidence.gradient_uncertainty_bound,
            ),
            "hessian_eigenvalue_1": points["hessian_eigenvalues"][:, 0],
            "hessian_eigenvalue_2": points["hessian_eigenvalues"][:, 1],
            "kind": points["kind"],
            "kappa": points["kappa"],
            "kx": points["kx"],
            "kx_pred": points["kx_pred"],
            "ky": points["ky"],
            "ky_pred": points["ky_pred"],
            "morse_index": points["morse_index"],
            "negative_boundary_resolution_ratio": np.full(
                point_count,
                evidence.negative_boundary_ratio,
            ),
            "omega": points["omega"],
            "omega_pred": points["omega_pred"],
            "positive_boundary_resolution_ratio": np.full(
                point_count,
                evidence.positive_boundary_ratio,
            ),
            "theta": points["theta"],
        },
    )

    order = evidence.order
    hessian = np.asarray(points["hessian_eigenvalues"])[order]
    write_source_csv(
        source_dir / "panel_c_hessian.csv",
        {
            "artifact_row": order,
            "hessian_eigenvalue_1": hessian[:, 0],
            "hessian_eigenvalue_2": hessian[:, 1],
            "hessian_uncertainty_upper_bound": (
                evidence.hessian_uncertainty_upper_bound[order]
            ),
            "kind": points["kind"][order],
            "minimum_hessian_to_uncertainty": np.full(
                point_count,
                evidence.hessian_ratio,
            ),
            "morse_index": points["morse_index"][order],
            "point_number": np.arange(1, point_count + 1, dtype=np.int64),
            "theta": points["theta"][order],
        },
    )
    write_source_csv(
        source_dir / "panel_d_prediction_errors.csv",
        {
            "artifact_row": order,
            "cartesian_location_error_over_kappa0": (
                evidence.cartesian_location_error_over_kappa0[order]
            ),
            "delta_kappa_over_kappa0": evidence.delta_kappa_over_kappa0[order],
            "delta_omega_over_omega0": evidence.delta_omega_over_omega0[order],
            "kind": points["kind"][order],
            "point_number": np.arange(1, point_count + 1, dtype=np.int64),
            "theta": points["theta"][order],
        },
    )


def _panel_label(ax: Axes, label: str) -> None:
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


def _style_geometry_axis(ax: Axes, points: dict[str, NDArray[np.generic]]) -> None:
    ax.set_aspect("equal", adjustable="box")
    outer = max(
        float(np.max(np.abs(points["kx_grid"]))),
        float(np.max(np.abs(points["ky_grid"]))),
    )
    padding = 0.025 * outer
    ax.set_xlim(-outer - padding, outer + padding)
    ax.set_ylim(-outer - padding, outer + padding)
    ax.set_xlabel("$k_x h$")
    ax.set_ylabel("$k_y h$")
    ax.tick_params(direction="out", length=2.5, width=0.7)


def _draw_registered_circles(
    ax: Axes,
    circles: dict[str, NDArray[np.float64]],
    *,
    show_ring: bool,
) -> None:
    for prefix in ("annulus_inner", "annulus_outer"):
        ax.plot(
            circles[f"{prefix}_kx"],
            circles[f"{prefix}_ky"],
            color=PALETTE["neutral"],
            linewidth=0.65,
            linestyle=(0, (2.5, 2.0)),
            alpha=0.72,
            zorder=4,
        )
    if show_ring:
        ax.plot(
            circles["ring_kx"],
            circles["ring_ky"],
            color=PALETTE["isotropic"],
            linewidth=1.8,
            zorder=5,
        )


def _draw_panel_a(
    ax: Axes,
    surface_kx: NDArray[np.float64],
    surface_ky: NDArray[np.float64],
    omega_iso: NDArray[np.float64],
    points: dict[str, NDArray[np.generic]],
    circles: dict[str, NDArray[np.float64]],
    levels: NDArray[np.float64],
) -> None:
    ax.contour(
        surface_kx,
        surface_ky,
        omega_iso,
        levels=levels,
        colors=PALETTE["neutral"],
        linewidths=0.55,
        alpha=0.74,
    )
    _draw_registered_circles(ax, circles, show_ring=True)
    ring_index = circles["ring_kx"].size // 4
    ax.annotate(
        "critical ring",
        xy=(
            float(circles["ring_kx"][ring_index]),
            float(circles["ring_ky"][ring_index]),
        ),
        xytext=(0.5, 1.06),
        textcoords="axes fraction",
        color=PALETTE["isotropic"],
        ha="center",
        va="bottom",
        fontweight="bold",
        annotation_clip=False,
        arrowprops={
            "arrowstyle": "-",
            "color": PALETTE["isotropic"],
            "lw": 0.55,
        },
    )
    _style_geometry_axis(ax, points)
    _panel_label(ax, "a")


def _draw_panel_b(
    ax: Axes,
    surface_kx: NDArray[np.float64],
    surface_ky: NDArray[np.float64],
    omega_aniso: NDArray[np.float64],
    points: dict[str, NDArray[np.generic]],
    circles: dict[str, NDArray[np.float64]],
    levels: NDArray[np.float64],
) -> None:
    contour_palette = mpl.colors.LinearSegmentedColormap.from_list(
        "morse_full_wave",
        ["#FFFFFF", "#E4EEF8", PALETTE["anisotropic"]],
    )
    ax.contourf(
        surface_kx,
        surface_ky,
        omega_aniso,
        levels=levels,
        cmap=contour_palette,
        alpha=0.76,
        extend="both",
    )
    ax.contour(
        surface_kx,
        surface_ky,
        omega_aniso,
        levels=levels,
        colors=PALETTE["anisotropic"],
        linewidths=0.5,
        alpha=0.76,
    )
    _draw_registered_circles(ax, circles, show_ring=False)

    kinds = np.asarray(points["kind"])
    for kind in ("minimum", "saddle"):
        selected = kinds == kind
        ax.scatter(
            points["kx"][selected],
            points["ky"][selected],
            s=32,
            marker=MARKERS[kind],
            facecolor=PALETTE[kind],
            edgecolor="white",
            linewidth=0.7,
            zorder=7,
            label=kind,
        )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=8.4,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.6,
    )
    minimum_index = int(np.flatnonzero(kinds == "minimum")[0])
    saddle_index = int(np.flatnonzero(kinds == "saddle")[0])
    # In the reserved title gap rather than inside the axes: the filled
    # contour field covers the panel, so an opaque in-axes box would hide a
    # solid rectangle of the very surface being described.
    ax.text(
        0.50,
        1.005,
        "eight resolved roots in the declared local annulus",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        # Sized to the panel width: at 8.4 pt this line overhung both edges.
        fontsize=6.6,
        color=PALETTE["neutral"],
        zorder=8,
    )
    _style_geometry_axis(ax, points)
    _panel_label(ax, "b")


def _draw_panel_c(
    ax: Axes,
    points: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
) -> None:
    order = evidence.order
    point_number = np.arange(1, order.size + 1, dtype=np.float64)
    kinds = np.asarray(points["kind"])[order]
    hessian = np.asarray(points["hessian_eigenvalues"], dtype=np.float64)[order]
    uncertainty = evidence.hessian_uncertainty_upper_bound[order]
    offsets = (-0.10, 0.10)
    for kind in ("minimum", "saddle"):
        selected = kinds == kind
        for column, offset in enumerate(offsets):
            ax.errorbar(
                point_number[selected] + offset,
                hessian[selected, column],
                yerr=uncertainty[selected],
                linestyle="none",
                marker=MARKERS[kind],
                markersize=4.1,
                markerfacecolor=PALETTE[kind] if column == 0 else "white",
                markeredgecolor=PALETTE[kind],
                markeredgewidth=0.8,
                ecolor=PALETTE[kind],
                elinewidth=0.65,
                capsize=1.7,
                zorder=4,
            )
    uncertainty_band = float(np.max(uncertainty))
    ax.axhspan(
        -uncertainty_band,
        uncertainty_band,
        color=PALETTE["uncertainty"],
        alpha=0.55,
        linewidth=0.0,
        zorder=1,
    )
    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.65, zorder=2)
    minimum_absolute_eigenvalue = float(np.min(np.abs(hessian)))
    ax.set_yscale(
        "symlog",
        linthresh=max(
            0.20 * minimum_absolute_eigenvalue,
            2.0 * uncertainty_band,
        ),
    )
    ax.set_ylim(
        2.0 * (float(np.min(hessian)) - uncertainty_band),
        2.0 * (float(np.max(hessian)) + uncertainty_band),
    )
    ax.set_xticks(point_number)
    ax.set_xlabel("critical point (angular order)")
    ax.set_ylabel("Cartesian Hessian eigenvalue")
    ax.text(
        0.98,
        0.05,
        rf"min $|\lambda|/u_H={evidence.hessian_ratio:.2g}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        color=PALETTE["neutral"],
    )
    # Retained deliberately: tests/figures/test_figure04.py looks this key up
    # by its "filled:" prefix and asserts its headroom, so it is part of the
    # figure contract even though the caption also states the convention.
    ax.text(
        0.02,
        0.83,
        r"filled: $\lambda_1$   open: $\lambda_2$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.4,
        color=PALETTE["neutral"],
        zorder=6,
    )
    _panel_label(ax, "c")


def _draw_panel_d(
    ax: Axes,
    points: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
) -> None:
    order = evidence.order
    point_number = np.arange(1, order.size + 1, dtype=np.float64)
    kinds = np.asarray(points["kind"])[order]
    radial_error = evidence.delta_kappa_over_kappa0[order]
    frequency_error = evidence.delta_omega_over_omega0[order]
    ax.plot(
        point_number,
        radial_error,
        color=PALETTE["neutral"],
        linewidth=0.65,
        alpha=0.65,
    )
    ax.plot(
        point_number,
        frequency_error,
        color=PALETTE["neutral"],
        linewidth=0.65,
        linestyle="--",
        alpha=0.65,
    )
    for kind in ("minimum", "saddle"):
        selected = kinds == kind
        ax.scatter(
            point_number[selected],
            radial_error[selected],
            s=25,
            marker=MARKERS[kind],
            facecolor=PALETTE[kind],
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
        ax.scatter(
            point_number[selected],
            frequency_error[selected],
            s=25,
            marker=MARKERS[kind],
            facecolor="white",
            edgecolor=PALETTE[kind],
            linewidth=0.8,
            zorder=4,
        )
    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.65)
    error_scale = float(
        np.max(np.abs(np.concatenate((radial_error, frequency_error))))
    )
    ax.set_ylim(-1.10 * error_scale, 0.22 * error_scale)
    ax.set_xticks(point_number)
    ax.set_xlabel("critical point (angular order)")
    ax.set_ylabel("relative signed error")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _panel_label(ax, "d")


def build(
    data_dir: Path,
    output_dir: Path,
    source_dir: Path,
) -> dict[str, Path]:
    """Build Figure 4 from two strictly validated registered artifacts."""

    isotropic, isotropic_metadata = load_figure_artifact(data_dir, "isotropic_zgv")
    points, points_metadata = load_figure_artifact(data_dir, "critical_points")
    evidence = _validate_evidence(
        isotropic,
        isotropic_metadata,
        points,
        points_metadata,
    )

    surface_kx, surface_ky = np.meshgrid(points["kx_grid"], points["ky_grid"])
    omega_iso = np.asarray(points["omega_iso_grid"], dtype=np.float64)
    omega_aniso = np.asarray(points["omega_aniso_grid"], dtype=np.float64)
    circles = _circle_arrays(omega_iso.size, evidence)
    levels = np.linspace(
        min(float(np.min(omega_iso)), float(np.min(omega_aniso))),
        max(float(np.max(omega_iso)), float(np.max(omega_aniso))),
        9,
    )
    _write_source_data(
        source_dir,
        points,
        evidence,
        surface_kx,
        surface_ky,
        circles,
    )

    apply_publication_style()
    fig = plt.figure(layout="constrained", facecolor="white")
    axes = fig.subplot_mosaic(
        [["a", "b"], ["c", "d"]],
        width_ratios=[0.94, 1.06],
        height_ratios=[1.08, 0.92],
    )
    _draw_panel_a(
        axes["a"],
        surface_kx,
        surface_ky,
        omega_iso,
        points,
        circles,
        levels,
    )
    _draw_panel_b(
        axes["b"],
        surface_kx,
        surface_ky,
        omega_aniso,
        points,
        circles,
        levels,
    )
    _draw_panel_c(axes["c"], points, evidence)
    _draw_panel_d(axes["d"], points, evidence)
    return save_publication_figure(
        fig,
        output_dir / "figure_04_morse_points",
        _SPEC,
    )
