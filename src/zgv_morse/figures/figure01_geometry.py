"""Figure 1: isotropic ZGV ring, anisotropic Morse splitting, and mechanism."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


_SPEC = FigureSpec(
    "1",
    "anisotropy changes the topology and dimension of the stationary set",
    "schematic-led composite",
    183.0,
    140.0,
)


def _assert_scientific_contract(
    iso: dict[str, NDArray[np.generic]],
    angular: dict[str, NDArray[np.generic]],
    points: dict[str, NDArray[np.generic]],
    metadata: dict[str, object],
) -> None:
    """Fail before rendering unless the declared local evidence is intact."""

    kinds = np.asarray(points["kind"])
    minimum = kinds == "minimum"
    saddle = kinds == "saddle"
    assert np.count_nonzero(minimum) == 4 and np.count_nonzero(saddle) == 4, (
        "count: expected four minima and four saddles"
    )

    theta = np.mod(np.asarray(points["theta"], dtype=float), 2.0 * np.pi)
    order = np.argsort(theta, kind="stable")
    ordered_kinds = kinds[order]
    angular_gaps = np.diff(np.r_[theta[order], theta[order][0] + 2.0 * np.pi])
    assert np.all(np.isfinite(theta)) and np.all(angular_gaps > 0.0), (
        "alternation: critical-point angles must be finite and distinct"
    )
    assert np.all(ordered_kinds != np.roll(ordered_kinds, 1)), (
        "alternation: minima and saddles must alternate in angle"
    )

    morse_index = np.asarray(points["morse_index"], dtype=np.int64)
    assert int(np.sum(morse_index)) == 0, "index sum: total gradient index must be zero"
    expected_index = np.where(minimum, 1, -1)
    assert np.array_equal(morse_index, expected_index), (
        "pointwise index: every minimum must have index +1 and every saddle index -1"
    )

    hessian = np.asarray(points["hessian_eigenvalues"], dtype=float)
    minimum_inertia = np.all(hessian[minimum] > 0.0, axis=1)
    saddle_inertia = np.prod(hessian[saddle], axis=1) < 0.0
    assert np.isfinite(hessian).all() and minimum_inertia.all() and saddle_inertia.all(), (
        "Hessian inertia: minima must be positive definite and saddles indefinite"
    )

    gradient = np.asarray(points["gradient_residual"], dtype=float)
    assert np.isfinite(gradient).all() and np.all(gradient >= 0.0), (
        "finite gradient: residuals must be finite and nonnegative"
    )

    assert metadata.get("artifact") == "critical_points", (
        "certificate: metadata must belong to critical_points"
    )
    tolerances = metadata.get("tolerances")
    assert isinstance(tolerances, dict), "certificate: tolerances must be registered"

    def metric(name: str) -> float:
        value = tolerances.get(name)
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"certificate: {name} must be numeric"
        )
        result = float(value)
        assert np.isfinite(result) and result >= 0.0, (
            f"certificate: {name} must be finite and nonnegative"
        )
        return result

    registered_gradient = metric("maximum_gradient_residual")
    observed_gradient = float(np.max(gradient))
    assert np.isclose(observed_gradient, registered_gradient, rtol=1.0e-12, atol=0.0), (
        "bounded gradient: array maximum must match the registered certificate"
    )
    independent_gradient_uncertainty = min(
        metric("positive_maximum_boundary_gradient_uncertainty"),
        metric("negative_maximum_boundary_gradient_uncertainty"),
    )
    assert registered_gradient <= independent_gradient_uncertainty, (
        "bounded gradient: point residual must not exceed independent gradient uncertainty"
    )

    assert metric("minimum_hessian_to_uncertainty") > 10.0, (
        "resolved Hessian: eigenvalues must exceed ten times their uncertainty"
    )
    for sign in ("positive", "negative"):
        boundary_gradient = metric(f"{sign}_minimum_boundary_gradient")
        boundary_uncertainty = metric(
            f"{sign}_maximum_boundary_gradient_uncertainty"
        )
        assert boundary_gradient > 10.0 * boundary_uncertainty, (
            "noncritical boundary: gradient margin must exceed ten times uncertainty"
        )

    assert np.isfinite(float(iso["kappa0"])) and float(iso["kappa0"]) > 0.0
    assert np.isfinite(float(iso["curvature_a"])) and float(iso["curvature_a"]) > 0.0
    assert np.isfinite(float(angular["V4"])) and float(angular["V4"]) != 0.0


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


def _style_geometry_axis(ax: Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$k_x h$")
    ax.set_ylabel("$k_y h$")
    ax.tick_params(direction="out", length=2.5, width=0.7)


def _draw_panel_a(
    ax: Axes,
    surface_kx: NDArray[np.float64],
    surface_ky: NDArray[np.float64],
    omega_iso: NDArray[np.float64],
    ring_kx: NDArray[np.float64],
    ring_ky: NDArray[np.float64],
    levels: NDArray[np.float64],
) -> None:
    ax.contour(
        surface_kx,
        surface_ky,
        omega_iso,
        levels=levels,
        colors=PALETTE["neutral"],
        linewidths=0.55,
        alpha=0.72,
    )
    ax.plot(
        ring_kx,
        ring_ky,
        color=PALETTE["isotropic"],
        linewidth=2.0,
        solid_capstyle="round",
        zorder=3,
    )
    # Centre-anchored in the clear band just outside the ring.  A bottom
    # anchor let the text grow upwards into the ring's lower arc.
    ax.text(
        0.50,
        0.033,
        "continuous ZGV ring",
        transform=ax.transAxes,
        color=PALETTE["isotropic"],
        fontweight="bold",
        ha="center",
        va="center",
        zorder=6,
    )
    # Same title pad as panel b, whose title is lifted to make room for the
    # note below it.  Matching the pad keeps the two panel titles on one
    # baseline instead of leaving them 30 px apart.
    _style_geometry_axis(ax)
    _panel_label(ax, "a")


def _ring_splitting_anomaly(
    surface_kx: NDArray[np.float64],
    surface_ky: NDArray[np.float64],
    omega_iso: NDArray[np.float64],
    omega_aniso: NDArray[np.float64],
    *,
    bins: int = 40,
) -> NDArray[np.float64]:
    """Angular part of the perturbation, which is what splits the ring.

    The full difference ``omega_aniso - omega_iso`` is dominated by a
    radial, angle-independent term: that part shifts the carrier
    (``Omega_0 + eps V_0``) and moves the ring without breaking it.  What
    breaks the ring is the remaining angular anomaly, obtained by removing
    the mean over angle at each radius.  Isolating it is the same split the
    analysis performs, and it is a pure regrouping of the two stored
    surfaces rather than a new computation.
    """

    radius = np.hypot(surface_kx, surface_ky)
    anomaly = np.asarray(omega_aniso, dtype=float) - np.asarray(omega_iso, dtype=float)
    edges = np.linspace(float(radius.min()), float(radius.max()), bins + 1)
    for index in range(bins):
        shell = (radius >= edges[index]) & (radius < edges[index + 1])
        if int(shell.sum()) > 3:
            anomaly[shell] -= float(anomaly[shell].mean())
    return anomaly


def _draw_panel_b(
    ax: Axes,
    surface_kx: NDArray[np.float64],
    surface_ky: NDArray[np.float64],
    anomaly: NDArray[np.float64],
    points: dict[str, NDArray[np.generic]],
    _levels: NDArray[np.float64],
) -> None:
    # Diverging map centred on zero: troughs carry the colour used for
    # minima and crests the colour used for saddles, so the field and the
    # markers on it share one convention.  Marker shape still duplicates
    # colour for grayscale reproduction.
    contour_palette = mpl.colors.LinearSegmentedColormap.from_list(
        "zgv_anomaly",
        ["#C9DCF0", "#FFFFFF", "#F6DCB8"],
    )
    # Robust span: the outermost radial shell holds only the canvas corners,
    # so its mean removal is unreliable and its residual would otherwise set
    # the colour scale and wash the physical signal out to white.
    span = float(np.percentile(np.abs(anomaly), 99.0))
    anomaly_levels = np.linspace(-span, span, 13)
    ax.contourf(
        surface_kx,
        surface_ky,
        anomaly,
        levels=anomaly_levels,
        cmap=contour_palette,
        extend="both",
    )
    ax.contour(
        surface_kx,
        surface_ky,
        anomaly,
        levels=anomaly_levels,
        colors=PALETTE["neutral"],
        linewidths=0.5,
        alpha=0.55,
    )

    kinds = np.asarray(points["kind"])
    for kind in ("minimum", "saddle"):
        selected = kinds == kind
        ax.scatter(
            np.asarray(points["kx"])[selected],
            np.asarray(points["ky"])[selected],
            s=34,
            marker=MARKERS[kind],
            facecolor=PALETTE[kind],
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
            label=kind,
        )
    # Keyed by marker instead of by leader labels: the anomaly field fills
    # this panel, so any in-axes text covered at least 18% of it.
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
        0.5,
        1.005,
        "eight resolved roots in the declared local annulus",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=PALETTE["neutral"],
        zorder=6,
    )
    _style_geometry_axis(ax)
    _panel_label(ax, "b")


def _mechanism_box(
    ax: Axes,
    x: float,
    title: str,
    dimension: str,
    decay: str,
    edgecolor: str,
) -> None:
    box = FancyBboxPatch(
        (x, 0.16),
        0.31,
        0.68,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.text(
        x + 0.155,
        0.70,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontweight="bold",
        color=edgecolor,
    )
    ax.text(
        x + 0.155,
        0.47,
        dimension,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=PALETTE["neutral"],
    )
    ax.text(
        x + 0.155,
        0.28,
        decay,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=PALETTE["prediction"],
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )


def _draw_panel_c(ax: Axes) -> None:
    ax.set_axis_off()
    _panel_label(ax, "c")
    ax.text(
        0.5,
        0.98,
        "Stationary-set dimension and Hessian rank control decay",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontweight="bold",
        fontsize=8.4,
    )
    _mechanism_box(
        ax,
        0.04,
        "critical ring",
        "stationary dimension 1",
        "generic decay  t⁻¹⁄²",
        PALETTE["isotropic"],
    )
    _mechanism_box(
        ax,
        0.65,
        "isolated Morse points",
        "stationary dimension 0",
        "generic decay  t⁻¹",
        PALETTE["anisotropic"],
    )
    arrow = FancyArrowPatch(
        (0.39, 0.50),
        (0.61, 0.50),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.0,
        color=PALETTE["anisotropic"],
    )
    ax.add_patch(arrow)
    ax.text(
        0.50,
        0.59,
        "weak anisotropy",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=PALETTE["anisotropic"],
    )


def build(
    data_dir: Path,
    output_dir: Path,
    source_dir: Path,
) -> dict[str, Path]:
    """Build the four publication exports and exact panel-level source data."""

    iso, _iso_metadata = load_figure_artifact(data_dir, "isotropic_zgv")
    angular, _angular_metadata = load_figure_artifact(data_dir, "angular_sensitivity")
    points, points_metadata = load_figure_artifact(data_dir, "critical_points")
    _assert_scientific_contract(iso, angular, points, points_metadata)

    surface_kx, surface_ky = np.meshgrid(points["kx_grid"], points["ky_grid"])
    omega_iso = np.asarray(points["omega_iso_grid"], dtype=float)
    omega_aniso = np.asarray(points["omega_aniso_grid"], dtype=float)
    ring_theta = np.linspace(0.0, 2.0 * np.pi, omega_iso.size, endpoint=True)
    ring_kx = float(iso["kappa0"]) * np.cos(ring_theta)
    ring_ky = float(iso["kappa0"]) * np.sin(ring_theta)
    levels = np.linspace(
        min(float(np.min(omega_iso)), float(np.min(omega_aniso))),
        max(float(np.max(omega_iso)), float(np.max(omega_aniso))),
        9,
    )
    anomaly = _ring_splitting_anomaly(surface_kx, surface_ky, omega_iso, omega_aniso)

    write_source_csv(
        source_dir / "panel_a_ring.csv",
        {
            "kx": surface_kx,
            "ky": surface_ky,
            "omega_iso": omega_iso,
            "ring_kx": ring_kx,
            "ring_ky": ring_ky,
        },
    )
    write_source_csv(
        source_dir / "panel_b_surface.csv",
        {
            "kx": surface_kx,
            "ky": surface_ky,
            "omega_aniso": omega_aniso,
            "angular_anomaly": anomaly,
        },
    )
    write_source_csv(
        source_dir / "panel_b_points.csv",
        {
            "gradient_residual": points["gradient_residual"],
            "hessian_eigenvalue_1": points["hessian_eigenvalues"][:, 0],
            "hessian_eigenvalue_2": points["hessian_eigenvalues"][:, 1],
            "kind": points["kind"],
            "kx": points["kx"],
            "ky": points["ky"],
            "morse_index": points["morse_index"],
            "omega": points["omega"],
            "theta": points["theta"],
        },
    )

    apply_publication_style()
    fig = plt.figure(layout="constrained", facecolor="white")
    axes = fig.subplot_mosaic(
        [["a", "b"], ["c", "c"]],
        height_ratios=[1.0, 0.42],
        # Equal widths: panels a and b show the same k-space window and are
        # meant to be compared directly, so they must render at one common
        # scale.  Unequal ratios drew the anisotropic panel ~17% larger and
        # made the ring and the split points look differently sized.
        width_ratios=[1.0, 1.0],
    )
    _draw_panel_a(
        axes["a"],
        surface_kx,
        surface_ky,
        omega_iso,
        ring_kx,
        ring_ky,
        levels,
    )
    _draw_panel_b(
        axes["b"],
        surface_kx,
        surface_ky,
        anomaly,
        points,
        levels,
    )
    _draw_panel_c(axes["c"])
    return save_publication_figure(
        fig,
        output_dir / "figure_01_geometry_mechanism",
        _SPEC,
    )
