"""Main Figure 2: exact isotropic ZGV foundation."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray

from .common import (
    FigureSpec,
    PALETTE,
    apply_publication_style,
    load_figure_inputs,
    save_publication_figure,
    write_source_csv,
)


_SPEC: Final = FigureSpec(
    "2",
    "the chosen ZGV ring has a nondegenerate positive radial curvature",
    "quantitative grid",
    183.0,
    195.0,
)
_REGISTERED_ISOTROPIC_TOLERANCE: Final = 1.0e-7
_REGISTERED_CURVATURE_TOLERANCE: Final = 1.0e-4
_REGISTERED_EIGEN_RESIDUAL_TOLERANCE: Final = 1.0e-10
_REGISTERED_HERMITIAN_TOLERANCE: Final = 1.0e-12
_REGISTERED_ORTHOGONALITY_TOLERANCE: Final = 1.0e-10


def _scalar(arrays: dict[str, NDArray[np.generic]], name: str) -> float:
    value = float(arrays[name])
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_evidence(
    isotropic: dict[str, NDArray[np.generic]],
    convergence: dict[str, NDArray[np.generic]],
) -> tuple[float, float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Validate the coefficient-level claim before any plotting side effect."""

    kappa0 = _scalar(isotropic, "kappa0")
    omega0 = _scalar(isotropic, "omega0")
    curvature = _scalar(isotropic, "curvature_a")
    if curvature <= 0.0:
        raise ValueError("the selected branch must have positive radial curvature")

    local_q = np.asarray(isotropic["local_q"], dtype=np.float64)
    quadratic = omega0 + 0.5 * curvature * local_q**2
    np.testing.assert_allclose(
        isotropic["local_quadratic"],
        quadratic,
        rtol=2.0e-15,
        atol=2.0e-15,
        err_msg=(
            "stored local_quadratic must equal the coefficient-level formula "
            "omega0 + 0.5*curvature_a*q**2"
        ),
    )

    final = slice(-2, None)
    final_two_pass = (
        np.all(
            convergence["omega0_error"][final]
            < _REGISTERED_ISOTROPIC_TOLERANCE
        )
        and np.all(
            convergence["kappa0_error"][final]
            < _REGISTERED_ISOTROPIC_TOLERANCE
        )
        and np.all(
            convergence["curvature_error"][final]
            < _REGISTERED_CURVATURE_TOLERANCE
        )
        and np.all(
            convergence["eigen_residual"][final]
            < _REGISTERED_EIGEN_RESIDUAL_TOLERANCE
        )
        and np.all(
            convergence["hermitian_residual"][final]
            < _REGISTERED_HERMITIAN_TOLERANCE
        )
        and np.all(
            convergence["mass_orthogonality"][final]
            < _REGISTERED_ORTHOGONALITY_TOLERANCE
        )
        and np.all(
            convergence["eigengap"][final]
            > 10.0
            * np.maximum(
                convergence["eigen_residual"][final],
                np.finfo(float).eps,
            )
        )
    )
    if not final_two_pass:
        raise ValueError(
            "the final two polynomial orders must satisfy all registered convergence gates"
        )

    mode_magnitude = np.abs(np.asarray(isotropic["mode_u"]))
    mode_scale = float(np.max(mode_magnitude))
    squared_displacement = np.asarray(
        isotropic["mode_squared_displacement"], dtype=np.float64
    )
    squared_displacement_scale = float(np.max(squared_displacement))
    if (
        not np.isfinite(mode_magnitude).all()
        or not np.isfinite(squared_displacement).all()
        or np.any(squared_displacement < 0.0)
        or mode_scale <= 0.0
        or squared_displacement_scale <= 0.0
    ):
        raise ValueError("the through-thickness mode profile must be finite and normalizable")
    return (
        kappa0,
        omega0,
        curvature,
        mode_magnitude / mode_scale,
        squared_displacement / squared_displacement_scale,
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


def _direct_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    offset: tuple[float, float],
    *,
    align: str = "left",
) -> None:
    """Label a curve at a point.  ``align`` lets a label sit to the left of
    its anchor so long strings stay inside the axes."""

    ax.annotate(
        text,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        color=color,
        fontsize=8.4,
        ha=align,
        va="center",
        annotation_clip=False,
    )


def _leader_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    position: tuple[float, float],
) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xycoords="data",
        xytext=position,
        textcoords=ax.transAxes,
        color=color,
        fontsize=8.4,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.45},
        annotation_clip=False,
    )


def _write_source_data(
    source_dir: Path,
    isotropic: dict[str, NDArray[np.generic]],
    convergence: dict[str, NDArray[np.generic]],
    kappa0: float,
    omega0: float,
    curvature: float,
    mode_magnitude: NDArray[np.float64],
    squared_displacement: NDArray[np.float64],
) -> None:
    branch_count, sample_count = isotropic["omega_symmetric"].shape
    branch_rows = branch_count * sample_count
    write_source_csv(
        source_dir / "panel_a_branches.csv",
        {
            "branch_label": np.repeat(
                isotropic["branch_labels"].astype(str), sample_count
            ),
            "kappa": np.tile(isotropic["kappa"], branch_count),
            "omega": isotropic["omega_symmetric"].reshape(-1),
            "zgv_kappa0": np.full(branch_rows, kappa0),
            "zgv_omega0": np.full(branch_rows, omega0),
        },
    )
    local_count = isotropic["local_q"].size
    write_source_csv(
        source_dir / "panel_b_local_quadratic.csv",
        {
            "curvature_a": np.full(local_count, curvature),
            "local_q": isotropic["local_q"],
            "local_omega": isotropic["local_omega"],
            "local_quadratic": isotropic["local_quadratic"],
            "omega0": np.full(local_count, omega0),
        },
    )
    write_source_csv(
        source_dir / "panel_c_convergence.csv",
        {
            name: convergence[name]
            for name in (
                "polynomial_order",
                "omega0_error",
                "kappa0_error",
                "curvature_error",
                "eigen_residual",
                "hermitian_residual",
                "mass_orthogonality",
                "eigengap",
            )
        },
    )
    write_source_csv(
        source_dir / "panel_d_mode_profile.csv",
        {
            "z_over_h": isotropic["mode_z"],
            "u_x_magnitude_normalized": mode_magnitude[:, 0],
            "u_y_magnitude_normalized": mode_magnitude[:, 1],
            "u_z_magnitude_normalized": mode_magnitude[:, 2],
            "squared_displacement_proxy_normalized": squared_displacement,
        },
    )


def _draw_branches(
    ax: plt.Axes,
    isotropic: dict[str, NDArray[np.generic]],
    kappa0: float,
    omega0: float,
) -> None:
    kappa = isotropic["kappa"]
    branches = isotropic["omega_symmetric"]
    labels = isotropic["branch_labels"].astype(str)
    colors = (PALETTE["anisotropic"], PALETTE["minimum"], PALETTE["neutral"])
    for index, (branch, label) in enumerate(zip(branches, labels, strict=True)):
        color = colors[index % len(colors)]
        ax.plot(kappa, branch, color=color, linewidth=1.5)
        # Anchored in axes coordinates at a measured-clear position above
        # the bowl; offsetting along the curve kept the text on the line.
        ax.text(
            0.52,
            0.92,
            str(label),
            transform=ax.transAxes,
            color=color,
            fontsize=8.4,
            ha="center",
            va="center",
        )
    ax.scatter(
        [kappa0],
        [omega0],
        s=27,
        marker="o",
        facecolor="white",
        edgecolor=PALETTE["saddle"],
        linewidth=1.2,
        zorder=5,
    )
    # Far enough along the rising right branch that the label clears both
    # the marker and the curve; nearer placements touch one or the other.
    # Up and to the left of the marker: measured clear of the branch, which
    # rises steeply on the right of the minimum.
    _direct_label(ax, kappa0, omega0, "ZGV", PALETTE["saddle"], (-20.0, 13.0))
    ax.set_xlabel(r"dimensionless wavenumber $\kappa=kh$")
    ax.set_ylabel(r"dimensionless frequency $\Omega=\omega h/c_T$")
    ax.grid(color="#E8E8E8", linewidth=0.45)


def _draw_local_quadratic(
    ax: plt.Axes,
    isotropic: dict[str, NDArray[np.generic]],
    curvature: float,
) -> None:
    local_q = isotropic["local_q"]
    exact = isotropic["local_omega"]
    quadratic = isotropic["local_quadratic"]
    ax.plot(
        local_q,
        exact,
        color=PALETTE["anisotropic"],
        linewidth=1.2,
        marker="o",
        markersize=2.5,
        markevery=4,
    )
    ax.plot(
        local_q,
        quadratic,
        color=PALETTE["prediction"],
        linewidth=1.35,
        linestyle=(0, (4, 2)),
    )
    positive = np.flatnonzero(local_q > 0.0)
    exact_index = int(positive[len(positive) // 2])
    prediction_index = int(positive[-2])
    # Placed by measurement in the panel's clear left-of-centre band; an
    # offset from a point on the curve kept the text against the line.
    ax.text(
        0.55,
        0.32,
        "exact branch",
        transform=ax.transAxes,
        color=PALETTE["anisotropic"],
        fontsize=8.4,
        ha="center",
        va="center",
    )
    _direct_label(
        ax,
        float(local_q[prediction_index]),
        float(quadratic[prediction_index]),
        "quadratic prediction",
        PALETTE["prediction"],
        (-63.0, 10.0),
    )
    ax.text(
        0.28,
        0.78,
        f"a = {curvature:.4g} > 0",
        transform=ax.transAxes,
        color=PALETTE["anisotropic"],
        fontsize=8.4,
        # Centre-anchored so the placement matches the measured position:
        # a left/bottom anchor grew the text away from the clear spot.
        ha="center",
        va="center",
    )
    ax.set_xlabel(r"radial offset $q=\kappa-\kappa_0$")
    ax.set_ylabel(r"dimensionless frequency $\Omega=\omega h/c_T$")
    ax.grid(color="#E8E8E8", linewidth=0.45)


def _draw_convergence(
    ax: plt.Axes,
    convergence: dict[str, NDArray[np.generic]],
) -> None:
    order = convergence["polynomial_order"]
    primary = (
        (
            "kappa0_error",
            r"$\kappa_0$ error",
            PALETTE["minimum"],
            "o",
            (0.60, 0.67),
        ),
        (
            "omega0_error",
            r"$\omega_0$ error",
            PALETTE["saddle"],
            "s",
            (0.60, 0.26),
        ),
        (
            "curvature_error",
            r"curvature error",
            PALETTE["anisotropic"],
            "D",
            (0.60, 0.92),
        ),
        (
            "eigen_residual",
            r"eigen residual",
            PALETTE["neutral"],
            "^",
            (0.60, 0.60),
        ),
        (
            "mass_orthogonality",
            r"mass orthog.",
            "#8B8B8B",
            "v",
            (0.60, 0.19),
        ),
        (
            "hermitian_residual",
            r"Herm. resid.",
            "#B0B0B0",
            "P",
            (0.60, 0.12),
        ),
    )
    handles: list[Line2D] = []
    # The mass-orthogonality and Hermitian defects sit two decades below the
    # rest and are described in the caption, so they are drawn but left out
    # of the key: seven entries could not fit inside the panel without
    # crossing a trace.
    keyed = {"mass_orthogonality", "hermitian_residual"}
    for name, label, color, marker, _position in primary:
        values = convergence[name]
        (line,) = ax.plot(
            order,
            values,
            color=color,
            linewidth=0.95,
            marker=marker,
            markersize=3.1,
            label=label,
        )
        if name not in keyed:
            handles.append(line)
    ax.set_yscale("log")
    ax.set_xlabel("polynomial order")
    ax.set_ylabel("relative error or residual")
    ax.set_xticks(order)
    ax.grid(which="major", color="#E8E8E8", linewidth=0.45)

    gap_axis = ax.twinx()
    gap = convergence["eigengap"]
    gap_axis.plot(
        order,
        gap,
        color=PALETTE["prediction"],
        linewidth=1.0,
        linestyle=(0, (2, 2)),
    )
    gap_axis.set_ylabel("relative eigengap")
    lower = max(0.0, float(np.min(gap)) * 0.92)
    upper = float(np.max(gap)) * 1.08
    gap_axis.set_ylim(lower, upper)
    gap_axis.spines["top"].set_visible(False)
    (gap_line,) = gap_axis.plot([], [], color=PALETTE["prediction"],
                                linewidth=1.0, linestyle=(0, (2, 2)),
                                label="eigengap (right axis)")
    handles.append(gap_line)
    # One legend for both axes, below the panel so it covers no data.
    # Kept inside the panel: an anchor below the axes spilled the legend
    # past both side margins and out of the panel's own cell.  Seven entries
    # in two columns fill the empty upper-right quadrant, which the curves
    # leave clear because every trace descends to the left.
    # The full-width row leaves a clear band on the right (every trace
    # descends to the left), so the legend sits there in two compact
    # columns at the same size as the rest of the figure text.
    # Above the panel, centred: inside the axes the key always landed on one
    # of the seven traces, and this row spans the full figure width so there
    # is room for four columns outside the data area.
    # Extra decades below the data: the seven traces otherwise fill the
    # view, leaving nowhere inside the panel for the key.
    _lo, _hi = ax.get_ylim()
    ax.set_ylim(_lo * 1e-3, _hi)
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=8.4,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.9,
        handletextpad=0.35,
        labelspacing=0.25,
    )


def _draw_mode_profile(
    ax: plt.Axes,
    z: NDArray[np.generic],
    mode_magnitude: NDArray[np.float64],
    squared_displacement: NDArray[np.float64],
) -> None:
    curves = (
        (mode_magnitude[:, 0], r"$|u_x|$", PALETTE["minimum"], "-", 0),
        (
            mode_magnitude[:, 1],
            r"$|u_y|$",
            PALETTE["saddle"],
            (0, (2, 1)),
            int(np.argmax(mode_magnitude[:, 1])),
        ),
        (
            mode_magnitude[:, 2],
            r"$|u_z|$",
            PALETTE["anisotropic"],
            (0, (4, 2)),
            int(np.argmax(mode_magnitude[:, 2])),
        ),
        (
            squared_displacement,
            "squared displacement",
            PALETTE["neutral"],
            (0, (1, 1)),
            squared_displacement.size - 1,
        ),
    )
    # Four labels floating among four overlapping profiles read as clutter,
    # so they are collected into one legend in the panel's empty lower-left
    # corner, where the log axis leaves the profiles far to the right.
    for values, label, color, linestyle, _label_index in curves:
        ax.plot(
            values,
            z,
            color=color,
            linewidth=1.25,
            linestyle=linestyle,
            label=label,
        )
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.5, 0.82),
        ncol=4,
        fontsize=8.4,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.6,
        labelspacing=0.4,
    )
    positive_values = np.concatenate(
        (
            mode_magnitude[mode_magnitude > 0.0],
            squared_displacement[squared_displacement > 0.0],
        )
    )
    ax.set_xscale("log")
    ax.set_xlim(float(np.min(positive_values)) * 0.6, 1.5)
    ax.set_ylim(float(np.min(z)), float(np.max(z)))
    ax.set_xlabel("normalized magnitude or squared-displacement proxy")
    ax.set_ylabel(r"thickness coordinate $z/h$")
    ax.grid(which="major", color="#E8E8E8", linewidth=0.45)


def build(
    data_dir: Path,
    output_dir: Path,
    source_dir: Path,
) -> dict[str, Path]:
    """Build Figure 2 exclusively from the two validated registered artifacts."""

    isotropic = load_figure_inputs(data_dir, "isotropic_zgv")
    convergence = load_figure_inputs(data_dir, "convergence")
    (
        kappa0,
        omega0,
        curvature,
        mode_magnitude,
        squared_displacement,
    ) = _validate_evidence(isotropic, convergence)
    _write_source_data(
        source_dir,
        isotropic,
        convergence,
        kappa0,
        omega0,
        curvature,
        mode_magnitude,
        squared_displacement,
    )

    apply_publication_style()
    fig = plt.figure(layout="constrained", facecolor="white")
    # Panel c carries seven traces plus a twin axis, which is more than a
    # half-width cell can show clearly, and it left no room for a legend.
    # Giving it a full-width row spreads the traces over twice the
    # horizontal range and opens the space its legend now occupies.
    axes = fig.subplot_mosaic(
        [["a", "b"], ["c", "c"], ["d", "d"]],
        width_ratios=(0.96, 1.04),
        height_ratios=(1.15, 1.45, 0.95),
    )
    _draw_branches(axes["a"], isotropic, kappa0, omega0)
    _draw_local_quadratic(axes["b"], isotropic, curvature)
    _draw_convergence(axes["c"], convergence)
    _draw_mode_profile(
        axes["d"], isotropic["mode_z"], mode_magnitude, squared_displacement
    )
    for label, ax in axes.items():
        _panel_label(ax, label)
    return save_publication_figure(
        fig,
        output_dir / "figure_02_isotropic_zgv",
        _SPEC,
    )
