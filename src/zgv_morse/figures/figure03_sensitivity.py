"""Main Figure 3: elastic sensitivity and the cubic fourfold potential."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .common import (
    FigureSpec,
    PALETTE,
    apply_publication_style,
    load_figure_artifact,
    save_publication_figure,
    write_source_csv,
)


_SPEC: Final = FigureSpec(
    "3",
    "cubic anisotropy produces the predicted fourfold unfolding potential",
    "quantitative grid",
    183.0,
    # Raised from 125 mm: a polar axes cannot exceed its row height, so the
    # rose was capped well below the width of its neighbour.  The extra
    # height lets the top row grow and the rose fill its cell.
    158.0,
)


@dataclass(frozen=True, slots=True)
class _Evidence:
    """Coefficient-level quantities checked before any output is created."""

    V0: float
    V4: float
    Q4: float
    angular_period: float
    relative_V_error: float
    relative_B_error: float


def _finite_scalar(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _metadata_mapping(metadata: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = metadata.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"validated metadata must contain the {name} mapping")
    return value


def _metadata_scalar(metadata: Mapping[str, Any], name: str) -> float:
    if name not in metadata:
        raise ValueError(f"validated metadata must contain {name}")
    return _finite_scalar(metadata[name], f"metadata {name}")


def _relative_error(reference: NDArray[np.generic], comparison: NDArray[np.generic]) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator <= 0.0:
        raise ValueError("analytic sensitivity norm must be positive")
    return float(np.linalg.norm(comparison - reference) / denominator)


def _validate_lineage(
    sensitivity_metadata: Mapping[str, Any],
    convergence_metadata: Mapping[str, Any],
) -> None:
    inputs = _metadata_mapping(convergence_metadata, "input_artifacts")
    sensitivity_input = inputs.get("sensitivity")
    if not isinstance(sensitivity_input, Mapping):
        raise ValueError("convergence lineage must register the sensitivity artifact")
    expected = {
        "artifact": sensitivity_metadata.get("artifact"),
        "cache_key": sensitivity_metadata.get("cache_key"),
        "output_sha256": sensitivity_metadata.get("output_sha256"),
    }
    observed = {name: sensitivity_input.get(name) for name in expected}
    if observed != expected or expected["artifact"] != "angular_sensitivity":
        raise ValueError("convergence lineage does not match angular_sensitivity")


def _validate_evidence(
    sensitivity: dict[str, NDArray[np.generic]],
    sensitivity_metadata: Mapping[str, Any],
    convergence: dict[str, NDArray[np.generic]],
    convergence_metadata: Mapping[str, Any],
) -> _Evidence:
    """Reject convention, reconstruction, and validation failures pre-render."""

    V0 = _finite_scalar(sensitivity["V0"], "V0")
    V4 = _finite_scalar(sensitivity["V4"], "V4")
    if V4 == 0.0:
        raise ValueError("V4 must be nonzero for a resolved cubic unfolding")

    theta = np.asarray(sensitivity["theta"], dtype=np.float64)
    if theta.size < 4 or not np.all(theta[1:] > theta[:-1]):
        raise ValueError("theta must be a strictly increasing angular grid")
    angular_steps = theta[1:] - theta[:-1]
    angular_step = float(angular_steps[0])
    np.testing.assert_allclose(
        angular_steps,
        angular_step,
        rtol=2.0e-14,
        atol=2.0e-15,
        err_msg="theta must use one uniform angular origin",
    )
    angular_period = angular_step * theta.size
    if not np.isclose(theta[0], 0.0, rtol=0.0, atol=2.0e-15) or not np.isclose(
        angular_period, 2.0 * np.pi, rtol=2.0e-14, atol=2.0e-15
    ):
        raise ValueError("theta must use the registered zero origin and full period")

    reconstruction_formula = V0 + V4 * np.cos(4.0 * theta)
    np.testing.assert_allclose(
        sensitivity["V_reconstruction"],
        reconstruction_formula,
        rtol=2.0e-14,
        atol=2.0e-15,
        err_msg="fourfold reconstruction must equal V0 + V4*cos(4*theta)",
    )
    centered_norm = float(np.linalg.norm(sensitivity["V"] - V0))
    if centered_norm <= 0.0:
        raise ValueError("fourfold reconstruction requires nonconstant V(theta)")
    reconstruction_error = float(
        np.linalg.norm(sensitivity["V"] - sensitivity["V_reconstruction"]) / centered_norm
    )

    sensitivity_tolerances = _metadata_mapping(sensitivity_metadata, "tolerances")
    registered_tolerance = _metadata_scalar(
        sensitivity_tolerances, "configured_relative_sensitivity"
    )
    if registered_tolerance <= 0.0:
        raise ValueError("registered sensitivity tolerance must be positive")
    if reconstruction_error >= registered_tolerance:
        raise ValueError("fourfold reconstruction exceeds the registered tolerance")
    reported_leakage = _metadata_scalar(sensitivity_tolerances, "non_cubic_leakage")
    if not np.isclose(
        reconstruction_error,
        reported_leakage,
        rtol=2.0e-12,
        atol=2.0e-15,
    ):
        raise ValueError("fourfold reconstruction disagrees with validated metadata")

    harmonic_order = np.asarray(sensitivity["harmonic_order"], dtype=np.int64)
    harmonic_amplitude = np.asarray(sensitivity["harmonic_amplitude"], dtype=np.float64)
    if not np.all(harmonic_order[1:] > harmonic_order[:-1]):
        raise ValueError("harmonic orders must be strictly increasing")
    if not np.all(harmonic_amplitude > 0.0):
        raise ValueError("all stored harmonic amplitudes must be positive on the log axis")
    is_m4 = harmonic_order == 4
    if np.count_nonzero(is_m4) != 1 or not np.isclose(
        float(harmonic_amplitude[is_m4][0]),
        abs(V4),
        rtol=2.0e-14,
        atol=2.0e-15,
    ):
        raise ValueError("m=4 harmonic must equal the stored V4 coefficient")

    epsilon = np.asarray(sensitivity["epsilon"], dtype=np.float64)
    delta_c = np.asarray(sensitivity["delta_c"], dtype=np.float64)
    physical_shift = np.asarray(sensitivity["physical_V4_shift"], dtype=np.float64)
    np.testing.assert_allclose(
        physical_shift,
        epsilon * V4,
        rtol=2.0e-14,
        atol=2.0e-15,
        err_msg="physical shift must apply epsilon exactly once",
    )
    nonzero = epsilon != 0.0
    if not np.any(nonzero):
        raise ValueError("Q4 coefficient relation requires a nonzero epsilon")
    delta_first = delta_c[nonzero] / epsilon[nonzero]
    delta_first_reference = float(delta_first[0])
    if delta_first_reference == 0.0 or not np.isfinite(delta_first_reference):
        raise ValueError("Q4 coefficient relation requires nonzero Delta_C^(1)")
    np.testing.assert_allclose(
        delta_first,
        delta_first_reference,
        rtol=2.0e-14,
        atol=2.0e-15,
        err_msg="Q4 coefficient relation requires Delta_C(epsilon)=epsilon*Delta_C^(1)",
    )
    if np.any(~nonzero):
        np.testing.assert_allclose(
            delta_c[~nonzero],
            0.0,
            rtol=0.0,
            atol=2.0e-15,
            err_msg="Q4 coefficient relation requires Delta_C(0)=0",
        )
    Q4 = V4 / delta_first_reference
    np.testing.assert_allclose(
        physical_shift,
        Q4 * delta_c,
        rtol=2.0e-14,
        atol=2.0e-15,
        err_msg="Q4 coefficient relation must satisfy epsilon*V4=Q4*Delta_C",
    )

    relative_V_error = _relative_error(sensitivity["V"], sensitivity["V_fd"])
    if relative_V_error >= registered_tolerance:
        raise ValueError("analytic/finite-difference V exceeds the registered tolerance")
    relative_B_error = _relative_error(sensitivity["B"], sensitivity["B_fd"])
    if relative_B_error >= registered_tolerance:
        raise ValueError("analytic/finite-difference B exceeds the registered tolerance")
    for name, calculated in (
        ("relative_V_error", relative_V_error),
        ("relative_B_error", relative_B_error),
    ):
        reported = _metadata_scalar(sensitivity_tolerances, name)
        if not np.isclose(calculated, reported, rtol=2.0e-12, atol=2.0e-15):
            raise ValueError(f"{name} disagrees with validated sensitivity metadata")

    step = np.asarray(convergence["sensitivity_step"], dtype=np.float64)
    V4_error = np.asarray(convergence["V4_fd_error"], dtype=np.float64)
    B_error = np.asarray(convergence["B_fd_error"], dtype=np.float64)
    if (
        not np.all(step > 0.0)
        or not np.all(step[1:] < step[:-1])
        or not np.all(V4_error > 0.0)
        or not np.all(B_error > 0.0)
        or not np.all(V4_error[1:] < V4_error[:-1])
        or not np.all(B_error[1:] < B_error[:-1])
        or not np.all(V4_error[-2:] < registered_tolerance)
        or not np.all(B_error[-2:] < registered_tolerance)
    ):
        raise ValueError("finite-difference step convergence must resolve both coefficient errors")
    convergence_tolerances = _metadata_mapping(convergence_metadata, "tolerances")
    for name, value in (
        ("final_V4_fd_relative_error", float(V4_error[-1])),
        ("final_B_fd_relative_error", float(B_error[-1])),
    ):
        if not np.isclose(
            value,
            _metadata_scalar(convergence_tolerances, name),
            rtol=2.0e-12,
            atol=2.0e-15,
        ):
            raise ValueError("finite-difference step convergence disagrees with validated metadata")

    _validate_lineage(sensitivity_metadata, convergence_metadata)
    return _Evidence(
        V0=V0,
        V4=V4,
        Q4=Q4,
        angular_period=angular_period,
        relative_V_error=relative_V_error,
        relative_B_error=relative_B_error,
    )


def _closed(values: NDArray[np.generic]) -> NDArray[np.generic]:
    return np.concatenate((values, values[:1]))


def _write_source_data(
    source_dir: Path,
    sensitivity: dict[str, NDArray[np.generic]],
    convergence: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
) -> None:
    theta_closed = np.concatenate(
        (
            sensitivity["theta"],
            sensitivity["theta"][:1] + evidence.angular_period,
        )
    )
    write_source_csv(
        source_dir / "panel_a_polar.csv",
        {
            "theta": theta_closed,
            "V_minus_V0": _closed(sensitivity["V"]) - evidence.V0,
            "fourfold_reconstruction": _closed(sensitivity["V_reconstruction"]) - evidence.V0,
        },
    )
    is_m4 = np.asarray(sensitivity["harmonic_order"] == 4, dtype=np.int64)
    write_source_csv(
        source_dir / "panel_b_harmonics.csv",
        {
            "harmonic_order": sensitivity["harmonic_order"],
            "harmonic_amplitude": sensitivity["harmonic_amplitude"],
            "is_m4": is_m4,
        },
    )
    count = sensitivity["epsilon"].size
    write_source_csv(
        source_dir / "panel_c_physical_shift.csv",
        {
            "epsilon": sensitivity["epsilon"],
            "delta_c": sensitivity["delta_c"],
            "physical_epsilon_V4_shift": sensitivity["physical_V4_shift"],
            "Q4": np.full(count, evidence.Q4),
            "Q4_delta_c_prediction": evidence.Q4 * sensitivity["delta_c"],
        },
    )
    write_source_csv(
        source_dir / "panel_d_angular_fd.csv",
        {name: sensitivity[name] for name in ("theta", "V", "V_fd", "B", "B_fd")},
    )
    write_source_csv(
        source_dir / "panel_d_step_convergence.csv",
        {name: convergence[name] for name in ("sensitivity_step", "V4_fd_error", "B_fd_error")},
    )


def _panel_label(ax: Axes, label: str, *, x: float = -0.14) -> None:
    """Panel tag, set outside the axes at the bottom left in parentheses."""

    ax.text(
        -0.14,
        -0.16,
        f"({label})",
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        ha="right",
        va="top",
    )


def _settle_constrained_layout(fig: Figure, spec: FigureSpec) -> None:
    """Iterate the layout engine to a fixed point before multi-format export."""

    fig.set_size_inches(spec.width_mm / 25.4, spec.height_mm / 25.4)
    previous: tuple[tuple[float, float, float, float], ...] | None = None
    for _iteration in range(12):
        fig.canvas.draw()
        current = tuple(
            tuple(float(value) for value in ax.get_position().bounds) for ax in fig.axes
        )
        if previous is not None and max(
            abs(value - prior)
            for bounds, prior_bounds in zip(current, previous, strict=True)
            for value, prior in zip(bounds, prior_bounds, strict=True)
        ) <= 16.0 * np.finfo(np.float64).eps:
            # Constrained layout can alternate by one floating-point ulp after
            # convergence. Freeze the roundoff-level fixed point so every
            # export backend sees exactly the same axes geometry.
            fig.set_layout_engine(None)
            return
        previous = current
    raise RuntimeError("constrained layout did not converge to a deterministic fixed point")


def _angular_ticks(ax: Axes, period: float) -> None:
    ax.set_xlim(0.0, period)
    ax.set_xticks((0.0, period / 4.0, period / 2.0, 3.0 * period / 4.0, period))
    ax.set_xticklabels(("0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"))


def _draw_polar(
    ax: Axes,
    sensitivity: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
) -> None:
    theta = np.concatenate(
        (
            sensitivity["theta"],
            sensitivity["theta"][:1] + evidence.angular_period,
        )
    )
    centered = _closed(sensitivity["V"]) - evidence.V0
    reconstruction = _closed(sensitivity["V_reconstruction"]) - evidence.V0
    scale = float(max(np.max(np.abs(centered)), np.max(np.abs(reconstruction))))
    radial_offset = 1.25 * scale
    ax.plot(
        theta,
        centered + radial_offset,
        color=PALETTE["anisotropic"],
        linewidth=1.6,
        clip_on=False,
        label=r"$V(\theta)-V_0$",
    )
    ax.plot(
        theta,
        reconstruction + radial_offset,
        color=PALETTE["prediction"],
        linewidth=1.15,
        linestyle=(0, (4, 2)),
        clip_on=False,
        label=r"$V_4\cos 4\theta$",
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(
        (
            0.0,
            evidence.angular_period / 4.0,
            evidence.angular_period / 2.0,
            3.0 * evidence.angular_period / 4.0,
        )
    )
    ax.set_xticklabels((r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"))
    ax.set_ylim(radial_offset - 1.12 * scale, radial_offset + 1.90 * scale)
    # On the 45 deg label ray the rose sits at its innermost radius, which is
    # exactly the -scale tick.  That one tick is therefore left unlabelled so
    # no tick text is printed on top of the curve; the ring stays drawn and
    # the scale is fixed by the remaining two labels.
    # A fourth, unlabelled tick beyond +scale carries the outer frame, so the
    # "+scale" label no longer sits on the outermost ring and has clear space
    # around it.  The caption relies on these labels for the radial scale, so
    # they are kept rather than dropped.
    signed_ticks = (-scale, 0.0, scale, 1.90 * scale)
    ax.set_yticks(tuple(radial_offset + value for value in signed_ticks))
    # Radial tick labels give the amplitude scale.  They are placed on the
    # 45 deg ray, where the rose is at its innermost radius, and only the
    # two informative rings are labelled.
    ax.set_yticklabels(("", "0", f"+{scale:.3g}", ""), fontsize=8.4)
    ax.set_rlabel_position(45.0)
    ax.grid(color="#DADADA", linewidth=0.45)
    for gridline in (*ax.get_xgridlines(), *ax.get_ygridlines()):
        gridline.set_clip_on(False)
    # Outside the polar axes: inside, the key always landed on a petal, a
    # ring or the boundary circle.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        fontsize=8.4,
        handlelength=1.8,
        columnspacing=3.0,
        handletextpad=0.4,
        borderpad=0.2,
        frameon=False,
    )
    _panel_label(ax, "a", x=-0.08)


def _draw_harmonics(
    ax: Axes,
    sensitivity: dict[str, NDArray[np.generic]],
) -> None:
    order = np.asarray(sensitivity["harmonic_order"], dtype=np.int64)
    amplitude = np.asarray(sensitivity["harmonic_amplitude"], dtype=np.float64)
    is_m4 = order == 4
    colors = np.where(is_m4, PALETTE["minimum"], "#A9A9A9")
    floor = float(np.min(amplitude)) / 2.5
    ax.vlines(
        order,
        floor,
        amplitude,
        colors=colors,
        linewidth=1.0,
        clip_on=False,
    )
    ax.scatter(
        order[~is_m4],
        amplitude[~is_m4],
        s=14,
        marker="o",
        facecolor="#A9A9A9",
        edgecolor="white",
        linewidth=0.5,
        clip_on=False,
        zorder=3,
    )
    ax.scatter(
        order[is_m4],
        amplitude[is_m4],
        s=30,
        marker="o",
        facecolor=PALETTE["minimum"],
        edgecolor="white",
        linewidth=0.55,
        clip_on=False,
        zorder=4,
    )
    m4_index = int(np.flatnonzero(is_m4)[0])
    ax.set_yscale("log")
    ax.set_ylim(floor, float(np.max(amplitude)) * 2.5)
    ax.set_xticks(order[::2])
    ax.set_xlabel("angular harmonic order $m$")
    ax.set_ylabel(r"harmonic amplitude $|V_m|$")
    ax.grid(axis="y", which="major", color="#E2E2E2", linewidth=0.45)
    for gridline in ax.get_ygridlines():
        gridline.set_clip_on(False)
    _panel_label(ax, "b")


def _draw_physical_shift(
    ax: Axes,
    sensitivity: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
) -> None:
    delta_c = np.asarray(sensitivity["delta_c"], dtype=np.float64)
    physical_shift = np.asarray(sensitivity["physical_V4_shift"], dtype=np.float64)
    order = np.argsort(delta_c, kind="stable")
    prediction = evidence.Q4 * delta_c
    ax.plot(
        delta_c[order],
        prediction[order],
        color=PALETTE["prediction"],
        linewidth=1.15,
        linestyle=(0, (4, 2)),
        clip_on=False,
        label=r"$Q_4\Delta_C$",
    )
    ax.scatter(
        delta_c,
        physical_shift,
        s=25,
        marker="o",
        facecolor=PALETTE["minimum"],
        edgecolor="white",
        linewidth=0.55,
        clip_on=False,
        zorder=3,
        label=r"stored $\varepsilon V_4$",
    )
    ax.axhline(0.0, color="#BDBDBD", linewidth=0.5, clip_on=False, zorder=0)
    ax.axvline(0.0, color="#BDBDBD", linewidth=0.5, clip_on=False, zorder=0)
    ax.set_xlabel(r"cubic stiffness perturbation $\Delta_C(\varepsilon)$")
    ax.set_ylabel(r"physical frequency shift $\varepsilon V_4$")
    ax.grid(color="#E8E8E8", linewidth=0.45)
    for gridline in (*ax.get_xgridlines(), *ax.get_ygridlines()):
        gridline.set_clip_on(False)
    ax.legend(loc="upper left", fontsize=8.4, handlelength=2.0)
    ax.text(
        0.98,
        0.05,
        f"$Q_4={evidence.Q4:.5g}$; one application of $\\varepsilon$",
        transform=ax.transAxes,
        fontsize=8.4,
        color=PALETTE["neutral"],
        ha="right",
        va="bottom",
    )
    _panel_label(ax, "c")


def _draw_validation(
    ax: Axes,
    step_axis: Axes,
    sensitivity: dict[str, NDArray[np.generic]],
    convergence: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
) -> None:
    theta = np.asarray(sensitivity["theta"], dtype=np.float64)
    curves = (
        (sensitivity["V"], "$V$ analytic", PALETTE["anisotropic"], "-", None),
        (
            sensitivity["V_fd"],
            "$V$ centered FD",
            PALETTE["anisotropic"],
            "None",
            "o",
        ),
        (sensitivity["B"], "$B$ analytic", PALETTE["saddle"], "-", None),
        (
            sensitivity["B_fd"],
            "$B$ centered FD",
            PALETTE["saddle"],
            "None",
            "s",
        ),
    )
    for values, label, color, linestyle, marker in curves:
        ax.plot(
            theta,
            values,
            color=color,
            linewidth=1.05,
            linestyle=linestyle,
            marker=marker,
            markersize=2.4,
            markevery=8,
            markerfacecolor="white",
            markeredgewidth=0.55,
            clip_on=False,
            label=label,
        )
    # The four sinusoids fill the default view, leaving no room for a key.
    # Extending the lower limit opens a clear band at the bottom of the
    # panel that the legend occupies without covering any curve.
    finite = np.concatenate([
        np.asarray(sensitivity[key], dtype=np.float64)
        for key in ("V", "V_fd", "B", "B_fd")
    ])
    lowest = float(np.min(finite))
    highest = float(np.max(finite))
    span = highest - lowest
    ax.set_ylim(lowest - 0.75 * span, highest + 0.06 * span)
    _angular_ticks(ax, evidence.angular_period)
    ax.set_xlabel(r"angle $\theta$ from the registered cubic axis", labelpad=1.0)
    ax.set_ylabel("sensitivity coefficient")
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.39, 0.21),
        ncol=2,
        fontsize=8.4,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
        handletextpad=0.4,
        labelspacing=0.25,
    )

    step = convergence["sensitivity_step"]
    step_axis.loglog(
        step,
        convergence["V4_fd_error"],
        color=PALETTE["minimum"],
        marker="o",
        markersize=2.2,
        linewidth=0.85,
        clip_on=False,
        label=r"$V_4$",
    )
    step_axis.loglog(
        step,
        convergence["B_fd_error"],
        color=PALETTE["saddle"],
        marker="s",
        markersize=2.2,
        linewidth=0.85,
        clip_on=False,
        label=r"$B$",
    )
    step_axis.invert_xaxis()
    step_axis.set_xlabel(r"finite-difference step $h_{\rm FD}$", fontsize=8.4, labelpad=0.5)
    step_axis.set_ylabel("relative error", fontsize=8.4, labelpad=4.0)
    step_axis.tick_params(axis="both", which="both", labelsize=8.4, length=2.2)
    # Lifted clear of the rotated y-axis label, whose box reaches the
    # default panel-label slot on this panel.
    _panel_label(ax, "d", x=-0.20)


def build(
    data_dir: Path,
    output_dir: Path,
    source_dir: Path,
) -> dict[str, Path]:
    """Build Figure 3 only from validated sensitivity and convergence artifacts."""

    sensitivity, sensitivity_metadata = load_figure_artifact(data_dir, "angular_sensitivity")
    convergence, convergence_metadata = load_figure_artifact(data_dir, "convergence")
    evidence = _validate_evidence(
        sensitivity,
        sensitivity_metadata,
        convergence,
        convergence_metadata,
    )
    _write_source_data(source_dir, sensitivity, convergence, evidence)

    apply_publication_style()
    plt.rcParams["path.simplify"] = False
    fig = plt.figure(layout="constrained", facecolor="white")

    # A polar axes stays circular, so its size is set by whichever of its
    # cell's width or height is smaller.  Here the row height is the binding
    # constraint, so widening the cell only produced dead space: the taller
    # top row below is what actually enlarges the rose, and the cell is kept
    # just wide enough to hold it plus its radial labels.
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(0.72, 1.28),
        height_ratios=(1.45, 0.62, 0.38),
        hspace=0.08,
        wspace=0.06,
    )
    polar = fig.add_subplot(grid[0, 0], projection="polar")
    harmonics = fig.add_subplot(grid[0, 1])
    physical = fig.add_subplot(grid[1:, 0])
    validation = fig.add_subplot(grid[1, 1])
    step_axis = fig.add_subplot(grid[2, 1])
    _draw_polar(polar, sensitivity, evidence)
    _draw_harmonics(harmonics, sensitivity)
    _draw_physical_shift(physical, sensitivity, evidence)
    _draw_validation(validation, step_axis, sensitivity, convergence, evidence)
    _settle_constrained_layout(fig, _SPEC)
    return save_publication_figure(
        fig,
        output_dir / "figure_03_angular_sensitivity",
        _SPEC,
    )
