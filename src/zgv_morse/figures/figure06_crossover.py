"""Main Figure 6: two numerical asymptotic slices and their proved overlap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
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
from zgv_morse.workflows.green import (
    _MORSE_COMPARISON_START as _REGISTERED_UNIFORM_TIME_START,
    _UNIFORM_TAU_MAXIMUM as _REGISTERED_UNIFORM_TAU_MAXIMUM,
)


_SPEC: Final = FigureSpec(
    "6",
    "joint-limit and fixed-anisotropy slices are connected by a proved overlap",
    "asymmetric mixed-modality figure",
    183.0,
    185.0,
)
_MAX_COLLAPSE_ERROR: Final = 0.08
_MAX_CROSSOVER_SLOPE_ERROR: Final = 0.10
_MAX_MORSE_RMS_ERROR: Final = 0.055
_MAX_MORSE_POINTWISE_ERROR: Final = 0.17
_MAX_CANCELLATION_RMS: Final = 0.09
_MAX_FREQUENCY_SEPARATION_ERROR: Final = 0.03
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
    uniform_indices: NDArray[np.int64]
    fixed_index: int
    v4: float
    crossover_slope: float
    comparison_start: float
    comparison_stop: float
    morse_rms_error: float
    morse_cancellation_rms: float
    cancellation_fraction: float
    cancellation_indices: NDArray[np.int64]
    morse_coherence: NDArray[np.float64]
    morse_coherence_threshold: float
    morse_normalization: float
    collapse_error: float
    late_fit_centers: NDArray[np.float64]
    late_fit_rms: NDArray[np.float64]
    registered_uniform_window: NDArray[np.bool_]


def _mapping(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"green-crossover metadata must contain mapping {key!r}")
    return value


def _metric(container: dict[str, Any], key: str) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"registered metric {key!r} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"registered metric {key!r} must be finite")
    return result


def _unique_index(values: NDArray[np.generic], target: float, label: str) -> int:
    matches = np.flatnonzero(np.asarray(values, dtype=np.float64) == target)
    if matches.size != 1:
        raise ValueError(f"{label} must identify exactly one registered epsilon row")
    return int(matches[0])


def _validate_context(
    green_metadata: dict[str, Any],
    angular_metadata: dict[str, Any],
    convergence_metadata: dict[str, Any],
) -> None:
    identities = (
        (green_metadata, "green_crossover"),
        (angular_metadata, "angular_sensitivity"),
        (convergence_metadata, "convergence"),
    )
    for metadata, identity in identities:
        if metadata.get("artifact") != identity:
            raise ValueError(f"scientific context: expected artifact {identity}")
    for field in _CONTEXT_FIELDS:
        values = {metadata.get(field) for metadata, _identity in identities}
        if len(values) != 1:
            raise ValueError(f"scientific context: registered {field} values differ")

    expected_sensitivity = {
        "artifact": "angular_sensitivity",
        "cache_key": angular_metadata.get("cache_key"),
        "output_sha256": angular_metadata.get("output_sha256"),
    }
    for metadata, identity in (
        (green_metadata, "green_crossover"),
        (convergence_metadata, "convergence"),
    ):
        inputs = metadata.get("input_artifacts")
        if not isinstance(inputs, dict) or inputs.get("sensitivity") != (
            expected_sensitivity
        ):
            raise ValueError(
                f"scientific context: {identity} sensitivity lineage is inconsistent"
            )


def _validate_evidence(
    green: dict[str, NDArray[np.generic]],
    green_metadata: dict[str, Any],
    angular: dict[str, NDArray[np.generic]],
    angular_metadata: dict[str, Any],
    convergence: dict[str, NDArray[np.generic]],
    convergence_metadata: dict[str, Any],
) -> _Evidence:
    """Fail before writing unless every asymptotic claim has registered evidence."""

    _validate_context(green_metadata, angular_metadata, convergence_metadata)

    early_slope = float(green["slope_early"][0])
    late_slope = float(green["slope_late"][0])
    if not np.isfinite(early_slope) or abs(early_slope + 0.5) > 0.05:
        raise ValueError("registered early slope must be within 0.05 of -1/2")
    if not np.isfinite(late_slope) or abs(late_slope + 1.0) > 0.05:
        raise ValueError("registered late slope must be within 0.05 of -1")

    v4 = float(angular["V4"])
    if not np.isfinite(v4) or v4 == 0.0:
        raise ValueError("V4 must be finite and nonzero")

    tolerances = _mapping(green_metadata, "tolerances")
    collapse_error = _metric(tolerances, "maximum_bessel_collapse_error")
    if collapse_error < 0.0 or collapse_error > _MAX_COLLAPSE_ERROR:
        raise ValueError("registered Bessel collapse error exceeds its fixed gate")
    per_epsilon = _mapping(tolerances, "bessel_collapse_error_by_epsilon")
    if not per_epsilon or any(
        _metric(per_epsilon, key) < 0.0 or _metric(per_epsilon, key) > collapse_error
        for key in per_epsilon
    ):
        raise ValueError("registered Bessel collapse errors are inconsistent")

    crossover_slope = _metric(tolerances, "measured_crossover_log_slope")
    if abs(crossover_slope + 1.0) > _MAX_CROSSOVER_SLOPE_ERROR:
        raise ValueError("registered crossover is not proportional to 1/|epsilon V4|")

    morse_rms = _metric(tolerances, "fixed_epsilon_morse_relative_rms_error")
    morse_max = _metric(tolerances, "fixed_epsilon_morse_relative_max_error")
    cancellation_rms = _metric(
        tolerances, "fixed_epsilon_cancellation_region_normalized_rms"
    )
    cancellation_fraction = _metric(
        tolerances, "fixed_epsilon_cancellation_fraction"
    )
    if (
        morse_rms < 0.0
        or morse_rms > _MAX_MORSE_RMS_ERROR
        or morse_max < 0.0
        or morse_max > _MAX_MORSE_POINTWISE_ERROR
        or cancellation_rms < 0.0
        or cancellation_rms > _MAX_CANCELLATION_RMS
        or cancellation_fraction < 0.0
        or cancellation_fraction >= 1.0
    ):
        raise ValueError("registered exact-Morse fixed-epsilon comparison fails")

    phase_limit = _metric(tolerances, "configured_phase_error")
    phase_error = np.asarray(green["phase_error"], dtype=np.float64)
    registered_phase = _metric(tolerances, "maximum_phase_error")
    if (
        not np.isfinite(phase_error).all()
        or np.any(phase_error < 0.0)
        or not np.isclose(
            float(np.max(phase_error)), registered_phase, rtol=1.0e-12, atol=0.0
        )
        or registered_phase > phase_limit
        or phase_limit > 0.05
        or float(convergence["phase_error"][-1]) > phase_limit
    ):
        raise ValueError("registered phase error exceeds 0.05 or is inconsistent")

    epsilon = np.asarray(green["epsilon"], dtype=np.float64)
    time = np.asarray(green["time"], dtype=np.float64)
    uniform_values = green_metadata.get("uniform_rows")
    if not isinstance(uniform_values, list) or not uniform_values:
        raise ValueError("uniform_rows must be a nonempty registered list")
    uniform_indices = np.asarray(
        [_unique_index(epsilon, float(value), "uniform_rows") for value in uniform_values],
        dtype=np.int64,
    )
    if np.unique(uniform_indices).size != uniform_indices.size:
        raise ValueError("uniform_rows must not contain duplicate epsilon values")
    expected_error_keys = {f"{epsilon[index]:.8g}" for index in uniform_indices}
    if set(per_epsilon) != expected_error_keys:
        raise ValueError("Bessel collapse evidence must match the registered uniform rows")
    registered_uniform_window = (
        (time[np.newaxis, :] >= _REGISTERED_UNIFORM_TIME_START)
        & (np.abs(green["tau"]) <= _REGISTERED_UNIFORM_TAU_MAXIMUM)
    )
    if registered_uniform_window.shape != green["J0"].shape:
        raise ValueError("registered Bessel collapse window does not align with responses")
    reconstructed_errors: list[float] = []
    for index in uniform_indices:
        window = registered_uniform_window[index]
        if np.count_nonzero(window) < 10:
            raise ValueError("registered Bessel collapse window is under-resolved")
        error = float(
            np.max(np.abs(green["scaled_response"][index][window] - green["J0"][index][window]))
        )
        registered = _metric(per_epsilon, f"{epsilon[index]:.8g}")
        if not np.isclose(error, registered, rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("registered Bessel collapse window is inconsistent")
        reconstructed_errors.append(error)
    if not np.isclose(
        max(reconstructed_errors), collapse_error, rtol=1.0e-12, atol=1.0e-15
    ):
        raise ValueError("registered Bessel collapse maximum is inconsistent")

    fixed_epsilon = green_metadata.get("fixed_morse_epsilon")
    if isinstance(fixed_epsilon, bool) or not isinstance(fixed_epsilon, (int, float)):
        raise ValueError("fixed_morse_epsilon must be numeric")
    fixed_index = _unique_index(epsilon, float(fixed_epsilon), "fixed_morse_epsilon")
    if green_metadata.get("fixed_morse_row_excluded_from_first_order_uniform_gate") is not True:
        raise ValueError("the fixed-Morse row must be excluded from the uniform gate")
    if fixed_index in uniform_indices:
        raise ValueError("fixed-Morse and first-order uniform rows must remain distinct")
    if green_metadata.get("no_fitted_amplitude_phase_frequency_or_time_shift") is not True:
        raise ValueError("the comparison forbids fitted alignment parameters")

    evidence = _mapping(green_metadata, "fixed_morse_cancellation_evidence")
    comparison_start = _metric(evidence, "comparison_start_time")
    comparison_stop = _metric(evidence, "comparison_stop_time")
    if comparison_start >= comparison_stop or not np.any(
        (time >= comparison_start) & (time <= comparison_stop)
    ):
        raise ValueError("exact-Morse comparison window must be ordered and sampled")
    if comparison_start != _REGISTERED_UNIFORM_TIME_START:
        raise ValueError("registered Bessel and exact-Morse start times must agree")
    cancellation_values = evidence.get("cancellation_time_indices")
    if not isinstance(cancellation_values, list) or not cancellation_values:
        raise ValueError("exact-Morse cancellation indices must be registered")
    if any(type(value) is not int for value in cancellation_values):
        raise ValueError("exact-Morse cancellation indices must be integers")
    cancellation_indices = np.asarray(cancellation_values, dtype=np.int64)
    if (
        np.unique(cancellation_indices).size != cancellation_indices.size
        or np.any(cancellation_indices < 0)
        or np.any(cancellation_indices >= time.size)
    ):
        raise ValueError("exact-Morse cancellation indices must be unique and in range")
    comparison_window = (time >= comparison_start) & (time <= comparison_stop)
    coherence_threshold = _metric(evidence, "coherence_threshold")
    if coherence_threshold <= 0.0 or coherence_threshold >= 1.0:
        raise ValueError("exact-Morse cancellation coherence threshold is invalid")
    try:
        morse_coherence = np.asarray(evidence["coherence"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("exact-Morse cancellation coherence must be numeric") from error
    if (
        morse_coherence.shape != time.shape
        or not np.isfinite(morse_coherence).all()
        or np.any(morse_coherence < 0.0)
        or np.any(morse_coherence > 1.0)
    ):
        raise ValueError("exact-Morse cancellation coherence is invalid")
    expected_cancellation_indices = np.flatnonzero(
        comparison_window & (morse_coherence < coherence_threshold)
    )
    if not np.array_equal(cancellation_indices, expected_cancellation_indices):
        raise ValueError("exact-Morse cancellation coherence and indices disagree")
    if not np.all(comparison_window[cancellation_indices]) or not np.isclose(
        cancellation_indices.size / np.count_nonzero(comparison_window),
        cancellation_fraction,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise ValueError("exact-Morse cancellation fraction is inconsistent")
    cancellation_window = np.zeros(time.size, dtype=np.bool_)
    cancellation_window[cancellation_indices] = True
    certified_window = comparison_window & ~cancellation_window
    fixed_full = green["G_full"][fixed_index]
    fixed_morse = green["G_morse"][fixed_index]
    normalization = float(np.sqrt(np.mean(np.abs(fixed_morse[certified_window]) ** 2)))
    registered_normalization = _metric(
        evidence, "noncancellation_morse_rms_normalization"
    )
    if not np.isclose(
        normalization, registered_normalization, rtol=1.0e-12, atol=1.0e-15
    ):
        raise ValueError(
            "registered exact-Morse metrics have inconsistent noncancellation normalization"
        )
    complex_error = fixed_full - fixed_morse
    reconstructed_morse_rms = float(
        np.sqrt(np.mean(np.abs(complex_error[certified_window]) ** 2)) / normalization
    )
    reconstructed_morse_max = float(
        np.max(np.abs(complex_error[certified_window])) / normalization
    )
    reconstructed_cancellation_rms = float(
        np.sqrt(np.mean(np.abs(complex_error[comparison_window & cancellation_window]) ** 2))
        / normalization
    )
    if not (
        np.isclose(reconstructed_morse_rms, morse_rms, rtol=1.0e-12, atol=1.0e-15)
        and np.isclose(reconstructed_morse_max, morse_max, rtol=1.0e-12, atol=1.0e-15)
        and np.isclose(
            reconstructed_cancellation_rms,
            cancellation_rms,
            rtol=1.0e-12,
            atol=1.0e-15,
        )
    ):
        raise ValueError("registered exact-Morse metrics are inconsistent with arrays")

    late_evidence = _mapping(green_metadata, "late_fit_evidence")
    late_epsilon = _metric(late_evidence, "epsilon")
    if late_epsilon != float(epsilon[fixed_index]):
        raise ValueError("late-fit evidence must use the fixed-Morse epsilon")
    late_slope_evidence = _metric(late_evidence, "slope")
    if late_slope_evidence != late_slope:
        raise ValueError("late-fit evidence must reproduce the stored late slope")
    try:
        late_centers = np.asarray(
            late_evidence["geometric_time_centers"], dtype=np.float64
        )
        late_rms = np.asarray(late_evidence["direct_response_rms"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("late-fit centers and RMS evidence must be numeric") from error
    if (
        late_centers.ndim != 1
        or late_rms.shape != late_centers.shape
        or late_centers.size < 3
        or not np.isfinite(late_centers).all()
        or not np.isfinite(late_rms).all()
        or np.any(late_centers <= 0.0)
        or np.any(late_rms <= 0.0)
        or np.any(np.diff(late_centers) <= 0.0)
    ):
        raise ValueError("late-fit centers and RMS evidence must be finite and ordered")

    if not np.any(green["fit_window_early"][uniform_indices[0]]):
        raise ValueError("registered early fit window must be nonempty")
    if not np.any(green["fit_window_late"][fixed_index]):
        raise ValueError("registered late fit window must be nonempty")

    observed_separation = np.abs(
        np.asarray(green["omega_saddle"], dtype=np.float64)
        - np.asarray(green["omega_min"], dtype=np.float64)
    )
    predicted_separation = 2.0 * np.abs(epsilon * v4)
    relative_error = np.abs(observed_separation - predicted_separation) / predicted_separation
    if not np.isfinite(relative_error).all() or np.max(relative_error) > (
        _MAX_FREQUENCY_SEPARATION_ERROR
    ):
        raise ValueError("critical-frequency separation is inconsistent with 2|epsilon V4|")

    return _Evidence(
        uniform_indices=uniform_indices,
        fixed_index=fixed_index,
        v4=v4,
        crossover_slope=crossover_slope,
        comparison_start=comparison_start,
        comparison_stop=comparison_stop,
        morse_rms_error=morse_rms,
        morse_cancellation_rms=cancellation_rms,
        cancellation_fraction=cancellation_fraction,
        cancellation_indices=cancellation_indices,
        morse_coherence=morse_coherence,
        morse_coherence_threshold=coherence_threshold,
        morse_normalization=registered_normalization,
        collapse_error=collapse_error,
        late_fit_centers=late_centers,
        late_fit_rms=late_rms,
        registered_uniform_window=registered_uniform_window,
    )


def _write_source_data(
    source_dir: Path,
    green: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
) -> None:
    uniform = evidence.uniform_indices
    samples = green["time"].size
    selected_scaled = green["scaled_response"][uniform]
    selected_j0 = green["J0"][uniform]
    common_uniform = {
        "epsilon": np.repeat(green["epsilon"][uniform], samples),
        "time": np.tile(green["time"], uniform.size),
        "tau": green["tau"][uniform],
        "J0": selected_j0,
        "scaled_real": selected_scaled.real,
        "scaled_imaginary": selected_scaled.imag,
        "scaled_magnitude": np.abs(selected_scaled),
    }
    write_source_csv(source_dir / "panel_a_collapse.csv", common_uniform)
    write_source_csv(
        source_dir / "panel_b_absolute_error.csv",
        {
            **common_uniform,
            "absolute_complex_error": np.abs(selected_scaled - selected_j0),
            "registered_joint_limit_window": evidence.registered_uniform_window[
                uniform
            ].astype(np.int64),
            "registered_minimum_time": np.full(
                selected_j0.size, _REGISTERED_UNIFORM_TIME_START
            ),
            "registered_maximum_abs_tau": np.full(
                selected_j0.size, _REGISTERED_UNIFORM_TAU_MAXIMUM
            ),
        },
    )

    source_indices = np.asarray(
        [int(uniform[0]), evidence.fixed_index], dtype=np.int64
    )
    write_source_csv(
        source_dir / "panel_c_envelopes.csv",
        {
            "epsilon": np.repeat(green["epsilon"][source_indices], samples),
            "time": np.tile(green["time"], source_indices.size),
            "envelope": green["envelope"][source_indices],
            "rms_envelope": green["rms_envelope"][source_indices],
            "local_slope": green["local_slope"][source_indices],
            "fit_window_early": green["fit_window_early"][source_indices].astype(
                np.int64
            ),
            "fit_window_late": green["fit_window_late"][source_indices].astype(
                np.int64
            ),
        },
    )
    write_source_csv(
        source_dir / "panel_c_late_fit.csv",
        {
            "epsilon": np.full(
                evidence.late_fit_centers.size, green["epsilon"][evidence.fixed_index]
            ),
            "geometric_time_center": evidence.late_fit_centers,
            "direct_response_rms": evidence.late_fit_rms,
            "stored_late_slope": np.full(
                evidence.late_fit_centers.size, float(green["slope_late"][0])
            ),
        },
    )

    inverse_rate = 1.0 / np.abs(green["epsilon"] * evidence.v4)
    write_source_csv(
        source_dir / "panel_d_crossover.csv",
        {
            "epsilon": green["epsilon"],
            "inverse_abs_epsilon_V4": inverse_rate,
            "crossover_time": green["crossover_time"],
            "epsilon_log_slope": np.full(
                green["epsilon"].size, evidence.crossover_slope
            ),
            "inverse_rate_log_slope": np.full(
                green["epsilon"].size, -evidence.crossover_slope
            ),
            "phase_error": green["phase_error"],
        },
    )

    fixed = evidence.fixed_index
    comparison_window = (
        (green["time"] >= evidence.comparison_start)
        & (green["time"] <= evidence.comparison_stop)
    )
    cancellation_region = np.zeros(samples, dtype=np.bool_)
    cancellation_region[evidence.cancellation_indices] = True
    certified_window = comparison_window & ~cancellation_region
    write_source_csv(
        source_dir / "panel_e_fixed_morse.csv",
        {
            "epsilon": np.full(samples, green["epsilon"][fixed]),
            "time": green["time"],
            "full_real": green["G_full"][fixed].real,
            "full_imaginary": green["G_full"][fixed].imag,
            "full_magnitude": np.abs(green["G_full"][fixed]),
            "morse_real": green["G_morse"][fixed].real,
            "morse_imaginary": green["G_morse"][fixed].imag,
            "morse_magnitude": np.abs(green["G_morse"][fixed]),
            "comparison_window": comparison_window.astype(np.int64),
            "cancellation_region": cancellation_region.astype(np.int64),
            "certified_noncancellation_window": certified_window.astype(np.int64),
            "morse_phasor_coherence": evidence.morse_coherence,
            "coherence_threshold": np.full(
                samples, evidence.morse_coherence_threshold
            ),
            "comparison_start_time": np.full(samples, evidence.comparison_start),
            "comparison_stop_time": np.full(samples, evidence.comparison_stop),
            "noncancellation_morse_rms_normalization": np.full(
                samples, evidence.morse_normalization
            ),
        },
    )

    fixed_epsilon = float(green["epsilon"][fixed])
    omega_min = float(green["omega_min"][fixed])
    omega_saddle = float(green["omega_saddle"][fixed])
    midpoint = 0.5 * (omega_min + omega_saddle)
    rate = abs(fixed_epsilon * evidence.v4)
    spectrum = np.asarray(green["spectrum"][fixed], dtype=np.float64)
    write_source_csv(
        source_dir / "panel_f_frequency_features.csv",
        {
            "epsilon": np.full(green["spectrum_omega"].size, fixed_epsilon),
            "spectrum_omega": green["spectrum_omega"],
            "detuning_from_critical_midpoint": green["spectrum_omega"] - midpoint,
            "spectrum": spectrum,
            "spectrum_normalized": spectrum / np.max(spectrum),
            "omega_min": np.full(spectrum.size, omega_min),
            "omega_saddle": np.full(spectrum.size, omega_saddle),
            "predicted_separation_2_abs_epsilon_V4": np.full(
                spectrum.size, 2.0 * rate
            ),
            "signed_modulation_rate_abs_epsilon_V4": np.full(
                spectrum.size, rate
            ),
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


def _epsilon_colors(count: int) -> list[tuple[float, float, float, float]]:
    return [mpl.colormaps["viridis"](value) for value in np.linspace(0.14, 0.82, count)]


def _draw_collapse(
    ax: plt.Axes,
    green: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
    colors: list[tuple[float, float, float, float]],
) -> None:
    theory = int(evidence.uniform_indices[-1])
    ax.plot(
        green["tau"][theory],
        green["J0"][theory],
        color=PALETTE["prediction"],
        linewidth=1.7,
        zorder=5,
    )
    for color, index in zip(colors, evidence.uniform_indices, strict=True):
        response = green["scaled_response"][index]
        ax.plot(green["tau"][index], response.real, color=color, linewidth=0.85)
        ax.plot(
            green["tau"][index],
            np.abs(response),
            color=color,
            linewidth=0.75,
            linestyle=(0, (3, 1.5)),
            alpha=0.9,
        )
    role_handles = (
        Line2D([], [], color=PALETTE["prediction"], lw=1.7, label=r"$J_0(\tau)$"),
        Line2D([], [], color=PALETTE["anisotropic"], lw=0.9, label="scaled Re"),
        Line2D(
            [],
            [],
            color=PALETTE["anisotropic"],
            lw=0.9,
            linestyle=(0, (3, 1.5)),
            label="scaled magnitude",
        ),
    )
    role_legend = ax.legend(
        handles=role_handles, loc="lower left", fontsize=8.4, ncol=1
    )
    epsilon_handles = tuple(
        Line2D(
            [],
            [],
            color=color,
            lw=1.2,
            label=f"{float(green['epsilon'][index]):g}",
        )
        for color, index in zip(colors, evidence.uniform_indices, strict=True)
    )
    ax.add_artist(role_legend)
    ax.legend(
        handles=epsilon_handles,
        title=r"$\varepsilon$",
        loc="upper center",
        bbox_to_anchor=(0.56, 0.99),
        fontsize=8.4,
        title_fontsize=8.4,
        ncol=len(epsilon_handles),
        handlelength=1.5,
        columnspacing=0.8,
    )
    ax.set_xlabel(r"transition variable $\tau=\varepsilon V_4t$")
    ax.set_ylabel("scaled response")
    # Retained deliberately: tests/figures/test_figure06.py asserts this
    # phrase is present in the SVG.
    ax.text(
        0.99,
        0.99,
        "without fitted alignment",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color=PALETTE["neutral"],
        zorder=6,
    )
    ax.grid(color="#E8E8E8", linewidth=0.4)


def _draw_absolute_error(
    ax: plt.Axes,
    green: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
    colors: list[tuple[float, float, float, float]],
) -> None:
    for color, index in zip(colors, evidence.uniform_indices, strict=True):
        error = np.abs(green["scaled_response"][index] - green["J0"][index])
        ax.plot(
            green["tau"][index],
            error,
            color=color,
            linewidth=0.55,
            alpha=0.30,
        )
        window = evidence.registered_uniform_window[index]
        ax.plot(
            green["tau"][index][window],
            error[window],
            color=color,
            linewidth=1.15,
        )
    ax.set_yscale("log")
    ax.text(
        0.97,
        0.95,
        f"max = {evidence.collapse_error:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        zorder=6,
        linespacing=1.75,
    )
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("absolute complex error")
    ax.grid(which="major", color="#E8E8E8", linewidth=0.4)


def _draw_envelopes(
    ax: plt.Axes,
    green: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
) -> None:
    early = int(evidence.uniform_indices[0])
    fixed = evidence.fixed_index
    time = green["time"]
    ax.plot(
        time,
        green["envelope"][early],
        color=PALETTE["minimum"],
        linewidth=0.55,
        alpha=0.50,
    )
    ax.plot(
        time,
        green["rms_envelope"][early],
        color=PALETTE["minimum"],
        linewidth=1.15,
    )
    ax.plot(
        time,
        green["envelope"][fixed],
        color=PALETTE["saddle"],
        linewidth=0.55,
        alpha=0.48,
    )
    ax.plot(
        time,
        green["rms_envelope"][fixed],
        color=PALETTE["saddle"],
        linewidth=1.15,
    )
    early_window = green["fit_window_early"][early]
    late_window = green["fit_window_late"][fixed]
    ax.plot(
        time[early_window],
        green["envelope"][early][early_window],
        color=PALETTE["minimum"],
        linewidth=5.0,
        alpha=0.22,
        solid_capstyle="butt",
        zorder=1,
    )
    ax.plot(
        time[late_window],
        green["rms_envelope"][fixed][late_window],
        color=PALETTE["saddle"],
        linewidth=5.0,
        alpha=0.22,
        solid_capstyle="butt",
        zorder=1,
    )
    ax.plot(
        evidence.late_fit_centers,
        evidence.late_fit_rms,
        color=PALETTE["saddle"],
        marker="s",
        markerfacecolor="white",
        markeredgewidth=0.7,
        markersize=3.5,
        linewidth=0.7,
        zorder=6,
    )
    ax.text(
        0.04,
        0.08,
        f"joint-limit slice: slope {float(green['slope_early'][0]):.3f}\n"
        f"fixed-ε slice: slope {float(green['slope_late'][0]):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"time $t\,c_T/h$")
    ax.set_ylabel("response envelope")
    ax.grid(which="major", color="#E8E8E8", linewidth=0.4)


def _draw_crossover(
    ax: plt.Axes,
    green: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
) -> None:
    inverse_rate = 1.0 / np.abs(green["epsilon"] * evidence.v4)
    order = np.argsort(inverse_rate)
    ax.plot(
        inverse_rate[order],
        green["crossover_time"][order],
        color=PALETTE["anisotropic"],
        marker="o",
        markersize=3.2,
        linewidth=1.0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(
        0.05,
        0.94,
        f"slope = {-evidence.crossover_slope:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
    )
    ax.set_xlabel(r"$1/|\varepsilon V_4|$")
    ax.set_ylabel("crossover time")
    ax.grid(which="major", color="#E8E8E8", linewidth=0.4)


def _draw_fixed_morse(
    ax: plt.Axes,
    green: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
) -> None:
    fixed = evidence.fixed_index
    window = (
        (green["time"] >= evidence.comparison_start)
        & (green["time"] <= evidence.comparison_stop)
    )
    ax.plot(
        green["time"][window],
        np.abs(green["G_full"][fixed][window]),
        color=PALETTE["anisotropic"],
        linewidth=1.0,
    )
    ax.plot(
        green["time"][window],
        np.abs(green["G_morse"][fixed][window]),
        color=PALETTE["prediction"],
        linewidth=1.0,
        linestyle=(0, (4, 2)),
    )
    cancellation = np.zeros(green["time"].size, dtype=np.bool_)
    cancellation[evidence.cancellation_indices] = True
    displayed_cancellation = window & cancellation
    ax.scatter(
        green["time"][displayed_cancellation],
        np.abs(green["G_full"][fixed][displayed_cancellation]),
        s=8,
        marker="o",
        facecolor="white",
        edgecolor=PALETTE["uncertainty"],
        linewidth=0.5,
        zorder=5,
    )
    ax.text(
        0.97,
        0.94,
        "exact-Morse stationary sum\n"
        f"RMS {evidence.morse_rms_error:.3f} / "
        f"{evidence.morse_cancellation_rms:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        zorder=6,
    )
    ax.set_xlabel(r"time $t\,c_T/h$")
    ax.set_ylabel(r"$|G(t)|$")
    ax.grid(color="#E8E8E8", linewidth=0.4)


def _draw_frequency_features(
    ax: plt.Axes,
    green: dict[str, NDArray[np.generic]],
    evidence: _Evidence,
) -> None:
    fixed = evidence.fixed_index
    epsilon = float(green["epsilon"][fixed])
    omega_min = float(green["omega_min"][fixed])
    omega_saddle = float(green["omega_saddle"][fixed])
    midpoint = 0.5 * (omega_min + omega_saddle)
    x = green["spectrum_omega"] - midpoint
    spectrum = green["spectrum"][fixed]
    normalized = spectrum / np.max(spectrum)
    min_x = omega_min - midpoint
    saddle_x = omega_saddle - midpoint
    rate = abs(epsilon * evidence.v4)
    ax.plot(x, normalized, color=PALETTE["neutral"], linewidth=0.9)
    ax.axvline(min_x, color=PALETTE["minimum"], linewidth=0.8)
    ax.axvline(saddle_x, color=PALETTE["saddle"], linewidth=0.8)
    blended = ax.get_xaxis_transform()
    ax.plot(
        [min_x],
        [0.98],
        marker=MARKERS["minimum"],
        color=PALETTE["minimum"],
        markersize=4,
        transform=blended,
        clip_on=False,
    )
    ax.plot(
        [saddle_x],
        [0.98],
        marker=MARKERS["saddle"],
        color=PALETTE["saddle"],
        markersize=4,
        transform=blended,
        clip_on=False,
    )
    span = 6.0 * max(abs(min_x), abs(saddle_x), rate)
    ax.set_xlim(-span, span)
    ax.set_xlabel(r"angular-frequency detuning $\omega-\omega_{\mathrm{mid}}$")
    ax.set_ylabel("normalized spectrum")
    ax.grid(color="#E8E8E8", linewidth=0.4)


def build(data_dir: Path, output_dir: Path, source_dir: Path) -> dict[str, Path]:
    """Build Figure 6 from validated, preregistered crossover evidence only."""

    green, green_metadata = load_figure_artifact(data_dir, "green_crossover")
    angular, angular_metadata = load_figure_artifact(data_dir, "angular_sensitivity")
    convergence, convergence_metadata = load_figure_artifact(data_dir, "convergence")
    evidence = _validate_evidence(
        green,
        green_metadata,
        angular,
        angular_metadata,
        convergence,
        convergence_metadata,
    )
    _write_source_data(source_dir, green, evidence)

    apply_publication_style()
    fig = plt.figure(layout="constrained", facecolor="white")
    axes = fig.subplot_mosaic(
        [
            ["a", "a", "b"],
            ["a", "a", "d"],
            ["c", "c", "e"],
            ["c", "c", "f"],
        ],
        width_ratios=(1.0, 1.0, 1.05),
        height_ratios=(1.0, 1.0, 0.94, 1.06),
    )
    colors = _epsilon_colors(evidence.uniform_indices.size)
    _draw_collapse(axes["a"], green, evidence, colors)
    _draw_absolute_error(axes["b"], green, evidence, colors)
    _draw_envelopes(axes["c"], green, evidence)
    _draw_crossover(axes["d"], green, evidence)
    _draw_fixed_morse(axes["e"], green, evidence)
    _draw_frequency_features(axes["f"], green, evidence)
    for label, ax in axes.items():
        _panel_label(ax, label)
    return save_publication_figure(
        fig,
        output_dir / "figure_06_decay_crossover",
        _SPEC,
    )
