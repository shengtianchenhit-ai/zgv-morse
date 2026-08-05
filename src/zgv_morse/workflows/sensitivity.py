"""Generate full-elastic angular and radial anisotropy sensitivities."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..config import ReferenceConfig
from ..elasticity import cubic_family, isotropic_tensor
from ..mode_tracking import seed_tracked_mode, track_mode
from ..perturbation import (
    cubic_harmonic_report,
    extract_angular_harmonics,
    physical_cubic_shift,
    sensitivity_from_plate,
)
from ..spectral_plate import (
    WavevectorDerivatives,
    assemble_plate_matrices,
    assemble_wavevector_derivatives,
    solve_plate_modes,
)
from ..zgv import find_s1_zgv
from .common import validate_stage_inputs, write_stage_artifact


_PROFILE = MappingProxyType(
    {
        "smoke": MappingProxyType({"order": 8, "angular_nodes": 32, "fd_step": 0.005}),
        "full": MappingProxyType({"order": 10, "angular_nodes": 64, "fd_step": 0.0025}),
    }
)


def _tracked_frequency(
    radius: float,
    epsilon: float,
    cosine: float,
    sine: float,
    reference: object,
    cfg: ReferenceConfig,
    order: int,
) -> float:
    tensor = cubic_family(cfg.lam, cfg.mu, cfg.delta, epsilon)[0]
    matrices = assemble_plate_matrices(
        radius * cosine,
        radius * sine,
        tensor,
        cfg.rho,
        cfg.h,
        order=order,
    )
    modes = solve_plate_modes(matrices, 12)
    tracked = track_mode(
        reference,  # type: ignore[arg-type]
        modes,
        min_mac=0.8,
        predicted_eigenvalue=reference.eigenvalue,  # type: ignore[attr-defined]
    )
    return tracked.omega


def compute_sensitivity_arrays(
    cfg: ReferenceConfig,
    profile: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compute reusable full-plate sensitivity arrays for one profile."""

    settings = _PROFILE[profile]
    exact = find_s1_zgv(cfg, dps=60)
    order = int(settings["order"])
    angular_nodes = int(settings["angular_nodes"])
    fd_step = float(settings["fd_step"])
    theta = 2.0 * np.pi * np.arange(angular_nodes) / angular_nodes
    base_tensor = isotropic_tensor(cfg.lam, cfg.mu)
    auxiliary_step = 0.25
    plus_auxiliary = cubic_family(
        cfg.lam,
        cfg.mu,
        cfg.delta,
        auxiliary_step,
    )[0]
    minus_auxiliary = cubic_family(
        cfg.lam,
        cfg.mu,
        cfg.delta,
        -auxiliary_step,
    )[0]
    V = np.empty(angular_nodes)
    B = np.empty(angular_nodes)
    V_fd = np.empty(angular_nodes)
    B_fd = np.empty(angular_nodes)
    minimum_gap = np.inf
    maximum_residual = 0.0

    for index, angle in enumerate(theta):
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        kx, ky = exact.kappa0 * cosine, exact.kappa0 * sine
        base = assemble_plate_matrices(
            kx,
            ky,
            base_tensor,
            cfg.rho,
            cfg.h,
            order=order,
        )
        base_derivatives = assemble_wavevector_derivatives(
            kx,
            ky,
            base_tensor,
            cfg.rho,
            cfg.h,
            order=order,
        )
        modes = solve_plate_modes(base, 12)
        reference = seed_tracked_mode(
            modes,
            int(np.argmin(abs(modes.omega - exact.omega0))),
        )
        plus = assemble_plate_matrices(
            kx,
            ky,
            plus_auxiliary,
            cfg.rho,
            cfg.h,
            order=order,
        )
        minus = assemble_plate_matrices(
            kx,
            ky,
            minus_auxiliary,
            cfg.rho,
            cfg.h,
            order=order,
        )
        plus_derivatives = assemble_wavevector_derivatives(
            kx,
            ky,
            plus_auxiliary,
            cfg.rho,
            cfg.h,
            order=order,
        )
        minus_derivatives = assemble_wavevector_derivatives(
            kx,
            ky,
            minus_auxiliary,
            cfg.rho,
            cfg.h,
            order=order,
        )
        perturbation = replace(
            base,
            stiffness=(plus.stiffness - minus.stiffness) / (2.0 * auxiliary_step),
            mass=base.mass,
            nodes=base.nodes,
        )
        derivative_names = ("dkx", "dky", "dkx2", "dkx_dky", "dky2")
        perturbation_derivatives = WavevectorDerivatives(
            *(
                (getattr(plus_derivatives, name) - getattr(minus_derivatives, name))
                / (2.0 * auxiliary_step)
                for name in derivative_names
            )
        )
        analytic = sensitivity_from_plate(
            reference,
            base,
            base_derivatives,
            perturbation,
            perturbation_derivatives,
            float(angle),
        )
        V[index] = analytic.V
        B[index] = analytic.B
        minimum_gap = min(minimum_gap, reference.eigengap)
        maximum_residual = max(maximum_residual, reference.residual)

        omega_plus = _tracked_frequency(
            exact.kappa0,
            fd_step,
            cosine,
            sine,
            reference,
            cfg,
            order,
        )
        omega_minus = _tracked_frequency(
            exact.kappa0,
            -fd_step,
            cosine,
            sine,
            reference,
            cfg,
            order,
        )
        V_fd[index] = (omega_plus - omega_minus) / (2.0 * fd_step)
        mixed = (
            _tracked_frequency(
                exact.kappa0 + fd_step,
                fd_step,
                cosine,
                sine,
                reference,
                cfg,
                order,
            )
            - _tracked_frequency(
                exact.kappa0 + fd_step,
                -fd_step,
                cosine,
                sine,
                reference,
                cfg,
                order,
            )
            - _tracked_frequency(
                exact.kappa0 - fd_step,
                fd_step,
                cosine,
                sine,
                reference,
                cfg,
                order,
            )
            + _tracked_frequency(
                exact.kappa0 - fd_step,
                -fd_step,
                cosine,
                sine,
                reference,
                cfg,
                order,
            )
        )
        B_fd[index] = mixed / (4.0 * fd_step**2)

    max_order = 12 if angular_nodes == 32 else 16
    report = cubic_harmonic_report(theta, V, max_order=max_order)
    harmonics = extract_angular_harmonics(theta, V, max_order=max_order)
    relative_V_error = float(np.linalg.norm(V_fd - V) / np.linalg.norm(V))
    relative_B_error = float(np.linalg.norm(B_fd - B) / np.linalg.norm(B))
    if relative_V_error >= cfg.sensitivity_match_tolerance:
        raise RuntimeError("finite-difference V does not meet the registered tolerance")
    if relative_B_error >= cfg.sensitivity_match_tolerance:
        raise RuntimeError("finite-difference B does not meet the registered tolerance")
    if minimum_gap <= 10.0 * max(maximum_residual, np.finfo(float).eps):
        raise RuntimeError("sensitivity branch eigengap is unresolved")
    shift = physical_cubic_shift(cfg.epsilon_values, cfg.delta, report.V4)
    arrays = {
        "theta": theta,
        "V": V,
        "B": B,
        "V_reconstruction": report.reconstruction,
        "harmonic_order": harmonics.order,
        "harmonic_amplitude": np.hypot(harmonics.cosine, harmonics.sine),
        "V0": np.array(report.V0),
        "V4": np.array(report.V4),
        "V8": np.array(report.V8),
        "epsilon": np.asarray(cfg.epsilon_values),
        "delta_c": shift.delta_c,
        "physical_V4_shift": shift.frequency_shift,
        "V_fd": V_fd,
        "B_fd": B_fd,
    }
    diagnostics = {
        "relative_V_error": relative_V_error,
        "relative_B_error": relative_B_error,
        "minimum_relative_eigengap": float(minimum_gap),
        "maximum_eigen_residual": float(maximum_residual),
        "periodicity_defect": report.periodicity_defect,
        "mirror_defect": report.mirror_defect,
        "non_cubic_leakage": report.non_cubic_leakage,
    }
    return arrays, diagnostics


def run(cfg: ReferenceConfig, output_dir: Path, profile: str) -> Path:
    """Run the registered sensitivity stage."""

    config, directory, selected_profile = validate_stage_inputs(cfg, output_dir, profile)
    arrays, diagnostics = compute_sensitivity_arrays(config, selected_profile)
    units = {
        "theta": "rad",
        "V": "Omega per epsilon",
        "B": "Omega per epsilon per kappa",
        "V_reconstruction": "Omega per epsilon",
        "harmonic_order": "integer",
        "harmonic_amplitude": "Omega per epsilon",
        "V0": "Omega per epsilon",
        "V4": "Omega per epsilon",
        "V8": "Omega per epsilon",
        "epsilon": "1",
        "delta_c": "stiffness",
        "physical_V4_shift": "Omega",
        "V_fd": "Omega per epsilon",
        "B_fd": "Omega per epsilon per kappa",
    }
    tolerances = {
        **diagnostics,
        "configured_relative_sensitivity": config.sensitivity_match_tolerance,
    }
    return write_stage_artifact(
        "sensitivity",
        config,
        directory,
        selected_profile,
        arrays,
        units,
        tolerances,
    )
