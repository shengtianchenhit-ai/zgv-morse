"""Generate the exact and full-wave isotropic ZGV reference artifact."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..config import ReferenceConfig
from ..dispersion import RingAnchoredSpectralEvaluator
from ..elasticity import isotropic_tensor
from ..mode_tracking import relative_eigengap, symmetric_lamb_parity_score
from ..spectral_plate import assemble_plate_matrices, solve_plate_modes
from ..zgv import find_s1_zgv
from .common import validate_stage_inputs, write_stage_artifact


_PROFILE = MappingProxyType(
    {
        "smoke": MappingProxyType({"order": 9, "radial_nodes": 41, "local_nodes": 21}),
        "full": MappingProxyType({"order": 12, "radial_nodes": 81, "local_nodes": 41}),
    }
)


def _tracked_radial_branch(
    evaluator: RingAnchoredSpectralEvaluator,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero_index = int(np.flatnonzero(q == 0.0)[0])
    omega = np.empty(q.size)
    uncertainty = np.empty(q.size)
    gap = np.empty(q.size)
    for direction, indices in (
        (-1, range(zero_index, -1, -1)),
        (1, range(zero_index, q.size)),
    ):
        tracker = evaluator.radial_tracker(0.0, direction)
        for index in indices:
            sample = tracker(np.array([evaluator.k0 + q[index], 0.0]))
            if index == zero_index and direction == 1:
                omega[index] = 0.5 * (omega[index] + sample.frequency.omega)
                uncertainty[index] = max(
                    uncertainty[index],
                    sample.frequency.frequency_uncertainty,
                )
                gap[index] = min(gap[index], sample.relative_eigengap)
            else:
                omega[index] = sample.frequency.omega
                uncertainty[index] = sample.frequency.frequency_uncertainty
                gap[index] = sample.relative_eigengap
    return omega, uncertainty, gap


def run(cfg: ReferenceConfig, output_dir: Path, profile: str) -> Path:
    """Run the registered isotropic reference stage."""

    config, directory, selected_profile = validate_stage_inputs(cfg, output_dir, profile)
    settings = _PROFILE[selected_profile]
    exact = find_s1_zgv(config, dps=60 if selected_profile == "smoke" else 80)
    tensor = isotropic_tensor(config.lam, config.mu)
    order = int(settings["order"])
    evaluator = RingAnchoredSpectralEvaluator(
        tensor,
        rho=config.rho,
        half_thickness=config.h,
        k0=exact.kappa0,
        target_omega=exact.omega0,
        order=order,
        num_modes=12,
        angular_sectors=8,
    )
    q = np.linspace(-0.20 * exact.kappa0, 0.20 * exact.kappa0, int(settings["radial_nodes"]))
    omega, frequency_uncertainty, eigengap = _tracked_radial_branch(evaluator, q)
    anchor_index = int(np.flatnonzero(q == 0.0)[0])
    relative_anchor_error = abs(omega[anchor_index] - exact.omega0) / exact.omega0
    if relative_anchor_error > config.isotropic_match_tolerance:
        raise RuntimeError("full-wave isotropic ZGV frequency misses the exact regression")

    local_q = np.linspace(
        -0.04 * exact.kappa0,
        0.04 * exact.kappa0,
        int(settings["local_nodes"]),
    )
    local_omega, local_uncertainty, local_gap = _tracked_radial_branch(
        evaluator,
        local_q,
    )
    local_quadratic = exact.omega0 + 0.5 * exact.curvature_a * local_q**2

    matrices = assemble_plate_matrices(
        exact.kappa0,
        0.0,
        tensor,
        config.rho,
        config.h,
        order=order + 4,
    )
    modes = solve_plate_modes(matrices, num_modes=12)
    parity = np.array(
        [
            symmetric_lamb_parity_score(
                modes.vectors[:, index],
                matrices.nodes,
                matrices.mass,
            )
            for index in range(modes.omega.size)
        ]
    )
    symmetric = np.flatnonzero(parity > 0.5)
    mode_index = int(symmetric[np.argmin(abs(modes.omega[symmetric] - exact.omega0))])
    mode_u = np.asarray(modes.vectors[:, mode_index]).reshape((-1, 3))
    mode_squared_displacement = np.sum(np.abs(mode_u) ** 2, axis=1)
    selected_gap = relative_eigengap(modes.eigenvalues, mode_index)
    minimum_gap = min(float(np.min(eigengap)), float(np.min(local_gap)), selected_gap)

    arrays = {
        "kappa": exact.kappa0 + q,
        "omega_symmetric": omega[np.newaxis, :],
        "branch_labels": np.array(["selected symmetric ZGV branch"]),
        "kappa0": np.array(exact.kappa0),
        "omega0": np.array(exact.omega0),
        "curvature_a": np.array(exact.curvature_a),
        "local_q": local_q,
        "local_omega": local_omega,
        "local_quadratic": local_quadratic,
        "mode_z": np.asarray(matrices.nodes),
        "mode_u": np.asarray(mode_u),
        "mode_squared_displacement": mode_squared_displacement,
    }
    units = {
        "kappa": "1",
        "omega_symmetric": "1",
        "branch_labels": "label",
        "kappa0": "1",
        "omega0": "1",
        "curvature_a": "1",
        "local_q": "1",
        "local_omega": "1",
        "local_quadratic": "1",
        "mode_z": "h",
        "mode_u": "mass-normalized",
        "mode_squared_displacement": "squared-displacement proxy",
    }
    tolerances = {
        "relative_isotropic_frequency": relative_anchor_error,
        "maximum_frequency_uncertainty": float(
            max(np.max(frequency_uncertainty), np.max(local_uncertainty))
        ),
        "minimum_relative_eigengap": minimum_gap,
        "configured_isotropic_match": config.isotropic_match_tolerance,
    }
    return write_stage_artifact(
        "isotropic",
        config,
        directory,
        selected_profile,
        arrays,
        units,
        tolerances,
    )
