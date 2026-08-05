"""Finite-anisotropy [001] silicon stress test, separate from the proof."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..config import ReferenceConfig
from ..critical_points import Annulus, locate_critical_points, verify_annular_exhaustion
from ..dispersion import RingAnchoredSpectralEvaluator
from ..elasticity import cubic_tensor
from ..mode_tracking import symmetric_lamb_parity_score
from ..spectral_plate import assemble_plate_matrices, solve_plate_modes
from .common import validate_stage_inputs, write_stage_artifact


_PROFILE = MappingProxyType(
    {
        "smoke": MappingProxyType({"order": 9, "grid_nodes": 20}),
        "full": MappingProxyType({"order": 12, "grid_nodes": 40}),
    }
)
_C11_GPA = 165.7
_C12_GPA = 63.9
_C44_GPA = 79.6
_SOURCE_ID = "doi:10.1107/S1600577514004962"


def _anchor_seed(tensor: np.ndarray, order: int) -> tuple[float, float]:
    radii = np.linspace(0.70, 1.00, 13)
    candidates: list[tuple[float, float]] = []
    for theta in (0.0, 0.25 * np.pi):
        branch = np.empty(radii.size)
        for index, radius in enumerate(radii):
            matrices = assemble_plate_matrices(
                radius * np.cos(theta),
                radius * np.sin(theta),
                tensor,
                1.0,
                1.0,
                order=order,
            )
            modes = solve_plate_modes(matrices, 12)
            symmetric = np.array(
                [
                    symmetric_lamb_parity_score(
                        modes.vectors[:, mode_index],
                        matrices.nodes,
                        matrices.mass,
                    )
                    for mode_index in range(modes.omega.size)
                ]
            )
            indices = np.flatnonzero(symmetric > 0.5)
            selected = int(indices[np.argmin(abs(modes.omega[indices] - 2.2))])
            branch[index] = modes.omega[selected]
        minimum = int(np.argmin(branch))
        candidates.append((float(radii[minimum]), float(branch[minimum])))
    return (
        float(np.mean([candidate[0] for candidate in candidates])),
        float(np.mean([candidate[1] for candidate in candidates])),
    )


def run(cfg: ReferenceConfig, output_dir: Path, profile: str) -> Path:
    """Run the registered silicon stress-test stage."""

    config, directory, selected_profile = validate_stage_inputs(cfg, output_dir, profile)
    settings = _PROFILE[selected_profile]
    order = int(settings["order"])
    tensor = cubic_tensor(
        _C11_GPA / _C44_GPA,
        _C12_GPA / _C44_GPA,
        1.0,
    )
    k0, target_omega = _anchor_seed(tensor, order)
    evaluator = RingAnchoredSpectralEvaluator(
        tensor,
        rho=1.0,
        half_thickness=1.0,
        k0=k0,
        target_omega=target_omega,
        order=order,
        num_modes=12,
        angular_sectors=8,
    )
    annulus = Annulus(k0, 0.15 * k0)
    points = locate_critical_points(
        evaluator,
        annulus,
        n_radial=9,
        n_theta=32,
        hessian_step=3.0e-3,
    )
    report = verify_annular_exhaustion(evaluator, annulus, points, 16)
    if len(points) != 8 or not report.boundary_is_noncritical or not report.index_closes:
        raise RuntimeError("silicon stress test did not resolve an exhaustive eight-point set")

    grid_nodes = int(settings["grid_nodes"])
    kx_grid = np.linspace(-1.10 * k0, 1.10 * k0, grid_nodes)
    ky_grid = np.linspace(-1.10 * k0, 1.10 * k0, grid_nodes)
    omega_grid = np.empty((grid_nodes, grid_nodes))
    for row, ky in enumerate(ky_grid):
        for column, kx in enumerate(kx_grid):
            omega_grid[row, column] = evaluator(np.array([kx, ky])).omega

    tracking_gap = np.empty(len(points))
    for index, point in enumerate(points):
        direction = -1 if point.radius < k0 else 1
        tracker = evaluator.radial_tracker(point.theta, direction)
        sample = tracker(np.array([point.kx, point.ky]))
        tracking_gap[index] = sample.relative_eigengap
    arrays = {
        "kx_grid": kx_grid,
        "ky_grid": ky_grid,
        "omega_grid": omega_grid,
        "kx": np.array([point.kx for point in points]),
        "ky": np.array([point.ky for point in points]),
        "omega": np.array([point.omega for point in points]),
        "hessian_eigenvalues": np.array(
            [point.hessian_eigenvalues for point in points]
        ),
        "kind": np.array([point.kind for point in points]),
        "tracking_gap": tracking_gap,
        "gradient_residual": np.array(
            [point.gradient_residual for point in points]
        ),
    }
    units = {
        "kx_grid": "1/h",
        "ky_grid": "1/h",
        "omega_grid": "sqrt(C44/rho)/h",
        "kx": "1/h",
        "ky": "1/h",
        "omega": "sqrt(C44/rho)/h",
        "hessian_eigenvalues": "frequency per k squared",
        "kind": "label",
        "tracking_gap": "relative",
        "gradient_residual": "group velocity",
    }
    tolerances = {
        "minimum_tracking_gap": float(np.min(tracking_gap)),
        "maximum_gradient_residual": float(
            max(point.gradient_residual for point in points)
        ),
        "minimum_boundary_gradient": report.minimum_boundary_gradient,
        "maximum_boundary_gradient_uncertainty": report.maximum_gradient_uncertainty,
    }
    return write_stage_artifact(
        "silicon",
        config,
        directory,
        selected_profile,
        arrays,
        units,
        tolerances,
        extra_metadata={
            "scope": "finite-anisotropy stress test only",
            "material_source_id": _SOURCE_ID,
            "material_constants_GPa": {
                "C11": _C11_GPA,
                "C12": _C12_GPA,
                "C44": _C44_GPA,
            },
            "orientation": "[001] plate normal",
            "annular_boundary_certificate": {
                "annulus_center_kappa": k0,
                "annulus_half_width": annulus.half_width,
                "coarse_boundary_nodes": 16,
                "fine_boundary_nodes": 32,
                "boundary_is_noncritical": report.boundary_is_noncritical,
                "index_closes": report.index_closes,
            },
        },
    )
