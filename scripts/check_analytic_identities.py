"""Executable symbolic and high-precision checks for the isotropic ZGV ring.

The symbolic functions in this module mirror the algebra in
``docs/derivations/01_isotropic_rayleigh_lamb.tex``.  The numerical check uses
the float-valued Task-4 point only as an initial seed, independently refines
the two equations ``D = D_k = 0`` with 80-digit :mod:`mpmath` arithmetic, and
then compares the refined result with both the Task-4 API and the generated
reference artifact.  Independence of the full spectral formulation is tested
elsewhere; it is not claimed by this scalar-determinant check.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import brentq

from zgv_morse.asymptotics import (
    MorseContribution,
    critical_frequency_separation,
    morse_stationary_phase_response,
    signed_modulation_rate,
)
from zgv_morse.config import load_reference_config
from zgv_morse.elasticity import cubic_perturbation_tensor, isotropic_tensor
from zgv_morse.green_response import normal_impulse_amplitude
from zgv_morse.mode_tracking import seed_tracked_mode
from zgv_morse.perturbation import frequency_sensitivity, radial_frequency_sensitivity
from zgv_morse.rayleigh_lamb import det_symmetric_mp
from zgv_morse.spectral_plate import assemble_plate_matrices, solve_plate_modes
from zgv_morse.zgv import find_s1_zgv


@dataclass(frozen=True, slots=True)
class ReferenceSubstitutionReport:
    """Residuals and cross-source errors at the reference ZGV point."""

    kappa0: float
    omega0: float
    curvature_a: float
    determinant_residual: float
    production_determinant_residual: float
    group_velocity: float
    d_omega: float
    task4_kappa_error: float
    task4_omega_error: float
    task4_curvature_error: float
    artifact_kappa_error: float
    artifact_omega_error: float
    artifact_curvature_error: float


def sine_ratio_series() -> sp.Expr:
    """Return the analytic continuation of ``sin(s*h)/s`` through ``s**6``."""

    s, h = sp.symbols("s h")
    return sp.expand(sp.series(sp.sin(s * h) / s, s, 0, 8).removeO())


def traction_determinant_residual() -> sp.Expr:
    """Return ``det(T diag(1,1/s)) - D_S`` as a simplified expression."""

    k, p, s, h = sp.symbols("k p s h")
    shear_factor = s**2 - k**2
    physical = sp.Matrix(
        [
            [-2 * sp.I * k * p * sp.sin(p * h), shear_factor * sp.sin(s * h)],
            [-shear_factor * sp.cos(p * h), 2 * sp.I * k * s * sp.cos(s * h)],
        ]
    )
    regular = physical * sp.diag(1, 1 / s)
    determinant = shear_factor**2 * sp.sin(s * h) / s * sp.cos(p * h) + 4 * k**2 * p * sp.cos(
        s * h
    ) * sp.sin(p * h)
    return sp.simplify(sp.expand_trig(regular.det() - determinant))


def shear_cutoff_limit_residual() -> sp.Expr:
    """Return the residual of the regular determinant's exact ``s -> 0`` limit."""

    k, p, s, h = sp.symbols("k p s h")
    shear_factor = s**2 - k**2
    determinant = shear_factor**2 * sp.sin(s * h) / s * sp.cos(p * h) + 4 * k**2 * p * sp.cos(
        s * h
    ) * sp.sin(p * h)
    expected = k**4 * h * sp.cos(p * h) + 4 * k**2 * p * sp.sin(p * h)
    return sp.simplify(sp.limit(determinant, s, 0) - expected)


def implicit_derivative_formulas() -> dict[str, sp.Expr]:
    """Differentiate ``D(k, omega(k)) = 0`` twice and solve for the derivatives."""

    k = sp.symbols("k")
    omega = sp.Function("omega")(k)
    determinant = sp.Function("D")(k, omega)
    omega_k_derivative = sp.diff(omega, k)
    omega_kk_derivative = sp.diff(omega, k, 2)

    d_k, d_omega, d_kk, d_komega, d_omegaomega = sp.symbols(
        "D_k D_omega D_kk D_komega D_omegaomega"
    )
    replacements = {
        sp.Derivative(determinant, omega): d_omega,
        sp.Derivative(determinant, (omega, 2)): d_omegaomega,
    }

    # SymPy represents partial derivatives of a multivariate undefined
    # function with Subs objects after applying the chain rule.  xreplace on
    # these exact objects keeps this check tied to actual differentiation.
    first_chain = sp.diff(determinant, k)
    second_chain = sp.diff(determinant, k, 2)
    derivative_atoms = first_chain.atoms(sp.Subs) | second_chain.atoms(sp.Subs)
    for atom in derivative_atoms:
        derivative = atom.expr
        variables = tuple(derivative.variables)
        positions = tuple(derivative.expr.args.index(variable) for variable in variables)
        if len(positions) == 1:
            replacements[atom] = d_k if positions == (0,) else d_omega
        elif len(positions) == 2:
            if positions == (0, 0):
                replacements[atom] = d_kk
            elif positions == (1, 1):
                replacements[atom] = d_omegaomega
            else:
                replacements[atom] = d_komega

    first_algebraic = sp.expand(first_chain.xreplace(replacements))
    second_algebraic = sp.expand(second_chain.xreplace(replacements))
    first_solution = sp.solve(sp.Eq(first_algebraic, 0), omega_k_derivative, dict=False)[0]
    second_general = sp.solve(
        sp.Eq(second_algebraic, 0),
        omega_kk_derivative,
        dict=False,
    )[0]
    second_solution = sp.simplify(second_general.subs(omega_k_derivative, first_solution))

    return {
        "omega_k": sp.factor(first_solution),
        "omega_kk": sp.factor(second_solution),
        "zgv_curvature": sp.simplify(second_solution.subs(d_k, 0)),
    }


def radial_hessian_diagonalization() -> tuple[sp.Matrix, sp.Matrix]:
    """Differentiate a generic radial two-jet in Cartesian coordinates."""

    x, y = sp.symbols("x y", real=True)
    theta = sp.symbols("theta", real=True)
    r_positive = sp.symbols("r_positive", positive=True)
    r, omega_r, omega_rr, a = sp.symbols("r omega_r omega_rr a")
    radius = sp.sqrt(x**2 + y**2)
    radial_two_jet = omega_r * (radius - r_positive) + omega_rr * (radius - r_positive) ** 2 / 2
    cartesian_hessian = sp.hessian(radial_two_jet, (x, y)).subs(
        {x: r_positive * sp.cos(theta), y: r_positive * sp.sin(theta)}
    )
    radial = sp.Matrix([sp.cos(theta), sp.sin(theta)])
    tangent = sp.Matrix([-sp.sin(theta), sp.cos(theta)])
    basis = sp.Matrix.hstack(radial, tangent)
    diagonal = (basis.T * cartesian_hessian * basis).applyfunc(
        lambda entry: sp.simplify(sp.trigsimp(entry)).subs(r_positive, r)
    )
    at_zgv = diagonal.subs({omega_r: 0, omega_rr: a})
    return diagonal, at_zgv


def independent_traction_determinant_mp(
    k: object,
    omega: object,
    c_l: object,
    c_t: object,
    h: object,
) -> object:
    """Form the regular determinant directly from the physical traction columns.

    This check intentionally does not call the production determinant.  It
    reconstructs the physical two-by-two matrix, applies the ``1/s`` shear
    amplitude rescaling, and takes the determinant entry by entry.
    """

    k_value, omega_value, c_l_value, c_t_value, h_value = map(mp.mpc, (k, omega, c_l, c_t, h))
    p = mp.sqrt((omega_value / c_l_value) ** 2 - k_value**2)
    s = mp.sqrt((omega_value / c_t_value) ** 2 - k_value**2)
    sine_ratio = h_value if s == 0 else mp.sin(s * h_value) / s
    shear_factor = s**2 - k_value**2
    t11 = -2j * k_value * p * mp.sin(p * h_value)
    t12 = shear_factor * sine_ratio
    t21 = -shear_factor * mp.cos(p * h_value)
    t22 = 2j * k_value * mp.cos(s * h_value)
    return t11 * t22 - t12 * t21


def _real_mp(name: str, value: object) -> mp.mpf:
    if not mp.isfinite(value):
        raise RuntimeError(f"{name} is non-finite")
    real = mp.re(value)
    imaginary = mp.im(value)
    if abs(imaginary) > 100 * mp.eps * max(mp.mpf(1), abs(real)):
        raise RuntimeError(f"{name} is not real")
    return real


def check_reference_substitution(
    project_root: Path,
    dps: int = 80,
) -> ReferenceSubstitutionReport:
    """Refine and check the Task-4 ZGV values at arbitrary precision."""

    root = Path(project_root).resolve()
    if type(dps) is not int:
        raise TypeError("dps must be an integer")
    if dps < 60:
        raise ValueError("dps must be at least 60")

    config = load_reference_config(root / "config" / "reference.yaml")

    with mp.workdps(dps):
        c_l = mp.mpf(str(config.c_l))
        c_t = mp.mpf(str(config.c_t))
        h = mp.mpf(str(config.h))

        def determinant(kappa: mp.mpf, omega: mp.mpf) -> object:
            return _real_mp(
                "independent traction determinant",
                independent_traction_determinant_mp(kappa, omega, c_l, c_t, h),
            )

        def determinant_k(kappa: mp.mpf, omega: mp.mpf) -> object:
            return mp.diff(lambda argument: determinant(argument, omega), kappa)

        kappa0_raw, omega0_raw = mp.findroot(
            (determinant, determinant_k),
            (mp.mpf("0.8"), mp.mpf("2.85")),
        )
        kappa0 = _real_mp("kappa0", kappa0_raw)
        omega0 = _real_mp("omega0", omega0_raw)
        residual = abs(determinant(kappa0, omega0))
        production_residual = abs(det_symmetric_mp(kappa0, omega0, c_l, c_t, h))
        d_k = _real_mp("D_k", determinant_k(kappa0, omega0))
        d_omega = _real_mp(
            "D_omega",
            mp.diff(lambda argument: determinant(kappa0, argument), omega0),
        )
        d_kk = _real_mp(
            "D_kk",
            mp.diff(lambda argument: determinant(argument, omega0), kappa0, 2),
        )
        if d_omega == 0:
            raise RuntimeError("D_omega vanishes at the reference point")
        group_velocity = -d_k / d_omega
        curvature = -d_kk / d_omega

        task4 = find_s1_zgv(config, dps=dps)
        task4_errors = (
            abs(kappa0 - mp.mpf(str(task4.kappa0))),
            abs(omega0 - mp.mpf(str(task4.omega0))),
            abs(curvature - mp.mpf(str(task4.curvature_a))),
        )

        artifact_path = root / "data" / "generated" / "isotropic_zgv.npz"
        with np.load(artifact_path, allow_pickle=False) as artifact:
            artifact_values = tuple(
                mp.mpf(str(float(artifact[key]))) for key in ("kappa0", "omega0", "curvature_a")
            )
        artifact_errors = tuple(
            abs(actual - recorded)
            for actual, recorded in zip((kappa0, omega0, curvature), artifact_values, strict=True)
        )

        report = ReferenceSubstitutionReport(
            kappa0=float(kappa0),
            omega0=float(omega0),
            curvature_a=float(curvature),
            determinant_residual=float(residual),
            production_determinant_residual=float(production_residual),
            group_velocity=float(group_velocity),
            d_omega=float(d_omega),
            task4_kappa_error=float(task4_errors[0]),
            task4_omega_error=float(task4_errors[1]),
            task4_curvature_error=float(task4_errors[2]),
            artifact_kappa_error=float(artifact_errors[0]),
            artifact_omega_error=float(artifact_errors[1]),
            artifact_curvature_error=float(artifact_errors[2]),
        )

    return report


def cubic_tensor_decomposition_error(delta: float = 1.0) -> float:
    """Check the invariant decomposition of the registered cubic perturbation."""

    delta_value = float(delta)
    identity = np.eye(3)
    decomposition = -(delta_value / 3.0) * np.einsum("ij,kl->ijkl", identity, identity)
    for axis in range(3):
        direction = identity[:, axis]
        decomposition += delta_value * np.einsum(
            "i,j,k,l->ijkl", direction, direction, direction, direction
        )
    production = cubic_perturbation_tensor(delta_value)
    return float(np.max(np.abs(production - decomposition)))


def cubic_strain_contraction_residual() -> sp.Expr:
    """Contract a component-built cubic tensor with the full complex strain."""

    theta = sp.symbols("theta", real=True)
    e_real, e_imag, z_real, z_imag, s_real, s_imag = sp.symbols(
        "E_R E_I Z_R Z_I S_R S_I", real=True
    )
    c11, c12, c44, delta = sp.symbols("C11 C12 C44 delta", real=True)
    longitudinal = e_real + sp.I * e_imag
    normal = z_real + sp.I * z_imag
    shear = s_real + sp.I * s_imag
    cosine, sine = sp.cos(theta), sp.sin(theta)
    strain = sp.Matrix(
        [
            [longitudinal * cosine**2, longitudinal * cosine * sine, shear * cosine],
            [longitudinal * cosine * sine, longitudinal * sine**2, shear * sine],
            [shear * cosine, shear * sine, normal],
        ]
    )

    tensor = np.empty((3, 3, 3, 3), dtype=object)
    tensor.fill(sp.Integer(0))
    for first in range(3):
        tensor[first, first, first, first] = c11
        for second in range(3):
            if first == second:
                continue
            tensor[first, first, second, second] = c12
            tensor[first, second, first, second] = c44
            tensor[first, second, second, first] = c44

    direct = sp.Integer(0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for ell in range(3):
                    direct += sp.conjugate(strain[i, j]) * tensor[i, j, k, ell] * strain[k, ell]

    diagonal = (strain[0, 0], strain[1, 1], strain[2, 2])
    off_diagonal = (strain[0, 1], strain[0, 2], strain[1, 2])
    component_formula = c11 * sum(abs(value) ** 2 for value in diagonal)
    component_formula += (
        2
        * c12
        * sp.re(
            sp.conjugate(diagonal[0]) * diagonal[1]
            + sp.conjugate(diagonal[0]) * diagonal[2]
            + sp.conjugate(diagonal[1]) * diagonal[2]
        )
    )
    component_formula += 4 * c44 * sum(abs(value) ** 2 for value in off_diagonal)
    general_residual = sp.simplify(sp.expand_complex(direct - component_formula))

    registered = sp.simplify(direct.subs({c11: 2 * delta / 3, c12: -delta / 3, c44: 0}))
    longitudinal_squared = e_real**2 + e_imag**2
    normal_squared = z_real**2 + z_imag**2
    cross_real = e_real * z_real + e_imag * z_imag
    fourfold_target = delta * (
        sp.Rational(5, 12) * longitudinal_squared
        + sp.Rational(2, 3) * normal_squared
        - sp.Rational(2, 3) * cross_real
        + sp.Rational(1, 4) * longitudinal_squared * sp.cos(4 * theta)
    )
    registered_residual = sp.simplify(
        sp.trigsimp(sp.expand_trig(sp.expand_complex(registered - fourfold_target)))
    )
    return sp.simplify(general_residual**2 + registered_residual**2)


def closed_form_cubic_coefficients(
    project_root: Path,
    *,
    order: int = 10,
) -> dict[str, float]:
    """Evaluate the analytic ``V0`` and ``V4`` thickness integrals.

    The calculation uses one isotropic spectral eigenmode but no angular
    Fourier fit and no perturbed eigenproblem.  It is therefore independent of
    the extraction route used to create ``angular_sensitivity.npz``.
    """

    root = Path(project_root).resolve()
    config = load_reference_config(root / "config/reference.yaml")
    exact = find_s1_zgv(config, dps=60)
    matrices = assemble_plate_matrices(
        exact.kappa0,
        0.0,
        isotropic_tensor(config.lam, config.mu),
        config.rho,
        config.h,
        order=order,
    )
    modes = solve_plate_modes(matrices, 12)
    tracked = seed_tracked_mode(modes, int(np.argmin(abs(modes.omega - exact.omega0))))
    vector = np.asarray(tracked.vector)
    mass_norm = float(np.real(np.vdot(vector, matrices.mass @ vector)))

    integral_u_squared = 0.0
    integral_constant = 0.0
    for element in matrices.mesh.elements:
        connectivity = np.asarray(element.connectivity)
        in_plane = vector[3 * connectivity]
        normal = vector[3 * connectivity + 2]
        longitudinal_strain = 1j * exact.kappa0 * in_plane
        normal_strain = element.derivative @ normal
        integral_u_squared += float(np.sum(element.weights * np.abs(in_plane) ** 2))
        integrand = (
            (5.0 / 12.0) * np.abs(longitudinal_strain) ** 2
            + (2.0 / 3.0) * np.abs(normal_strain) ** 2
            - (2.0 / 3.0) * np.real(np.conj(longitudinal_strain) * normal_strain)
        )
        integral_constant += float(np.sum(element.weights * integrand))

    q4 = exact.kappa0**2 * integral_u_squared / (8.0 * tracked.omega * mass_norm)
    v4 = config.delta * q4
    v0 = config.delta * integral_constant / (2.0 * tracked.omega * mass_norm)
    return {
        "V0": float(v0),
        "V4": float(v4),
        "Q4": float(q4),
        "mass_norm": mass_norm,
    }


def synthetic_sensitivity_formula_errors() -> dict[str, float]:
    """Compare the documented reduced-resolvent formulas with public APIs."""

    omega = 2.0
    eigenvalue = omega**2
    mass = np.diag([1.2, 0.8, 1.5])
    eigenvalues = np.array([4.0, 9.0, 16.0])
    stiffness = mass @ np.diag(eigenvalues)
    radial = np.array([[0.8, -0.6, 0.2], [-0.6, 0.1, -0.3], [0.2, -0.3, -0.4]])
    perturbation = np.array([[0.7, 0.3, -0.1], [0.3, -0.2, 0.25], [-0.1, 0.25, 0.05]])
    mixed = np.array([[-0.2, 0.05, 0.07], [0.05, 0.12, -0.04], [0.07, -0.04, -0.08]])
    generalized_modes = np.diag(1.0 / np.sqrt(np.diag(mass)))
    normalized_mode = generalized_modes[:, 0]
    scale = 3.7 * np.exp(0.41j)
    arbitrary_mode = scale * normalized_mode

    mode_derivative = np.zeros(3)
    for index in (1, 2):
        basis = generalized_modes[:, index]
        mode_derivative += (
            basis * (basis @ radial @ normalized_mode) / (eigenvalue - eigenvalues[index])
        )
    lambda_radial = float(normalized_mode @ radial @ normalized_mode)
    lambda_epsilon = float(normalized_mode @ perturbation @ normalized_mode)
    lambda_radial_epsilon = float(
        normalized_mode @ mixed @ normalized_mode
        + 2.0 * normalized_mode @ perturbation @ mode_derivative
    )
    expected_v = lambda_epsilon / (2.0 * omega)
    expected_b = lambda_radial_epsilon / (2.0 * omega) - (
        lambda_epsilon * lambda_radial / (4.0 * omega**3)
    )

    scale_derivative = (-0.2 + 0.6j) * scale
    arbitrary_mode_derivative = scale_derivative * normalized_mode + scale * mode_derivative
    norm = complex(np.vdot(arbitrary_mode, mass @ arbitrary_mode))
    norm_derivative = complex(
        np.vdot(arbitrary_mode_derivative, mass @ arbitrary_mode)
        + np.vdot(arbitrary_mode, mass @ arbitrary_mode_derivative)
    )
    strain_form = complex(np.vdot(arbitrary_mode, perturbation @ arbitrary_mode))
    strain_form_derivative = complex(
        np.vdot(arbitrary_mode_derivative, perturbation @ arbitrary_mode)
        + np.vdot(arbitrary_mode, mixed @ arbitrary_mode)
        + np.vdot(arbitrary_mode, perturbation @ arbitrary_mode_derivative)
    )
    arbitrary_b = (strain_form_derivative - strain_form * norm_derivative / norm) / (
        2.0 * omega * norm
    ) - strain_form * lambda_radial / (4.0 * omega**3 * norm)

    api_v = frequency_sensitivity(
        omega,
        arbitrary_mode,
        mass,
        perturbation,
    )
    api = radial_frequency_sensitivity(
        stiffness,
        mass,
        omega,
        arbitrary_mode,
        radial,
        perturbation,
        mixed,
    )
    phase_aligned = api.mode_radial_derivative * np.exp(-0.41j)
    return {
        "V": float(max(abs(api_v - expected_v), abs(api.V - expected_v))),
        "B": float(abs(api.B - expected_b)),
        "arbitrary_B": float(abs(arbitrary_b - expected_b)),
        "lambda_epsilon": float(abs(api.lambda_epsilon - lambda_epsilon)),
        "lambda_radial_epsilon": float(abs(api.lambda_radial_epsilon - lambda_radial_epsilon)),
        "mode": float(np.linalg.norm(phase_aligned.real - mode_derivative)),
    }


def morse_splitting_data(
    harmonic: int,
    *,
    epsilon: float,
    V_m: float,
    k0: float = 2.0,
    curvature: float = 3.0,
    B0: float = 0.3,
) -> dict[str, object]:
    """Find and classify pure-harmonic critical points independently in Cartesian space."""

    if type(harmonic) is not int or harmonic <= 0 or harmonic % 2:
        raise ValueError("harmonic must be a positive even integer")
    parameters = np.asarray((epsilon, V_m, k0, curvature, B0), dtype=float)
    if not np.isfinite(parameters).all():
        raise ValueError("normal-form parameters must be finite")
    if k0 <= 0.0 or curvature <= 0.0:
        raise ValueError("k0 and radial curvature must be positive")
    if epsilon == 0.0 or V_m == 0.0 or epsilon * V_m == 0.0:
        raise ValueError("epsilon and V_m must define a resolved first-order splitting")

    def angular_derivative(angle: float) -> float:
        return float(-harmonic * V_m * np.sin(harmonic * angle))

    sample_count = 32 * harmonic
    angular_step = 2.0 * np.pi / sample_count
    grid = 0.37 * angular_step + angular_step * np.arange(sample_count + 1)
    roots: list[float] = []
    for left, right in zip(grid[:-1], grid[1:], strict=True):
        left_value = angular_derivative(left)
        right_value = angular_derivative(right)
        if np.signbit(left_value) == np.signbit(right_value):
            continue
        root = float(np.mod(brentq(angular_derivative, left, right), 2.0 * np.pi))
        if not any(
            min(abs(root - previous), 2.0 * np.pi - abs(root - previous)) < 1.0e-10
            for previous in roots
        ):
            roots.append(root)
    angles = np.sort(np.asarray(roots))
    if angles.size != 2 * harmonic:
        raise RuntimeError("the independent angular root search did not resolve 2m roots")

    critical_radius = k0 - epsilon * B0 / curvature
    if critical_radius <= 0.0:
        raise ValueError("the resolved radial critical points must have positive radius")
    tangential_scale = abs(epsilon * V_m) * harmonic**2 / critical_radius**2
    roundoff_floor = (
        512.0
        * np.finfo(float).eps
        * max(
            curvature,
            abs(epsilon * B0) / critical_radius,
            tangential_scale,
        )
    )
    if tangential_scale <= roundoff_floor:
        raise ValueError("the first-order angular curvature is not numerically resolved")

    def potential(x: float, y: float) -> float:
        radius = float(np.hypot(x, y))
        angle = float(np.arctan2(y, x))
        q = radius - k0
        return 0.5 * curvature * q**2 + epsilon * (V_m * np.cos(harmonic * angle) + B0 * q)

    kinds: list[str] = []
    indices: list[int] = []
    tangential_curvatures: list[float] = []
    frequencies: list[float] = []
    for angle in angles:
        radial_direction = np.array([np.cos(angle), np.sin(angle)])
        tangent_direction = np.array([-np.sin(angle), np.cos(angle)])
        point = critical_radius * radial_direction
        center = potential(*point)
        target_scale = max(
            abs(center),
            abs(epsilon * V_m),
            np.finfo(float).tiny,
        )
        step = (
            critical_radius
            * (
                np.finfo(float).eps
                * target_scale
                / (max(curvature, tangential_scale) * critical_radius**2)
            )
            ** 0.25
        )
        step = float(
            np.clip(
                step,
                32.0 * np.finfo(float).eps * critical_radius,
                1.0e-2 * critical_radius,
            )
        )

        def local_cartesian_hessian(step_size: float) -> np.ndarray:
            """Return Cartesian directional second differences in the local frame."""

            radial_plus = point + step_size * radial_direction
            radial_minus = point - step_size * radial_direction
            tangent_plus = point + step_size * tangent_direction
            tangent_minus = point - step_size * tangent_direction
            mixed = (
                potential(*(point + step_size * (radial_direction + tangent_direction)))
                - potential(*(point + step_size * (radial_direction - tangent_direction)))
                - potential(*(point + step_size * (-radial_direction + tangent_direction)))
                + potential(*(point - step_size * (radial_direction + tangent_direction)))
            ) / (4.0 * step_size**2)
            return np.array(
                [
                    [
                        (potential(*radial_plus) - 2.0 * center + potential(*radial_minus))
                        / step_size**2,
                        mixed,
                    ],
                    [
                        mixed,
                        (potential(*tangent_plus) - 2.0 * center + potential(*tangent_minus))
                        / step_size**2,
                    ],
                ]
            )

        coarse_hessian = local_cartesian_hessian(step)
        fine_hessian = local_cartesian_hessian(step / 2.0)
        hessian = (4.0 * fine_hessian - coarse_hessian) / 3.0

        def schur_complement(matrix: np.ndarray) -> float:
            return float(matrix[1, 1] - matrix[0, 1] ** 2 / matrix[0, 0])

        radial_pivot = float(hessian[0, 0])
        fine_schur = schur_complement(fine_hessian)
        tangential_schur = schur_complement(hessian)
        schur_error = 16.0 * abs(tangential_schur - fine_schur)
        resolution = max(roundoff_floor, schur_error)
        if (
            not np.isfinite(hessian).all()
            or not np.isfinite((radial_pivot, tangential_schur, resolution)).all()
            or abs(radial_pivot) <= roundoff_floor
            or abs(tangential_schur) <= resolution
        ):
            raise ValueError("the Cartesian Hessian inertia is not numerically resolved")
        if radial_pivot > 0.0 and tangential_schur > 0.0:
            kinds.append("minimum")
            indices.append(1)
        elif radial_pivot < 0.0 and tangential_schur < 0.0:
            kinds.append("maximum")
            indices.append(1)
        else:
            kinds.append("saddle")
            indices.append(-1)
        tangential_curvatures.append(tangential_schur)
        frequencies.append(center)

    frequency_values = np.asarray(frequencies)
    return {
        "theta": angles,
        "kind": np.asarray(kinds),
        "index": np.asarray(indices),
        "tangential_curvature": np.asarray(tangential_curvatures),
        "radial_shift": np.full(angles.size, critical_radius - k0),
        "frequency_separation": float(np.max(frequency_values) - np.min(frequency_values)),
    }


def normal_form_hessian_identity() -> tuple[sp.Matrix, sp.Expr]:
    """Check the full Cartesian--polar Hessian map and its critical Schur pivot."""

    x, y = sp.symbols("x y", real=True)
    f_r, f_theta, f_rr, f_rtheta, f_thetatheta = sp.symbols(
        "f_r f_theta f_rr f_rtheta f_thetatheta", real=True
    )
    epsilon, b_prime, v_second = sp.symbols("epsilon B_prime V_second", real=True)
    a, k0 = sp.symbols("a k_0", positive=True)
    radius = sp.sqrt(x**2 + y**2)
    angle = sp.atan(y / x)
    radial_offset = radius - k0
    phase_two_jet = (
        f_r * radial_offset
        + f_theta * angle
        + f_rr * radial_offset**2 / 2
        + f_rtheta * radial_offset * angle
        + f_thetatheta * angle**2 / 2
    )
    transformed = sp.hessian(phase_two_jet, (x, y)).subs({x: k0, y: 0}).applyfunc(sp.simplify)
    general_documented = sp.Matrix(
        [
            [f_rr, f_rtheta / k0 - f_theta / k0**2],
            [
                f_rtheta / k0 - f_theta / k0**2,
                f_r / k0 + f_thetatheta / k0**2,
            ],
        ]
    )
    critical = transformed.subs(
        {
            f_r: 0,
            f_theta: 0,
            f_rr: a,
            f_rtheta: epsilon * b_prime,
            f_thetatheta: epsilon * v_second,
        }
    )
    schur = sp.simplify(critical[1, 1] - critical[0, 1] ** 2 / critical[0, 0])
    expected_schur = epsilon * v_second / k0**2 - epsilon**2 * b_prime**2 / (a * k0**2)
    return sp.simplify(transformed - general_documented), sp.simplify(schur - expected_schur)


def exceptional_second_order_splitting(
    epsilon: float,
    *,
    harmonic: int,
    amplitude: float,
) -> dict[str, float | int]:
    """Resolve the ``V_m=0`` fixture split by a second-order angular potential."""

    if type(harmonic) is not int or harmonic <= 0 or harmonic % 2:
        raise ValueError("harmonic must be a positive even integer")
    epsilon_value = float(epsilon)
    amplitude_value = float(amplitude)
    if not np.isfinite((epsilon_value, amplitude_value)).all():
        raise ValueError("epsilon and second-order amplitude must be finite")
    if epsilon_value == 0.0 or amplitude_value == 0.0:
        raise ValueError("second-order splitting requires nonzero epsilon and amplitude")
    angles = np.arange(2 * harmonic, dtype=float) * np.pi / harmonic
    angular_curvature = -(
        epsilon_value**2 * amplitude_value * harmonic**2 * np.cos(harmonic * angles)
    )
    if not np.isfinite(angular_curvature).all() or not np.all(angular_curvature != 0.0):
        raise ValueError("second-order angular curvature must be finite and resolved")
    minimum_count = int(np.count_nonzero(angular_curvature > 0.0))
    saddle_count = int(np.count_nonzero(angular_curvature < 0.0))
    return {
        "point_count": minimum_count + saddle_count,
        "minimum_count": minimum_count,
        "saddle_count": saddle_count,
        "curvature_scale": float(np.max(np.abs(angular_curvature))),
    }


def angular_bessel_identity_errors(taus: tuple[float, ...]) -> dict[float, float]:
    """Independently quadrature-check the fourfold angular Bessel identity."""

    errors: dict[float, float] = {}
    with mp.workdps(80):
        breakpoints = [mp.pi * index / 8 for index in range(17)]
        for raw_tau in taus:
            tau = mp.mpf(str(raw_tau))
            integral = mp.quad(
                lambda theta: mp.exp(-mp.j * tau * mp.cos(4 * theta)),
                breakpoints,
            )
            expected = 2 * mp.pi * mp.besselj(0, tau)
            errors[float(raw_tau)] = float(abs(integral - expected))
    return errors


def regularized_fresnel_identity_error() -> float:
    """Check the convergent Gaussian continuation of the radial Fresnel factor."""

    with mp.workdps(80):
        eta = mp.mpf("0.9")
        curvature = mp.mpf("1.7")
        time = mp.mpf("4.25")
        coefficient = eta + mp.j * curvature * time / 2
        integral = mp.quad(
            lambda q: mp.exp(-coefficient * q**2),
            [-mp.inf, -4, -2, -1, 0, 1, 2, 4, mp.inf],
        )
        expected = mp.sqrt(mp.pi / coefficient)
        large_time = mp.mpf("1e60")
        explicit_fresnel = mp.exp(-mp.j * mp.pi / 4) * mp.sqrt(2 * mp.pi / curvature)
        scaled_limit = mp.sqrt(large_time) * mp.sqrt(
            mp.pi / (eta + mp.j * curvature * large_time / 2)
        )
        return float(max(abs(integral - expected), abs(scaled_limit - explicit_fresnel)))


def large_bessel_phase_errors() -> np.ndarray:
    """Return relative leading-order J0 errors at phase-safe large arguments."""

    errors: list[float] = []
    with mp.workdps(80):
        for multiplier in (20, 40, 80):
            tau = 2 * mp.pi * multiplier + mp.pi / 4
            leading = mp.sqrt(2 / (mp.pi * tau)) * mp.cos(tau - mp.pi / 4)
            signed_errors = [
                abs(mp.besselj(0, sign * tau) - leading) / abs(mp.besselj(0, sign * tau))
                for sign in (-1, 1)
            ]
            errors.append(float(max(signed_errors)))
    return np.asarray(errors)


def quadratic_signature_phase_errors() -> dict[str, float]:
    """Recover Morse signature phases from exact regularized Gaussian integrals."""

    eta = 0.8
    time = 1.0e8
    hessians = {
        "minimum": np.array([2.0, 3.0]),
        "saddle": np.array([2.0, -3.0]),
        "maximum": np.array([-2.0, -3.0]),
    }
    errors: dict[str, float] = {}
    for name, eigenvalues in hessians.items():
        exact_integral = np.pi / (
            np.sqrt(eta + 0.5j * time * eigenvalues[0])
            * np.sqrt(eta + 0.5j * time * eigenvalues[1])
        )
        scaled = exact_integral * time * np.sqrt(abs(np.prod(eigenvalues))) / (2.0 * np.pi)
        signature = int(np.count_nonzero(eigenvalues > 0.0) - np.count_nonzero(eigenvalues < 0.0))
        expected = np.exp(-0.25j * np.pi * signature)
        errors[name] = float(abs(scaled - expected))
    return errors


def oscillator_positive_frequency_residual() -> sp.Expr:
    """Verify the undoubled positive-frequency component of an impulse oscillator."""

    omega, time = sp.symbols("omega t", positive=True, real=True)
    positive = sp.I * sp.exp(-sp.I * omega * time) / (2 * omega)
    negative = -sp.I * sp.exp(sp.I * omega * time) / (2 * omega)
    return sp.simplify(sp.expand_complex(positive + negative - sp.sin(omega * time) / omega))


def synthetic_branch_projection_errors() -> dict[str, float]:
    """Check normalization invariance and the co-located normal impulse channel."""

    omega = 2.7
    mass = np.array(
        [
            [1.4, 0.1 - 0.05j, 0.0],
            [0.1 + 0.05j, 1.1, -0.08j],
            [0.0, 0.08j, 0.9],
        ],
        dtype=np.complex128,
    )
    mode = np.array([0.7 + 0.2j, -0.3 + 0.4j, 0.5 - 0.1j])
    source = np.array([0.2 - 0.1j, 0.8 + 0.3j, -0.4 + 0.2j])
    detector = np.array([0.6 + 0.2j, -0.1j, 0.5 + 0.4j])

    def projected_amplitude(vector: np.ndarray) -> complex:
        return complex(
            1j
            * np.vdot(detector, vector)
            * np.vdot(vector, source)
            / (2.0 * omega * np.vdot(vector, mass @ vector))
        )

    reference = projected_amplitude(mode)
    scaled = projected_amplitude((3.2 - 1.7j) * mode)
    mass_norm = float(np.real(np.vdot(mode, mass @ mode)))
    normalized_mode = mode / np.sqrt(mass_norm)
    top_normal_component = complex(normalized_mode[-1])
    normal_expected = 1j * abs(top_normal_component) ** 2 / (2.0 * omega)
    impulse_amplitude = normal_impulse_amplitude(omega, top_normal_component)
    time = 1.4
    positive_response = impulse_amplitude * np.exp(-1j * omega * time)
    physical = 2.0 * np.real(positive_response)
    oscillator_response = abs(top_normal_component) ** 2 * np.sin(omega * time) / omega
    return {
        "normalization": float(abs(reference - scaled)),
        "normal_impulse": float(abs(impulse_amplitude - normal_expected)),
        "physical_reconstruction": float(abs(physical - oscillator_response)),
    }


def polar_jacobian_cancellation_error() -> float:
    """Check the polar-measure cancellation for an exact Cartesian Morse Hessian."""

    radius = 1.8
    angle = 0.63
    amplitude = 0.7 - 0.2j
    cartesian = np.array([[2.3, 0.41], [0.41, -0.72]])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    coordinate_map = rotation @ np.diag([1.0, radius])
    polar = coordinate_map.T @ cartesian @ coordinate_map
    determinant_error = abs(np.linalg.det(polar) - radius**2 * np.linalg.det(cartesian))
    polar_weight = radius * amplitude / np.sqrt(abs(np.linalg.det(polar)))
    cartesian_weight = amplitude / np.sqrt(abs(np.linalg.det(cartesian)))
    return float(max(determinant_error, abs(polar_weight - cartesian_weight)))


def bessel_morse_overlap_error() -> float:
    """Match the full leading large-J0 expression to all eight Morse points."""

    relative_errors: list[float] = []
    with mp.workdps(80):
        normalization = (2 * mp.pi) ** -2
        k0 = mp.mpf("1.7")
        amplitude = mp.mpc("0.8", "-0.3")
        curvature = mp.mpf("2.4")
        time = mp.mpf("230")
        carrier = mp.mpf("5.2")
        for delta in (mp.mpf("-0.037"), mp.mpf("0.037")):
            transition = abs(delta) * time
            prefactor = (
                normalization
                * 2
                * mp.pi
                * k0
                * amplitude
                * mp.exp(-mp.j * mp.pi / 4)
                * mp.sqrt(2 * mp.pi / curvature)
            )
            bessel_late = (
                prefactor
                * mp.exp(-mp.j * carrier * time)
                * time ** (-mp.mpf("0.5"))
                * mp.sqrt(2 / (mp.pi * transition))
                * mp.cos(transition - mp.pi / 4)
            )

            point_sum = mp.mpc(0)
            for index in range(8):
                theta = mp.pi * index / 4
                cosine = mp.cos(4 * theta)
                tangential_eigenvalue = -16 * delta * cosine / k0**2
                signature = 2 if tangential_eigenvalue > 0 else 0
                frequency = carrier + delta * cosine
                point_sum += (
                    amplitude
                    * mp.exp(-mp.j * frequency * time - mp.j * mp.pi * signature / 4)
                    / mp.sqrt(abs(curvature * tangential_eigenvalue))
                )
            morse_sum = 2 * mp.pi * normalization * point_sum / time
            relative_errors.append(
                float(abs(bessel_late - morse_sum) / max(abs(bessel_late), abs(morse_sum)))
            )
    return max(relative_errors)


def production_eight_point_overlap_error() -> float:
    """Cross-check the production Morse sum against the explicit late-Bessel limit."""

    normalization = (2.0 * np.pi) ** -2
    carrier = 4.0
    k0 = 2.0
    curvature = 1.5
    amplitude = 0.43 - 0.27j
    transition = 40.0 * np.pi + 0.31
    relative_errors: list[float] = []
    for delta in (-0.03125, 0.03125):
        time = transition / abs(delta)
        contributions: list[MorseContribution] = []
        for index in range(8):
            angle = index * np.pi / 4.0
            cosine = (-1.0) ** index
            tangential_eigenvalue = -16.0 * delta * cosine / k0**2
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            hessian = rotation @ np.diag([curvature, tangential_eigenvalue]) @ rotation.T
            contributions.append(
                MorseContribution(
                    carrier + delta * cosine,
                    hessian,
                    amplitude,
                    0.0,
                    0.0,
                )
            )
        production = morse_stationary_phase_response(
            np.array([time]),
            contributions,
            carrier,
            normalization,
        ).demodulated[0]
        prefactor = (
            normalization
            * 2.0
            * np.pi
            * k0
            * amplitude
            * np.exp(-0.25j * np.pi)
            * np.sqrt(2.0 * np.pi / curvature)
        )
        explicit = (
            prefactor
            * time**-0.5
            * np.sqrt(2.0 / (np.pi * transition))
            * np.cos(transition - np.pi / 4.0)
        )
        relative_errors.append(float(abs(production - explicit) / abs(explicit)))
    return max(relative_errors)


def growing_tau_overlap_errors() -> np.ndarray:
    """Test the two-parameter overlap with nonconstant amplitude and W2eff."""

    number_of_angles = 2**16
    angles = 2.0 * np.pi * np.arange(number_of_angles) / number_of_angles
    effective_second_order = 0.23 * np.cos(8.0 * angles) + 0.11 * np.cos(4.0 * angles)
    amplitude_correction = 0.17 * np.cos(4.0 * angles) - 0.09 * np.cos(8.0 * angles)
    normalized_errors: list[float] = []
    for multiplier in (10, 20, 40):
        transition = 2.0 * np.pi * multiplier + np.pi / 4.0
        epsilon = transition**-2
        time = transition / epsilon
        amplitude = 1.0 + epsilon * amplitude_correction
        phase = time * (epsilon * np.cos(4.0 * angles) + epsilon**2 * effective_second_order)
        quadrature = 2.0 * np.pi * np.mean(amplitude * np.exp(-1j * phase))
        first_order_stationary_phase = (
            2.0 * np.pi * np.sqrt(2.0 / (np.pi * transition)) * np.cos(transition - np.pi / 4.0)
        )
        normalized_errors.append(
            float(abs(quadrature - first_order_stationary_phase) * np.sqrt(transition))
        )
    return np.asarray(normalized_errors)


def frequency_modulation_errors() -> dict[str, float]:
    """Compare the public frequency observables with the perturbative identities."""

    epsilon = -0.037
    coefficient = 0.62
    angles = np.arange(8, dtype=float) * np.pi / 4.0
    first_order_frequencies = epsilon * coefficient * np.cos(4.0 * angles)
    expected_separation = float(np.max(first_order_frequencies) - np.min(first_order_frequencies))
    expected_rate = expected_separation / 2.0
    modulation = signed_modulation_rate(epsilon, coefficient)
    separation = critical_frequency_separation(epsilon, coefficient)
    return {
        "separation": float(abs(separation - expected_separation)),
        "modulation": float(abs(modulation - expected_rate)),
        "factor_two": float(abs(separation - 2.0 * modulation)),
    }


def weighted_bessel_identity_errors() -> dict[float, float]:
    """Quadrature-check the general angular Fourier--Bessel selection rule."""

    raw_coefficients = {
        0: 0.3 - 0.4j,
        4: -0.2 + 0.7j,
        -4: 0.5 + 0.1j,
        8: -0.15 - 0.25j,
        -8: 0.33 - 0.12j,
        2: 0.91 + 0.17j,
    }
    errors: dict[float, float] = {}
    with mp.workdps(80):
        coefficients = {
            harmonic: mp.mpc(str(value.real), str(value.imag))
            for harmonic, value in raw_coefficients.items()
        }
        breakpoints = [mp.pi * index / 8 for index in range(17)]
        for raw_tau in (-5.5, -1.25, 0.0, 2.75, 9.0):
            tau = mp.mpf(str(raw_tau))

            def integrand(theta: mp.mpf) -> mp.mpc:
                weight = sum(
                    coefficient * mp.exp(mp.j * harmonic * theta)
                    for harmonic, coefficient in coefficients.items()
                )
                return weight * mp.exp(-mp.j * tau * mp.cos(4 * theta))

            quadrature = mp.quad(integrand, breakpoints) / (2 * mp.pi)
            series = mp.mpc(0)
            for harmonic, coefficient in coefficients.items():
                if harmonic % 4:
                    continue
                order = -harmonic // 4
                series += coefficient * (-mp.j) ** order * mp.besselj(order, tau)
            errors[raw_tau] = float(abs(quadrature - series))
    return errors


def _assert_symbolic_identities() -> None:
    if traction_determinant_residual() != 0:
        raise AssertionError("scaled traction determinant does not equal D_S")
    if shear_cutoff_limit_residual() != 0:
        raise AssertionError("the documented shear-cutoff limit is incorrect")
    formulas = implicit_derivative_formulas()
    d_kk, d_omega = sp.symbols("D_kk D_omega")
    if sp.simplify(formulas["zgv_curvature"] + d_kk / d_omega) != 0:
        raise AssertionError("the ZGV curvature identity is incorrect")
    general, at_zgv = radial_hessian_diagonalization()
    r, omega_r, omega_rr, a = sp.symbols("r omega_r omega_rr a")
    if general != sp.diag(omega_rr, omega_r / r) or at_zgv != sp.diag(a, 0):
        raise AssertionError("the radial Hessian diagonalization is incorrect")


def _assert_anisotropic_identities(project_root: Path) -> dict[str, float]:
    if cubic_tensor_decomposition_error() >= 1.0e-15:
        raise AssertionError("the cubic invariant decomposition is incorrect")
    if cubic_strain_contraction_residual() != 0:
        raise AssertionError("the cubic strain contraction is incorrect")
    matrix_residual, schur_residual = normal_form_hessian_identity()
    if matrix_residual != sp.zeros(2) or schur_residual != 0:
        raise AssertionError("the normal-form Cartesian Hessian is incorrect")
    sensitivity_errors = synthetic_sensitivity_formula_errors()
    if max(sensitivity_errors.values()) >= 2.0e-14:
        raise AssertionError("the documented V or B formula disagrees with the public API")
    for harmonic in (2, 4, 6):
        splitting = morse_splitting_data(harmonic, epsilon=0.03, V_m=0.4)
        if (
            len(splitting["theta"]) != 2 * harmonic
            or np.count_nonzero(splitting["kind"] == "minimum") != harmonic
            or np.count_nonzero(splitting["kind"] == "saddle") != harmonic
            or int(np.sum(splitting["index"])) != 0
        ):
            raise AssertionError("the alternating Morse count or index closure is incorrect")
    exceptional = exceptional_second_order_splitting(0.04, harmonic=4, amplitude=0.7)
    if exceptional["point_count"] != 8 or exceptional["curvature_scale"] <= 0.0:
        raise AssertionError("the V4=0 second-order exception is not represented")
    coefficients = closed_form_cubic_coefficients(project_root, order=10)
    with np.load(
        project_root / "data/generated/angular_sensitivity.npz", allow_pickle=False
    ) as data:
        if (
            abs(coefficients["V0"] - float(data["V0"])) >= 2.0e-13
            or abs(coefficients["V4"] - float(data["V4"])) >= 2.0e-13
        ):
            raise AssertionError("closed cubic coefficients disagree with the full artifact")
    return coefficients


def _assert_green_identities() -> None:
    if oscillator_positive_frequency_residual() != 0:
        raise AssertionError("the positive-frequency oscillator decomposition is incorrect")
    if max(angular_bessel_identity_errors((-11.25, -2.0, 0.0, 3.75, 14.0)).values()) >= 1.0e-45:
        raise AssertionError("the fourfold angular Bessel identity is incorrect")
    if regularized_fresnel_identity_error() >= 1.0e-45:
        raise AssertionError("the regularized radial Fresnel identity is incorrect")
    large_errors = large_bessel_phase_errors()
    if large_errors[0] >= 4.5e-6 or not np.all(large_errors[1:] < 0.27 * large_errors[:-1]):
        raise AssertionError("the large-J0 phase or error order is incorrect")
    if max(quadratic_signature_phase_errors().values()) >= 1.0e-8:
        raise AssertionError("the Morse signature phases are incorrect")
    if max(synthetic_branch_projection_errors().values()) >= 2.0e-14:
        raise AssertionError("the branch projection or normal impulse amplitude is incorrect")
    if polar_jacobian_cancellation_error() >= 2.0e-14:
        raise AssertionError("the polar Jacobian does not cancel against the Hessian map")
    if bessel_morse_overlap_error() >= 2.0e-14:
        raise AssertionError("the eight-point Morse sum does not match the large-J0 limit")
    if production_eight_point_overlap_error() >= 2.0e-14:
        raise AssertionError("the production Morse sum has an incorrect normalization")
    overlap_errors = growing_tau_overlap_errors()
    if overlap_errors[0] >= 1.9e-2 or not np.all(overlap_errors[1:] < 0.51 * overlap_errors[:-1]):
        raise AssertionError("the growing-tau overlap does not have the predicted error order")
    if max(frequency_modulation_errors().values()) >= 2.0e-15:
        raise AssertionError("frequency separation and modulation disagree")
    if max(weighted_bessel_identity_errors().values()) >= 1.0e-40:
        raise AssertionError("the weighted Fourier--Bessel selection rule is incorrect")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing config/ and data/generated/",
    )
    parser.add_argument("--dps", type=int, default=80, help="mpmath decimal precision")
    arguments = parser.parse_args()

    _assert_symbolic_identities()
    cubic_coefficients = _assert_anisotropic_identities(arguments.project_root)
    _assert_green_identities()
    report = check_reference_substitution(arguments.project_root, dps=arguments.dps)
    if report.determinant_residual >= 10.0 ** (-(arguments.dps - 10)):
        raise AssertionError("high-precision determinant residual is too large")
    if report.production_determinant_residual >= 10.0 ** (-(arguments.dps - 10)):
        raise AssertionError("production determinant disagrees at the refined root")
    if abs(report.group_velocity) >= 10.0 ** (-(arguments.dps - 10)):
        raise AssertionError("high-precision group velocity is too large")
    if abs(report.d_omega - (-9.856471487573888)) >= 5.0e-14:
        raise AssertionError("fixed-normalization D_omega regression failed")
    if (
        max(
            report.task4_kappa_error,
            report.task4_omega_error,
            report.artifact_kappa_error,
            report.artifact_omega_error,
        )
        >= 5.0e-16
    ):
        raise AssertionError("reference location disagrees with Task 4 or its artifact")
    if max(report.task4_curvature_error, report.artifact_curvature_error) >= 5.0e-15:
        raise AssertionError("reference curvature disagrees with Task 4 or its artifact")

    print("symbolic and asymptotic identities: PASS")
    print(
        "reference substitution: "
        f"k0={report.kappa0:.17g}, omega0={report.omega0:.17g}, "
        f"a={report.curvature_a:.17g}"
    )
    print(
        "high-precision residuals: "
        f"|D|={report.determinant_residual:.3e}, "
        f"|D_production|={report.production_determinant_residual:.3e}, "
        f"|omega_k|={abs(report.group_velocity):.3e}, "
        f"D_omega={report.d_omega:.17g}"
    )
    print(
        "cubic coefficient closure: "
        f"V0={cubic_coefficients['V0']:.17g}, "
        f"V4={cubic_coefficients['V4']:.17g}, "
        f"Q4={cubic_coefficients['Q4']:.17g}"
    )


if __name__ == "__main__":
    main()
