"""Run and export the exact-versus-spectral isotropic validation."""

from __future__ import annotations

from pathlib import Path

from zgv_morse.config import load_reference_config
from zgv_morse.validation import run_isotropic_validation, write_isotropic_validation


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_reference_config(ROOT / "config/reference.yaml")
    exact, rows, split = run_isotropic_validation(config)
    write_isotropic_validation(
        exact,
        rows,
        split,
        ROOT / "data/generated/isotropic_validation.json",
        ROOT / "data/generated/isotropic_convergence.csv",
    )
    final = rows[-1]
    print(f"k0={final.k_zgv:.15g} omega0={final.omega_zgv:.15g} a={final.curvature:.15g}")


if __name__ == "__main__":
    main()
