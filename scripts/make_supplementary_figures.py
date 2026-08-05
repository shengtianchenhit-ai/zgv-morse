"""Generate all six supplementary figures from validated records only."""

from pathlib import Path

from zgv_morse.figures.supplementary import build_all


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    build_all(
        ROOT / "data/generated",
        ROOT / "data/generated/isotropic_validation.json",
        ROOT / "data/generated/isotropic_convergence.csv",
        ROOT / "config/reference.yaml",
        ROOT / "figures/supplementary",
        ROOT / "data/source_data/supplementary",
    )
