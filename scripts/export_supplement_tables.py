"""Generate supplementary TeX tables from validated records and configuration."""

from pathlib import Path

from zgv_morse.figures.supplementary import export_tables


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    export_tables(
        ROOT / "data/generated",
        ROOT / "data/generated/isotropic_validation.json",
        ROOT / "data/generated/isotropic_convergence.csv",
        ROOT / "config/reference.yaml",
        ROOT / "paper/generated",
    )
