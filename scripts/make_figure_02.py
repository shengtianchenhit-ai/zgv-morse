"""Generate main Figure 2 and its panel-level source data."""

from pathlib import Path

from zgv_morse.figures.figure02_isotropic import build


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    build(
        ROOT / "data/generated",
        ROOT / "figures/main",
        ROOT / "data/source_data/figure_02",
    )
