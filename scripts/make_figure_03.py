"""Generate main Figure 3 and its panel-level source data."""

from pathlib import Path

from zgv_morse.figures.figure03_sensitivity import build


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    build(
        ROOT / "data/generated",
        ROOT / "figures/main",
        ROOT / "data/source_data/figure_03",
    )
