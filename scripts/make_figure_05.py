"""Generate main Figure 5 and its exact panel-level source data."""

from pathlib import Path

from zgv_morse.figures.figure05_scaling import build


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    build(
        ROOT / "data/generated",
        ROOT / "figures/main",
        ROOT / "data/source_data/figure_05",
    )
