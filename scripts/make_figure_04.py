"""Generate main-text Figure 4 and its panel-level source data."""

from pathlib import Path

from zgv_morse.figures.figure04_morse import build


if __name__ == "__main__":
    build(
        Path("data/generated"),
        Path("figures/main"),
        Path("data/source_data/figure_04"),
    )
