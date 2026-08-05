"""Generate main Figure 6 and its exact panel-level source data."""

from pathlib import Path

from zgv_morse.figures.figure06_crossover import build


if __name__ == "__main__":
    build(
        Path("data/generated"),
        Path("figures/main"),
        Path("data/source_data/figure_06"),
    )
