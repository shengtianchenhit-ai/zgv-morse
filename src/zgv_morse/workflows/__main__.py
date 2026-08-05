"""Command-line adapter for the seven deterministic workflow stages."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..config import load_reference_config
from . import STAGES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", type=Path, default=Path("config/reference.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/generated"))
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Parse registered CLI choices, run one stage, and return its artifact."""

    arguments = _parser().parse_args(argv)
    return STAGES[arguments.stage](
        load_reference_config(arguments.config),
        arguments.output,
        arguments.profile,
    )


if __name__ == "__main__":
    main()
