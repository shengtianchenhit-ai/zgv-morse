"""Export validated artifact values as deterministic LaTeX commands."""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile

import numpy as np

from zgv_morse.artifact_schema import validate_artifact


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/generated"
OUTPUT = ROOT / "paper/generated/results_macros.tex"

VALUES = {
    "ZGVKappa": ("isotropic_zgv", "kappa0"),
    "ZGVOmega": ("isotropic_zgv", "omega0"),
    "ZGVCurvature": ("isotropic_zgv", "curvature_a"),
    "VFour": ("angular_sensitivity", "V4"),
    "SplittingSlope": ("perturbation_scaling", "slope_splitting"),
    "RemainderSlope": ("perturbation_scaling", "slope_remainder"),
    "EarlyDecaySlope": ("green_crossover", "slope_early"),
    "LateDecaySlope": ("green_crossover", "slope_late"),
    "MaxPhaseError": ("green_crossover", "phase_error"),
}


def _latex_number(value: float) -> str:
    """Render one finite value with six significant digits and TeX exponents."""

    if not math.isfinite(value):
        raise ValueError("manuscript macro values must be finite")
    rendered = format(value, ".6g")
    if "e" not in rendered:
        return rendered
    mantissa, exponent = rendered.split("e", maxsplit=1)
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def _load_artifact(name: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / f"{name}.npz"
    arrays, _metadata = validate_artifact(path, path.with_suffix(".json"))
    return arrays


def _scalarize(name: str, values: np.ndarray) -> float:
    array = np.asarray(values)
    if name == "MaxPhaseError":
        if array.size == 0:
            raise ValueError("MaxPhaseError cannot be computed from an empty array")
        return float(np.max(array))
    if array.size != 1:
        raise ValueError(f"{name} requires exactly one artifact value, got {array.size}")
    return float(array.reshape(-1)[0])


def render_macros() -> str:
    """Validate all inputs and return the complete canonical macro file."""

    cache: dict[str, dict[str, np.ndarray]] = {}
    lines: list[str] = []
    for macro, (artifact, key) in VALUES.items():
        if artifact not in cache:
            cache[artifact] = _load_artifact(artifact)
        value = _scalarize(macro, cache[artifact][key])
        lines.append(rf"\newcommand{{\{macro}}}{{{_latex_number(value)}}}")

    critical = _load_artifact("critical_points")
    kinds = np.asarray(critical["kind"])
    minimum_count = int(np.count_nonzero(kinds == "minimum"))
    saddle_count = int(np.count_nonzero(kinds == "saddle"))
    if kinds.size != 8 or minimum_count != 4 or saddle_count != 4:
        raise ValueError(
            "critical_points must contain exactly four minima and four saddles "
            "before count macros are exported"
        )
    lines.extend(
        (
            rf"\newcommand{{\MorseMinimumCount}}{{{minimum_count}}}",
            rf"\newcommand{{\MorseSaddleCount}}{{{saddle_count}}}",
        )
    )
    return "\n".join(lines) + "\n"


def write_macros(text: str, output: Path = OUTPUT) -> None:
    """Atomically replace the generated file with UTF-8, newline-stable text."""

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    write_macros(render_macros())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
