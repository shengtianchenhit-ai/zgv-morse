from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np
import pytest

from zgv_morse.config import ReferenceConfig, load_reference_config
from zgv_morse.workflows import critical_points, green, scaling
from zgv_morse.workflows.common import write_stage_artifact


ROOT = Path(__file__).resolve().parents[1]
DependencyLoader = Callable[[ReferenceConfig, Path, str], dict[str, np.ndarray]]


def _sensitivity_arrays() -> dict[str, np.ndarray]:
    theta = 2.0 * np.pi * np.arange(16) / 16
    V = 0.2 + 0.03 * np.cos(4.0 * theta)
    B = -0.1 + 0.02 * np.cos(4.0 * theta)
    epsilon = np.array([0.005, 0.01, 0.02], dtype=np.float64)
    return {
        "theta": theta,
        "V": V,
        "B": B,
        "V_reconstruction": V.copy(),
        "harmonic_order": np.array([0, 4, 8], dtype=np.int64),
        "harmonic_amplitude": np.array([0.2, 0.03, 0.0]),
        "V0": np.array(0.2),
        "V4": np.array(0.03),
        "V8": np.array(0.0),
        "epsilon": epsilon,
        "delta_c": 0.01 * epsilon,
        "physical_V4_shift": 0.03 * epsilon,
        "V_fd": V.copy(),
        "B_fd": B.copy(),
    }


def _write_sensitivity(
    cfg: ReferenceConfig,
    output_dir: Path,
    profile: str,
) -> Path:
    arrays = _sensitivity_arrays()
    return write_stage_artifact(
        "sensitivity",
        cfg,
        output_dir,
        profile,
        arrays,
        {key: "1" for key in arrays},
        {"synthetic_dependency_fixture": 0.0},
    )


@pytest.fixture
def cfg() -> ReferenceConfig:
    return load_reference_config(ROOT / "config/reference.yaml")


@pytest.mark.parametrize(
    "loader",
    [
        critical_points._sensitivity_arrays,
        scaling._sensitivity_arrays,
        green._load_sensitivity,
    ],
)
def test_sensitivity_dependency_rejects_mismatched_profile(
    loader: DependencyLoader,
    cfg: ReferenceConfig,
    tmp_path: Path,
) -> None:
    _write_sensitivity(cfg, tmp_path, "smoke")

    with pytest.raises(RuntimeError, match="profile"):
        loader(cfg, tmp_path, "full")


@pytest.mark.parametrize(
    "loader",
    [
        critical_points._sensitivity_arrays,
        scaling._sensitivity_arrays,
        green._load_sensitivity,
    ],
)
def test_sensitivity_dependency_rejects_mismatched_config(
    loader: DependencyLoader,
    cfg: ReferenceConfig,
    tmp_path: Path,
) -> None:
    _write_sensitivity(cfg, tmp_path, "smoke")
    changed = replace(
        cfg,
        source_radius_over_h=1.1 * cfg.source_radius_over_h,
    )

    with pytest.raises(RuntimeError, match="config"):
        loader(changed, tmp_path, "smoke")


@pytest.mark.parametrize(
    "loader",
    [
        critical_points._sensitivity_arrays,
        scaling._sensitivity_arrays,
        green._load_sensitivity,
    ],
)
def test_sensitivity_dependency_rejects_tampered_npz(
    loader: DependencyLoader,
    cfg: ReferenceConfig,
    tmp_path: Path,
) -> None:
    path = _write_sensitivity(cfg, tmp_path, "smoke")
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="checksum|integrity"):
        loader(cfg, tmp_path, "smoke")


@pytest.mark.parametrize(
    ("module", "loader", "required"),
    [
        (
            critical_points,
            critical_points._sensitivity_arrays,
            {"theta", "V", "B", "V4"},
        ),
        (scaling, scaling._sensitivity_arrays, {"theta", "V", "B", "V4"}),
        (green, green._load_sensitivity, {"V0", "V4"}),
    ],
)
def test_missing_sensitivity_dependency_recomputes_standalone(
    module: ModuleType,
    loader: DependencyLoader,
    required: set[str],
    cfg: ReferenceConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _sensitivity_arrays()
    calls: list[tuple[ReferenceConfig, str]] = []

    def compute(config: ReferenceConfig, profile: str):
        calls.append((config, profile))
        return expected, {}

    monkeypatch.setattr(module, "compute_sensitivity_arrays", compute)

    loaded = loader(cfg, tmp_path, "full")

    assert calls == [(cfg, "full")]
    assert set(loaded) >= required
    assert not (tmp_path / "angular_sensitivity.npz").exists()
