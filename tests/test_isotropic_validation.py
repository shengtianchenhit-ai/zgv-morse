from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError, asdict, fields, replace
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from zgv_morse.config import ReferenceConfig, load_reference_config
from zgv_morse.validation import (
    IsotropicBenchmarkRow,
    run_isotropic_validation,
    write_isotropic_validation,
)
from zgv_morse.zgv import ZGVPoint


ROOT = Path(__file__).resolve().parents[1]
ROW_FIELDS = (
    "order",
    "elements",
    "k_zgv",
    "omega_zgv",
    "curvature",
    "relative_k_error",
    "relative_omega_error",
    "relative_curvature_error",
    "maximum_eigen_residual",
    "hermitian_defect",
    "mass_orthogonality_defect",
    "rotational_frequency_defect",
    "minimum_relative_eigengap",
)


@pytest.fixture
def cfg() -> ReferenceConfig:
    return load_reference_config(ROOT / "config/reference.yaml")


@pytest.fixture
def exact_point() -> ZGVPoint:
    return ZGVPoint(
        kappa0=0.804217319371518,
        omega0=2.851758774960090,
        curvature_a=1.196862725073930,
        det_residual=1.0e-70,
        group_velocity=0.0,
        branch_index=1,
    )


def _row(*, order: int = 24, elements: int = 1) -> IsotropicBenchmarkRow:
    return IsotropicBenchmarkRow(
        order=order,
        elements=elements,
        k_zgv=0.804217319371518,
        omega_zgv=2.851758774960090,
        curvature=1.196862725073930,
        relative_k_error=2.0e-12,
        relative_omega_error=3.0e-13,
        relative_curvature_error=4.0e-8,
        maximum_eigen_residual=5.0e-15,
        hermitian_defect=6.0e-17,
        mass_orthogonality_defect=7.0e-15,
        rotational_frequency_defect=8.0e-16,
        minimum_relative_eigengap=9.0e-2,
    )


def test_benchmark_row_has_the_frozen_slotted_schema() -> None:
    row = _row()

    assert tuple(field.name for field in fields(IsotropicBenchmarkRow)) == ROW_FIELDS
    assert IsotropicBenchmarkRow.__dataclass_params__.frozen
    assert tuple(IsotropicBenchmarkRow.__slots__) == ROW_FIELDS
    with pytest.raises(FrozenInstanceError):
        row.order = 28  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"order": True}, TypeError, "order must be a built-in integer"),
        ({"elements": 0}, ValueError, "elements must be positive"),
        ({"curvature": np.nan}, ValueError, "curvature must be finite"),
        ({"curvature": -1.0}, ValueError, "curvature must be positive"),
        ({"relative_k_error": -1.0}, ValueError, "relative_k_error must be nonnegative"),
        (
            {"minimum_relative_eigengap": 0.0},
            ValueError,
            "minimum_relative_eigengap must be positive",
        ),
    ],
)
def test_benchmark_row_rejects_invalid_field_values(
    changes: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    values = asdict(_row())
    values.update(changes)
    with pytest.raises(error, match=message):
        IsotropicBenchmarkRow(**values)  # type: ignore[arg-type]


def test_run_validation_requires_a_valid_reference_config(cfg: ReferenceConfig) -> None:
    with pytest.raises(TypeError, match="ReferenceConfig"):
        run_isotropic_validation(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="h must be positive"):
        run_isotropic_validation(replace(cfg, h=-1.0))


@pytest.mark.parametrize(
    ("orders", "error", "message"),
    [
        ((), ValueError, "nonempty"),
        ((12, 12), ValueError, "strictly increasing"),
        ((16, 12), ValueError, "strictly increasing"),
        ((12, True), TypeError, "built-in integers"),
        ((12, np.int64(16)), TypeError, "built-in integers"),
        ((1,), ValueError, "between 2 and 512"),
        ((513,), ValueError, "between 2 and 512"),
    ],
)
def test_run_validation_rejects_invalid_orders_before_computation(
    cfg: ReferenceConfig,
    orders: tuple[object, ...],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        run_isotropic_validation(cfg, orders=orders)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("num_modes", "error", "message"),
    [
        (True, TypeError, "built-in integer"),
        (np.int64(18), TypeError, "built-in integer"),
        (17, ValueError, "at least 18"),
    ],
)
def test_run_validation_requires_enough_reported_modes(
    cfg: ReferenceConfig,
    num_modes: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        run_isotropic_validation(cfg, num_modes=num_modes)  # type: ignore[arg-type]


def test_write_validation_is_sorted_finite_complete_and_deterministic(
    tmp_path: Path,
    exact_point: ZGVPoint,
) -> None:
    rows = (_row(order=20), _row(order=24))
    split = _row(order=24, elements=2)
    json_path = tmp_path / "nested" / "json" / "isotropic_validation.json"
    csv_path = tmp_path / "different" / "csv" / "isotropic_convergence.csv"

    write_isotropic_validation(exact_point, rows, split, json_path, csv_path)
    first_json = json_path.read_bytes()
    first_csv = csv_path.read_bytes()
    write_isotropic_validation(exact_point, rows, split, json_path, csv_path)

    assert json_path.read_bytes() == first_json
    assert csv_path.read_bytes() == first_csv
    assert b"NaN" not in first_json and b"Infinity" not in first_json
    payload = json.loads(first_json)
    assert list(payload) == sorted(payload)
    assert payload == {
        "exact": asdict(exact_point),
        "single_element": [asdict(row) for row in rows],
        "two_element": asdict(split),
    }

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
    assert tuple(reader.fieldnames or ()) == ROW_FIELDS
    assert len(csv_rows) == 3
    assert [(int(row["order"]), int(row["elements"])) for row in csv_rows] == [
        (20, 1),
        (24, 1),
        (24, 2),
    ]
    assert first_csv.endswith(b"\n")
    assert b"\r\n" not in first_csv


@pytest.mark.parametrize(
    ("rows", "split", "message"),
    [
        ((), _row(order=24, elements=2), "nonempty"),
        ((_row(order=20, elements=2),), _row(order=20, elements=2), "single-element"),
        ((_row(order=20),), _row(order=20), "two-element"),
        ((_row(order=20),), _row(order=24, elements=2), "same order"),
        ((_row(order=24), _row(order=20)), _row(order=20, elements=2), "strictly increasing"),
        ((_row(order=20), object()), _row(order=20, elements=2), "benchmark rows"),
    ],
)
def test_write_validation_rejects_empty_or_mismatched_rows(
    tmp_path: Path,
    exact_point: ZGVPoint,
    rows: tuple[object, ...],
    split: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        write_isotropic_validation(
            exact_point,
            rows,  # type: ignore[arg-type]
            split,  # type: ignore[arg-type]
            tmp_path / "result.json",
            tmp_path / "result.csv",
        )


@pytest.mark.parametrize(
    ("exact_transform", "message"),
    [
        (lambda exact: object(), "ZGVPoint"),
        (lambda exact: replace(exact, curvature_a=np.nan), "finite"),
        (lambda exact: replace(exact, kappa0=0.0), "kappa0 must be positive"),
        (lambda exact: replace(exact, omega0=-1.0), "omega0 must be positive"),
        (lambda exact: replace(exact, curvature_a=0.0), "curvature_a must be positive"),
        (lambda exact: replace(exact, det_residual=-1.0), "det_residual must be nonnegative"),
    ],
)
def test_write_validation_rejects_wrong_or_nonfinite_records_without_writing(
    tmp_path: Path,
    exact_point: ZGVPoint,
    exact_transform: object,
    message: str,
) -> None:
    exact_value = exact_transform(exact_point)  # type: ignore[operator]
    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"

    with pytest.raises((TypeError, ValueError), match=message):
        write_isotropic_validation(
            exact_value,  # type: ignore[arg-type]
            (_row(),),
            _row(elements=2),
            json_path,
            csv_path,
        )

    assert not json_path.exists()
    assert not csv_path.exists()


def test_write_validation_rejects_one_path_for_both_formats(
    tmp_path: Path,
    exact_point: ZGVPoint,
) -> None:
    path = tmp_path / "same"
    with pytest.raises(ValueError, match="distinct"):
        write_isotropic_validation(exact_point, (_row(),), _row(elements=2), path, path)


def test_validation_script_exposes_a_callable_main() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/validate_isotropic.py"))

    assert callable(namespace["main"])


@pytest.fixture(scope="module")
def converged_benchmark() -> tuple[
    ZGVPoint,
    tuple[IsotropicBenchmarkRow, ...],
    IsotropicBenchmarkRow,
]:
    config = load_reference_config(ROOT / "config/reference.yaml")
    return run_isotropic_validation(config, orders=(12, 16, 20, 24, 28))


@pytest.mark.slow
def test_converged_spectral_solver_matches_exact_zgv(
    converged_benchmark: tuple[
        ZGVPoint,
        tuple[IsotropicBenchmarkRow, ...],
        IsotropicBenchmarkRow,
    ],
) -> None:
    exact, rows, split = converged_benchmark
    final = rows[-1]

    assert tuple(row.order for row in rows) == (12, 16, 20, 24, 28)
    assert all(row.elements == 1 for row in rows)
    assert split.order == 28 and split.elements == 2
    assert final.relative_k_error < 1.0e-7
    assert final.relative_omega_error < 1.0e-7
    assert final.relative_curvature_error < 1.0e-8
    assert final.maximum_eigen_residual < 1.0e-10
    assert final.hermitian_defect < 1.0e-13
    assert final.mass_orthogonality_defect < 1.0e-10
    assert final.rotational_frequency_defect < 1.0e-10
    assert final.minimum_relative_eigengap > 1.0e-4
    assert exact.curvature_a > 0.0
    assert abs(final.omega_zgv - split.omega_zgv) / final.omega_zgv < 1.0e-8
    assert abs(final.k_zgv - split.k_zgv) / final.k_zgv < 1.0e-8
    assert abs(final.curvature - split.curvature) / final.curvature < 1.0e-7
    assert split.maximum_eigen_residual < 1.0e-10
    assert split.minimum_relative_eigengap > 1.0e-4


@pytest.mark.slow
def test_last_two_orders_form_a_frequency_and_curvature_plateau(
    converged_benchmark: tuple[
        ZGVPoint,
        tuple[IsotropicBenchmarkRow, ...],
        IsotropicBenchmarkRow,
    ],
) -> None:
    _exact, rows, _split = converged_benchmark
    order_20, order_24, order_28 = rows[-3:]

    assert (order_20.order, order_24.order, order_28.order) == (20, 24, 28)
    assert abs(order_28.omega_zgv - order_24.omega_zgv) / order_28.omega_zgv < 1.0e-8
    assert abs(order_28.curvature - order_24.curvature) / order_28.curvature < 1.0e-8
