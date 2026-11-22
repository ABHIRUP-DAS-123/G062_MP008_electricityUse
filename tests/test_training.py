"""Unit tests for MP008 data cleaning, preprocessing, training, PSI binning, and V2 promotion logic."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.mp008.cleaning import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMN,
    TARGET,
    clean_inference_data,
    clean_training_data,
)
from src.mp008.evaluation import (
    compute_group_mae,
    evaluate_model,
    evaluate_promotion_rule,
)
from src.mp008.monitoring import (
    build_psi_bin_edges,
    compute_feature_drift,
    compute_psi,
)
from src.mp008.training import (
    build_model_pipeline,
    prepare_evaluation_data,
    split_later_chronologically,
)


def make_synthetic_raw_dataset(n_rows: int = 40, seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic dataset matching the MP008 raw training schema."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_rows, freq="D").strftime("%Y-%m-%d")
    feeders = rng.choice(["Feeder-1", "Feeder-2", "Feeder-3"], size=n_rows)
    connections = rng.choice(["domestic", "domestic_plus", "shop_cum_home"], size=n_rows)
    rooms = rng.integers(1, 7, size=n_rows)
    members = rng.integers(1, 9, size=n_rows)
    ac_hours = rng.uniform(1.0, 12.0, size=n_rows).round(2)
    avg_temp = rng.uniform(15.0, 38.0, size=n_rows).round(1)
    area = rng.integers(400, 2800, size=n_rows)
    solar = rng.uniform(0.0, 6.0, size=n_rows).round(2)

    units = (
        rooms * 25.0
        + members * 15.0
        + ac_hours * 20.0
        + avg_temp * 3.0
        + area * 0.05
        - solar * 30.0
        + rng.normal(0.0, 5.0, size=n_rows)
    ).clip(min=20.0).round(2)
    billed = (units + rng.normal(0.0, 1.0, size=n_rows)).clip(min=10.0).round(2)

    # Inject Feeder-1 unit mix-up (minutes instead of hours)
    ac_recorded = ac_hours.copy()
    f1_mask = feeders == "Feeder-1"
    ac_recorded[f1_mask] = (ac_hours[f1_mask] * 60.0).round(2)

    # Inject a 10x target typo on row 0
    units_recorded = units.copy()
    units_recorded[0] = round(float(billed[0]) * 10.0, 2)

    return pd.DataFrame(
        {
            "record_id": [f"R{10000 + i}" for i in range(n_rows)],
            "date": dates,
            "rooms": rooms,
            "members": members,
            "ac_hours_per_day": ac_recorded,
            "avg_temp_c": avg_temp,
            "house_area_sqft": area,
            "solar_kw": solar,
            "connection": connections,
            "feeder": feeders,
            "billed_units_after_reading": billed,
            "units_kwh": units_recorded,
        }
    )


def test_leakage_column_excluded_from_features() -> None:
    """Test 1 & 2: Leakage column is excluded and cleaning produces the exact 8 expected features."""
    raw = make_synthetic_raw_dataset(n_rows=30)
    X_train, y_train = clean_training_data(raw)
    X_inf = clean_inference_data(raw)

    assert LEAKAGE_COLUMN not in X_train.columns
    assert LEAKAGE_COLUMN not in X_inf.columns
    assert "record_id" not in X_train.columns
    assert "date" not in X_train.columns
    assert TARGET not in X_train.columns
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_inf.columns) == FEATURE_COLUMNS
    assert len(X_train) == len(y_train) == 30
    assert not y_train.isna().any()

    # Verify Feeder-1 minutes -> hours conversion and 10x typo correction
    f1_max = X_train.loc[X_train["feeder"] == "Feeder-1", "ac_hours_per_day"].max()
    assert f1_max <= 24.0
    assert abs(float(y_train.iloc[0]) - float(raw["billed_units_after_reading"].iloc[0])) < 1e-5


def test_training_pipeline_can_fit_and_predict() -> None:
    """Test 3: Training pipeline fits on cleaned features (including missing values) and predicts."""
    raw = make_synthetic_raw_dataset(n_rows=50)
    X_train, y_train = clean_training_data(raw)

    # Introduce missing values to verify preprocessing imputation
    X_train_with_na = X_train.copy()
    X_train_with_na.loc[0, "avg_temp_c"] = np.nan
    X_train_with_na.loc[1, "connection"] = pd.NA

    for model_type, params in [
        ("RandomForestRegressor", {"n_estimators": 20, "max_depth": 5, "random_state": 42}),
        ("ExtraTreesRegressor", {"n_estimators": 20, "max_depth": 5, "random_state": 42}),
        ("HistGradientBoostingRegressor", {"max_iter": 20, "max_depth": 4, "random_state": 42}),
    ]:
        pipeline = build_model_pipeline(model_type=model_type, hyperparameters=params)
        pipeline.fit(X_train_with_na, y_train)
        preds = pipeline.predict(X_train_with_na)
        assert len(preds) == len(X_train_with_na)
        assert np.all(np.isfinite(preds))
        metrics = evaluate_model(pipeline, X_train, y_train)
        assert set(metrics.keys()) == {"mae", "rmse", "r2"}
        assert metrics["mae"] >= 0.0


def test_chronological_split_and_v2_promotion_threshold() -> None:
    """Test 7: Chronological split orders dates ascending, documents row-based vs date-grouped behavior, and checks 10% rule."""
    # Construct dataset with repeated dates across the 70% boundary to test both modes
    raw = make_synthetic_raw_dataset(n_rows=30)
    repeated_dates = [f"2026-07-{1 + (i // 3):02d}" for i in range(30)]
    raw["date"] = repeated_dates
    shuffled = raw.sample(frac=1.0, random_state=99).reset_index(drop=True)

    # Default faculty protocol: 70/30 row-based chronological split (21 rows / 9 rows)
    first_70, last_30 = split_later_chronologically(
        shuffled, date_column="date", train_fraction=0.70, date_grouped=False
    )
    assert len(first_70) == 21
    assert len(last_30) == 9
    assert first_70["date"].max() <= last_30["date"].min()
    assert first_70["date"].is_monotonic_increasing
    assert last_30["date"].is_monotonic_increasing

    # Optional date-grouped split: keeps calendar dates unsplit across subsets
    first_grp, last_grp = split_later_chronologically(
        shuffled, date_column="date", train_fraction=0.70, date_grouped=True
    )
    assert first_grp["date"].max() < last_grp["date"].min()
    assert set(first_grp["date"]).isdisjoint(set(last_grp["date"]))

    # Verify 10% threshold: V2_MAE <= V1_MAE * 0.90
    promoted_res = evaluate_promotion_rule(v1_mae=100.0, v2_mae=89.5, improvement_threshold=0.10)
    assert promoted_res["promoted"] is True
    assert promoted_res["decision"] == "promote"

    exact_boundary = evaluate_promotion_rule(v1_mae=100.0, v2_mae=90.0, improvement_threshold=0.10)
    assert exact_boundary["promoted"] is True
    assert exact_boundary["decision"] == "promote"

    rejected_res = evaluate_promotion_rule(v1_mae=100.0, v2_mae=90.5, improvement_threshold=0.10)
    assert rejected_res["promoted"] is False
    assert rejected_res["decision"] == "keep"


def test_psi_duplicate_quantiles_and_constant_features() -> None:
    """Test robust PSI binning for discrete features with repeated quantile edges and constant features."""
    # Discrete feature with only 4 distinct levels across 200 samples (10-quantile calculation has duplicates)
    rng = np.random.default_rng(123)
    ref_discrete = pd.Series(rng.choice([1, 2, 3, 4], size=200, p=[0.4, 0.3, 0.2, 0.1]))
    cur_discrete_same = ref_discrete.copy()
    cur_discrete_shifted = pd.Series(rng.choice([1, 2, 3, 4], size=200, p=[0.1, 0.2, 0.3, 0.4]))

    edges = build_psi_bin_edges(ref_discrete, buckets=10)
    # Must have strictly increasing boundaries (no zero-width bins)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == -np.inf and edges[-1] == np.inf

    psi_same = compute_psi(ref_discrete, cur_discrete_same, buckets=10)
    psi_shifted = compute_psi(ref_discrete, cur_discrete_shifted, buckets=10)
    assert abs(psi_same) < 1e-9
    assert psi_shifted > 0.25

    # Constant-valued reference feature
    ref_const = pd.Series(np.full(100, 5.0))
    cur_const_same = pd.Series(np.full(100, 5.0))
    cur_const_shifted = pd.Series(np.full(100, 8.0))

    const_edges = build_psi_bin_edges(ref_const, buckets=10)
    assert np.all(np.diff(const_edges) > 0)

    psi_const_same = compute_psi(ref_const, cur_const_same, buckets=10)
    psi_const_shifted = compute_psi(ref_const, cur_const_shifted, buckets=10)
    assert abs(psi_const_same) < 1e-9
    assert np.isfinite(psi_const_shifted) and psi_const_shifted > 1.0


def test_group_mae_and_psi_monitoring() -> None:
    """Verify feeder-wise MAE breakdown and PSI drift calculation on synthetic data."""
    raw = make_synthetic_raw_dataset(n_rows=45)
    X_train, y_train = clean_training_data(raw)
    pipeline = build_model_pipeline(
        model_type="RandomForestRegressor",
        hyperparameters={"n_estimators": 15, "random_state": 42},
    ).fit(X_train, y_train)

    _, _, eval_frame = prepare_evaluation_data(raw)
    group_res = compute_group_mae(pipeline, eval_frame, group_column="feeder")
    assert set(group_res["mae_by_group"].keys()) == {"Feeder-1", "Feeder-2", "Feeder-3"}
    assert group_res["worst_served_group"] in {"Feeder-1", "Feeder-2", "Feeder-3"}

    shifted = X_train.copy()
    shifted["solar_kw"] = shifted["solar_kw"] + 5.0
    drift = compute_feature_drift(X_train, shifted)
    assert drift["most_drifted_feature"] == "solar_kw"
    assert drift["psi"]["solar_kw"] > 0.25


def test_champion_model_can_be_loaded_from_joblib() -> None:
    """Test 8: Champion model artifact exists at models/model.joblib, can be loaded, and predicts."""
    model_path = Path("models/model.joblib")
    assert model_path.exists(), f"Expected champion model at {model_path}"
    loaded_model = joblib.load(model_path)

    sample = clean_inference_data(make_synthetic_raw_dataset(n_rows=5))
    preds = loaded_model.predict(sample)
    assert len(preds) == 5
    assert np.all(np.isfinite(preds))
