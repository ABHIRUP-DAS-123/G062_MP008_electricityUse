"""Unit and integration tests for MP008 holdout predictions (including non-negativity)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.mp008.cleaning import clean_training_data
from src.mp008.training import build_model_pipeline, generate_holdout_predictions
from tests.test_training import make_synthetic_raw_dataset


class _NegativeDummyModel:
    """Dummy model returning negative values to verify non-negative clipping in generate_holdout_predictions."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.linspace(-25.5, 100.0, len(X))


def test_generate_holdout_predictions_schema_and_count(tmp_path: Path) -> None:
    """Test 4 & 5: Predictions contain exact columns (record_id, prediction) and matching row count."""
    raw_train = make_synthetic_raw_dataset(n_rows=40, seed=7)
    X_train, y_train = clean_training_data(raw_train)
    pipeline = build_model_pipeline(
        model_type="RandomForestRegressor",
        hyperparameters={"n_estimators": 15, "random_state": 42},
    ).fit(X_train, y_train)

    holdout_synthetic = make_synthetic_raw_dataset(n_rows=25, seed=11).drop(
        columns=["billed_units_after_reading", "units_kwh"]
    )
    out_csv = tmp_path / "holdout_predictions.csv"

    result_df = generate_holdout_predictions(
        model=pipeline,
        holdout_df=holdout_synthetic,
        output_path=out_csv,
    )

    assert out_csv.exists()
    saved_df = pd.read_csv(out_csv)
    assert list(saved_df.columns) == ["record_id", "prediction"]
    assert len(saved_df) == len(holdout_synthetic) == 25
    assert list(saved_df["record_id"]) == list(holdout_synthetic["record_id"])
    assert not saved_df["prediction"].isna().any()
    assert pd.api.types.is_numeric_dtype(saved_df["prediction"])
    assert np.all(np.isfinite(saved_df["prediction"]))
    assert (saved_df["prediction"] >= 0.0).all()


def test_generate_holdout_predictions_clips_negative_values(tmp_path: Path) -> None:
    """Verify negative raw model outputs are clipped to >= 0.0 before writing holdout_predictions.csv."""
    holdout_synthetic = make_synthetic_raw_dataset(n_rows=10, seed=19).drop(
        columns=["billed_units_after_reading", "units_kwh"]
    )
    out_csv = tmp_path / "clipped_predictions.csv"
    result_df = generate_holdout_predictions(
        model=_NegativeDummyModel(),
        holdout_df=holdout_synthetic,
        output_path=out_csv,
    )

    assert (result_df["prediction"] >= 0.0).all()
    assert float(result_df["prediction"].iloc[0]) == 0.0
    saved_df = pd.read_csv(out_csv)
    assert (saved_df["prediction"] >= 0.0).all()


def test_actual_holdout_predictions_artifact() -> None:
    """Verify the generated predictions/holdout_predictions.csv against data/raw/holdout_features.csv."""
    holdout_input_path = Path("data/raw/holdout_features.csv")
    predictions_path = Path("predictions/holdout_predictions.csv")

    assert holdout_input_path.exists()
    assert predictions_path.exists()

    holdout_in = pd.read_csv(holdout_input_path)
    preds_out = pd.read_csv(predictions_path)

    assert list(preds_out.columns) == ["record_id", "prediction"]
    assert len(preds_out) == len(holdout_in) == 250
    assert list(preds_out["record_id"]) == list(holdout_in["record_id"])
    assert not preds_out["prediction"].isna().any()
    assert pd.api.types.is_numeric_dtype(preds_out["prediction"])
    assert np.all(np.isfinite(preds_out["prediction"]))
    assert (preds_out["prediction"] >= 0.0).all(), "Holdout predictions must not contain negative values"
