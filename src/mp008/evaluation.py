"""Evaluation metrics and group-level breakdown utilities for MP008."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.mp008.cleaning import TARGET, clean_inference_data


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate MAE, RMSE, and R² regression metrics."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_t, y_p)),
        "rmse": float(root_mean_squared_error(y_t, y_p)),
        "r2": float(r2_score(y_t, y_p)),
    }


def evaluate_model(
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Predict with a fitted model pipeline and return MAE, RMSE, and R²."""
    predictions = model.predict(X_eval)
    return compute_regression_metrics(y_eval, predictions)


def compute_group_mae(
    model: Any,
    eval_frame: pd.DataFrame,
    group_column: str = "feeder",
    target_column: str = TARGET,
) -> dict[str, Any]:
    """Compute MAE, RMSE, R², and row count separately for each group value (feeder).

    Also identifies the worst-served group (highest MAE).
    """
    if group_column not in eval_frame.columns:
        raise ValueError(f"Group column '{group_column}' missing from evaluation frame.")
    if target_column not in eval_frame.columns:
        raise ValueError(f"Target column '{target_column}' missing from evaluation frame.")

    X_eval = clean_inference_data(eval_frame)
    y_eval = pd.to_numeric(eval_frame[target_column], errors="raise").astype(float)
    predictions = pd.Series(model.predict(X_eval), index=eval_frame.index, dtype=float)

    groups = eval_frame[group_column].astype("string").str.strip()
    mae_by_group: dict[str, float] = {}
    details_by_group: dict[str, dict[str, float | int]] = {}

    for group_val in sorted(groups.dropna().unique()):
        mask = groups.eq(group_val)
        y_g = y_eval.loc[mask]
        p_g = predictions.loc[mask]
        metrics = compute_regression_metrics(y_g, p_g)
        mae_by_group[str(group_val)] = round(metrics["mae"], 4)
        details_by_group[str(group_val)] = {
            "count": int(mask.sum()),
            "mae": round(metrics["mae"], 4),
            "rmse": round(metrics["rmse"], 4),
            "r2": round(metrics["r2"], 4),
        }

    if not mae_by_group:
        raise ValueError("No valid groups found when computing group MAE.")

    worst_group = max(mae_by_group, key=lambda k: mae_by_group[k])
    return {
        "group_column": group_column,
        "mae_by_group": mae_by_group,
        "details_by_group": details_by_group,
        "worst_served_group": worst_group,
        "worst_served_mae": mae_by_group[worst_group],
    }


def evaluate_promotion_rule(
    v1_mae: float,
    v2_mae: float,
    improvement_threshold: float = 0.10,
) -> dict[str, Any]:
    """Evaluate the 10% MAE reduction promotion rule: V2_MAE <= V1_MAE * (1 - threshold)."""
    if v1_mae <= 0:
        raise ValueError(f"v1_mae must be positive, got {v1_mae}")
    required_max_v2_mae = float(v1_mae * (1.0 - improvement_threshold))
    relative_improvement = float((v1_mae - v2_mae) / v1_mae)
    promoted = bool(v2_mae <= required_max_v2_mae + 1e-12)
    return {
        "promoted": promoted,
        "decision": "promote" if promoted else "keep",
        "v1_mae": round(float(v1_mae), 4),
        "v2_mae": round(float(v2_mae), 4),
        "improvement_threshold": float(improvement_threshold),
        "required_max_v2_mae": round(required_max_v2_mae, 4),
        "relative_improvement": round(relative_improvement, 4),
        "relative_improvement_pct": round(relative_improvement * 100.0, 2),
    }
