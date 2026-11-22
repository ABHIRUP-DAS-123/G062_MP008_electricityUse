"""End-to-end training, retraining, promotion, and reporting for MP008."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline

from src.mp008.cleaning import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMN,
    TARGET,
    clean_inference_data,
    clean_training_data,
    clean_training_frame,
)
from src.mp008.data_checks import sha256_file
from src.mp008.evaluation import (
    compute_group_mae,
    evaluate_model,
    evaluate_promotion_rule,
)
from src.mp008.mlflow_utils import (
    configure_mlflow,
    log_training_run,
    register_run_model_version,
    set_champion_alias,
)
from src.mp008.monitoring import compute_feature_drift
from src.mp008.preprocessing import build_preprocessor, load_params

MODEL_REGISTRY_CLASSES: dict[str, type] = {
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
}


def build_estimator(model_type: str, hyperparameters: dict[str, Any]) -> Any:
    """Instantiate a candidate regression estimator from configuration."""
    if model_type not in MODEL_REGISTRY_CLASSES:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {sorted(MODEL_REGISTRY_CLASSES)}"
        )
    cls = MODEL_REGISTRY_CLASSES[model_type]
    return cls(**hyperparameters)


def build_model_pipeline(
    model_type: str,
    hyperparameters: dict[str, Any],
    preprocessing_cfg: dict[str, Any] | None = None,
    feature_cfg: dict[str, Any] | None = None,
) -> Pipeline:
    """Construct a complete scikit-learn Pipeline (ColumnTransformer + estimator)."""
    prep_cfg = preprocessing_cfg or {}
    feat_cfg = feature_cfg or {}

    preprocessor = build_preprocessor(
        numeric_features=feat_cfg.get("numeric_features"),
        categorical_features=feat_cfg.get("categorical_features"),
        numeric_imputer_strategy=prep_cfg.get("numeric_imputer_strategy", "median"),
        categorical_imputer_strategy=prep_cfg.get("categorical_imputer_strategy", "most_frequent"),
        scale_numeric=bool(prep_cfg.get("scale_numeric", True)),
    )
    estimator = build_estimator(model_type, hyperparameters)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )


def prepare_evaluation_data(raw_later: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare evaluation frame using clean_inference_data (later.csv is already clean per PROBLEM.md) while excluding leakage."""
    cleaned_frame = raw_later.copy(deep=True)
    for column in ("connection", "feeder"):
        if column in cleaned_frame:
            cleaned_frame[column] = cleaned_frame[column].astype("string").str.strip()
    X_eval = clean_inference_data(cleaned_frame)
    if LEAKAGE_COLUMN in X_eval.columns:
        raise ValueError(f"Leakage column '{LEAKAGE_COLUMN}' must never be in X_eval.")
    if list(X_eval.columns) != FEATURE_COLUMNS:
        raise ValueError(f"Unexpected feature columns: {list(X_eval.columns)}")
    y_eval = pd.to_numeric(cleaned_frame[TARGET], errors="raise").astype(float)
    return X_eval, y_eval, cleaned_frame


def split_later_chronologically(
    raw_later: pd.DataFrame,
    date_column: str = "date",
    train_fraction: float = 0.70,
    date_grouped: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort later dataset chronologically by date ascending and split into 70% / 30%.

    Split protocol note:
    - By default (`date_grouped=False`), this follows the exact faculty-required 70/30
      row-based chronological split (`PROBLEM.md` Section 7: "Sort later.csv by date.
      Retrain v2 on cleaned train.csv plus the first 70% of later.csv. Compare v1 and
      v2 on the last 30%"). Sorting uses stable mergesort (`kind='mergesort'`) so records
      within the same date preserve their original arrival order (`record_id` ascending).
      For the 300-row `later.csv`, the first 70% is exactly 210 rows (`2026-07-01` to
      `2026-08-14`, records `R20001`..`R20210`) and the final 30% is 90 rows
      (`2026-08-14` to `2026-08-31`, records `R20211`..`R20300`), meaning the boundary
      date `2026-08-14` has its earlier arrivals (3 rows) in the 70% training slice and
      its later arrivals (6 rows) in the 30% evaluation slice.
    - When `date_grouped=True`, whole dates are kept together by choosing the date boundary
      whose cumulative row count is closest to `round(len(work) * train_fraction)` without
      splitting any single calendar date across subsets (for `later.csv`, `<= 2026-08-13`
      gives 207 rows and `>= 2026-08-14` gives 93 rows).
    """
    if date_column not in raw_later.columns:
        raise ValueError(f"Date column '{date_column}' not found in later dataset.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")

    work = raw_later.copy(deep=True)
    work["_parsed_date"] = pd.to_datetime(work[date_column], errors="raise")
    work = work.sort_values("_parsed_date", ascending=True, kind="mergesort")
    work = work.drop(columns=["_parsed_date"]).reset_index(drop=True)

    target_index = int(round(len(work) * train_fraction))
    if target_index <= 0 or target_index >= len(work):
        raise ValueError(
            f"Invalid chronological split index {target_index} for {len(work)} rows."
        )

    if date_grouped:
        date_counts = work.groupby(date_column, sort=False).size()
        cum_counts = date_counts.cumsum()
        valid_boundaries = [int(c) for c in cum_counts.values if 0 < int(c) < len(work)]
        if not valid_boundaries:
            raise ValueError("Cannot perform date-grouped split: all rows share a single date.")
        split_index = min(valid_boundaries, key=lambda c: (abs(c - target_index), c))
    else:
        split_index = target_index

    first_part = work.iloc[:split_index].reset_index(drop=True)
    last_part = work.iloc[split_index:].reset_index(drop=True)
    return first_part, last_part


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_findings_yaml(
    findings_path: str | Path,
    tracking_info: dict[str, Any],
    monitoring_info: dict[str, Any],
) -> dict[str, Any]:
    """Update tracking and monitoring sections in reports/findings.yaml while preserving data_issues."""
    path = Path(findings_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            findings = yaml.safe_load(handle) or {}
    else:
        findings = {"group_id": "G062", "problem_id": "MP008", "data_issues": {}}

    findings["group_id"] = findings.get("group_id", "G062")
    findings["problem_id"] = findings.get("problem_id", "MP008")
    findings["tracking"] = {
        "experiment_name": tracking_info["experiment_name"],
        "best_run_id": tracking_info["best_run_id"],
        "registered_model_name": tracking_info["registered_model_name"],
        "champion_version": int(tracking_info["champion_version"]),
    }
    findings["monitoring"] = {
        "psi": monitoring_info["psi"],
        "most_drifted_feature": monitoring_info["most_drifted_feature"],
        "mae_by_group": monitoring_info["mae_by_group"],
        "worst_served_group": monitoring_info["worst_served_group"],
        "mae_v1_on_check": monitoring_info["mae_v1_on_check"],
        "mae_v2_on_check": monitoring_info["mae_v2_on_check"],
        "decision": monitoring_info["decision"],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(findings, sort_keys=False), encoding="utf-8")
    return findings


def generate_holdout_predictions(
    model: Any,
    holdout_df: pd.DataFrame,
    output_path: str | Path | None = None,
    id_column: str = "record_id",
) -> pd.DataFrame:
    """Generate non-negative holdout predictions with exact schema: record_id,prediction."""
    if id_column not in holdout_df.columns:
        raise ValueError(f"Identifier column '{id_column}' missing from holdout dataset.")

    X_holdout = clean_inference_data(holdout_df)
    if LEAKAGE_COLUMN in X_holdout.columns:
        raise ValueError(f"Leakage column '{LEAKAGE_COLUMN}' present in holdout features.")

    preds = model.predict(X_holdout)
    clipped_preds = pd.Series(preds, dtype=float).clip(lower=0.0).round(4)
    result = pd.DataFrame(
        {
            "record_id": holdout_df[id_column].astype(str).values,
            "prediction": clipped_preds.values,
        }
    )

    if list(result.columns) != ["record_id", "prediction"]:
        raise ValueError(f"Invalid prediction output columns: {list(result.columns)}")
    if result["prediction"].isna().any():
        raise ValueError("Generated holdout predictions contain missing values.")
    if (result["prediction"] < 0).any():
        raise ValueError("Generated holdout predictions contain negative electricity values.")
    if len(result) != len(holdout_df):
        raise ValueError(
            f"Holdout prediction count mismatch: expected {len(holdout_df)}, got {len(result)}"
        )

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_file, index=False)

    return result


def run_training_pipeline(params_path: str | Path = "params.yaml") -> dict[str, Any]:
    """Execute complete Phase 2 workflow: V1 candidate sweep, V1 registration, V2 retraining, promotion, reports, and holdout predictions."""
    params = load_params(params_path)
    data_cfg = params["data"]
    prep_cfg = params.get("preprocessing", {})
    mlflow_cfg = params["mlflow"]
    train_cfg = params["training"]
    mon_cfg = params.get("monitoring", {})
    out_cfg = params["outputs"]

    train_path = Path(data_cfg["train_path"])
    later_path = Path(data_cfg["later_path"])
    holdout_path = Path(data_cfg["holdout_path"])

    for file_path in (train_path, later_path, holdout_path):
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset file missing: {file_path}. "
                "Restore original CSVs from the faculty ZIP."
            )

    train_sha256 = sha256_file(train_path)
    later_sha256 = sha256_file(later_path)
    holdout_sha256 = sha256_file(holdout_path)

    expected_train_sha = data_cfg.get("expected_train_sha256")
    if expected_train_sha and train_sha256.lower() != str(expected_train_sha).lower():
        raise ValueError(
            f"SHA-256 mismatch for {train_path}: expected {expected_train_sha}, got {train_sha256}"
        )

    raw_train = pd.read_csv(train_path)
    raw_later = pd.read_csv(later_path)
    raw_holdout = pd.read_csv(holdout_path)

    X_train_v1, y_train_v1 = clean_training_data(raw_train)
    X_later_full, y_later_full, cleaned_later_full = prepare_evaluation_data(raw_later)

    if LEAKAGE_COLUMN in X_train_v1.columns or LEAKAGE_COLUMN in X_later_full.columns:
        raise RuntimeError(f"Leakage column '{LEAKAGE_COLUMN}' detected in feature matrix!")

    experiment_id = configure_mlflow(
        tracking_uri=mlflow_cfg["tracking_uri"],
        experiment_name=mlflow_cfg["experiment_name"],
    )

    candidates_cfg: dict[str, dict[str, Any]] = train_cfg["candidates"]
    v1_runs: list[dict[str, Any]] = []
    fitted_v1_models: dict[str, Pipeline] = {}

    for candidate_key, candidate_spec in candidates_cfg.items():
        model_type = candidate_spec["model_type"]
        hyperparams = dict(candidate_spec.get("hyperparameters", {}))

        pipeline = build_model_pipeline(
            model_type=model_type,
            hyperparameters=hyperparams,
            preprocessing_cfg=prep_cfg,
            feature_cfg=data_cfg,
        )
        pipeline.fit(X_train_v1, y_train_v1)
        fitted_v1_models[candidate_key] = pipeline

        metrics = evaluate_model(pipeline, X_later_full, y_later_full)
        group_eval = compute_group_mae(
            pipeline,
            cleaned_later_full,
            group_column=data_cfg.get("group_column", "feeder"),
            target_column=data_cfg.get("target_column", TARGET),
        )

        extra_metrics = {
            "mae_later": metrics["mae"],
            "rmse_later": metrics["rmse"],
            "r2_later": metrics["r2"],
            "mae_feeder_1": group_eval["mae_by_group"].get("Feeder-1", 0.0),
            "mae_feeder_2": group_eval["mae_by_group"].get("Feeder-2", 0.0),
            "mae_feeder_3": group_eval["mae_by_group"].get("Feeder-3", 0.0),
        }

        run_record = log_training_run(
            run_name=f"v1_{candidate_key}",
            model_name=model_type,
            pipeline=pipeline,
            hyperparameters=hyperparams,
            metrics=metrics,
            X_train=X_train_v1,
            X_val=X_later_full,
            data_sha256=train_sha256,
            stage_tag="v1_candidate",
            extra_params={"evaluation_dataset": "later.csv_full"},
            extra_metrics=extra_metrics,
            registered_model_name=None,
            serialization_format=mlflow_cfg.get("serialization_format", "cloudpickle"),
        )
        run_record["candidate_key"] = candidate_key
        run_record["hyperparameters"] = hyperparams
        run_record["raw_metrics"] = {k: round(v, 4) for k, v in metrics.items()}
        run_record["feeder_evaluation"] = group_eval
        v1_runs.append(run_record)

    v1_runs_sorted = sorted(v1_runs, key=lambda r: r["raw_metrics"]["mae"])
    best_v1_run = v1_runs_sorted[0]
    best_v1_key = best_v1_run["candidate_key"]
    best_v1_model = fitted_v1_models[best_v1_key]
    best_v1_spec = candidates_cfg[best_v1_key]

    registered_model_name = mlflow_cfg["registered_model_name"]
    v1_version = register_run_model_version(
        run_id=best_v1_run["run_id"],
        registered_model_name=registered_model_name,
        model_uri=best_v1_run["model_uri"],
    )
    best_v1_run["registered_version"] = v1_version

    # Chronological split of later.csv (first 70% additional training, final 30% V2 evaluation)
    v2_fraction = float(train_cfg.get("v2_train_fraction", 0.70))
    later_first_70_raw, later_last_30_raw = split_later_chronologically(
        raw_later,
        date_column=data_cfg.get("date_column", "date"),
        train_fraction=v2_fraction,
    )

    X_later_70, y_later_70, _ = prepare_evaluation_data(later_first_70_raw)
    X_later_30, y_later_30, cleaned_later_30 = prepare_evaluation_data(later_last_30_raw)

    X_train_v2 = pd.concat([X_train_v1, X_later_70], ignore_index=True)
    y_train_v2 = pd.concat([y_train_v1, y_later_70], ignore_index=True)

    v2_model_type = best_v1_spec["model_type"]
    v2_hyperparams = dict(best_v1_spec.get("hyperparameters", {}))
    v2_pipeline = build_model_pipeline(
        model_type=v2_model_type,
        hyperparameters=v2_hyperparams,
        preprocessing_cfg=prep_cfg,
        feature_cfg=data_cfg,
    )
    v2_pipeline.fit(X_train_v2, y_train_v2)

    v2_metrics_30 = evaluate_model(v2_pipeline, X_later_30, y_later_30)
    v1_metrics_30 = evaluate_model(best_v1_model, X_later_30, y_later_30)
    v2_group_eval_30 = compute_group_mae(
        v2_pipeline,
        cleaned_later_30,
        group_column=data_cfg.get("group_column", "feeder"),
        target_column=data_cfg.get("target_column", TARGET),
    )

    v2_run_record = log_training_run(
        run_name=f"v2_{best_v1_key}_retrained",
        model_name=v2_model_type,
        pipeline=v2_pipeline,
        hyperparameters=v2_hyperparams,
        metrics=v2_metrics_30,
        X_train=X_train_v2,
        X_val=X_later_30,
        data_sha256=train_sha256,
        stage_tag="v2_retrained",
        extra_params={
            "evaluation_dataset": "later.csv_final_30pct_chronological",
            "v2_train_fraction": v2_fraction,
            "base_train_rows": len(X_train_v1),
            "added_later_rows": len(X_later_70),
        },
        extra_metrics={
            "mae_check_30pct": v2_metrics_30["mae"],
            "v1_mae_full_later": best_v1_run["raw_metrics"]["mae"],
            "v1_mae_check_30pct": v1_metrics_30["mae"],
        },
        registered_model_name=registered_model_name,
        serialization_format=mlflow_cfg.get("serialization_format", "cloudpickle"),
    )
    v2_version = int(v2_run_record["registered_version"])

    threshold = float(train_cfg.get("promotion_improvement_threshold", 0.10))
    v1_full_later_mae = float(best_v1_run["raw_metrics"]["mae"])
    v1_check_30_mae = float(v1_metrics_30["mae"])
    v2_check_30_mae = float(v2_metrics_30["mae"])

    promo_vs_full = evaluate_promotion_rule(
        v1_mae=v1_full_later_mae,
        v2_mae=v2_check_30_mae,
        improvement_threshold=threshold,
    )
    promo_vs_check30 = evaluate_promotion_rule(
        v1_mae=v1_check_30_mae,
        v2_mae=v2_check_30_mae,
        improvement_threshold=threshold,
    )

    # V2 must satisfy the 10% threshold under the protocol
    promoted = promo_vs_full["promoted"]
    champion_version = v2_version if promoted else v1_version
    champion_model = v2_pipeline if promoted else best_v1_model
    champion_stage = "V2" if promoted else "V1"
    champion_run_id = v2_run_record["run_id"] if promoted else best_v1_run["run_id"]

    alias_info = set_champion_alias(
        registered_model_name=registered_model_name,
        version=champion_version,
        alias=mlflow_cfg.get("champion_alias", "champion"),
    )

    champion_model_path = Path(out_cfg["champion_model_path"])
    champion_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_model, champion_model_path)

    holdout_preds_df = generate_holdout_predictions(
        model=champion_model,
        holdout_df=raw_holdout,
        output_path=out_cfg["holdout_predictions_path"],
        id_column=data_cfg.get("id_column", "record_id"),
    )

    drift_report = compute_feature_drift(
        train_features=X_train_v1,
        later_features=X_later_full,
        numeric_columns=data_cfg.get("numeric_features"),
        buckets=int(mon_cfg.get("psi_buckets", 10)),
    )

    v1_metrics_report = {
        "stage": "V1",
        "selected_model_name": best_v1_run["model_name"],
        "run_id": best_v1_run["run_id"],
        "registered_model_name": registered_model_name,
        "registered_version": v1_version,
        "data_sha256": train_sha256,
        "train_rows": len(X_train_v1),
        "evaluation_dataset": "later.csv (complete, 300 rows)",
        "evaluation_rows": len(X_later_full),
        "metrics_full_later": best_v1_run["raw_metrics"],
        "metrics_last_30pct_later": {k: round(v, 4) for k, v in v1_metrics_30.items()},
        "feeder_mae": best_v1_run["feeder_evaluation"]["mae_by_group"],
        "worst_served_group": best_v1_run["feeder_evaluation"]["worst_served_group"],
        "hyperparameters": best_v1_run["hyperparameters"],
    }

    v2_metrics_report = {
        "stage": "V2",
        "selected_model_name": v2_model_type,
        "run_id": v2_run_record["run_id"],
        "registered_model_name": registered_model_name,
        "registered_version": v2_version,
        "data_sha256": train_sha256,
        "train_rows_total": len(X_train_v2),
        "train_rows_original": len(X_train_v1),
        "train_rows_later_first_70pct": len(X_later_70),
        "later_first_70pct_date_range": {
            "start": str(later_first_70_raw["date"].min()),
            "end": str(later_first_70_raw["date"].max()),
        },
        "evaluation_dataset": "later.csv (final 30% chronological, 90 rows)",
        "evaluation_rows": len(X_later_30),
        "later_last_30pct_date_range": {
            "start": str(later_last_30_raw["date"].min()),
            "end": str(later_last_30_raw["date"].max()),
        },
        "metrics_last_30pct_later": {k: round(v, 4) for k, v in v2_metrics_30.items()},
        "feeder_mae_last_30pct": v2_group_eval_30["mae_by_group"],
        "hyperparameters": v2_hyperparams,
    }

    feeder_metrics_report = {
        "v1_on_full_later": best_v1_run["feeder_evaluation"],
        "v2_on_last_30pct_later": v2_group_eval_30,
        "feature_drift_psi": drift_report,
    }

    promotion_decision_report = {
        "registered_model_name": registered_model_name,
        "v1_version": v1_version,
        "v1_run_id": best_v1_run["run_id"],
        "v2_version": v2_version,
        "v2_run_id": v2_run_record["run_id"],
        "protocol_note": (
            "Per project protocol, V1 is evaluated on the complete later.csv dataset (300 rows) "
            "and also checked on the chronological final 30% segment (90 rows), while V2 is trained "
            "on cleaned train.csv + first 70% chronological later.csv (1210 rows) and evaluated on "
            "the chronological final 30% of later.csv (90 rows). Because complete later.csv and its "
            "final 30% segment are different evaluation splits, both comparisons are reported transparently."
        ),
        "comparison_v1_full_later_vs_v2_final_30pct": promo_vs_full,
        "comparison_v1_final_30pct_vs_v2_final_30pct": promo_vs_check30,
        "promoted": promoted,
        "decision": "promote" if promoted else "keep",
        "champion_stage": champion_stage,
        "champion_version": champion_version,
        "champion_run_id": champion_run_id,
        "champion_alias": alias_info["alias"],
        "champion_model_path": str(champion_model_path),
    }

    model_comparison_report = {
        "experiment_name": mlflow_cfg["experiment_name"],
        "experiment_id": experiment_id,
        "tracking_uri": mlflow_cfg["tracking_uri"],
        "data_sha256": {
            "train.csv": train_sha256,
            "later.csv": later_sha256,
            "holdout_features.csv": holdout_sha256,
        },
        "v1_candidate_runs": [
            {
                "run_id": r["run_id"],
                "run_name": r["run_name"],
                "model_name": r["model_name"],
                "hyperparameters": r["hyperparameters"],
                "metrics_on_full_later": r["raw_metrics"],
                "feeder_mae_on_full_later": r["feeder_evaluation"]["mae_by_group"],
                "registered_version": r.get("registered_version"),
            }
            for r in v1_runs_sorted
        ],
        "selected_v1": v1_metrics_report,
        "retrained_v2": v2_metrics_report,
        "promotion_decision": promotion_decision_report,
        "holdout_predictions": {
            "output_path": out_cfg["holdout_predictions_path"],
            "row_count": len(holdout_preds_df),
            "columns": list(holdout_preds_df.columns),
        },
    }

    _write_json(v1_metrics_report, out_cfg["v1_metrics_path"])
    _write_json(v2_metrics_report, out_cfg["v2_metrics_path"])
    _write_json(feeder_metrics_report, out_cfg["feeder_metrics_path"])
    _write_json(promotion_decision_report, out_cfg["promotion_decision_path"])
    _write_json(model_comparison_report, out_cfg["model_comparison_path"])

    update_findings_yaml(
        findings_path=out_cfg["findings_path"],
        tracking_info={
            "experiment_name": mlflow_cfg["experiment_name"],
            "best_run_id": best_v1_run["run_id"],
            "registered_model_name": registered_model_name,
            "champion_version": champion_version,
        },
        monitoring_info={
            "psi": drift_report["psi"],
            "most_drifted_feature": drift_report["most_drifted_feature"],
            "mae_by_group": best_v1_run["feeder_evaluation"]["mae_by_group"],
            "worst_served_group": best_v1_run["feeder_evaluation"]["worst_served_group"],
            "mae_v1_on_check": round(v1_check_30_mae, 4),
            "mae_v2_on_check": round(v2_check_30_mae, 4),
            "decision": "promote" if promoted else "keep",
        },
    )

    return model_comparison_report
