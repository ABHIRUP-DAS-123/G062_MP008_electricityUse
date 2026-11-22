"""CLI script to evaluate the saved champion model on later.csv and verify evaluation reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mp008.cleaning import TARGET, clean_training_data
from src.mp008.evaluation import compute_group_mae, evaluate_model
from src.mp008.monitoring import compute_feature_drift
from src.mp008.preprocessing import load_params
from src.mp008.training import prepare_evaluation_data, split_later_chronologically


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate champion model and drift metrics for MP008.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml.")
    parser.add_argument("--model", default=None, help="Optional override path to model.joblib.")
    args = parser.parse_args()

    params = load_params(args.params)
    data_cfg = params["data"]
    out_cfg = params["outputs"]
    mon_cfg = params.get("monitoring", {})
    train_cfg = params["training"]

    model_path = Path(args.model or out_cfg["champion_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Champion model file not found at {model_path}. Run scripts/train.py first."
        )

    champion_model = joblib.load(model_path)

    raw_train = pd.read_csv(data_cfg["train_path"])
    raw_later = pd.read_csv(data_cfg["later_path"])

    X_train, _ = clean_training_data(raw_train)
    X_later, y_later, cleaned_later = prepare_evaluation_data(raw_later)

    full_later_metrics = evaluate_model(champion_model, X_later, y_later)
    feeder_eval = compute_group_mae(
        champion_model,
        cleaned_later,
        group_column=data_cfg.get("group_column", "feeder"),
        target_column=data_cfg.get("target_column", TARGET),
    )

    _, later_30_raw = split_later_chronologically(
        raw_later,
        date_column=data_cfg.get("date_column", "date"),
        train_fraction=float(train_cfg.get("v2_train_fraction", 0.70)),
    )
    X_later_30, y_later_30, _ = prepare_evaluation_data(later_30_raw)
    last_30_metrics = evaluate_model(champion_model, X_later_30, y_later_30)

    drift = compute_feature_drift(
        train_features=X_train,
        later_features=X_later,
        numeric_columns=data_cfg.get("numeric_features"),
        buckets=int(mon_cfg.get("psi_buckets", 10)),
    )

    payload = {
        "champion_model_path": str(model_path),
        "full_later_metrics": {k: round(v, 4) for k, v in full_later_metrics.items()},
        "last_30pct_later_metrics": {k: round(v, 4) for k, v in last_30_metrics.items()},
        "feeder_evaluation": feeder_eval,
        "feature_drift_psi": drift,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
