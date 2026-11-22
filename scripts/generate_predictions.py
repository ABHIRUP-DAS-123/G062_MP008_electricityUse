"""CLI script to generate holdout predictions using the saved champion model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mp008.preprocessing import load_params
from src.mp008.training import generate_holdout_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate holdout predictions for MP008.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml.")
    parser.add_argument("--model", default=None, help="Optional override path to model.joblib.")
    parser.add_argument("--holdout", default=None, help="Optional override path to holdout_features.csv.")
    parser.add_argument("--output", default=None, help="Optional override output CSV path.")
    args = parser.parse_args()

    params = load_params(args.params)
    data_cfg = params["data"]
    out_cfg = params["outputs"]

    model_path = Path(args.model or out_cfg["champion_model_path"])
    holdout_path = Path(args.holdout or data_cfg["holdout_path"])
    output_path = Path(args.output or out_cfg["holdout_predictions_path"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Champion model file not found at {model_path}. Run scripts/train.py first."
        )
    if not holdout_path.exists():
        raise FileNotFoundError(f"Holdout features file not found at {holdout_path}.")

    model = joblib.load(model_path)
    holdout_df = pd.read_csv(holdout_path)

    result = generate_holdout_predictions(
        model=model,
        holdout_df=holdout_df,
        output_path=output_path,
        id_column=data_cfg.get("id_column", "record_id"),
    )
    print(f"Generated {len(result)} predictions -> {output_path}")


if __name__ == "__main__":
    main()
