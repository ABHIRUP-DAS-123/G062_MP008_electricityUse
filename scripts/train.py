"""CLI script to train V1 candidates, retrain V2, register models in MLflow, and save the champion model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mp008.training import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V1 and V2 models for MP008.")
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Path to params.yaml configuration file.",
    )
    args = parser.parse_args()

    summary = run_training_pipeline(params_path=args.params)
    print(json.dumps(summary["promotion_decision"], indent=2))


if __name__ == "__main__":
    main()
