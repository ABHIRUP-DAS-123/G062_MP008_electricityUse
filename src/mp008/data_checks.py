"""Data-quality checks for Group G062, problem MP008.

Run from the repository root:
    python -m src.mp008.data_checks --train data/raw/train.csv \
        --later data/raw/later.csv --holdout data/raw/holdout_features.csv

The script reads the source CSV files only. It never writes to or modifies them.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import yaml

TARGET = "units_kwh"
ID_COLUMNS = {"record_id", "date", TARGET}
NUMERIC_FEATURES = [
    "rooms", "members", "ac_hours_per_day",
    "avg_temp_c", "house_area_sqft", "solar_kw",
]
EXPECTED_SHA256 = {
    "train.csv": "9aea26c5aa76831f4a4e58a3dfafd06d47144136c13ace14834010bfcf472781",
    "later.csv": "4c161defd6502ae4c6336aef612118ac329579730f31ee879d265e5df692db8c",
    "holdout_features.csv": "c701755d2e84f1e877edf12a2aab797bba448807816635c8d4ceded8226cf896",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_category(value: object) -> str:
    value = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def _find_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame) -> str | None:
    absent = [
        c for c in train.columns
        if c not in holdout.columns and c not in ID_COLUMNS
    ]
    for column in absent:
        if column not in train.columns or not pd.api.types.is_numeric_dtype(train[column]):
            continue
        valid = train[[TARGET, column]].dropna()
        # Exclude the obvious 10x target-entry errors while testing the relationship.
        if "billed_units_after_reading" in train.columns:
            billed = train["billed_units_after_reading"]
            ratio = train[TARGET] / billed.replace(0, np.nan)
            valid = train.loc[ratio < 5, [TARGET, column]].dropna()
        if len(valid) >= 30 and abs(valid[TARGET].corr(valid[column])) >= 0.98:
            return column
    return absent[0] if len(absent) == 1 else None


def _find_unit_mixup(train: pd.DataFrame) -> dict:
    column = "ac_hours_per_day"
    if column not in train or "feeder" not in train:
        return {"present": False, "column": None, "group_value": None, "factor": None}

    medians = train.groupby("feeder")[column].median()
    if len(medians) < 2:
        return {"present": False, "column": None, "group_value": None, "factor": None}

    common_factors = (10, 12, 60, 100, 1000)
    for group_value, median in medians.items():
        others = medians.drop(index=group_value)
        baseline = float(others.median())
        if baseline <= 0:
            continue
        ratio = float(median / baseline)
        factor = min(common_factors, key=lambda x: abs(ratio - x) / x)
        if abs(ratio - factor) / factor <= 0.20:
            return {
                "present": True,
                "column": column,
                "group_value": str(group_value),
                "factor": int(factor),
            }
    return {"present": False, "column": None, "group_value": None, "factor": None}


def _find_sentinel(train: pd.DataFrame) -> dict:
    # The handbook describes a special impossible value repeated many times.
    candidates = {-1, -9, -99, -999, 999, 9999}
    for column in train.select_dtypes(include=np.number).columns:
        counts = train[column].value_counts()
        for value in candidates:
            count = int(counts.get(value, 0))
            if count >= 3:
                return {
                    "present": True,
                    "column": column,
                    "value": int(value),
                    "count": count,
                }
    return {"present": False, "column": None, "value": None, "count": None}


def _find_stuck_sensor(train: pd.DataFrame, min_days: int = 7) -> dict:
    """Find a numeric field constant for every observation on consecutive dates."""
    work = train.copy()
    work["_date"] = pd.to_datetime(work["date"], errors="coerce")
    for column in work.select_dtypes(include=np.number).columns:
        if column == TARGET or column == "record_id":
            continue
        daily = (
            work.dropna(subset=["_date"])
            .groupby("_date")[column]
            .agg(nunique="nunique", value="first")
            .sort_index()
        )
        best: list[tuple[pd.Timestamp, object]] = []
        current: list[tuple[pd.Timestamp, object]] = []
        for date, row in daily.iterrows():
            if row["nunique"] == 1:
                if (
                    current
                    and date == current[-1][0] + pd.Timedelta(days=1)
                    and row["value"] == current[-1][1]
                ):
                    current.append((date, row["value"]))
                else:
                    if len(current) > len(best):
                        best = current
                    current = [(date, row["value"])]
            else:
                if len(current) > len(best):
                    best = current
                current = []
        if len(current) > len(best):
            best = current
        if len(best) >= min_days:
            return {
                "present": True,
                "column": column,
                "start_date": best[0][0].strftime("%Y-%m-%d"),
                "end_date": best[-1][0].strftime("%Y-%m-%d"),
            }
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def _find_target_typos(train: pd.DataFrame) -> dict:
    if "billed_units_after_reading" not in train:
        return {"present": False, "count": None}
    billed = train["billed_units_after_reading"].replace(0, np.nan)
    ratio = train[TARGET] / billed
    typo_mask = ratio.between(8.0, 12.0, inclusive="both")
    return {"present": bool(typo_mask.any()), "count": int(typo_mask.sum()) if typo_mask.any() else None}


def _find_spelling_variants(train: pd.DataFrame, later: pd.DataFrame) -> dict:
    for column in train.select_dtypes(include=["object"]).columns:
        if column in {"record_id", "date"}:
            continue
        raw_values = set(train[column].dropna().astype(str))
        normalised = {_normalise_category(v) for v in raw_values}
        if len(raw_values) > len(normalised):
            return {
                "present": True,
                "column": column,
                "true_levels": int(later[column].nunique()) if column in later else int(len(normalised)),
            }
    return {"present": False, "column": None, "true_levels": None}


def analyse_data(train_path: str | Path, later_path: str | Path, holdout_path: str | Path) -> dict:
    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)

    duplicate_count = int(train.duplicated().sum())
    leak = _find_leaking_column(train, holdout)
    unit_mixup = _find_unit_mixup(train)
    sentinel = _find_sentinel(train)
    stuck = _find_stuck_sensor(train)
    target_typos = _find_target_typos(train)
    spelling = _find_spelling_variants(train, later)

    return {
        "train_sha256": sha256_file(train_path),
        "later_sha256": sha256_file(later_path),
        "holdout_sha256": sha256_file(holdout_path),
        "rows": {"train": len(train), "later": len(later), "holdout": len(holdout)},
        "data_issues": {
            "leaking_column": {"present": leak is not None, "column": leak},
            "unit_mixup": unit_mixup,
            "sentinel_value": sentinel,
            "duplicate_rows": {
                "present": duplicate_count > 0,
                "count": duplicate_count if duplicate_count > 0 else None,
            },
            "stuck_sensor": stuck,
            "target_typos": target_typos,
            "spelling_variants": spelling,
        },
    }


def write_findings_template(results: dict, output_path: str | Path) -> None:
    issues = results["data_issues"]
    payload = {
        "group_id": "G062",
        "problem_id": "MP008",
        "data_issues": issues,
        "tracking": {
            "experiment_name": None,
            "best_run_id": None,
            "registered_model_name": None,
            "champion_version": None,
        },
        "monitoring": {
            "psi": {name: None for name in NUMERIC_FEATURES},
            "most_drifted_feature": None,
            "mae_by_group": {"Feeder-1": None, "Feeder-2": None, "Feeder-3": None},
            "worst_served_group": None,
            "mae_v1_on_check": None,
            "mae_v2_on_check": None,
            "decision": None,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--later", default="data/raw/later.csv")
    parser.add_argument("--holdout", default="data/raw/holdout_features.csv")
    parser.add_argument("--findings", default="reports/findings.yaml")
    args = parser.parse_args()

    results = analyse_data(args.train, args.later, args.holdout)
    print(json.dumps(results, indent=2))
    write_findings_template(results, args.findings)
    print(f"\nData findings written to: {args.findings}")


if __name__ == "__main__":
    main()
