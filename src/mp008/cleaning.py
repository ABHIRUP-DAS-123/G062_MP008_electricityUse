"""Cleaning functions for the MP008 household electricity dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "units_kwh"
FEATURE_COLUMNS = [
    "rooms",
    "members",
    "ac_hours_per_day",
    "avg_temp_c",
    "house_area_sqft",
    "solar_kw",
    "connection",
    "feeder",
]
LEAKAGE_COLUMN = "billed_units_after_reading"


def clean_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned training frame without modifying the input frame.

    Corrections supported by the supplied G062 data:
    - Remove exact duplicate rows.
    - Convert Feeder-1 AC readings from minutes to hours (divide by 60).
    - Correct 10x target-entry errors using the after-reading bill column.
    """
    df = raw.copy(deep=True)
    df = df.drop_duplicates().reset_index(drop=True)

    for column in ("connection", "feeder"):
        if column in df:
            df[column] = df[column].astype("string").str.strip()

    feeder_mask = df["feeder"].eq("Feeder-1")
    df.loc[feeder_mask, "ac_hours_per_day"] = (
        pd.to_numeric(df.loc[feeder_mask, "ac_hours_per_day"], errors="coerce") / 60.0
    )

    if LEAKAGE_COLUMN in df and TARGET in df:
        billed = pd.to_numeric(df[LEAKAGE_COLUMN], errors="coerce").replace(0, np.nan)
        ratio = pd.to_numeric(df[TARGET], errors="coerce") / billed
        typo_mask = ratio.between(8.0, 12.0, inclusive="both")
        df.loc[typo_mask, TARGET] = df.loc[typo_mask, TARGET] / 10.0

    return df


def make_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only the eight fields allowed by PROBLEM.md / holdout schema."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required model feature columns: {missing}")

    features = frame[FEATURE_COLUMNS].copy()
    for column in ("connection", "feeder"):
        features[column] = features[column].astype("string").str.strip()
    return features


def clean_training_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean training data and return (X, y), excluding post-event leakage."""
    cleaned = clean_training_frame(raw)
    X = make_model_features(cleaned)
    y = pd.to_numeric(cleaned[TARGET], errors="raise").astype(float)
    if y.isna().any():
        raise ValueError("Target contains missing values after cleaning.")
    return X, y


def clean_inference_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Prepare later/holdout feature rows without applying train-only corrections."""
    return make_model_features(raw)
