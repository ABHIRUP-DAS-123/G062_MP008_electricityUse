"""Preprocessing pipelines for the MP008 household electricity dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.mp008.cleaning import FEATURE_COLUMNS

NUMERIC_FEATURES: list[str] = [
    "rooms",
    "members",
    "ac_hours_per_day",
    "avg_temp_c",
    "house_area_sqft",
    "solar_kw",
]
CATEGORICAL_FEATURES: list[str] = [
    "connection",
    "feeder",
]


def _coerce_categorical_to_object(frame: Any) -> Any:
    """Convert pandas StringDtype / pd.NA to object dtype with np.nan for SimpleImputer compatibility."""
    if isinstance(frame, pd.DataFrame):
        return frame.astype(object).where(frame.notna(), np.nan)
    if isinstance(frame, pd.Series):
        return frame.astype(object).where(frame.notna(), np.nan)
    return frame


def load_params(params_path: str | Path = "params.yaml") -> dict[str, Any]:
    """Read and parse project configuration from params.yaml."""
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    numeric_imputer_strategy: str = "median",
    categorical_imputer_strategy: str = "most_frequent",
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create a scikit-learn ColumnTransformer for numeric and categorical features.

    All transformations are fit strictly on training data and applied
    reproducibly to validation and holdout datasets.
    """
    num_cols = list(numeric_features) if numeric_features is not None else list(NUMERIC_FEATURES)
    cat_cols = list(categorical_features) if categorical_features is not None else list(CATEGORICAL_FEATURES)

    expected_set = set(FEATURE_COLUMNS)
    configured_set = set(num_cols) | set(cat_cols)
    if configured_set != expected_set:
        raise ValueError(
            f"Preprocessor features {sorted(configured_set)} do not match "
            f"approved model features {sorted(expected_set)}"
        )

    num_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=numeric_imputer_strategy)),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("to_object", FunctionTransformer(_coerce_categorical_to_object, validate=False)),
            ("imputer", SimpleImputer(strategy=categorical_imputer_strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, num_cols),
            ("categorical", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )
