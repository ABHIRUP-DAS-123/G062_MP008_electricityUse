"""Data drift monitoring utilities (PSI) for MP008."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.mp008.preprocessing import NUMERIC_FEATURES


def build_psi_bin_edges(
    reference: pd.Series | np.ndarray,
    buckets: int = 10,
) -> np.ndarray:
    """Derive strictly increasing PSI bin edges from the reference distribution.

    - Uses `buckets` quantile intervals (`np.linspace(0, 100, buckets + 1)`) when
      the reference feature has enough distinct values.
    - Deduplicates repeated quantile boundaries so discrete numeric features
      (e.g., `rooms`, `members`) never produce zero-width bins.
    - Handles constant-valued reference features safely by creating a finite
      narrow interval around the constant value bounded by `(-inf, +inf)`.
    """
    if buckets < 1:
        raise ValueError(f"buckets must be >= 1, got {buckets}")

    ref = np.asarray(
        pd.to_numeric(pd.Series(reference), errors="coerce").dropna(),
        dtype=float,
    )
    if len(ref) == 0:
        raise ValueError("Reference sample must be non-empty to build PSI bin edges.")

    ref_min = float(np.min(ref))
    ref_max = float(np.max(ref))

    # Constant-valued reference feature: avoid zero-width bins
    if np.isclose(ref_min, ref_max, rtol=0.0, atol=1e-12):
        delta = max(abs(ref_min) * 1e-6, 1e-6)
        return np.array([-np.inf, ref_min - delta, ref_max + delta, np.inf], dtype=float)

    raw_quantiles = np.percentile(ref, np.linspace(0.0, 100.0, buckets + 1))
    internal_edges = np.unique(raw_quantiles[1:-1])

    # If all internal quantiles collapsed to the minimum or maximum (e.g., extreme point mass),
    # fall back to the midpoint between min and max so at least 2 valid bins exist.
    if len(internal_edges) == 0:
        midpoint = 0.5 * (ref_min + ref_max)
        internal_edges = np.array([midpoint], dtype=float)

    edges = np.concatenate(([-np.inf], internal_edges, [np.inf])).astype(float)
    edges = np.unique(edges)
    if np.any(np.diff(edges) <= 0):
        raise RuntimeError(f"Non-monotonic PSI bin edges generated: {edges}")
    return edges


def compute_psi(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    buckets: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index (PSI) between reference and current samples.

    Handles discrete features (duplicate quantile boundaries) and constant features
    safely without creating zero-width bins.
    """
    ref = np.asarray(
        pd.to_numeric(pd.Series(reference), errors="coerce").dropna(),
        dtype=float,
    )
    cur = np.asarray(
        pd.to_numeric(pd.Series(current), errors="coerce").dropna(),
        dtype=float,
    )
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("Reference and current arrays must be non-empty for PSI calculation.")

    edges = build_psi_bin_edges(ref, buckets=buckets)

    ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_share = np.histogram(cur, bins=edges)[0] / len(cur)

    ref_share = np.clip(ref_share, eps, None)
    cur_share = np.clip(cur_share, eps, None)

    psi_value = float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
    return max(0.0, psi_value)


def compute_feature_drift(
    train_features: pd.DataFrame,
    later_features: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    buckets: int = 10,
) -> dict[str, Any]:
    """Compute PSI for all numeric features between cleaned train and later datasets."""
    cols = list(numeric_columns) if numeric_columns is not None else list(NUMERIC_FEATURES)
    psi_scores: dict[str, float] = {}
    bin_counts: dict[str, int] = {}
    for col in cols:
        if col not in train_features.columns or col not in later_features.columns:
            raise ValueError(f"Column '{col}' missing when computing PSI drift.")
        edges = build_psi_bin_edges(train_features[col], buckets=buckets)
        psi_val = compute_psi(train_features[col], later_features[col], buckets=buckets)
        psi_scores[col] = round(psi_val, 4)
        bin_counts[col] = int(len(edges) - 1)

    most_drifted = max(psi_scores, key=lambda k: psi_scores[k])
    return {
        "buckets": buckets,
        "effective_bins": bin_counts,
        "psi": psi_scores,
        "most_drifted_feature": most_drifted,
        "max_psi": psi_scores[most_drifted],
    }
