"""MLflow experiment tracking, model registration, and alias management for MP008."""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature


def _sqlite_path_from_uri(tracking_uri: str) -> Path | None:
    """Extract the local SQLite file path from a sqlite:/// tracking URI."""
    if not tracking_uri.startswith("sqlite:///"):
        return None
    raw_path = tracking_uri[len("sqlite:///") :]
    return Path(raw_path)


def make_artifact_paths_portable(
    tracking_uri: str = "sqlite:///mlflow.db",
    artifact_root_dir: str = "mlruns",
) -> int:
    """Normalize machine-specific absolute artifact URIs in mlflow.db to portable relative paths.

    Rewrites entries such as:
        file:///C:/.../mlruns/1/... or file:C:/.../mlruns/1/...
    to:
        mlruns/1/...
    across `experiments`, `runs`, `logged_models`, and `model_versions`.
    """
    db_path = _sqlite_path_from_uri(tracking_uri)
    if db_path is None or not db_path.exists():
        return 0

    pattern = re.compile(
        rf"^(?:file:(?:///?)?)?(?:[A-Za-z]:)?(?:[^'\"]*[/\\])?({re.escape(artifact_root_dir)}(?:[/\\].*)?)$"
    )

    def _to_relative(uri_val: str | None) -> str | None:
        if not uri_val or not isinstance(uri_val, str):
            return uri_val
        normalized = uri_val.replace("\\", "/")
        match = pattern.match(normalized)
        if match:
            return match.group(1)
        return uri_val

    updated_count = 0
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        table_Specs = [
            ("experiments", "experiment_id", "artifact_location"),
            ("runs", "run_uuid", "artifact_uri"),
            ("logged_models", "model_id", "artifact_location"),
            ("model_versions", "rowid", "storage_location"),
        ]
        for table, pk_col, uri_col in table_Specs:
            exists = cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not exists:
                continue
            rows = cur.execute(f"SELECT {pk_col}, {uri_col} FROM {table}").fetchall()
            for pk_val, uri_val in rows:
                new_val = _to_relative(uri_val)
                if new_val != uri_val:
                    cur.execute(
                        f"UPDATE {table} SET {uri_col} = ? WHERE {pk_col} = ?",
                        (new_val, pk_val),
                    )
                    updated_count += 1
        conn.commit()
    finally:
        conn.close()

    return updated_count


def configure_mlflow(
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "MP008_Electricity_Consumption",
    artifact_location: str = "mlruns/1",
) -> str:
    """Configure MLflow SQLite tracking URI and active experiment with portable artifact location."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
    else:
        exp_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    make_artifact_paths_portable(tracking_uri=tracking_uri)
    return str(exp_id)


def log_training_run(
    run_name: str,
    model_name: str,
    pipeline: Any,
    hyperparameters: dict[str, Any],
    metrics: dict[str, float],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    data_sha256: str,
    stage_tag: str = "v1",
    extra_params: dict[str, Any] | None = None,
    extra_metrics: dict[str, float] | None = None,
    registered_model_name: str | None = None,
    serialization_format: str = "cloudpickle",
) -> dict[str, Any]:
    """Execute a complete MLflow tracking run and log model artifact with signature."""
    if not data_sha256 or not isinstance(data_sha256, str):
        raise ValueError("data_sha256 must be a non-empty string hash of original train.csv.")

    feature_names = list(X_train.columns)
    input_example = X_train.head(min(5, len(X_train))).copy()
    sample_preds = pipeline.predict(input_example)
    signature = infer_signature(input_example, sample_preds)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("stage", stage_tag)
        mlflow.set_tag("group_id", "G062")
        mlflow.set_tag("problem_id", "MP008")

        base_params: dict[str, Any] = {
            "model_name": model_name,
            "data_sha256": data_sha256,
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "feature_names": ",".join(feature_names),
            "feature_count": int(len(feature_names)),
            "stage": stage_tag,
        }
        for key, val in hyperparameters.items():
            base_params[str(key)] = "None" if val is None else val
        if extra_params:
            for key, val in extra_params.items():
                base_params[str(key)] = "None" if val is None else val

        mlflow.log_params(base_params)

        logged_metrics: dict[str, float] = {
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "r2": float(metrics["r2"]),
        }
        if extra_metrics:
            for key, val in extra_metrics.items():
                logged_metrics[str(key)] = float(val)

        mlflow.log_metrics(logged_metrics)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
            serialization_format=serialization_format,
        )

    make_artifact_paths_portable(tracking_uri=mlflow.get_tracking_uri())

    registered_version: int | None = None
    if registered_model_name:
        if getattr(model_info, "registered_model_version", None) is not None:
            registered_version = int(model_info.registered_model_version)
        else:
            registered_version = get_version_for_run(
                registered_model_name=registered_model_name,
                run_id=run_id,
            )

    return {
        "run_id": run_id,
        "run_name": run_name,
        "model_name": model_name,
        "model_uri": model_info.model_uri,
        "registered_model_name": registered_model_name,
        "registered_version": registered_version,
        "metrics": logged_metrics,
        "params": base_params,
    }


def wait_for_model_version_ready(
    client: MlflowClient,
    registered_model_name: str,
    version: int | str,
    timeout_seconds: float = 30.0,
) -> int:
    """Wait until an MLflow registered model version reaches READY status."""
    version_str = str(version)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        mv = client.get_model_version(name=registered_model_name, version=version_str)
        status = str(getattr(mv, "status", "READY")).upper()
        if status == "READY":
            return int(mv.version)
        if status in {"FAILED_REGISTRATION", "CANCELLED"}:
            raise RuntimeError(
                f"Model registration failed for {registered_model_name} v{version_str}: {status}"
            )
        time.sleep(0.25)
    return int(version_str)


def get_version_for_run(
    registered_model_name: str,
    run_id: str,
    client: MlflowClient | None = None,
) -> int:
    """Look up the registered model version corresponding to a given run_id."""
    mlflow_client = client or MlflowClient()
    versions = mlflow_client.search_model_versions(f"name='{registered_model_name}'")
    matching = [int(mv.version) for mv in versions if mv.run_id == run_id]
    if not matching:
        raise ValueError(
            f"No model version found for model '{registered_model_name}' and run_id '{run_id}'."
        )
    latest_ver = max(matching)
    return wait_for_model_version_ready(mlflow_client, registered_model_name, latest_ver)


def register_run_model_version(
    run_id: str,
    registered_model_name: str,
    model_uri: str | None = None,
    client: MlflowClient | None = None,
) -> int:
    """Register a logged run's model artifact as a new version in MLflow Model Registry."""
    mlflow_client = client or MlflowClient()
    try:
        existing = get_version_for_run(registered_model_name, run_id, client=mlflow_client)
        make_artifact_paths_portable(tracking_uri=mlflow.get_tracking_uri())
        return existing
    except ValueError:
        pass

    source_uri = model_uri or f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=source_uri, name=registered_model_name)
    ready_ver = wait_for_model_version_ready(mlflow_client, registered_model_name, mv.version)
    make_artifact_paths_portable(tracking_uri=mlflow.get_tracking_uri())
    return ready_ver


def set_champion_alias(
    registered_model_name: str,
    version: int | str,
    alias: str = "champion",
    client: MlflowClient | None = None,
) -> dict[str, Any]:
    """Assign the 'champion' alias to the specified registered model version and verify it."""
    mlflow_client = client or MlflowClient()
    ready_version = wait_for_model_version_ready(
        mlflow_client, registered_model_name, version
    )
    mlflow_client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=str(ready_version),
    )
    make_artifact_paths_portable(tracking_uri=mlflow.get_tracking_uri())
    resolved = mlflow_client.get_model_version_by_alias(
        name=registered_model_name,
        alias=alias,
    )
    return {
        "registered_model_name": registered_model_name,
        "alias": alias,
        "version": int(resolved.version),
        "run_id": resolved.run_id,
        "status": resolved.status,
    }
