"""Tests for MLflow experiment tracking, data_sha256 logging, portable artifact loading, and champion alias."""
from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

from src.mp008.cleaning import clean_inference_data, clean_training_data
from src.mp008.evaluation import evaluate_model
from src.mp008.mlflow_utils import (
    configure_mlflow,
    log_training_run,
    register_run_model_version,
    set_champion_alias,
)
from src.mp008.preprocessing import load_params
from src.mp008.training import build_model_pipeline
from tests.test_training import make_synthetic_raw_dataset


def test_mlflow_logging_and_registration_isolated(tmp_path: Path) -> None:
    """Test 6 (isolated): MLflow run logs data_sha256, MAE, model artifact, and champion alias."""
    db_file = (tmp_path / "test_mlflow.db").as_posix()
    tracking_uri = f"sqlite:///{db_file}"
    exp_name = "test_mp008_exp"
    reg_name = "test_mp008_model"
    expected_sha = "9aea26c5aa76831f4a4e58a3dfafd06d47144136c13ace14834010bfcf472781"

    prev_uri = mlflow.get_tracking_uri()
    try:
        configure_mlflow(
            tracking_uri=tracking_uri,
            experiment_name=exp_name,
            artifact_location=(tmp_path / "mlruns" / "1").as_posix(),
        )

        raw = make_synthetic_raw_dataset(n_rows=30)
        X, y = clean_training_data(raw)
        pipe = build_model_pipeline(
            model_type="RandomForestRegressor",
            hyperparameters={"n_estimators": 10, "random_state": 42},
        ).fit(X, y)
        metrics = evaluate_model(pipe, X, y)

        rec = log_training_run(
            run_name="synthetic_v1",
            model_name="RandomForestRegressor",
            pipeline=pipe,
            hyperparameters={"n_estimators": 10, "random_state": 42},
            metrics=metrics,
            X_train=X,
            X_val=X,
            data_sha256=expected_sha,
            registered_model_name=reg_name,
        )

        client = MlflowClient()
        run = client.get_run(rec["run_id"])
        assert run.data.params.get("data_sha256") == expected_sha
        assert "mae" in run.data.metrics
        assert "rmse" in run.data.metrics
        assert "r2" in run.data.metrics

        ver = register_run_model_version(rec["run_id"], reg_name, rec["model_uri"], client=client)
        alias_res = set_champion_alias(reg_name, ver, alias="champion", client=client)
        assert alias_res["version"] == ver
    finally:
        mlflow.set_tracking_uri(prev_uri)


def test_repository_mlflow_database_and_registry() -> None:
    """Verify the repository root mlflow.db contains >= 3 runs, data_sha256, 2 versions, champion alias, and relative paths."""
    params = load_params("params.yaml")
    mlflow_cfg = params["mlflow"]
    expected_sha = params["data"]["expected_train_sha256"]
    exp_name = mlflow_cfg["experiment_name"]
    reg_name = mlflow_cfg["registered_model_name"]

    db_path = Path("mlflow.db")
    assert db_path.exists(), "mlflow.db must exist at the repository root"

    # Verify no machine-specific absolute paths remain in mlflow.db
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for table, col in [
            ("experiments", "artifact_location"),
            ("runs", "artifact_uri"),
            ("logged_models", "artifact_location"),
            ("model_versions", "storage_location"),
        ]:
            uris = [r[0] for r in cur.execute(f"SELECT {col} FROM {table}").fetchall() if r[0]]
            for uri in uris:
                assert "C:/" not in uri and "c:/" not in uri and "Users" not in uri, (
                    f"Non-portable machine-specific URI found in {table}.{col}: {uri}"
                )
                assert uri.startswith("mlruns/"), f"Expected relative 'mlruns/...' URI, got {uri}"
    finally:
        conn.close()

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    client = MlflowClient()

    experiment = client.get_experiment_by_name(exp_name)
    assert experiment is not None, f"Experiment {exp_name} not found in mlflow.db"

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 3, f"Expected at least 3 runs, found {len(runs)}"

    for run in runs:
        assert run.data.params.get("data_sha256") == expected_sha
        assert "mae" in run.data.metrics
        assert "rmse" in run.data.metrics
        assert "r2" in run.data.metrics

    versions = client.search_model_versions(f"name='{reg_name}'")
    assert len(versions) >= 2, f"Expected at least 2 registered model versions, found {len(versions)}"

    champion_mv = client.get_model_version_by_alias(name=reg_name, alias="champion")
    assert int(champion_mv.version) == 2


def test_mlflow_registered_artifacts_portable_across_copied_directory(tmp_path: Path) -> None:
    """Verify both registered model versions and @champion alias load from a separate copied project directory."""
    repo_root = Path.cwd()
    shutil.copy2(repo_root / "mlflow.db", tmp_path / "mlflow.db")
    shutil.copytree(repo_root / "mlruns", tmp_path / "mlruns")

    sample_X = clean_inference_data(make_synthetic_raw_dataset(n_rows=5))

    prev_cwd = os.getcwd()
    prev_uri = mlflow.get_tracking_uri()
    try:
        os.chdir(tmp_path)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        v1_model = mlflow.sklearn.load_model("models:/MP008_Electricity_Model/1")
        v2_model = mlflow.sklearn.load_model("models:/MP008_Electricity_Model/2")
        champ_model = mlflow.sklearn.load_model("models:/MP008_Electricity_Model@champion")

        preds_v1 = v1_model.predict(sample_X)
        preds_v2 = v2_model.predict(sample_X)
        preds_champ = champ_model.predict(sample_X)

        assert len(preds_v1) == 5 and np.all(np.isfinite(preds_v1))
        assert len(preds_v2) == 5 and np.all(np.isfinite(preds_v2))
        assert np.allclose(preds_v2, preds_champ)
    finally:
        os.chdir(prev_cwd)
        mlflow.set_tracking_uri(prev_uri)
