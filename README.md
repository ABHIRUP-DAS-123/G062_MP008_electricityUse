# G062 | MP008: Household Electricity Use Prediction

**Course:** SCSE3040 MLOps Mini Project  
**Group:** G062  
**Problem:** MP008 - Household Electricity Use (Lucknow)  
**Faculty:** Dr. Avantika Saklani  
**Batch:** 90

## Team

| Enrollment number | Member |
|---|---|
| S24CSEU0417 | Shreya Kaushal |
| S24CSEU0185 | Vishakha |
| S24CSEU0601 | Naivedya Shilpi |
| S24CSEU1288 | Abhirup Das |

## Project objective

Build an end-to-end MLOps pipeline to predict household electricity consumption (`units_kwh`) for households in Lucknow. The assignment covers data quality investigation and cleaning, DVC data versioning, model training and MLflow tracking, a FastAPI prediction service, Docker, automated tests and CI, Kubernetes manifests, data-drift monitoring, retraining, and holdout predictions.

## Repository structure

```text
.
├── .dvc/
├── .dvcignore
├── data/
│   └── raw/
│       ├── holdout_features.csv
│       ├── train.csv.dvc
│       └── later.csv.dvc
├── reports/
│   └── findings.yaml
├── scripts/
│   └── setup_dvc.ps1
└── src/
    └── mp008/
        ├── __init__.py
        ├── cleaning.py
        └── data_checks.py
```

The repository is being built in stages. This README will be updated as the model, API, deployment, monitoring, tests, and prediction artifacts are integrated.

## Prerequisites

- Windows PowerShell
- Python 3.12 (recommended by the assignment Docker requirement)
- Git
- DVC

## Obtain the assignment data

Use only the original CSV files supplied to Group G062 in the faculty ZIP. Place them at:

```text
data/raw/train.csv
data/raw/later.csv
data/raw/holdout_features.csv
```

Do not open and save `train.csv` in Excel. The project setup script verifies the original SHA-256 fingerprints before DVC tracking.

The handbook requires `train.csv` and `later.csv` to be tracked with DVC. `holdout_features.csv` is kept as a normal repository file as specified in the handbook's repository layout.

## Install Phase 1 dependencies

From the repository root:

```powershell
python -m pip install dvc pandas numpy pyyaml
```

## Verify original data and initialize DVC

Run this from the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_dvc.ps1
```

The script verifies the original file fingerprints, initializes DVC if needed, and runs `dvc add` for `train.csv` and `later.csv`.

Review the generated pointer files and repository status:

```powershell
Get-Content .\dataaw	rain.csv.dvc
Get-Content .\dataaw\later.csv.dvc
git status
```

Commit the DVC pointer files and DVC configuration. Do not commit `train.csv` or `later.csv` as ordinary Git files.

**DVC remote note:** A DVC remote has not been configured in this repository yet. A fresh clone cannot retrieve the raw DVC-tracked CSVs with `dvc pull` until the group configures and uses an accessible shared remote. Until then, each team member needs the original faculty ZIP.

## Run the Phase 1 data investigation

Install the Python packages if they are not already installed:

```powershell
python -m pip install pandas numpy pyyaml
```

Run:

```powershell
python -m src.mp008.data_checks
```

To specify paths explicitly:

```powershell
python -m src.mp008.data_checks --train data/raw/train.csv --later data/raw/later.csv --holdout data/raw/holdout_features.csv --findings reports/findings.yaml
```

The script prints the source-file SHA-256 values, row counts, and the seven data-quality checks. It writes the data-issue findings to `reports/findings.yaml`. MLflow and monitoring fields are left for the later project stages.

## Use the cleaning functions

The current cleaning module exposes:

```python
from src.mp008.cleaning import (
    clean_training_data,
    clean_inference_data,
)
```

- `clean_training_data(raw_df)` returns `(X, y)` for model training.
- `clean_inference_data(raw_df)` returns the approved prediction-time feature columns.
- The model feature order is `rooms`, `members`, `ac_hours_per_day`, `avg_temp_c`, `house_area_sqft`, `solar_kw`, `connection`, `feeder`.
- `billed_units_after_reading` is not a model feature because it is unavailable in the holdout input.

## Phase 1 status

The data investigation and cleaning modules are present. The remaining project stages will add model training and MLflow, API and deployment, CI and tests, Kubernetes, monitoring/retraining, and holdout predictions. Do not treat the project as complete until those artifacts have been implemented, executed, and checked against `STUDENT_HANDBOOK.md` and `PROBLEM.md`.
