
VENV_PY := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,.venv/bin/python)

.PHONY: setup precommit-install lint format data-ingest mlflow-ui prefect-start test train-baseline

setup:
	python -m venv .venv
	$(VENV_PY) -m pip install -U pip
	$(VENV_PY) -m pip install -e ".[dev]"

precommit-install:
	$(VENV_PY) -m pre_commit install

lint:
	$(VENV_PY) -m ruff check src

format:
	$(VENV_PY) -m black src

data-ingest:
	$(VENV_PY) -m m5_forecasting.data.ingest

mlflow-ui:
	$(VENV_PY) -m mlflow ui --backend-store-uri ./mlruns --port 5000

prefect-start:
	$(VENV_PY) -m prefect server start

test:
	$(VENV_PY) -m pytest -q

train-baseline:
	$(VENV_PY) -m m5_forecasting.experiments.baseline_lgbm
