# M5 Forecasting (Quantile)


- **Stack**: Polars, DuckDB, LightGBM, Optuna, MLflow, Prefect, Evidently, Power BI
- **Goal**: P10/P50/P90 forecasts, rolling-origin backtest (MASE, wMAPE), batch inference → PostgreSQL


## Quickstart
1. `python -m venv .venv && . .venv/Scripts/activate`
2. `pip install -U pip && pip install -e .[dev]`
3. Put Kaggle CSVs under `data/raw/`
4. `pre-commit install`
5. `make data-ingest` (or `python -m m5_forecasting.data.ingest`)
6. `make mlflow-ui` to browse runs
7. `python -m m5_forecasting.pipelines.flows` to execute the basic flow
