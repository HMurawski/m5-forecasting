from __future__ import annotations

import os

import mlflow


def run_experiment() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("baseline_lgbm_quantile")


with mlflow.start_run(run_name="smoke_check"):
    mlflow.log_param("horizon", 28)
    mlflow.log_param("quantiles", [0.1, 0.5, 0.9])
    # Placeholder metric before we wire a real baseline/model
    mlflow.log_metric("dummy_metric", 0.0)


# TODO: train 3 LightGBM models with objective=quantile for alphas [0.1,0.5,0.9]
# and log artifacts (feature importance, plots) + metrics.


if __name__ == "__main__":
    run_experiment()
