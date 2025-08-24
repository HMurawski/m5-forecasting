from __future__ import annotations

from prefect import flow, task

from ..data.ingest import main as ingest_main
from ..experiments.baseline_lgbm import run_experiment


@task
def ingest_task() -> None:
    ingest_main()


@task
def train_baseline_task() -> None:
    run_experiment()


@flow(name="m5_main")
def m5_flow() -> None:
    ingest_task()
    train_baseline_task()


if __name__ == "__main__":
    m5_flow()
