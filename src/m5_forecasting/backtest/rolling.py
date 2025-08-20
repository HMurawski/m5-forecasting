from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Split:
    train_end: np.datetime64
    test_start: np.datetime64
    test_end: np.datetime64


def rolling_splits(
    dates: np.ndarray,
    horizon: int = 28,
    step: int = 7,
    min_train_days: int = 365,
) -> Iterable[Split]:
    """Yield rolling-origin splits over a sorted array of dates (unique).


    Keep it generic; downstream code will slice per-series.
    """
    unique_days = np.unique(dates)
    if len(unique_days) < min_train_days + horizon:
        return

    start_idx = min_train_days - 1
    while start_idx + horizon < len(unique_days):
        train_end = unique_days[start_idx]
        test_start = unique_days[start_idx + 1]
        test_end = unique_days[start_idx + horizon]
        logger.info(f"Split: train to {train_end}, test from {test_start} to {test_end}")
        yield Split(train_end, test_start, test_end)
        start_idx += step


def evaluate_split(
    df: pl.DataFrame,
    split: Split,
    id_col: str = "id",
    date_col: str = "date",
    target_col: str = "qty",
    seasonality: int = 7,
) -> pl.DataFrame:
    """Example placeholder that expects df to already contain columns with predictions.


    In the next step we will plug baselines (seasonal naïve + quantile variants) and models.
    Returns per-series metrics for this split.
    """
    # This is a skeleton; it will be filled when we wire a predictor
    return pl.DataFrame({"id": [], "mase": [], "wmape": [], "split_end": []})
