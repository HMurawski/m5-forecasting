from __future__ import annotations

import numpy as np


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE = sum(|e|) / sum(|y|). Safe for zeros-only targets.


    Args:
    y_true: actuals
    y_pred: predictions (same length)
    """
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return np.abs(y_true - y_pred).sum() / denom


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: np.ndarray,
    seasonality: int = 7,
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler).


    We scale MAE of forecast errors by the MAE of a seasonal-naïve method on the
    *in-sample* history.
    """
    if len(y_insample) <= seasonality:
        return float("nan")
    naive_diff = np.abs(y_insample[seasonality:] - y_insample[:-seasonality])
    mae_naive = naive_diff.mean()
    if mae_naive == 0:
        return float("nan")
    return np.abs(y_true - y_pred).mean() / mae_naive
