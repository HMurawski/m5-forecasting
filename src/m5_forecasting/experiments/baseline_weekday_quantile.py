from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl


@dataclass(frozen=True)
class OneSeriesForecast:
    dates: list[date]
    q10: list[float]
    q50: list[float]
    q90: list[float]


def weekday_quantile_forecast_one(
    df_series: pl.DataFrame,
    forecast_start: date,
    horizon: int = 28,
    k_weeks: int = 13,
    backoff_weeks: int = 26,
    min_nonzero: int = 3,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pl.DataFrame:
    """
    Empirical baseline forecast:
    For each day in the forecast horizon, take the last K historical values
    from the SAME weekday and compute the requested quantiles.

    df_series: only ONE time series; columns ['date','qty','weekday'], sorted ascending by date.
               Historical data must end on (forecast_start - 1).
    Returns: DataFrame with columns ['date','q10','q50','q90'].
    """
    assert set(("date", "qty", "weekday")).issubset(df_series.columns)
    hist = df_series.filter(pl.col("date") < pl.lit(forecast_start)).select("date", "qty", "weekday").sort("date")

    out_dates: list[date] = []
    q_vals = {q: [] for q in quantiles}

    for i in range(horizon):
        d = forecast_start + timedelta(days=i)
        wname = d.strftime("%A")

        arr_w = hist.filter(pl.col("weekday") == wname).select("qty").tail(k_weeks).to_series().to_numpy()

        arr_b = hist.tail(backoff_weeks * 7).select("qty").to_series().to_numpy()

        use = arr_w
        if use.size == 0 or (use > 0).sum() < min_nonzero:
            use = arr_b if arr_b.size > 0 else np.array([0.0], dtype=float)

        use = use.astype(float)
        use = np.clip(use, 0, None)  # na wszelki wypadek

        qs = {q: float(np.quantile(use, q)) for q in quantiles}

        if 0.1 in qs and 0.5 in qs:
            qs[0.1] = min(qs[0.1], qs[0.5])
        if 0.9 in qs and 0.5 in qs:
            qs[0.9] = max(qs[0.9], qs[0.5])

        out_dates.append(d)
        for q in quantiles:
            q_vals[q].append(qs[q])

    out = pl.DataFrame({"date": out_dates})
    if 0.1 in quantiles:
        out = out.with_columns(pl.Series("q10", q_vals[0.1]))
    if 0.5 in quantiles:
        out = out.with_columns(pl.Series("q50", q_vals[0.5]))
    if 0.9 in quantiles:
        out = out.with_columns(pl.Series("q90", q_vals[0.9]))
    return out
