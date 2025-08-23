from pathlib import Path

import polars as pl


def test_fact_counts():
    fact = pl.read_parquet(Path("data/processed/fact_sales.parquet"))
    assert fact.select(pl.len()).item() == 30490 * 1913

    assert (
        fact.select((pl.concat_str([pl.col("id"), pl.col("date").cast(pl.Utf8)], separator="|")).n_unique()).item()
        == 30490 * 1913
    )


def test_types_calendar_prices():
    fact = pl.read_parquet("data/processed/fact_sales.parquet").head(1)
    assert fact.schema["wm_yr_wk"] in (pl.Int32, pl.Int64)
