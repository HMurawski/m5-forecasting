"""M5 ingest: unpivot sales, join calendar & prices, persist to Parquet and DuckDB.


Run: python -m m5_forecasting.data.ingest
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from ..utils.io import DUCKDB_PATH, PROCESSED_DIR, RAW_DIR, ensure_dirs
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _read_sales(path: Path) -> pl.DataFrame:
    try:
        df = pl.read_csv(path)
        logger.info(f"Sales data loaded: {path.name} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Sales file not found at {path}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error while reading sales data from {path}: {e}")
        raise


def _read_calendar(path: Path) -> pl.DataFrame:
    try:
        df = pl.read_csv(path)
        logger.info(f"Calendar data loaded: {path.name} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Calendar file not found at {path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while reading calendar data from {path}: {e}")
        raise


def _read_prices(path: Path) -> pl.DataFrame:
    try:
        df = pl.read_csv(path)
        logger.info(f"Prices file loaded: {path.name} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Prices file not found at {path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while reading prices data from path {path}: {e}")
        raise


def unpivot_sales(df: pl.DataFrame) -> pl.DataFrame:
    """Wide (d_1..d_1913) -> long with (d, units)."""
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    value_vars = [c for c in df.columns if c.startswith("d_")]

    long_df = df.unpivot(index=id_vars, on=value_vars, variable_name="d", value_name="units")

    long_df = long_df.with_columns(
        [
            pl.col("units").cast(pl.Int32),
            pl.col("d").str.replace("d_", "").cast(pl.Int32).alias("d_num"),
        ]
    )
    return long_df


def join_calendar_prices(sales_long: pl.DataFrame, calendar: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    """Join long sales with per-day calendar and per-week prices.


    - calendar maps `d` -> actual `date` + `wm_yr_wk`
    - prices join via (store_id, item_id, wm_yr_wk)
    - create a unified `snap` flag based on state"""
    cal = calendar.select(
        [
            pl.col("d"),
            pl.col("date").str.strptime(pl.Date, fmt="%Y-%m-%d"),
            pl.col("wm_yr_wk").cast(pl.Int32),
            "wday",
            "week",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ]
    )
    joined = sales_long.join(cal, on="d", how="left")
    joined = joined.with_columns(
        pl.when(pl.col("state_id") == "CA")
        .then(pl.col("snap_CA"))
        .when(pl.col("state_id") == "TX")
        .then(pl.col("snap_TX"))
        .otherwise(pl.col("snap_WI"))
        .alias("snap")
        .cast(pl.Int8)
    ).drop(["snap_CA", "snap_TX", "snap_WI"])

    price_cols = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    prices_sel = prices.select(price_cols)
    out = joined.join(prices_sel, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    # Optional ordering for readability
    out = out.select(
        [
            "date",
            "d",
            "d_num",
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            pl.col("units").alias("qty"),
            "sell_price",
            "wm_yr_wk",
            "wday",
            "week",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap",
        ]
    )
    return out


def write_outputs(df: pl.DataFrame) -> Path:
    ensure_dirs()
    parquet_path = PROCESSED_DIR / "fact_sales.parquet"
    df.write_parquet(parquet_path)

    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS m5")
    # DuckDB can read Parquet directly — zero copy into table
    con.execute(
        f"""
    CREATE OR REPLACE TABLE m5.fact_sales AS
    SELECT *
    FROM read_parquet('{parquet_path.as_posix()}')
    """
    )
    con.close()
    return parquet_path


def main() -> None:
    # Check raw files
    required = [
        RAW_DIR / "sales_train_validation.csv",
        RAW_DIR / "calendar.csv",
        RAW_DIR / "sell_prices.csv",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files under {RAW_DIR}: {missing}")

    sales = _read_sales(RAW_DIR / "sales_train_validation.csv")
    calendar = _read_calendar(RAW_DIR / "calendar.csv")
    prices = _read_prices(RAW_DIR / "sell_prices.csv")

    sales_long = unpivot_sales(sales)
    fact = join_calendar_prices(sales_long, calendar, prices)
    path = write_outputs(fact)

    logger.info(f"Ingest OK → {len(fact):,} rows to {path} and DuckDB at {DUCKDB_PATH.as_posix()}")


if __name__ == "__main__":
    main()
