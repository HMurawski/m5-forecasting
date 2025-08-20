import polars as pl

from m5_forecasting.data.ingest import unpivot_sales


def test_unpivot_sales():
    # Prepare mock input DataFrame (wide format)
    df = pl.DataFrame(
        {
            "id": ["FOO_1", "FOO_2"],
            "item_id": ["ITEM_1", "ITEM_2"],
            "dept_id": ["DEPT_1", "DEPT_2"],
            "cat_id": ["CAT_1", "CAT_2"],
            "store_id": ["STORE_1", "STORE_2"],
            "state_id": ["CA", "TX"],
            "d_1": [5, 10],
            "d_2": [7, 12],
        }
    )

    # Apply melt_sales
    result = unpivot_sales(df)
    expected_cols = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "units", "d_num"}

    # Expected shape
    assert set(result.columns) == expected_cols
    # Check expected columns
    expected_cols = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "units", "d_num"}
    assert set(result.columns) == expected_cols

    # Check specific values
    assert result.filter(pl.col("id") == "FOO_1").filter(pl.col("d") == "d_1")["units"].item() == 5
    assert result.filter(pl.col("id") == "FOO_2").filter(pl.col("d") == "d_2")["units"].item() == 12
    assert result.filter(pl.col("d") == "d_1")["d_num"].unique().item() == 1
