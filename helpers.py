# This files contains all the helper functions for the repo
import datetime as dt

import polars as pl
import yfinance as yf


def get_AAPL_timeseries(
    start: dt.date = dt.date(2000, 1, 1), end: dt.date | None = None
) -> pl.DataFrame:
    """Get AAPL closing prices as a Polars DataFrame."""
    if end is None:
        end = dt.date.today()

    df = pl.from_pandas(
        yf.download("AAPL", start="2015-01-01", end="2026-01-01").reset_index()
    )
    df = df.with_columns(pl.col(df.columns[0]).alias("Date").cast(dt.date))
    df = df.with_columns(pl.col(df.columns[1]).alias("Close").cast(pl.Float64))
    df = df.select(pl.col("Date"), pl.col("Close"))
    return df.filter(pl.col("Date") >= start, pl.col("Date") <= end)


print(get_AAPL_timeseries())
