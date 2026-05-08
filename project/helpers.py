# This files contains all the helper functions for the repo
import datetime as dt
from math import e
import os

import polars as pl
import yfinance as yf

os.environ["PATH"] = r"C:\Program Files\Graphviz\bin;" + os.environ["PATH"]

r = 0.01
T = 0.5


def get_AAPL_timeseries(
    ticker: str = "AAPL",
    start: dt.date = dt.date(2000, 1, 1),
    end: dt.date | None = None,
) -> pl.DataFrame:
    """Get AAPL closing prices as a Polars DataFrame."""
    end = dt.date.today() if end is None else end + dt.timedelta(days=1)

    df = pl.from_pandas(yf.download(ticker, start, end).reset_index())
    df = df.with_columns(pl.col(df.columns[0]).alias("Date").cast(dt.date))
    df = df.with_columns(pl.col(df.columns[1]).alias("Close").cast(pl.Float64))
    df = df.select(pl.col("Date"), pl.col("Close"))
    return df.filter(pl.col("Date") >= start, pl.col("Date") <= end)


def get_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_returns")
    ).drop_nulls()


def get_vol(df: pl.DataFrame) -> float:
    if "log_returns" in df:
        return (
            df.select(pl.col("log_returns").std(ddof=1).alias("vol")).item() * 250**0.5
        )
    else:
        df = get_log_returns(df)
        return (
            df.select(pl.col("log_returns").std(ddof=1).alias("vol")).item() * 250**0.5
        )


def get_binomial_tree(aapl: pl.DataFrame, n: int) -> pl.DataFrame:
    vol = get_vol(aapl)
    delta_T = 0.5 / n  # 0.5 as 6 months until maturity
    u = e ** (vol * (delta_T**0.5))
    d = e ** (-vol * (delta_T**0.5))
    S0 = (
        aapl.filter(pl.col("Date") == aapl.select(pl.col("Date").max()).item())
        .select(pl.col("Close"))
        .item()
    )

    current = pl.DataFrame(
        {
            "t": pl.Series([0], dtype=pl.Int64),
            "path_id": pl.Series([1], dtype=pl.Int64),
            "S": [S0],
            "C": [S0],
            "move": [""],
            "history": [""],
        }
    )
    layers = [current]
    for t in range(1, n + 1):
        up = current.select(
            pl.lit(t, dtype=pl.Int64).alias("t"),
            (pl.col("path_id") * 2).alias("path_id"),
            (pl.col("S") * u).alias("S"),
            (pl.col("C") + pl.col("S") * u).alias("C"),
            pl.lit("U").alias("move"),
            (pl.col("history") + pl.lit("U")).alias("history"),
        )

        down = current.select(
            pl.lit(t, dtype=pl.Int64).alias("t"),
            (pl.col("path_id") * 2 + 1).alias("path_id"),  # just to index the tree
            (pl.col("S") * d).alias("S"),  # New spot price
            (pl.col("C") + pl.col("S") * d).alias("C"),  # Add to the sum
            pl.lit("D").alias("move"),  # show the move
            (pl.col("history") + pl.lit("D")).alias("history"),
        )

        current = pl.concat([up, down])
        layers.append(current)
        tree = pl.concat(layers)

    tree = tree.with_columns(
        pl.when(pl.col("path_id") == 1)
        .then(None)
        .otherwise(pl.col("path_id") // 2)
        .alias("parent_id")
    )

    return tree


def get_q(df: pl.DataFrame, vol: float, r: float = 0.01, strict=False) -> float:
    n = df.select(pl.col("t").max()).item()
    delta_t = T / n
    u = e ** (vol * delta_t**0.5)
    d = e ** (-vol * delta_t**0.5)
    q = (e ** (r * delta_t) - d) / (u - d)
    return q


def normalise(df: pl.DataFrame, col: str = "C") -> pl.DataFrame:
    """Adds a column called normalised_C which is the cumulative prices normalised with
    mean and std (ddof = 1)"""
    std = df.select(pl.col(col)).std(ddof=1).item()
    mean = df.select(pl.col(col).mean()).item()
    df = df.with_columns((((pl.col(col)) - mean) / std).alias(f"normalised_{col}"))
    return df


def get_arithmetic_avg(df: pl.DataFrame) -> pl.DataFrame:
    """Adds S_bar column to df"""
    df = df.with_columns((pl.col("C") / (pl.col("t") + 1)).alias("S_bar"))
    return df


def get_payoff(df: pl.DataFrame) -> pl.DataFrame:
    if "S_bar" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("t") == (df.select(pl.col("t").max()).item()))
            .then((pl.col("S") - pl.col("S_bar")).clip(lower_bound=0))
            .otherwise(None)
            .alias("V")
        )
    else:
        df = get_arithmetic_avg(df)
        df = df.with_columns(
            (pl.col("S") - pl.col("S_bar")).clip(lower_bound=0).alias("V")
        )
    return df


def backward_induction(df: pl.DataFrame, vol) -> pl.DataFrame:
    n = int(df.select(pl.col("t").max()).item())
    dt = T / n
    discount = e ** (-r * dt)
    q = get_q(df, vol)

    for t in range(n - 1, -1, -1):
        children = (
            df.filter(pl.col("t") == t + 1)
            .select(["parent_id", "move", "V"])
            .pivot(
                values="V",
                index="parent_id",
                on="move",
            )
            .rename(
                {
                    "parent_id": "path_id",
                    "U": "V_up",
                    "D": "V_down",
                }
            )
            .with_columns(
                (discount * (q * pl.col("V_up") + (1 - q) * pl.col("V_down"))).alias(
                    "V_new"
                )
            )
            .select(["path_id", "V_new"])
        )

        df = (
            df.join(children, on="path_id", how="left")
            .with_columns(
                pl.when(pl.col("t") == t)
                .then(pl.col("V_new"))
                .otherwise(pl.col("V"))
                .alias("V")
            )
            .drop("V_new")
        )

    return df


def get_v0(n: int):
    aapl = get_AAPL_timeseries(dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    aapl_lr = get_log_returns(aapl)
    vol = get_vol(aapl)
    aapl_tree = get_binomial_tree(aapl_lr, n)
    aapl_tree = normalise(aapl_tree)
    aapl_tree = get_arithmetic_avg(aapl_tree)
    aapl_tree = get_payoff(aapl_tree)
    aapl_tree = backward_induction(aapl_tree, vol)
    V0 = aapl_tree.filter(pl.col("t") == 0).select(pl.col("V")).item()
    return V0, aapl_tree


def get_robustness_matrix(df: pl.DataFrame) -> pl.DataFrame:
    vol = get_vol(get_AAPL_timeseries(dt.date(2020, 1, 1), dt.date(2026, 4, 28)))
    irates = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.2]
    robustness_matrix = pl.DataFrame(
        {
            "Interest rates / Volatility": irates,
            "0.2": [
                get_q(df, 0.2, 0.0),
                get_q(df, 0.2, 0.0025),
                get_q(df, 0.2, 0.005),
                get_q(df, 0.2, 0.0075),
                get_q(df, 0.2, 0.01),
                get_q(df, 0.2, 0.0125),
                get_q(df, 0.2, 0.015),
                get_q(df, 0.2, 0.0175),
                get_q(df, 0.2, 0.02),
            ],
            "0.25": [
                get_q(df, 0.25, 0.0),
                get_q(df, 0.25, 0.0025),
                get_q(df, 0.25, 0.005),
                get_q(df, 0.25, 0.0075),
                get_q(df, 0.25, 0.01),
                get_q(df, 0.25, 0.0125),
                get_q(df, 0.25, 0.015),
                get_q(df, 0.25, 0.0175),
                get_q(df, 0.25, 0.02),
            ],
            "0.3": [
                get_q(df, 0.3, 0.0),
                get_q(df, 0.3, 0.0025),
                get_q(df, 0.3, 0.005),
                get_q(df, 0.3, 0.0075),
                get_q(df, 0.3, 0.01),
                get_q(df, 0.3, 0.0125),
                get_q(df, 0.3, 0.015),
                get_q(df, 0.3, 0.0175),
                get_q(df, 0.3, 0.02),
            ],
            "0.3130 (actual)": [
                get_q(df, vol, 0.0),
                get_q(df, vol, 0.0025),
                get_q(df, vol, 0.005),
                get_q(df, vol, 0.0075),
                get_q(df, vol, 0.01),
                get_q(df, vol, 0.0125),
                get_q(df, vol, 0.015),
                get_q(df, vol, 0.0175),
                get_q(df, vol, 0.02),
            ],
            "0.35": [
                get_q(df, 0.35, 0.0),
                get_q(df, 0.35, 0.0025),
                get_q(df, 0.35, 0.005),
                get_q(df, 0.35, 0.0075),
                get_q(df, 0.35, 0.01),
                get_q(df, 0.35, 0.0125),
                get_q(df, 0.35, 0.015),
                get_q(df, 0.35, 0.0175),
                get_q(df, 0.35, 0.02),
            ],
            "0.4": [
                get_q(df, 0.4, 0.0),
                get_q(df, 0.4, 0.0025),
                get_q(df, 0.4, 0.005),
                get_q(df, 0.4, 0.0075),
                get_q(df, 0.4, 0.01),
                get_q(df, 0.4, 0.0125),
                get_q(df, 0.4, 0.015),
                get_q(df, 0.4, 0.0175),
                get_q(df, 0.4, 0.02),
            ],
        }
    )
    return robustness_matrix


def different_n(stop: int) -> dict:
    values = {}
    for i in range(1, stop + 1, 1):
        v0, _ = get_v0(i)
        values[i] = v0
    return values
