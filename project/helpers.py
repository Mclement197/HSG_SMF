# This files contains all the helper functions for the repo
import datetime as dt
from math import e, erf, pi, sqrt
import os

import polars as pl
import yfinance as yf

os.environ["PATH"] = r"C:\Program Files\Graphviz\bin;" + os.environ["PATH"]

r = 0.01
T = 0.5


def get_AAPL_timeseries(
    ticker: str = "AAPL",
    start: dt.date = dt.date(2020, 1, 1),
    end: dt.date | None = None,
) -> pl.DataFrame:
    """Get AAPL closing prices as a Polars DataFrame."""
    requested_end = dt.date(2026, 4, 28) if end is None else end
    download_end = requested_end + dt.timedelta(days=1)

    download = yf.download(
        ticker,
        start=start,
        end=download_end,
        auto_adjust=False,
        progress=False,
    )
    close = download["Close"]

    if hasattr(close, "columns"):
        close = close[ticker] if ticker in close.columns else close.iloc[:, 0]

    df = pl.from_pandas(close.rename("Close").reset_index())
    df = df.with_columns(pl.col("Date").cast(dt.date))
    df = df.with_columns(pl.col("Close").cast(pl.Float64))
    df = df.select(pl.col("Date"), pl.col("Close"))
    return df.filter(pl.col("Date") >= start, pl.col("Date") <= requested_end)


def get_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_returns")
    ).drop_nulls()


def get_vol(df: pl.DataFrame) -> float:
    if "log_returns" in df.columns:
        return (
            df.select(pl.col("log_returns").std(ddof=1).alias("vol")).item() * 250**0.5
        )

    df = get_log_returns(df)
    return df.select(pl.col("log_returns").std(ddof=1).alias("vol")).item() * 250**0.5


def get_binomial_tree(aapl: pl.DataFrame, n: int, vol: float) -> pl.DataFrame:
    delta_T = T / n
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
            (pl.col("path_id") * 2 + 1).alias("path_id"),
            (pl.col("S") * d).alias("S"),
            (pl.col("C") + pl.col("S") * d).alias("C"),
            pl.lit("D").alias("move"),
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

    if strict and not (0 < q < 1):
        raise ValueError(f"q={q} is outside (0,1). No-arbitrage condition fails.")

    return q


def normalise(df: pl.DataFrame, col: str = "C") -> pl.DataFrame:
    std = df.select(pl.col(col)).std(ddof=1).item()
    mean = df.select(pl.col(col).mean()).item()

    return df.with_columns(((pl.col(col) - mean) / std).alias(f"normalised_{col}"))


def get_arithmetic_avg(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns((pl.col("C") / (pl.col("t") + 1)).alias("S_bar"))


def get_payoff(df: pl.DataFrame) -> pl.DataFrame:
    n = df.select(pl.col("t").max()).item()

    if "S_bar" not in df.columns:
        df = get_arithmetic_avg(df)

    return df.with_columns(
        pl.when(pl.col("t") == n)
        .then((pl.col("S") - pl.col("S_bar")).clip(lower_bound=0))
        .otherwise(None)
        .alias("V")
    )


def backward_induction(
    df: pl.DataFrame,
    vol: float,
    q: float = -10,
    r: float = 0.01,
) -> pl.DataFrame:
    n = int(df.select(pl.col("t").max()).item())
    dt = T / n
    discount = e ** (-r * dt)

    if q == -10:
        q = get_q(df, vol, r)

    for t in range(n - 1, -1, -1):
        children = (
            df.filter(pl.col("t") == t + 1)
            .select(["parent_id", "move", "V"])
            .pivot(values="V", index="parent_id", on="move")
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


def get_dn(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns((pl.col("S") - pl.col("S_bar")).alias("Dn"))


def get_v0(n: int, return_tree: bool = False):
    aapl = get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    vol = get_vol(aapl)

    aapl_tree = get_binomial_tree(aapl, n, vol)
    aapl_tree = normalise(aapl_tree)
    aapl_tree = get_arithmetic_avg(aapl_tree)
    aapl_tree = get_payoff(aapl_tree)
    aapl_tree = backward_induction(aapl_tree, vol)
    aapl_tree = get_dn(aapl_tree)

    V0 = aapl_tree.filter(pl.col("t") == 0).select(pl.col("V")).item()

    if return_tree:
        return V0, aapl_tree

    return V0


def get_v0_2(
    n: int,
    q: float,
    vol: float,
    r: float,
    return_tree: bool = False,
):
    aapl = get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))

    aapl_tree = get_binomial_tree(aapl, n, vol)
    aapl_tree = normalise(aapl_tree)
    aapl_tree = get_arithmetic_avg(aapl_tree)
    aapl_tree = get_payoff(aapl_tree)
    aapl_tree = backward_induction(aapl_tree, vol, q, r)
    aapl_tree = get_dn(aapl_tree)

    V0 = aapl_tree.filter(pl.col("t") == 0).select(pl.col("V")).item()

    if return_tree:
        return V0, aapl_tree

    return V0


def get_robustness_matrix_q(df: pl.DataFrame) -> pl.DataFrame:
    actual_vol = get_vol(
        get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    )

    irates = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]
    vols = [0.2, 0.25, 0.3, actual_vol, 0.35, 0.4]

    data = {"Interest rates / Volatility": irates}

    for vol in vols:
        col_name = "actual" if vol == actual_vol else str(vol)
        data[col_name] = [get_q(df, vol, rate) for rate in irates]

    return pl.DataFrame(data)


def different_n(stop: int) -> dict:
    values = {}

    for i in range(1, stop + 1):
        v0 = get_v0(i)
        values[i] = v0

    return values


def get_robustness_matrix_v0(df: pl.DataFrame) -> pl.DataFrame:
    actual_vol = get_vol(
        get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    )

    irates = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]
    vols = [0.2, 0.25, 0.3, actual_vol, 0.35, 0.4]

    data = {"Interest rates / Volatility": irates}

    for vol in vols:
        col_name = "actual" if vol == actual_vol else str(vol)
        data[col_name] = [
            get_v0_2(25, get_q(df, vol, rate), vol, rate) for rate in irates
        ]

    return pl.DataFrame(data)


def normal_pdf(x: float = 1, mu: float = 0, sigma: float = 1) -> float:
    val = (sigma * (2 * pi) ** 0.5) ** (-1)
    val = val * e ** (-((x - mu) ** 2) / (2 * sigma**2))
    return val


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def get_v0_with_Dn(mu: float, var: float, r_: float = 0.01) -> float:
    sigma = var**0.5
    z = mu / sigma

    return e ** (-r_ * T) * (sigma * normal_pdf(z) + mu * normal_cdf(z))


def add_q_weights(df: pl.DataFrame, q: float) -> pl.DataFrame:
    return df.with_columns(
        pl.col("history").str.count_matches("U").alias("n_up"),
        pl.col("history").str.count_matches("D").alias("n_down"),
    ).with_columns(
        ((pl.lit(q) ** pl.col("n_up")) * (pl.lit(1 - q) ** pl.col("n_down"))).alias(
            "q_weight"
        )
    )


def get_dn_moments(tree: pl.DataFrame, q: float) -> tuple[float, float]:
    n = tree.select(pl.col("t").max()).item()

    terminal = tree.filter(pl.col("t") == n)
    terminal = add_q_weights(terminal, q)

    mu = terminal.select((pl.col("q_weight") * pl.col("Dn")).sum()).item()

    var = terminal.select((pl.col("q_weight") * (pl.col("Dn") - mu) ** 2).sum()).item()

    return mu, var


def diff_BI_Dn() -> float:
    Bi, tree = get_v0(25, return_tree=True)

    vol = get_vol(
        get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    )

    q = get_q(tree, vol, r=0.01)

    mu_Dn, var_Dn = get_dn_moments(tree, q)
    Dn_price = get_v0_with_Dn(mu_Dn, var_Dn)

    return Dn_price - Bi


def relative_diff_Bi_Dn() -> float:
    Bi, tree = get_v0(25, return_tree=True)

    vol = get_vol(
        get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    )

    q = get_q(tree, vol, r=0.01)

    mu_Dn, var_Dn = get_dn_moments(tree, q)
    Dn_price = get_v0_with_Dn(mu_Dn, var_Dn)

    return (Dn_price - Bi) / Bi

v, tree = get_v0(25, True)
print(tree.filter(pl.col("t") == 25).select(pl.col("C")))

c = tree.filter(pl.col("t") == 25).select(pl.col("C")).to_series()

mean = c.mean()
std = c.std()

# Standardized moments
skewness = ((c - mean) / std).pow(3).mean()
kurtosis = ((c - mean) / std).pow(4).mean()  # raw kurtosis (normal = 3)
excess_kurtosis = kurtosis - 3

print(f"Skewness:         {skewness:.6f}")
print(f"Kurtosis:         {kurtosis:.6f}")
print(f"Excess kurtosis:  {excess_kurtosis:.6f}")