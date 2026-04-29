# This files contains all the helper functions for the repo
import datetime as dt
from math import e
import os

from graphviz import Digraph
import matplotlib.pyplot as plt
import polars as pl
import yfinance as yf

os.environ["PATH"] = r"C:\Program Files\Graphviz\bin;" + os.environ["PATH"]

r = 0.01
T = 0.5


def get_AAPL_timeseries(
    start: dt.date = dt.date(2000, 1, 1), end: dt.date | None = None
) -> pl.DataFrame:
    """Get AAPL closing prices as a Polars DataFrame."""
    if end is None:
        end = dt.date.today()

    df = pl.from_pandas(yf.download("AAPL", start, end).reset_index())
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


def get_binomial_tree(n: int) -> pl.DataFrame:
    aapl = get_AAPL_timeseries(dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    aapl = get_log_returns(aapl)
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


def get_q(df: pl.DataFrame = get_binomial_tree(25)) -> float:
    n = df.select(pl.col("t").max()).item()
    delta_t = T / n
    vol = get_vol(get_AAPL_timeseries())
    u = e ** (vol * delta_t**0.5)
    d = e ** (-vol * delta_t**0.5)
    q = (e ** (r * delta_t) - d) / (u - d)  # IG we are in a continious framework ?
    return q


def normalise_terminal(df: pl.DataFrame, col: str = "C") -> pl.DataFrame:
    df = df.filter(pl.col("t") == df.select(pl.col("t").max()).item())
    std = df.select(pl.col(col)).std(ddof=1).item()
    mean = df.select(pl.col(col).mean()).item()
    df = df.with_columns((((pl.col(col)) - mean) / std).alias(f"normalised_{col}"))
    print(df)
    return df


def get_arithmetic_avg_terminal(df: pl.DataFrame) -> pl.DataFrame:
    n = df.select(pl.col("t").max()).item()

    return df.filter(pl.col("t") == n).with_columns(
        (pl.col("C") / (n + 1)).alias("S_bar")
    )


def get_terminal_payoff_asian_call(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(pl.col("t") == (df.select(pl.col("t").max()).item()))
    if "S_bar" in df.columns:
        df = df.with_columns(
            (pl.col("S") - pl.col("S_bar")).clip(lower_bound=0).alias("payoff")
        )
    else:
        df = get_arithmetic_avg_terminal(df)
        df = df.with_columns(
            (pl.col("S") - pl.col("S_bar")).clip(lower_bound=0).alias("payoff")
        )
    return df


def get_backward_induction(df: pl.DataFrame) -> pl.DataFrame:
    n = int(df.select(pl.col("t").max()).item())
    dt = T / n
    discount = e ** (-r * dt)
    q = get_q(get_binomial_tree(n))

    # terminal payoff
    df = df.with_columns(
        pl.when(pl.col("t") == n)
        .then((pl.col("S") - (pl.col("C") / (n + 1))).clip(lower_bound=0))
        .otherwise(None)
        .alias("V")
    )

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

## Graphing functions


def plot_binomial_tree(
    tree: pl.DataFrame,
    filename: str = "binomial_tree",
    show_average: bool = True,
    show_payoff: bool = True,
):
    dot = Digraph(format="png")
    dot.attr(rankdir="LR")

    n = int(tree.select(pl.col("t").max()).item())

    if filename == "binomial_tree":
        filename = f"binomial_tree_{n}n"

    # add S_bar and payoff if requested
    tree = tree.with_columns((pl.col("C") / (pl.col("t") + 1)).alias("S_bar"))

    tree = tree.with_columns(
        pl.when(pl.col("t") == n)
        .then((pl.col("S") - pl.col("S_bar")).clip(lower_bound=0))
        .otherwise(None)
        .alias("payoff")
    )

    rows = tree.to_dicts()

    for row in rows:
        node_id = str(row["path_id"])

        label = f"t={row['t']}\nS={row['S']:.2f}\nC={row['C']:.2f}"

        if show_average:
            label += f"\nAvg={row['S_bar']:.2f}"

        if show_payoff and row["payoff"] is not None:
            label += f"\nPayoff={row['payoff']:.2f}"

        if row["move"]:
            label += f"\n{row['move']}"

        dot.node(node_id, label)

        if row["parent_id"] is not None:
            dot.edge(str(row["parent_id"]), node_id)

    dot.render(filename, cleanup=True)


def plot_all_paths(
    tree: pl.DataFrame, value_col: str = "S", filename: str = "paths.png"
):
    if filename == "paths.png":
        filename = f"{value_col}_paths_{tree.select(pl.col('t').max()).item()}n.png"
    final_t = tree.select(pl.col("t").max()).item()

    final_paths = (
        tree.filter(pl.col("t") == final_t).select("history").to_series().to_list()
    )

    plt.figure()

    for path in final_paths:
        rows = []

        for k in range(len(path) + 1):
            prefix = path[:k]
            row = tree.filter(pl.col("history") == prefix)
            rows.append(row)

        path_df = pl.concat(rows).sort("t")

        plt.plot(path_df["t"].to_list(), path_df[value_col].to_list(), marker="o")

    plt.xlabel("Time")
    plt.ylabel(value_col)
    plt.title(f"Binomial Tree {value_col} Paths")

    plt.savefig(filename, dpi=300)
    plt.close()


def plot_terminal_distribution(
    tree: pl.DataFrame,
    value_col: str = "C",
    bins: int = 100,
):

    filename = f"{value_col}_terminal_distribution_{tree.select(pl.col('t').max()).item()}n.png"
    final_t = tree.select(pl.col("t").max()).item()

    terminal_values = (
        tree.filter(pl.col("t") == final_t).select(value_col).to_series().to_list()
    )

    plt.figure(figsize=(10, 6))
    plt.hist(terminal_values, bins=bins, density=True)

    plt.xlabel(f"Terminal {value_col}")
    plt.ylabel("Density")
    plt.title(f"Distribution of Terminal {value_col}")

    plt.savefig(filename, dpi=300)
    plt.close()


def plot_payoff_vs_terminal(tree: pl.DataFrame, filename="payoff_vs_ST.png"):
    n = tree.select(pl.col("t").max()).item()
    if filename == "payoff_vs_ST.png":
        filename = f"payoff_vs_ST_{n}n.png"

    df = tree.filter(pl.col("t") == n).with_columns(
        [
            (pl.col("C") / (n + 1)).alias("S_bar"),
            (pl.col("S") - pl.col("C") / (n + 1)).clip(lower_bound=0).alias("payoff"),
        ]
    )

    plt.figure()

    plt.scatter(df["S"].to_list(), df["payoff"].to_list(), alpha=0.5)

    plt.xlabel("Terminal Price $S_n$")
    plt.ylabel("Payoff")
    plt.title("Floating-strike Asian Call Payoff")

    plt.savefig(filename, dpi=300)
    plt.close()

def plot_option_value_by_time(
    priced_tree: pl.DataFrame,
    filename: str = "option_value_by_time.png",
):
    df = priced_tree.filter(pl.col("V").is_not_null())

    plt.figure(figsize=(10, 6))

    for t in sorted(df["t"].unique().to_list()):
        layer = df.filter(pl.col("t") == t)
        plt.scatter(
            layer["S"].to_list(),
            layer["V"].to_list(),
            alpha=0.6,
            label=f"t={t}"
        )

    plt.xlabel("Stock price S")
    plt.ylabel("Option value V")
    plt.title("Floating-strike Asian Call Value by Time Step")
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_backward_induction_tree(
    df: pl.DataFrame,
    filename: str = "backward_induction_tree",
):
    dot = Digraph(format="png")
    dot.attr(rankdir="LR")

    n = int(df.select(pl.col("t").max()).item())

    if filename == "backward_induction_tree":
        filename = f"backward_induction_tree_{n}n"

    df = df.with_columns(
        (pl.col("C") / (pl.col("t") + 1)).alias("S_bar")
    )

    rows = df.to_dicts()

    for row in rows:
        node_id = str(row["path_id"])

        label = (
            f"t={row['t']}\n"
            f"S={row['S']:.2f}\n"
            f"Avg={row['S_bar']:.2f}\n"
            f"V={row['V']:.2f}"
        )

        if row["t"] == n:
            label += f"\nPayoff={row['V']:.2f}"

        if row["move"]:
            label += f"\n{row['move']}"

        dot.node(node_id, label)

        if row["parent_id"] is not None:
            dot.edge(str(row["parent_id"]), node_id)

    dot.render(filename, cleanup=True)

ans = get_binomial_tree(3)
priced_tree = get_backward_induction(ans)
plot_option_value_by_time(priced_tree)


price = priced_tree.filter(pl.col("t") == 0).select("V").item()
plot_backward_induction_tree(priced_tree)
print(price)

#13.326957086019487