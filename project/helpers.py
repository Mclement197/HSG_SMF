# This file contains all the helper functions for the repo.
"""
Helper functions for downloading AAPL price data and pricing a path-dependent
option with a binomial tree.

The file is organised around one main workflow:
1. Download historical AAPL closing prices.
2. Estimate annualised volatility from daily log returns.
3. Build a recombining-style binary path tree of possible future stock prices.
4. Compute an arithmetic-average based payoff at maturity.
5. Price that payoff by backward induction under risk-neutral probabilities.

Most functions return Polars DataFrames rather than pandas DataFrames. Polars
uses expression syntax such as `pl.col("Close")`, which means "refer to this
column inside a DataFrame transformation".
"""
import datetime as dt
from math import e, erf, pi, sqrt
import os

import polars as pl
import yfinance as yf

# Make Graphviz executables available to any code that needs graph rendering.
# This changes the process PATH only while this Python process is running.
os.environ["PATH"] = r"C:\Program Files\Graphviz\bin;" + os.environ["PATH"]

# Global model assumptions used throughout the helper functions.
# r is the default continuously compounded risk-free interest rate.
# T is the option maturity in years; here, half a year.
r = 0.01
T = 0.5


def get_AAPL_timeseries(
    ticker: str = "AAPL",
    start: dt.date = dt.date(2020, 1, 1),
    end: dt.date | None = None,
) -> pl.DataFrame:
    """
    Download historical closing prices for a ticker as a Polars DataFrame.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol to download. The default is "AAPL".
    start:
        First date to keep in the returned dataset.
    end:
        Last date to keep in the returned dataset. If omitted, the project uses
        the fixed date 2026-04-28 so results are reproducible.

    Returns
    -------
    pl.DataFrame
        A DataFrame with two columns: `Date` as a Python date and `Close` as a
        floating-point closing price.

    Notes
    -----
    `yfinance.download` treats the `end` argument as exclusive, so the code adds
    one day before downloading and then filters back to the requested inclusive
    date range.
    """
    # Use a fixed default end date instead of today's date to avoid results
    # changing simply because the script is run on a different day.
    requested_end = dt.date(2026, 4, 28) if end is None else end

    # Yahoo Finance excludes the end date from downloads, so add one day to make
    # the requested end date available in the raw data.
    download_end = requested_end + dt.timedelta(days=1)

    # `auto_adjust=False` keeps the raw closing price column. `progress=False`
    # avoids printing yfinance's progress bar when helpers are used in notebooks
    # or scripts.
    download = yf.download(
        ticker,
        start=start,
        end=download_end,
        auto_adjust=False,
        progress=False,
    )

    # Extract only the closing price series. Depending on the yfinance version
    # and number of tickers, this may be a Series or a one-column DataFrame.
    close = download["Close"]

    # If `close` is a DataFrame, choose the requested ticker column when present;
    # otherwise fall back to the first available column.
    if hasattr(close, "columns"):
        close = close[ticker] if ticker in close.columns else close.iloc[:, 0]

    # Convert the pandas/yfinance output into Polars and standardise the column
    # names and types used by the rest of the project.
    df = pl.from_pandas(close.rename("Close").reset_index())
    df = df.with_columns(pl.col("Date").cast(dt.date))
    df = df.with_columns(pl.col("Close").cast(pl.Float64))
    df = df.select(pl.col("Date"), pl.col("Close"))

    # Keep only the inclusive requested date range.
    return df.filter(pl.col("Date") >= start, pl.col("Date") <= requested_end)


def get_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add daily log returns computed from the `Close` column.

    Parameters
    ----------
    df:
        DataFrame containing at least a `Close` price column.

    Returns
    -------
    pl.DataFrame
        The input data with an added `log_returns` column. The first row is
        dropped because it has no previous price to compare against.

    Notes
    -----
    The log return is `log(Close_t / Close_{t-1})`. Log returns are commonly
    used in finance because they add cleanly over time and are convenient for
    volatility estimation.
    """
    # `shift(1)` moves yesterday's close onto today's row, allowing each row to
    # compare today's price with the previous trading day's price.
    return df.with_columns(
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_returns")
    ).drop_nulls()


def get_vol(df: pl.DataFrame) -> float:
    """
    Estimate annualised volatility from daily log returns.

    Parameters
    ----------
    df:
        DataFrame with either a `log_returns` column or a `Close` column from
        which log returns can be calculated.

    Returns
    -------
    float
        Annualised volatility, calculated as the sample standard deviation of
        daily log returns multiplied by `sqrt(250)`.

    Notes
    -----
    The factor `250**0.5` assumes approximately 250 trading days per year.
    `ddof=1` requests the sample standard deviation rather than the population
    standard deviation.
    """
    # If log returns are already available, avoid recomputing them.
    if "log_returns" in df.columns:
        return (
            df.select(pl.col("log_returns").std(ddof=1).alias("vol")).item() * 250**0.5
        )

    # Otherwise derive log returns from closing prices first.
    df = get_log_returns(df)
    return df.select(pl.col("log_returns").std(ddof=1).alias("vol")).item() * 250**0.5


def get_binomial_tree(aapl: pl.DataFrame, n: int, vol: float) -> pl.DataFrame:
    """
    Build a binary tree of possible future AAPL price paths.

    Parameters
    ----------
    aapl:
        Historical price DataFrame containing `Date` and `Close` columns.
    n:
        Number of time steps in the tree.
    vol:
        Annualised volatility used to calculate up and down movement factors.

    Returns
    -------
    pl.DataFrame
        A tree DataFrame with one row per node. Important columns include:
        `t` for time step, `path_id` for node identity, `S` for stock price,
        `C` for cumulative stock-price sum along the path, `move` for the last
        move, `history` for all moves so far, and `parent_id` for linking each
        node to its predecessor.

    Notes
    -----
    `C` is not an option price here. It stores the running sum of stock prices
    along a path, which is later divided by the number of observations to get an
    arithmetic average.
    """
    # Split the total maturity T into n equal time intervals.
    delta_T = T / n

    # Cox-Ross-Rubinstein style up/down multipliers. Higher volatility produces
    # wider spacing between the up and down branches.
    u = e ** (vol * (delta_T**0.5))
    d = e ** (-vol * (delta_T**0.5))

    # Use the most recent available closing price as the starting stock price.
    S0 = (
        aapl.filter(pl.col("Date") == aapl.select(pl.col("Date").max()).item())
        .select(pl.col("Close"))
        .item()
    )

    # Root node of the tree. At time 0 there is only one path, no previous move,
    # and the cumulative price sum C equals the starting price S0.
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

    # Store each layer separately so they can be concatenated at the end.
    layers = [current]

    for t in range(1, n + 1):
        # Create the next layer's up nodes from every node in the current layer.
        # Multiplying path_id by 2 gives each up child a deterministic identifier.
        up = current.select(
            pl.lit(t, dtype=pl.Int64).alias("t"),
            (pl.col("path_id") * 2).alias("path_id"),
            (pl.col("S") * u).alias("S"),
            (pl.col("C") + pl.col("S") * u).alias("C"),
            pl.lit("U").alias("move"),
            (pl.col("history") + pl.lit("U")).alias("history"),
        )

        # Create the next layer's down nodes. The path_id formula mirrors a
        # binary heap: parent p has children 2p and 2p + 1.
        down = current.select(
            pl.lit(t, dtype=pl.Int64).alias("t"),
            (pl.col("path_id") * 2 + 1).alias("path_id"),
            (pl.col("S") * d).alias("S"),
            (pl.col("C") + pl.col("S") * d).alias("C"),
            pl.lit("D").alias("move"),
            (pl.col("history") + pl.lit("D")).alias("history"),
        )

        # The next current layer consists of all possible up and down children.
        current = pl.concat([up, down])
        layers.append(current)

    # Combine all time layers into one DataFrame.
    tree = pl.concat(layers)

    # Recover each node's parent id from the binary-heap numbering scheme. The
    # root node has no parent, so its parent_id is set to null.
    tree = tree.with_columns(
        pl.when(pl.col("path_id") == 1)
        .then(None)
        .otherwise(pl.col("path_id") // 2)
        .alias("parent_id")
    )

    return tree


def get_q(df: pl.DataFrame, vol: float, r: float = 0.01, strict=False) -> float:
    """
    Calculate the risk-neutral probability of an up move.

    Parameters
    ----------
    df:
        Tree DataFrame containing a `t` column. The maximum `t` determines the
        number of time steps.
    vol:
        Annualised volatility used for up/down movement factors.
    r:
        Continuously compounded risk-free interest rate.
    strict:
        If True, raise an error when the calculated probability is outside the
        valid no-arbitrage range `(0, 1)`.

    Returns
    -------
    float
        Risk-neutral probability `q` assigned to an up movement.

    Notes
    -----
    Under risk-neutral pricing, the expected one-step stock growth should match
    the risk-free growth rate. This leads to
    `q = (exp(r * delta_t) - d) / (u - d)`.
    """
    # Infer the number of time steps from the tree itself.
    n = df.select(pl.col("t").max()).item()
    delta_t = T / n

    # Rebuild the same up and down multipliers used in the binomial tree.
    u = e ** (vol * delta_t**0.5)
    d = e ** (-vol * delta_t**0.5)

    # Solve for the probability that makes the expected stock growth equal to
    # risk-free growth over one time step.
    q = (e ** (r * delta_t) - d) / (u - d)

    # A q outside (0, 1) means the model inputs violate the standard no-arbitrage
    # condition for this binomial setup.
    if strict and not (0 < q < 1):
        raise ValueError(f"q={q} is outside (0,1). No-arbitrage condition fails.")

    return q


def normalise(df: pl.DataFrame, col: str = "C") -> pl.DataFrame:
    """
    Add a z-score normalised version of one column.

    Parameters
    ----------
    df:
        DataFrame containing the column to normalise.
    col:
        Column name to standardise. The default is `C`.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with a new column named `normalised_<col>`.

    Notes
    -----
    A z-score is `(value - mean) / standard deviation`. It expresses each value
    in units of standard deviations from the mean.
    """
    # Standard deviation and mean are computed over the whole column.
    std = df.select(pl.col(col)).std(ddof=1).item()
    mean = df.select(pl.col(col).mean()).item()

    # Add the normalised column while keeping all existing columns.
    return df.with_columns(((pl.col(col) - mean) / std).alias(f"normalised_{col}"))


def get_arithmetic_avg(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add the arithmetic average stock price along each path.

    Parameters
    ----------
    df:
        Tree DataFrame containing `C`, the cumulative sum of stock prices, and
        `t`, the current time step.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an added `S_bar` column.

    Notes
    -----
    At time step `t`, a path has `t + 1` observed prices because it includes the
    initial stock price at time 0. Therefore `S_bar = C / (t + 1)`.
    """
    # Divide the cumulative path sum by the number of observations on the path.
    return df.with_columns((pl.col("C") / (pl.col("t") + 1)).alias("S_bar"))


def get_payoff(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add the terminal payoff column for the path-dependent option.

    Parameters
    ----------
    df:
        Tree DataFrame. It should contain `S`, `t`, and either `S_bar` or enough
        information for `get_arithmetic_avg` to create `S_bar`.

    Returns
    -------
    pl.DataFrame
        The input tree with a `V` column. At terminal nodes, `V` is the payoff;
        before maturity, `V` is left null until backward induction fills it in.

    Notes
    -----
    The payoff is `max(S - S_bar, 0)` at maturity. This resembles an Asian-style
    payoff because it depends on the path average, not only the final stock
    price.
    """
    # The final time step is the option maturity.
    n = df.select(pl.col("t").max()).item()

    # Ensure the arithmetic average exists before using it in the payoff.
    if "S_bar" not in df.columns:
        df = get_arithmetic_avg(df)

    # Only terminal nodes receive a payoff now. Earlier nodes remain null and
    # are valued later by backward induction.
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
    """
    Price the tree by working backward from terminal payoffs to time 0.

    Parameters
    ----------
    df:
        Tree DataFrame containing terminal payoffs in column `V`, node ids in
        `path_id`, parent links in `parent_id`, and last moves in `move`.
    vol:
        Annualised volatility used if `q` needs to be calculated.
    q:
        Risk-neutral up probability. The sentinel value `-10` means "calculate
        q automatically from the tree, volatility, and interest rate".
    r:
        Continuously compounded risk-free interest rate used for discounting.

    Returns
    -------
    pl.DataFrame
        The same tree with `V` filled in for every node. The time-0 row contains
        the model price.

    Notes
    -----
    For each parent node, the value is the discounted expected value of its two
    children under risk-neutral probabilities:
    `V = exp(-r * dt) * (q * V_up + (1 - q) * V_down)`.
    """
    # Number of tree steps and length of each step in years.
    n = int(df.select(pl.col("t").max()).item())
    dt = T / n

    # Present-value discount factor for one tree step.
    discount = e ** (-r * dt)

    # If no q is supplied, calculate the risk-neutral probability from inputs.
    if q == -10:
        q = get_q(df, vol, r)

    # Move backward one layer at a time, starting just before maturity and ending
    # at the root. Each loop fills in values for all nodes at time t.
    for t in range(n - 1, -1, -1):
        # Select children at time t + 1 and pivot them so each parent row has one
        # up-child value and one down-child value.
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
            # Compute the discounted expected child value for each parent.
            .with_columns(
                (discount * (q * pl.col("V_up") + (1 - q) * pl.col("V_down"))).alias(
                    "V_new"
                )
            )
            .select(["path_id", "V_new"])
        )

        # Join the newly calculated parent values back to the full tree. Only the
        # current time layer is updated; other rows keep their existing V values.
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
    """
    Add the difference between final price and path average.

    Parameters
    ----------
    df:
        Tree DataFrame containing `S` and `S_bar` columns.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with a new `Dn` column equal to `S - S_bar`.

    Notes
    -----
    Unlike the payoff in `V`, `Dn` is not clipped at zero. It keeps negative
    values and is later used to estimate moments of the underlying difference.
    """
    # Dn measures how far the node's stock price is above or below its path
    # average.
    return df.with_columns((pl.col("S") - pl.col("S_bar")).alias("Dn"))


def get_v0(n: int, return_tree: bool = False):
    """
    Run the default pricing pipeline and return the time-0 option value.

    Parameters
    ----------
    n:
        Number of binomial tree steps.
    return_tree:
        If True, return both the price and the fully annotated tree.

    Returns
    -------
    float or tuple[float, pl.DataFrame]
        The option value at time 0. If `return_tree` is True, the second return
        value is the full tree used to calculate that price.

    Notes
    -----
    This function estimates volatility from downloaded AAPL data and uses the
    module's default interest rate through `backward_induction`.
    """
    # Download the fixed historical sample and estimate volatility from it.
    aapl = get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    vol = get_vol(aapl)

    # Build and enrich the tree step by step. Each helper adds columns needed by
    # later steps in the pricing workflow.
    aapl_tree = get_binomial_tree(aapl, n, vol)
    aapl_tree = normalise(aapl_tree)
    aapl_tree = get_arithmetic_avg(aapl_tree)
    aapl_tree = get_payoff(aapl_tree)
    aapl_tree = backward_induction(aapl_tree, vol)
    aapl_tree = get_dn(aapl_tree)

    # After backward induction, the root node contains the option price.
    V0 = aapl_tree.filter(pl.col("t") == 0).select(pl.col("V")).item()

    # Returning the tree is useful for diagnostics, plots, and later moment
    # calculations.
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
    """
    Price the option using supplied model parameters.

    Parameters
    ----------
    n:
        Number of binomial tree steps.
    q:
        Risk-neutral up probability to use directly in backward induction.
    vol:
        Annualised volatility used to build the tree.
    r:
        Continuously compounded risk-free interest rate used for discounting.
    return_tree:
        If True, return both the price and the full tree.

    Returns
    -------
    float or tuple[float, pl.DataFrame]
        The time-0 option value, optionally together with the priced tree.

    Notes
    -----
    This is similar to `get_v0`, but it does not estimate volatility or q inside
    the function. That makes it useful for robustness checks over different
    parameter values.
    """
    # Historical data is still used to obtain the latest starting stock price.
    aapl = get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))

    # Build the same pricing tree as `get_v0`, but use the caller's q, vol, and r.
    aapl_tree = get_binomial_tree(aapl, n, vol)
    aapl_tree = normalise(aapl_tree)
    aapl_tree = get_arithmetic_avg(aapl_tree)
    aapl_tree = get_payoff(aapl_tree)
    aapl_tree = backward_induction(aapl_tree, vol, q, r)
    aapl_tree = get_dn(aapl_tree)

    # Extract the root value after backward induction has filled the tree.
    V0 = aapl_tree.filter(pl.col("t") == 0).select(pl.col("V")).item()

    if return_tree:
        return V0, aapl_tree

    return V0


def get_robustness_matrix_q(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build a table showing how risk-neutral q changes across rates and volatilities.

    Parameters
    ----------
    df:
        Tree DataFrame used by `get_q` to infer the number of time steps.

    Returns
    -------
    pl.DataFrame
        Matrix-like DataFrame where rows are interest rates and columns are
        volatility assumptions. Each cell contains the corresponding q value.

    Notes
    -----
    The column named `actual` uses volatility estimated from the historical AAPL
    sample. Other columns use manually chosen volatility scenarios.
    """
    # Estimate the baseline volatility from the same fixed AAPL history used
    # elsewhere in the project.
    actual_vol = get_vol(
        get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    )

    # Scenario grids for interest rates and volatility assumptions.
    irates = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]
    vols = [0.2, 0.25, 0.3, actual_vol, 0.35, 0.4]

    # Start the output table with the row labels.
    data = {"Interest rates / Volatility": irates}

    # For each volatility scenario, compute q at every interest rate.
    for vol in vols:
        col_name = "actual" if vol == actual_vol else str(vol)
        data[col_name] = [get_q(df, vol, rate) for rate in irates]

    return pl.DataFrame(data)


def different_n(stop: int) -> dict:
    """
    Calculate option values for tree sizes from 1 up to `stop`.

    Parameters
    ----------
    stop:
        Largest number of tree steps to evaluate.

    Returns
    -------
    dict
        Mapping from each `n` value to the corresponding time-0 price.

    Notes
    -----
    This is useful for checking whether prices stabilise as the binomial tree
    becomes finer.
    """
    # Store results as {number_of_steps: option_value}.
    values = {}

    # Evaluate every tree size between 1 and stop, inclusive.
    for i in range(1, stop + 1):
        v0 = get_v0(i)
        values[i] = v0

    return values


def get_robustness_matrix_v0(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build a table showing how the option price changes across model assumptions.

    Parameters
    ----------
    df:
        Tree DataFrame used by `get_q` to infer the number of time steps when
        calculating q for each scenario.

    Returns
    -------
    pl.DataFrame
        Matrix-like DataFrame where rows are interest rates and columns are
        volatility assumptions. Each cell contains the option value from
        `get_v0_2`.

    Notes
    -----
    The pricing tree size is fixed at 25 steps inside this robustness check.
    """
    # Estimate the baseline volatility from historical AAPL data.
    actual_vol = get_vol(
        get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    )

    # Same scenario grid as `get_robustness_matrix_q`.
    irates = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]
    vols = [0.2, 0.25, 0.3, actual_vol, 0.35, 0.4]

    # Start the table with the interest-rate labels.
    data = {"Interest rates / Volatility": irates}

    # For each volatility, compute a price at each interest rate. q is
    # recalculated consistently for every rate/volatility pair.
    for vol in vols:
        col_name = "actual" if vol == actual_vol else str(vol)
        data[col_name] = [
            get_v0_2(25, get_q(df, vol, rate), vol, rate) for rate in irates
        ]

    return pl.DataFrame(data)


def normal_pdf(x: float = 1, mu: float = 0, sigma: float = 1) -> float:
    """
    Evaluate the normal probability density function.

    Parameters
    ----------
    x:
        Point at which to evaluate the density.
    mu:
        Mean of the normal distribution.
    sigma:
        Standard deviation of the normal distribution.

    Returns
    -------
    float
        Density value of `N(mu, sigma^2)` at `x`.
    """
    # First part of the normal density: 1 / (sigma * sqrt(2*pi)).
    val = (sigma * (2 * pi) ** 0.5) ** (-1)

    # Exponential part of the normal density.
    val = val * e ** (-((x - mu) ** 2) / (2 * sigma**2))
    return val


def normal_cdf(x: float) -> float:
    """
    Evaluate the standard normal cumulative distribution function.

    Parameters
    ----------
    x:
        Point at which to evaluate the cumulative probability.

    Returns
    -------
    float
        Probability that a standard normal random variable is less than or equal
        to `x`.

    Notes
    -----
    The formula uses the error function `erf`, which is available in Python's
    standard `math` module.
    """
    return 0.5 * (1 + erf(x / sqrt(2)))


def get_v0_with_Dn(mu: float, var: float, r_: float = 0.01) -> float:
    """
    Approximate the discounted value of the positive part of a normal variable.

    Parameters
    ----------
    mu:
        Mean of the `Dn` distribution.
    var:
        Variance of the `Dn` distribution.
    r_:
        Continuously compounded risk-free interest rate for discounting.

    Returns
    -------
    float
        Discounted approximation to `E[max(Dn, 0)]` when `Dn` is treated as
        normally distributed.

    Notes
    -----
    If `Dn ~ N(mu, sigma^2)`, then
    `E[max(Dn, 0)] = sigma * phi(mu / sigma) + mu * Phi(mu / sigma)`, where
    `phi` is the standard normal PDF and `Phi` is the standard normal CDF.
    """
    # Convert variance to standard deviation.
    sigma = var**0.5

    # Standardised mean used inside the normal PDF/CDF formula.
    z = mu / sigma

    # Discount the expected positive payoff back over maturity T.
    return e ** (-r_ * T) * (sigma * normal_pdf(z) + mu * normal_cdf(z))


def add_q_weights(df: pl.DataFrame, q: float) -> pl.DataFrame:
    """
    Add risk-neutral path probabilities based on each node's move history.

    Parameters
    ----------
    df:
        Tree DataFrame containing a `history` string column made from "U" and
        "D" characters.
    q:
        Risk-neutral probability of an up move.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with `n_up`, `n_down`, and `q_weight` columns.

    Notes
    -----
    For a path with `n_up` up moves and `n_down` down moves, its probability is
    `q**n_up * (1 - q)**n_down`.
    """
    # Count how many up and down moves are present in the path history string.
    return df.with_columns(
        pl.col("history").str.count_matches("U").alias("n_up"),
        pl.col("history").str.count_matches("D").alias("n_down"),
    # Convert the move counts into a risk-neutral probability weight.
    ).with_columns(
        ((pl.lit(q) ** pl.col("n_up")) * (pl.lit(1 - q) ** pl.col("n_down"))).alias(
            "q_weight"
        )
    )


def get_dn_moments(tree: pl.DataFrame, q: float) -> tuple[float, float]:
    """
    Calculate the risk-neutral mean and variance of terminal `Dn` values.

    Parameters
    ----------
    tree:
        Tree DataFrame containing `t`, `history`, and `Dn` columns.
    q:
        Risk-neutral probability of an up move.

    Returns
    -------
    tuple[float, float]
        Mean and variance of `Dn` at maturity under risk-neutral path weights.

    Notes
    -----
    Only terminal nodes are used because `Dn` is being treated as the terminal
    payoff driver.
    """
    # Identify maturity as the largest time index in the tree.
    n = tree.select(pl.col("t").max()).item()

    # Keep terminal nodes and attach each path's risk-neutral probability.
    terminal = tree.filter(pl.col("t") == n)
    terminal = add_q_weights(terminal, q)

    # Weighted mean of terminal Dn values.
    mu = terminal.select((pl.col("q_weight") * pl.col("Dn")).sum()).item()

    # Weighted variance around the weighted mean.
    var = terminal.select((pl.col("q_weight") * (pl.col("Dn") - mu) ** 2).sum()).item()

    return mu, var


def diff_BI_Dn() -> float:
    """
    Compare the backward-induction price with the normal-Dn approximation.

    Returns
    -------
    float
        `Dn_price - Bi`, where `Bi` is the backward-induction price and
        `Dn_price` is the normal approximation based on terminal `Dn` moments.
    """
    # Price with the full backward-induction tree and keep the tree for moment
    # calculations.
    Bi, tree = get_v0(25, return_tree=True)

    # Estimate volatility so q can be calculated consistently with the tree.
    vol = get_vol(
        get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    )

    # Risk-neutral probability used to weight terminal paths.
    q = get_q(tree, vol, r=0.01)

    # Estimate the normal approximation from terminal Dn moments.
    mu_Dn, var_Dn = get_dn_moments(tree, q)
    Dn_price = get_v0_with_Dn(mu_Dn, var_Dn)

    # Positive means the normal-Dn approximation is higher than backward
    # induction; negative means it is lower.
    return Dn_price - Bi


def relative_diff_Bi_Dn() -> float:
    """
    Calculate the relative pricing difference between two valuation methods.

    Returns
    -------
    float
        `(Dn_price - Bi) / Bi`, where `Bi` is the backward-induction price.
    """
    # Get the reference backward-induction price and the tree used to produce it.
    Bi, tree = get_v0(25, return_tree=True)

    # Estimate volatility and risk-neutral probability for terminal path weights.
    vol = get_vol(
        get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    )

    q = get_q(tree, vol, r=0.01)

    # Price with the normal approximation based on Dn moments.
    mu_Dn, var_Dn = get_dn_moments(tree, q)
    Dn_price = get_v0_with_Dn(mu_Dn, var_Dn)

    # Express the difference as a fraction of the backward-induction price.
    return (Dn_price - Bi) / Bi


# The code below runs immediately when this file is executed or imported. It
# calculates a 25-step tree and prints summary information about the terminal
# cumulative path sums C. It is mainly diagnostic/exploratory output.
v, tree = get_v0(25, True)
print(tree.filter(pl.col("t") == 25).select(pl.col("C")))

# Extract terminal cumulative sums as a Series so scalar summary statistics can
# be calculated directly.
c = tree.filter(pl.col("t") == 25).select(pl.col("C")).to_series()

# Mean and standard deviation are used to standardise the terminal C values.
mean = c.mean()
std = c.std()

# Standardized moments describe the shape of the terminal C distribution beyond
# its mean and variance.
skewness = ((c - mean) / std).pow(3).mean()
kurtosis = ((c - mean) / std).pow(4).mean()  # raw kurtosis (normal = 3)
excess_kurtosis = kurtosis - 3

# Print the distribution-shape statistics with fixed decimal formatting.
print(f"Skewness:         {skewness:.6f}")
print(f"Kurtosis:         {kurtosis:.6f}")
print(f"Excess kurtosis:  {excess_kurtosis:.6f}")
