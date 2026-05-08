from pathlib import Path

from graphviz import Digraph
from helpers import different_n
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl


def plot_binomial_tree(
    tree: pl.DataFrame,
    n: int,
    show_average: bool = True,
    show_payoff: bool = True,
):
    """
    Plot a readable binomial tree up to time step n.

    Logic:
    - If show_payoff=True:
        terminal nodes are colored by payoff.
        payoff > 0  -> green
        payoff = 0  -> orange outline

    - If show_payoff=False:
        every node is colored by the move that produced it.
        U -> green
        D -> orange
    """

    # ── Get tree
    tree = tree.filter(pl.col("t") <= n)

    # ── Shared visual style
    BG = "#0f1117"
    PANEL = "#161b22"
    GRID = "#30363d"
    ACCENT = "#58a6ff"
    ORANGE = "#f59e0b"
    GREEN = "#22c55e"
    WHITE = "#f0f6fc"
    BLACK = "#000000"

    # ── Basic checks
    required_cols = {"t", "path_id", "S", "C", "move", "parent_id"}
    missing = required_cols - set(tree.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Actual plotted depth
    n_plot = int(tree.select(pl.col("t").max()).item())

    filename = f"binomial_tree_{n_plot}n"

    if show_average:
        filename += "_avg"

    if show_payoff:
        filename += "_payoff"
    else:
        filename += "_moves"

    output_dir = Path("graphs")
    output_dir.mkdir(exist_ok=True)

    # ── Add average only if needed
    if show_average or show_payoff:
        tree = tree.with_columns((pl.col("C") / (pl.col("t") + 1)).alias("S_bar"))

    # ── Add payoff only if requested
    if show_payoff:
        tree = tree.with_columns(
            pl.when(pl.col("t") == n_plot)
            .then((pl.col("S") - pl.col("S_bar")).clip(lower_bound=0))
            .otherwise(None)
            .alias("payoff")
        )

    rows = tree.to_dicts()

    # ── Graph setup
    dot = Digraph(format="png")

    dot.attr(
        rankdir="LR",
        bgcolor=BG,
        splines="line",
        nodesep="0.55",
        ranksep="0.95",
        pad="0.35",
        dpi="180",
    )

    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor=PANEL,
        color=GRID,
        fontcolor=WHITE,
        fontname="Helvetica",
        fontsize="11",
        penwidth="1.4",
        margin="0.22,0.16",
        width="1.65",
        height="0.75",
        fixedsize="false",
    )

    dot.attr(
        "edge",
        color=GRID,
        fontcolor=WHITE,
        arrowsize="0.7",
        penwidth="1.1",
    )

    def fmt(x: float) -> str:
        return f"{x:.2f}"

    # ── Add nodes and edges
    for row in rows:
        node_id = str(row["path_id"])

        label_lines = [
            f"t = {row['t']}",
            f"S = {fmt(row['S'])}",
            f"C = {fmt(row['C'])}",
        ]

        if show_average:
            label_lines.append(f"Avg = {fmt(row['S_bar'])}")

        if show_payoff and row["payoff"] is not None:
            label_lines.append(f"Payoff = {fmt(row['payoff'])}")

        if row["move"]:
            label_lines.append(f"Move = {row['move']}")

        # Left-aligned Graphviz label
        label = r"\l".join(label_lines) + r"\l"

        # ====================================================
        # COLOR LOGIC LINKED TO show_payoff
        # ====================================================

        if show_payoff:
            # Payoff mode: only terminal nodes are payoff-colored
            if row["t"] == n_plot:
                payoff = row["payoff"]

                if payoff is not None and payoff > 0:
                    fillcolor = GREEN
                    fontcolor = BLACK
                    bordercolor = GREEN
                else:
                    fillcolor = PANEL
                    fontcolor = WHITE
                    bordercolor = ORANGE
            else:
                fillcolor = PANEL
                fontcolor = WHITE
                bordercolor = ACCENT if row["t"] == 0 else GRID

        else:
            # Move mode: every node is colored by the move that led to it
            if row["t"] == 0:
                fillcolor = PANEL
                fontcolor = WHITE
                bordercolor = ACCENT

            elif row["move"] == "U":
                fillcolor = GREEN
                fontcolor = BLACK
                bordercolor = GREEN

            elif row["move"] == "D":
                fillcolor = ORANGE
                fontcolor = BLACK
                bordercolor = ORANGE

            else:
                fillcolor = PANEL
                fontcolor = WHITE
                bordercolor = GRID

        dot.node(
            node_id,
            label=label,
            fillcolor=fillcolor,
            fontcolor=fontcolor,
            color=bordercolor,
        )

        # ── Edge coloring
        if row["parent_id"] is not None:
            if show_payoff:
                edge_color = ACCENT if row["move"] == "U" else GRID
            else:
                edge_color = GREEN if row["move"] == "U" else ORANGE

            dot.edge(
                str(row["parent_id"]),
                node_id,
                color=edge_color,
            )

    output_path = dot.render(
        filename=filename,
        directory=str(output_dir),
        cleanup=True,
    )

    return output_path


def plot_all_paths(
    tree: pl.DataFrame,
    n: int,
    value_col: str = "S",
):
    """
    Plot all terminal paths of a binomial tree using the dark graph style.

    Parameters
    ----------
    tree:
        Binomial tree DataFrame.

    value_col:
        Column to plot on the y-axis, for example "S", "C", or "S_bar".

    filename:
        Output filename. Saved inside the graphs/ folder by default.
    """

    # ── Shared visual style
    BG = "#0f1117"
    PANEL = "#161b22"
    GRID = "#30363d"
    ACCENT = "#58a6ff"
    ORANGE = "#f59e0b"
    WHITE = "#f0f6fc"

    tree = tree.filter(pl.col("t") <= n)

    # ── Basic checks
    required_cols = {"t", "history", value_col}
    missing = required_cols - set(tree.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    final_t = int(tree.select(pl.col("t").max()).item())

    filename = f"{value_col}_paths_{final_t}n.png"

    output_dir = Path("graphs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / filename

    final_paths = (
        tree.filter(pl.col("t") == final_t).select("history").to_series().to_list()
    )

    # ── Figure setup
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    # ── Plot paths
    for path in final_paths:
        rows = []

        for k in range(len(path) + 1):
            prefix = path[:k]
            row = tree.filter(pl.col("history") == prefix)
            rows.append(row)

        path_df = pl.concat(rows).sort("t")

        ax.plot(
            path_df["t"].to_list(),
            path_df[value_col].to_list(),
            color=ACCENT,
            alpha=0.28,
            linewidth=0.9,
            marker="o",
            markersize=2.5,
            markeredgewidth=0,
        )

    # ── Baseline from initial value
    initial_value = tree.filter(pl.col("t") == 0).select(value_col).to_series().item()

    ax.axhline(
        initial_value,
        color=ORANGE,
        linewidth=1.2,
        linestyle="--",
        label=f"{value_col}₀ = {initial_value:.2f}",
    )

    # ── Labels and title
    ax.set_title(
        f"Binomial Tree {value_col} Paths ({len(final_paths)} paths)",
        fontweight="bold",
        color=WHITE,
    )

    ax.set_xlabel("Time step", color=WHITE)
    ax.set_ylabel(value_col, color=WHITE)

    # ── Styling
    ax.grid(color=GRID, alpha=0.35, linewidth=0.6)

    ax.tick_params(colors=WHITE)

    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.legend(
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=WHITE,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)

    return str(output_path)


def plot_terminal_distribution(
    tree: pl.DataFrame,
    n: int,
    value_col: str = "C",
    bins: int = 100,
    exclude_zero: bool = False,
):
    """
    Plot the terminal distribution of a chosen column of the binomial tree
    using the shared dark visual style.
    """

    # ── Shared visual style
    BG = "#0f1117"
    PANEL = "#161b22"
    GRID = "#30363d"
    ACCENT = "#58a6ff"
    ORANGE = "#f59e0b"
    GREEN = "#22c55e"
    WHITE = "#f0f6fc"

    # ── Filter tree up to step n
    tree = tree.filter(pl.col("t") <= n)

    final_t = int(tree.select(pl.col("t").max()).item())
    filename = f"{value_col}_terminal_distribution_{final_t}n.png"

    output_dir = Path("graphs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename

    if exclude_zero:
        tree = tree.filter(pl.col("V") != 0)

    # ── Extract terminal values
    terminal_values = (
        tree.filter(pl.col("t") == final_t).select(value_col).to_series().to_list()
    )

    mean_val = sum(terminal_values) / len(terminal_values)

    # ── Figure setup
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    # ── Histogram
    ax.hist(
        terminal_values,
        bins=bins,
        density=True,
        color=ACCENT,
        alpha=0.85,
        edgecolor="none",
    )

    # ── Reference lines
    ax.axvline(
        mean_val,
        color=ORANGE,
        linewidth=1.5,
        linestyle="--",
        label=f"Mean = {mean_val:.2f}",
    )

    ax.axvline(
        min(terminal_values),
        color=GREEN,
        linewidth=1.0,
        linestyle="--",
        label=f"Min = {min(terminal_values):.2f}",
    )

    # ── Labels and title
    ax.set_title(
        f"Distribution of Terminal {value_col} (t = {final_t})",
        fontweight="bold",
        color=WHITE,
    )
    ax.set_xlabel(f"Terminal {value_col}", color=WHITE)
    ax.set_ylabel("Density", color=WHITE)

    # ── Axis styling
    ax.grid(color=GRID, alpha=0.35, linewidth=0.6)
    ax.tick_params(colors=WHITE)

    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.legend(
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=WHITE,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)

    return str(output_path)


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
    tree: pl.DataFrame,
    n: int,
):
    """
    Scatter plot of option value V against stock price S,
    separated by time step, using the shared dark visual style.

    Only the orange and green reference lines are shown in the legend.
    """

    # ── Shared visual style
    BG = "#0f1117"
    PANEL = "#161b22"
    GRID = "#30363d"
    ACCENT = "#58a6ff"
    ORANGE = "#f59e0b"
    GREEN = "#22c55e"
    WHITE = "#f0f6fc"

    # ── Filter correctly
    df = tree.filter(pl.col("V").is_not_null() & (pl.col("t") <= n))

    if df.is_empty():
        raise ValueError(
            "No non-null option values V found up to the requested time step."
        )

    n_plot = int(df.select(pl.col("t").max()).item())

    filename = f"value_by_time_{n_plot}n.png"
    output_dir = Path("graphs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename

    # ── Figure setup
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    times = sorted(df["t"].unique().to_list())
    max_t = max(times) if times else 1

    # ── Scatter per layer
    for t in times:
        layer = df.filter(pl.col("t") == t)

        alpha = 0.25 + 0.60 * (t / max_t if max_t > 0 else 1.0)

        ax.scatter(
            layer["S"].to_list(),
            layer["V"].to_list(),
            color=ACCENT,
            alpha=alpha,
            s=28,
            edgecolors="none",
        )

    # ── Reference lines
    ax.axhline(
        0,
        color=ORANGE,
        linewidth=1.2,
        linestyle="--",
        label="V = 0",
    )

    final_layer = df.filter(pl.col("t") == n_plot)
    mean_final_v = final_layer["V"].mean()

    ax.axhline(
        mean_final_v,
        color=GREEN,
        linewidth=1.0,
        linestyle="--",
        label=f"Mean V at t={n_plot} = {mean_final_v:.2f}",
    )

    # ── Labels and title
    ax.set_title(
        "Floating-Strike Asian Call Value by Time Step",
        fontweight="bold",
        color=WHITE,
    )
    ax.set_xlabel("Stock price S", color=WHITE)
    ax.set_ylabel("Option value V", color=WHITE)

    # ── Styling
    ax.grid(color=GRID, alpha=0.35, linewidth=0.6)
    ax.tick_params(colors=WHITE)

    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.legend(
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=WHITE,
        fontsize=9,
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(
        output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight"
    )
    plt.close(fig)

    return str(output_path)


def plot_backward_induction_tree(
    df: pl.DataFrame,
    n: int,
):
    """
    Plot a backward-induction tree up to time step n.

    Assumes df already contains:
    - S_bar
    - V

    Visual logic:
    - terminal node with V > 0: full green
    - terminal node with V = 0: orange outline
    - root node: blue outline
    - intermediate nodes: grey outline

    Backward-induction logic:
    - arrows go from child -> parent
    - up-child contribution arrows are green
    - down-child contribution arrows are orange
    """

    # ── Filter first
    df = df.filter(pl.col("t") <= n)

    if df.is_empty():
        raise ValueError(f"No rows found for t <= {n}.")

    # ── Shared visual style
    BG = "#0f1117"
    PANEL = "#161b22"
    GRID = "#30363d"
    ACCENT = "#58a6ff"
    ORANGE = "#f59e0b"
    GREEN = "#22c55e"
    WHITE = "#f0f6fc"
    BLACK = "#000000"

    # ── Basic checks
    required_cols = {
        "t",
        "path_id",
        "S",
        "S_bar",
        "V",
        "move",
        "parent_id",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Actual plotted depth
    n_plot = int(df.select(pl.col("t").max()).item())

    filename = f"backward_induction_tree_{n_plot}n"

    output_dir = Path("graphs")
    output_dir.mkdir(exist_ok=True)

    rows = df.sort(["t", "path_id"]).to_dicts()

    # ── Graph setup
    dot = Digraph(format="png")

    dot.attr(
        rankdir="LR",
        bgcolor=BG,
        splines="line",
        nodesep="0.55",
        ranksep="0.95",
        pad="0.35",
        dpi="180",
    )

    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor=PANEL,
        color=GRID,
        fontcolor=WHITE,
        fontname="Helvetica",
        fontsize="11",
        penwidth="1.4",
        margin="0.22,0.16",
        width="1.75",
        height="0.75",
        fixedsize="false",
    )

    dot.attr(
        "edge",
        color=GRID,
        fontcolor=WHITE,
        arrowsize="0.8",
        penwidth="1.2",
    )

    def fmt(x: float | None) -> str:
        if x is None:
            return "NA"
        return f"{x:.2f}"

    # ── Add nodes and edges
    for row in rows:
        node_id = str(row["path_id"])

        label_lines = [
            f"t = {row['t']}",
            f"S = {fmt(row['S'])}",
            f"Avg = {fmt(row['S_bar'])}",
            f"V = {fmt(row['V'])}",
        ]

        if row["t"] == n_plot:
            label_lines.append(f"Payoff = {fmt(row['V'])}")

        if row["move"]:
            label_lines.append(f"Move = {row['move']}")

        label = r"\l".join(label_lines) + r"\l"

        # ── Node coloring
        if row["t"] == n_plot:
            # Terminal layer = starting point of backward induction
            payoff = row["V"]

            if payoff is not None and payoff > 0:
                fillcolor = GREEN
                fontcolor = BLACK
                bordercolor = GREEN
            else:
                fillcolor = PANEL
                fontcolor = WHITE
                bordercolor = ORANGE

        elif row["t"] == 0:
            # Root = final backward-induction result
            fillcolor = PANEL
            fontcolor = WHITE
            bordercolor = ACCENT

        else:
            # Intermediate continuation-value nodes
            fillcolor = PANEL
            fontcolor = WHITE
            bordercolor = GRID

        dot.node(
            node_id,
            label=label,
            fillcolor=fillcolor,
            fontcolor=fontcolor,
            color=bordercolor,
        )

        # ── Backward-induction arrows: child -> parent
        if row["parent_id"] is not None:
            if row["move"] == "U":
                edge_color = GREEN
            elif row["move"] == "D":
                edge_color = ORANGE
            else:
                edge_color = GRID

            dot.edge(
                node_id,  # child
                str(row["parent_id"]),  # parent
                color=edge_color,
            )

    output_path = dot.render(
        filename=filename,
        directory=str(output_dir),
        cleanup=True,
    )

    return output_path


def plot_aapl_timeseries(
    df: pl.DataFrame,
    date_col: str = "Date",
    price_col: str = "Close",
    show_mean: bool = True,
    show_last: bool = True,
    output_dir: str = "graphs",
    filename: str = "aapl_timeseries.png",
) -> str:
    """
    Plot AAPL closing price time series using the same dark theme
    as the binomial tree graph.

    Required columns:
    - Date
    - Close
    """

    # ── Shared visual style
    BG = "#0f1117"
    PANEL = "#161b22"
    GRID = "#30363d"
    ACCENT = "#58a6ff"
    ORANGE = "#f59e0b"
    GREEN = "#22c55e"
    WHITE = "#f0f6fc"

    # ── Basic checks
    required_cols = {date_col, price_col}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Clean and sort data
    plot_df = df.select(date_col, price_col).drop_nulls().sort(date_col)

    if plot_df.height == 0:
        raise ValueError("The dataframe is empty after removing null values.")

    dates = plot_df[date_col].to_list()
    prices = plot_df[price_col].to_numpy()

    first_date = dates[0]
    last_date = dates[-1]
    last_price = prices[-1]
    mean_price = prices.mean()

    # ── Output path
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    full_path = output_path / filename

    # ── Plot
    _fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax.set_facecolor(PANEL)

    ax.plot(
        dates,
        prices,
        color=ACCENT,
        linewidth=2.2,
        label="AAPL close",
    )

    if show_mean:
        ax.axhline(
            mean_price,
            color=ORANGE,
            linestyle="--",
            linewidth=1.6,
            label=f"Mean = {mean_price:.2f}",
        )

    if show_last:
        ax.axhline(
            last_price,
            color=GREEN,
            linestyle="--",
            linewidth=1.4,
            label=f"Last = {last_price:.2f}",
        )

        ax.scatter(
            last_date,
            last_price,
            color=GREEN,
            s=55,
            zorder=5,
        )

    # ── Labels and title
    ax.set_title(
        f"AAPL Closing Price Time Series\n{first_date} to {last_date}",
        color=WHITE,
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel("Date", color=WHITE, fontsize=11)
    ax.set_ylabel("Price", color=WHITE, fontsize=11)

    # ── Axis formatting
    ax.tick_params(axis="x", colors=WHITE)
    ax.tick_params(axis="y", colors=WHITE)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # ── Grid and borders
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.65)

    for spine in ax.spines.values():
        spine.set_color(GRID)

    # ── Legend
    legend = ax.legend(
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=WHITE,
        framealpha=1,
    )

    for text in legend.get_texts():
        text.set_color(WHITE)

    plt.tight_layout()
    plt.savefig(full_path, dpi=180, facecolor=BG)
    plt.close()

    return str(full_path)


def plot_values_by_n(
    values_by_n: dict[int, float],
    output_dir: str = "graphs",
    filename: str = "values_by_n.png",
    title: str = "Option Value by Number of Steps",
    y_label: str = "Value",
) -> str:
    """
    Plot values indexed by n.

    Example input:
    values_by_n = {
        1: 15.0,
        2: 12.4,
        3: 10.8,
    }
    """

    if not values_by_n:
        raise ValueError("values_by_n is empty.")

    # ── Shared visual style
    BG = "#0f1117"
    PANEL = "#161b22"
    GRID = "#30363d"
    ACCENT = "#58a6ff"
    ORANGE = "#f59e0b"
    GREEN = "#22c55e"
    WHITE = "#f0f6fc"

    # ── Sort by n
    sorted_items = sorted(values_by_n.items())
    n_values = [item[0] for item in sorted_items]
    y_values = [item[1] for item in sorted_items]

    # ── Output path
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    full_path = output_path / filename

    # ── Plot
    _fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax.set_facecolor(PANEL)

    ax.plot(
        n_values,
        y_values,
        color=ACCENT,
        linewidth=2.2,
        marker="o",
        markersize=5,
        label=y_label,
    )

    # ── Mean line
    mean_value = sum(y_values) / len(y_values)

    ax.axhline(
        mean_value,
        color=ORANGE,
        linestyle="--",
        linewidth=1.6,
        label=f"Mean = {mean_value:.4f}",
    )

    # ── Final value line
    final_n = n_values[-1]
    final_value = y_values[-1]

    ax.axhline(
        final_value,
        color=GREEN,
        linestyle="--",
        linewidth=1.4,
        label=f"Final value = {final_value:.4f}",
    )

    ax.scatter(
        final_n,
        final_value,
        color=GREEN,
        s=65,
        zorder=5,
    )

    # ── Labels and title
    ax.set_title(
        title,
        color=WHITE,
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel("Number of steps n", color=WHITE, fontsize=11)
    ax.set_ylabel(y_label, color=WHITE, fontsize=11)

    # ── Axis formatting
    ax.tick_params(axis="x", colors=WHITE)
    ax.tick_params(axis="y", colors=WHITE)

    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.65)

    for spine in ax.spines.values():
        spine.set_color(GRID)

    legend = ax.legend(
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=WHITE,
        framealpha=1,
    )

    for text in legend.get_texts():
        text.set_color(WHITE)

    plt.tight_layout()
    plt.savefig(full_path, dpi=180, facecolor=BG)
    plt.close()

    return str(full_path)


diff = different_n(25)
plot_values_by_n(diff)
