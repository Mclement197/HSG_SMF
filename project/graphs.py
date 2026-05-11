from pathlib import Path

from graphviz import Digraph
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import polars as pl

from helpers import get_v0
import datetime as dt

from helpers import (
    get_AAPL_timeseries,
    get_dn_moments,
    get_q,
    get_v0,
    get_v0_with_Dn,
    get_vol,
)

COLORS = {
    "paper": "#ffffff",
    "ink": "#111827",
    "muted": "#4b5563",
    "grid": "#d8dee9",
    "blue": "#2563eb",
    "blue_soft": "#dbeafe",
    "green": "#059669",
    "green_soft": "#d1fae5",
    "orange": "#d97706",
    "orange_soft": "#fef3c7",
    "purple": "#7c3aed",
    "purple_soft": "#ede9fe",
    "red": "#dc2626",
    "red_soft": "#fee2e2",
}

PLOT_STYLE = {
    "figure.facecolor": COLORS["paper"],
    "axes.facecolor": COLORS["paper"],
    "axes.edgecolor": COLORS["grid"],
    "axes.labelcolor": COLORS["ink"],
    "axes.titlecolor": COLORS["ink"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "grid.color": COLORS["grid"],
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 360,
}

COLUMN_LABELS = {
    "S": "Stock price $S_t$",
    "C": "Cumulative price sum $C_t$",
    "S_bar": "Arithmetic average $\\bar{S}_t$",
    "V": "Option value $V_t$",
    "normalised_C": "Normalised cumulative sum",
    "normalised_S_bar": "Normalised average price",
}


def _column_label(value_col: str) -> str:
    return COLUMN_LABELS.get(value_col, value_col.replace("_", " "))


def _format_value(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "NA"

    try:
        if not np.isfinite(value):
            return "NA"
    except TypeError:
        return str(value)

    return f"{value:,.{digits}f}"


def _prepare_output_path(
    output_dir: str | Path,
    filename: str | Path,
    default_suffix: str = ".png",
) -> Path:
    path = Path(filename)

    if not path.suffix:
        path = path.with_suffix(default_suffix)

    if not path.is_absolute() and path.parent == Path("."):
        path = Path(output_dir) / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _new_figure(figsize: tuple[float, float] = (6.8, 3.8)):
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor(COLORS["paper"])
    ax.set_facecolor(COLORS["paper"])
    return fig, ax


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    if title:
        ax.set_title(title, loc="left", fontweight="semibold", pad=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=COLORS["grid"], linewidth=0.65, alpha=0.72)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color(COLORS["grid"])
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(axis="both", colors=COLORS["muted"], length=3, width=0.8)


def _style_legend(ax, loc: str = "best") -> None:
    legend = ax.legend(
        loc=loc,
        frameon=True,
        facecolor=COLORS["paper"],
        edgecolor=COLORS["grid"],
        framealpha=0.96,
    )

    if legend is None:
        return

    for text in legend.get_texts():
        text.set_color(COLORS["ink"])


def _save_figure(fig, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())

    if output_path.suffix.lower() != ".pdf":
        fig.savefig(
            output_path.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    if output_path.suffix.lower() != ".png":
        fig.savefig(
            output_path.with_suffix(".png"),
            dpi=360,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    plt.close(fig)
    return str(output_path)


def _render_graphviz(dot: Digraph, filename: str, output_dir: str | Path) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_path = dot.render(
        filename=filename,
        directory=str(output_path),
        format="pdf",
        cleanup=True,
    )

    dot.render(
        filename=filename,
        directory=str(output_path),
        format="png",
        cleanup=True,
    )

    return pdf_path


def _with_arithmetic_average(tree: pl.DataFrame) -> pl.DataFrame:
    if "S_bar" in tree.columns:
        return tree

    required_cols = {"C", "t"}
    missing = required_cols - set(tree.columns)

    if missing:
        raise ValueError(f"Missing required columns to compute S_bar: {missing}")

    return tree.with_columns((pl.col("C") / (pl.col("t") + 1)).alias("S_bar"))


def _tree_graph() -> Digraph:
    dot = Digraph()

    dot.attr(
        rankdir="LR",
        bgcolor=COLORS["paper"],
        splines="line",
        nodesep="0.34",
        ranksep="0.82",
        pad="0.04",
        margin="0.02",
        outputorder="edgesfirst",
    )

    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor=COLORS["paper"],
        color=COLORS["grid"],
        fontcolor=COLORS["ink"],
        fontname="Helvetica",
        fontsize="9",
        penwidth="1.15",
        margin="0.10,0.07",
        width="1.05",
        height="0.38",
        fixedsize="false",
    )

    dot.attr(
        "edge",
        color=COLORS["grid"],
        arrowsize="0.55",
        penwidth="1.05",
    )

    return dot


def plot_binomial_tree(
    tree: pl.DataFrame,
    n: int,
    show_average: bool = True,
    show_payoff: bool = True,
    output_dir: str | Path = "graphs",
):
    """Plot a compact, LaTeX-friendly binomial tree up to time step n."""
    tree = tree.filter(pl.col("t") <= n)

    if tree.is_empty():
        raise ValueError(f"No rows found for t <= {n}.")

    required_cols = {"t", "path_id", "S", "move", "parent_id"}

    if show_average or show_payoff:
        required_cols.add("C")

    missing = required_cols - set(tree.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_plot = int(tree.select(pl.col("t").max()).item())
    filename = f"binomial_tree_{n_plot}n"

    if show_average:
        filename += "_avg"

    filename += "_payoff" if show_payoff else "_moves"

    if show_average or show_payoff:
        tree = _with_arithmetic_average(tree)

    if show_payoff:
        tree = tree.with_columns(
            pl.when(pl.col("t") == n_plot)
            .then((pl.col("S") - pl.col("S_bar")).clip(lower_bound=0))
            .otherwise(None)
            .alias("payoff")
        )

    rows = tree.sort(["t", "path_id"]).to_dicts()
    dot = _tree_graph()

    for row in rows:
        node_id = str(row["path_id"])
        label_lines = [
            f"t={row['t']}",
            f"S={_format_value(row['S'])}",
        ]

        if show_average:
            label_lines.append(f"Avg={_format_value(row['S_bar'])}")

        if show_payoff and row["payoff"] is not None:
            label_lines.append(f"Payoff={_format_value(row['payoff'])}")

        label = "\n".join(label_lines)

        if row["t"] == 0:
            fillcolor = COLORS["blue_soft"]
            bordercolor = COLORS["blue"]
        elif show_payoff and row["t"] == n_plot:
            payoff = row["payoff"]

            if payoff is not None and payoff > 0:
                fillcolor = COLORS["green_soft"]
                bordercolor = COLORS["green"]
            else:
                fillcolor = COLORS["orange_soft"]
                bordercolor = COLORS["orange"]
        elif row["move"] == "U":
            fillcolor = COLORS["green_soft"]
            bordercolor = COLORS["green"]
        elif row["move"] == "D":
            fillcolor = COLORS["orange_soft"]
            bordercolor = COLORS["orange"]
        else:
            fillcolor = COLORS["paper"]
            bordercolor = COLORS["grid"]

        dot.node(
            node_id,
            label=label,
            fillcolor=fillcolor,
            fontcolor=COLORS["ink"],
            color=bordercolor,
        )

        if row["parent_id"] is not None:
            edge_color = COLORS["green"] if row["move"] == "U" else COLORS["orange"]
            dot.edge(str(row["parent_id"]), node_id, color=edge_color)

    return _render_graphviz(dot, filename, output_dir)


def plot_all_paths(
    tree: pl.DataFrame,
    n: int,
    value_col: str = "S",
    output_dir: str | Path = "graphs",
):
    """Plot all terminal paths of a binomial tree."""
    tree = tree.filter(pl.col("t") <= n)

    if value_col == "S_bar":
        tree = _with_arithmetic_average(tree)

    required_cols = {"t", "history", value_col}
    missing = required_cols - set(tree.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if tree.is_empty():
        raise ValueError(f"No rows found for t <= {n}.")

    final_t = int(tree.select(pl.col("t").max()).item())
    filename = f"{value_col}_paths_{final_t}n.png"
    output_path = _prepare_output_path(output_dir, filename)

    final_paths = (
        tree.filter(pl.col("t") == final_t).select("history").to_series().to_list()
    )

    fig, ax = _new_figure(figsize=(6.8, 4.0))
    alpha = 1.0
    line_width = 0.85 if len(final_paths) <= 128 else 0.38
    marker_size = 2.0 if len(final_paths) <= 128 else 0

    for idx, path in enumerate(final_paths):
        rows = []

        for k in range(len(path) + 1):
            prefix = path[:k]
            row = tree.filter(pl.col("history") == prefix)
            rows.append(row)

        path_df = pl.concat(rows).sort("t")

        ax.plot(
            path_df["t"].to_list(),
            path_df[value_col].to_list(),
            color=COLORS["blue"],
            alpha=alpha,
            linewidth=line_width,
            marker="o" if marker_size else None,
            markersize=marker_size,
            markeredgewidth=0,
            solid_capstyle="round",
            label="Terminal paths" if idx == 0 else None,
        )

    initial_value = tree.filter(pl.col("t") == 0).select(value_col).to_series().item()
    ax.axhline(
        initial_value,
        color=COLORS["orange"],
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        label=f"Initial value = {_format_value(initial_value)}",
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(
        ax,
        title="",
        xlabel="Time step",
        ylabel=_column_label(value_col),
    )
    _style_legend(ax, loc="upper left")

    return _save_figure(fig, output_path)


def plot_terminal_distribution(
    tree: pl.DataFrame,
    n: int,
    value_col: str = "C",
    bins: int = 100,
    exclude_zero: bool = False,
    output_dir: str | Path = "graphs",
):
    """Plot the terminal distribution of a chosen binomial-tree column."""
    tree = tree.filter(pl.col("t") <= n)

    if value_col == "S_bar":
        tree = _with_arithmetic_average(tree)

    if tree.is_empty():
        raise ValueError(f"No rows found for t <= {n}.")

    if value_col not in tree.columns:
        raise ValueError(f"Missing required column: {value_col}")

    final_t = int(tree.select(pl.col("t").max()).item())
    suffix = "_excl_zero" if exclude_zero else ""
    filename = f"{value_col}_terminal_distribution_{final_t}n{suffix}.png"
    output_path = _prepare_output_path(output_dir, filename)

    terminal = tree.filter(pl.col("t") == final_t).select(value_col).drop_nulls()
    terminal_values = np.asarray(terminal[value_col].to_list(), dtype=float)

    if exclude_zero:
        terminal_values = terminal_values[terminal_values != 0]

    if terminal_values.size == 0:
        raise ValueError("No terminal values left to plot after filtering.")

    mean_val = float(np.mean(terminal_values))
    density = terminal_values.size > 1 and np.min(terminal_values) != np.max(
        terminal_values
    )

    fig, ax = _new_figure(figsize=(6.8, 3.9))
    ax.hist(
        terminal_values,
        bins=bins,
        density=density,
        color=COLORS["blue"],
        alpha=0.82,
        edgecolor=COLORS["paper"],
        linewidth=0.35,
    )

    ax.axvline(
        mean_val,
        color=COLORS["orange"],
        linewidth=1.35,
        linestyle=(0, (4, 3)),
        label=f"Mean = {_format_value(mean_val)}",
    )

    _style_axes(
        ax,
        title="",
        xlabel=_column_label(value_col),
        ylabel="Density" if density else "Count",
    )
    _style_legend(ax)

    return _save_figure(fig, output_path)


def plot_payoff_vs_terminal(
    tree: pl.DataFrame,
    filename: str | Path = "payoff_vs_ST.png",
    output_dir: str | Path = "graphs",
) -> str:
    """Plot terminal payoff against terminal stock price."""
    required_cols = {"t", "S", "C"}
    missing = required_cols - set(tree.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_plot = int(tree.select(pl.col("t").max()).item())

    if filename == "payoff_vs_ST.png":
        filename = f"payoff_vs_ST_{n_plot}n.png"

    output_path = _prepare_output_path(output_dir, filename)

    df = _with_arithmetic_average(tree.filter(pl.col("t") == n_plot)).with_columns(
        (pl.col("S") - pl.col("S_bar")).clip(lower_bound=0).alias("payoff")
    )

    x = np.asarray(df["S"].to_list(), dtype=float)
    y = np.asarray(df["payoff"].to_list(), dtype=float)

    fig, ax = _new_figure(figsize=(6.8, 3.9))

    if df.height > 5_000:
        hexbin = ax.hexbin(
            x,
            y,
            gridsize=48,
            cmap="Blues",
            mincnt=1,
            linewidths=0,
        )
        colorbar = fig.colorbar(hexbin, ax=ax, pad=0.015)
        colorbar.set_label("Number of terminal paths", color=COLORS["ink"])
        colorbar.ax.tick_params(colors=COLORS["muted"])
        colorbar.outline.set_edgecolor(COLORS["grid"])
    else:
        ax.scatter(
            x,
            y,
            color=COLORS["blue"],
            alpha=0.72,
            s=24,
            edgecolors=COLORS["paper"],
            linewidths=0.35,
        )

    ax.axhline(
        0,
        color=COLORS["muted"],
        linewidth=1.0,
        linestyle=(0, (4, 3)),
    )

    _style_axes(
        ax,
        title="Floating-strike Asian call payoff",
        xlabel="Terminal stock price $S_n$",
        ylabel="Payoff $\\max(S_n - \\bar{S}_n, 0)$",
    )

    return _save_figure(fig, output_path)


def plot_option_value_by_time(
    tree: pl.DataFrame,
    n: int,
    output_dir: str | Path = "graphs",
):
    """Scatter plot of option value V against stock price S by time step."""
    required_cols = {"t", "S", "V"}
    missing = required_cols - set(tree.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = tree.filter(pl.col("V").is_not_null() & (pl.col("t") <= n))

    if df.is_empty():
        raise ValueError(
            "No non-null option values V found up to the requested time step."
        )

    n_plot = int(df.select(pl.col("t").max()).item())
    filename = f"value_by_time_{n_plot}n.png"
    output_path = _prepare_output_path(output_dir, filename)

    fig, ax = _new_figure(figsize=(6.8, 4.0))
    scatter = ax.scatter(
        df["S"].to_numpy(),
        df["V"].to_numpy(),
        c=df["t"].to_numpy(),
        cmap="viridis",
        alpha=0.78,
        s=20,
        edgecolors="none",
    )

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    colorbar.set_label("Time step", color=COLORS["ink"])
    colorbar.ax.tick_params(colors=COLORS["muted"])
    colorbar.outline.set_edgecolor(COLORS["grid"])

    _style_axes(
        ax,
        title="",
        xlabel="Stock price $S_t$",
        ylabel="Option value $V_t$",
    )

    return _save_figure(fig, output_path)


def plot_backward_induction_tree(
    df: pl.DataFrame,
    n: int,
    output_dir: str | Path = "graphs",
):
    """Plot a compact backward-induction tree up to time step n."""
    df = df.filter(pl.col("t") <= n)

    if df.is_empty():
        raise ValueError(f"No rows found for t <= {n}.")

    required_cols = {"t", "path_id", "S", "S_bar", "V", "move", "parent_id"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_plot = int(df.select(pl.col("t").max()).item())
    filename = f"backward_induction_tree_{n_plot}n"
    rows = df.sort(["t", "path_id"]).to_dicts()
    dot = _tree_graph()

    for row in rows:
        node_id = str(row["path_id"])
        label = "\n".join(
            [
                f"t={row['t']}",
                f"S={_format_value(row['S'])}",
                f"Avg={_format_value(row['S_bar'])}",
                f"V={_format_value(row['V'])}",
            ]
        )

        if row["t"] == 0:
            fillcolor = COLORS["blue_soft"]
            bordercolor = COLORS["blue"]
        elif row["t"] == n_plot:
            if row["V"] is not None and row["V"] > 0:
                fillcolor = COLORS["green_soft"]
                bordercolor = COLORS["green"]
            else:
                fillcolor = COLORS["orange_soft"]
                bordercolor = COLORS["orange"]
        else:
            fillcolor = COLORS["paper"]
            bordercolor = COLORS["grid"]

        dot.node(
            node_id,
            label=label,
            fillcolor=fillcolor,
            fontcolor=COLORS["ink"],
            color=bordercolor,
        )

        if row["parent_id"] is not None:
            edge_color = COLORS["green"] if row["move"] == "U" else COLORS["orange"]
            dot.edge(node_id, str(row["parent_id"]), color=edge_color)

    return _render_graphviz(dot, filename, output_dir)


def plot_backward_induction_first_steps_tree(
    df: pl.DataFrame,
    steps: int = 3,
    full_n: int | None = None,
    output_dir: str | Path = "graphs",
    filename: str | None = None,
) -> str:
    """Plot the first steps of a full backward-induction tree."""
    required_cols = {"t", "path_id", "S", "S_bar", "V", "move", "parent_id"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    full_n = int(df.select(pl.col("t").max()).item()) if full_n is None else full_n
    plot_df = df.filter(pl.col("t") <= steps)

    if plot_df.is_empty():
        raise ValueError(f"No rows found for t <= {steps}.")

    if filename is None:
        filename = f"backward_induction_tree_{full_n}n_first_{steps}_steps"

    rows = plot_df.sort(["t", "path_id"]).to_dicts()
    dot = _tree_graph()

    for row in rows:
        node_id = str(row["path_id"])
        label = "\n".join(
            [
                f"t={row['t']}",
                f"S={_format_value(row['S'])}",
                f"Avg={_format_value(row['S_bar'])}",
                f"V={_format_value(row['V'])}",
            ]
        )

        if row["t"] == 0:
            fillcolor = COLORS["blue_soft"]
            bordercolor = COLORS["blue"]
        else:
            fillcolor = COLORS["paper"]
            bordercolor = COLORS["grid"]

        dot.node(
            node_id,
            label=label,
            fillcolor=fillcolor,
            fontcolor=COLORS["ink"],
            color=bordercolor,
        )

        if row["parent_id"] is not None:
            edge_color = COLORS["green"] if row["move"] == "U" else COLORS["orange"]
            dot.edge(node_id, str(row["parent_id"]), color=edge_color)

    return _render_graphviz(dot, filename, output_dir)


def plot_backward_induction_last_steps_tree(
    aapl: pl.DataFrame,
    vol: float,
    n: int = 25,
    steps: int = 3,
    output_dir: str | Path = "graphs",
    filename: str | None = None,
    prefix_history: str | None = None,
    r: float = 0.01,
) -> str:
    """Plot one local final subtree from an n-step backward-induction tree."""
    if steps >= n:
        raise ValueError("steps must be smaller than n.")

    from helpers import T, get_q

    delta_t = T / n
    u = np.exp(vol * delta_t**0.5)
    d = np.exp(-vol * delta_t**0.5)
    discount = np.exp(-r * delta_t)
    q = get_q(pl.DataFrame({"t": [0, n]}), vol, r)
    start_t = n - steps

    if prefix_history is None:
        prefix_history = "UD" * (start_t // 2) + "U" * (start_t % 2)

    if len(prefix_history) != start_t or any(move not in "UD" for move in prefix_history):
        raise ValueError(f"prefix_history must contain exactly {start_t} U/D moves.")

    s0 = (
        aapl.filter(pl.col("Date") == aapl.select(pl.col("Date").max()).item())
        .select(pl.col("Close"))
        .item()
    )

    def values_for(history: str) -> tuple[float, float, float]:
        s = float(s0)
        c = float(s0)

        for move in history:
            s *= u if move == "U" else d
            c += s

        s_bar = c / (len(history) + 1)
        return s, c, s_bar

    nodes: dict[str, dict[str, float | int | str | None]] = {}

    for depth in range(steps + 1):
        histories = [prefix_history]

        for _ in range(depth):
            histories = [history + move for history in histories for move in ("U", "D")]

        for history in histories:
            s, _, s_bar = values_for(history)
            nodes[history] = {
                "t": len(history),
                "history": history,
                "move": history[-1] if history else "",
                "parent": history[:-1] if history != prefix_history else None,
                "S": s,
                "S_bar": s_bar,
                "V": max(s - s_bar, 0) if len(history) == n else None,
            }

    for t in range(n - 1, start_t - 1, -1):
        for history, node in list(nodes.items()):
            if node["t"] != t:
                continue

            up_value = nodes[history + "U"]["V"]
            down_value = nodes[history + "D"]["V"]
            node["V"] = discount * (q * float(up_value) + (1 - q) * float(down_value))

    if filename is None:
        filename = f"backward_induction_tree_{n}n_last_{steps}_steps"

    dot = _tree_graph()

    for history, row in sorted(nodes.items(), key=lambda item: (item[1]["t"], item[0])):
        label = "\n".join(
            [
                f"t={row['t']}",
                f"S={_format_value(row['S'])}",
                f"Avg={_format_value(row['S_bar'])}",
                f"V={_format_value(row['V'])}",
            ]
        )

        if row["t"] == start_t:
            fillcolor = COLORS["blue_soft"]
            bordercolor = COLORS["blue"]
        elif row["t"] == n:
            if row["V"] is not None and row["V"] > 0:
                fillcolor = COLORS["green_soft"]
                bordercolor = COLORS["green"]
            else:
                fillcolor = COLORS["orange_soft"]
                bordercolor = COLORS["orange"]
        else:
            fillcolor = COLORS["paper"]
            bordercolor = COLORS["grid"]

        dot.node(
            history,
            label=label,
            fillcolor=fillcolor,
            fontcolor=COLORS["ink"],
            color=bordercolor,
        )

        if row["parent"] is not None:
            edge_color = COLORS["green"] if row["move"] == "U" else COLORS["orange"]
            dot.edge(history, str(row["parent"]), color=edge_color)

    return _render_graphviz(dot, filename, output_dir)


def plot_aapl_timeseries(
    df: pl.DataFrame,
    date_col: str = "Date",
    price_col: str = "Close",
    show_mean: bool = False,
    show_last: bool = False,
    output_dir: str | Path = "graphs",
    filename: str | Path = "aapl_timeseries.png",
) -> str:
    """Plot an AAPL closing-price time series for a LaTeX report."""
    required_cols = {date_col, price_col}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    plot_df = df.select(date_col, price_col).drop_nulls().sort(date_col)

    if plot_df.height == 0:
        raise ValueError("The dataframe is empty after removing null values.")

    dates = plot_df[date_col].to_list()
    prices = np.asarray(plot_df[price_col].to_list(), dtype=float)

    output_path = _prepare_output_path(output_dir, filename)

    fig, ax = _new_figure(figsize=(6.8, 3.8))
    ax.plot(
        dates,
        prices,
        color=COLORS["blue"],
        linewidth=1.85,
    )

    if show_mean:
        mean_price = float(np.mean(prices))
        ax.axhline(
            mean_price,
            color=COLORS["orange"],
            linestyle=(0, (4, 3)),
            linewidth=1.25,
            label=f"Mean = {_format_value(mean_price)}",
        )

    if show_last:
        last_date = dates[-1]
        last_price = float(prices[-1])
        ax.axhline(
            last_price,
            color=COLORS["green"],
            linestyle=(0, (1, 2)),
            linewidth=1.2,
            label=f"Last = {_format_value(last_price)}",
        )
        ax.scatter(
            last_date,
            last_price,
            color=COLORS["green"],
            s=34,
            zorder=5,
            edgecolors=COLORS["paper"],
            linewidths=0.45,
        )

    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    _style_axes(
        ax,
        title="",
        xlabel="Date",
        ylabel="Price",
    )

    if show_mean or show_last:
        _style_legend(ax)

    return _save_figure(fig, output_path)


def plot_values_by_n(
    values_by_n: dict[int, float],
    output_dir: str | Path = "graphs",
    filename: str | Path = "values_by_n.png",
    title: str = "",
    y_label: str = "Value",
) -> str:
    """Plot values indexed by n."""
    if not values_by_n:
        raise ValueError("values_by_n is empty.")

    sorted_items = sorted(values_by_n.items())
    n_values = [item[0] for item in sorted_items]
    y_values = [item[1] for item in sorted_items]
    output_path = _prepare_output_path(output_dir, filename)

    fig, ax = _new_figure(figsize=(6.8, 3.8))
    ax.plot(
        n_values,
        y_values,
        color=COLORS["blue"],
        linewidth=1.8,
        marker="o",
        markersize=4.2,
        markerfacecolor=COLORS["paper"],
        markeredgewidth=1.2,
    )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(
        ax,
        title=title,
        xlabel="Number of steps $n$",
        ylabel=y_label,
    )

    return _save_figure(fig, output_path)


def get_all_graphs(
    output_dir: str | Path | None = None,
    tree_n: int = 25,
    small_tree_n: int = 3,
    path_n: int = 12,
    values_stop: int = 25,
) -> dict[str, str]:
    """
    Regenerate the full graph set used by the LaTeX hand-in.

    The default parameters reproduce the existing structure in hand_in/graphs:
    - basic_binomial_trees/*_3n_*.png
    - backward_induction/backward_induction_tree_3n.png
    - distributions/*_terminal_distribution_25n*.png
    - paths/*_paths_12n.png
    - value_vs_S_by_time/value_by_time_25n.png
    - aapl_timeseries.png
    - values_by_n.png
    """
    from helpers import (
        AAPL_HISTORY_START,
        AAPL_TICKER,
        AAPL_TREE_END,
        AAPL_TREE_START,
        backward_induction,
        different_n,
        get_AAPL_timeseries,
        get_arithmetic_avg,
        get_binomial_tree,
        get_payoff,
        get_vol,
        normalise,
    )

    if output_dir is None:
        output_root = Path(__file__).resolve().parents[1] / "hand_in" / "graphs"
    else:
        output_root = Path(output_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    aapl_history = get_AAPL_timeseries(
        ticker=AAPL_TICKER,
        start=AAPL_HISTORY_START,
    )
    aapl_model = get_AAPL_timeseries(
        ticker=AAPL_TICKER,
        start=AAPL_TREE_START,
        end=AAPL_TREE_END,
    )
    vol = get_vol(aapl_model)

    outputs["aapl_timeseries"] = plot_aapl_timeseries(
        aapl_history,
        output_dir=output_root,
        filename="aapl_timeseries.png",
    )

    small_tree = get_binomial_tree(aapl_model, small_tree_n)
    small_tree = get_arithmetic_avg(small_tree)
    small_payoff_tree = get_payoff(small_tree)
    small_value_tree = backward_induction(small_payoff_tree, vol)
    basic_dir = output_root / "basic_binomial_trees"

    outputs["binomial_tree_moves"] = plot_binomial_tree(
        small_tree,
        small_tree_n,
        show_average=False,
        show_payoff=False,
        output_dir=basic_dir,
    )
    outputs["binomial_tree_avg_moves"] = plot_binomial_tree(
        small_tree,
        small_tree_n,
        show_average=True,
        show_payoff=False,
        output_dir=basic_dir,
    )
    outputs["binomial_tree_avg_payoff"] = plot_binomial_tree(
        small_tree,
        small_tree_n,
        show_average=True,
        show_payoff=True,
        output_dir=basic_dir,
    )
    outputs["backward_induction_tree"] = plot_backward_induction_tree(
        small_value_tree,
        small_tree_n,
        output_dir=output_root / "backward_induction",
    )

    path_tree = get_binomial_tree(aapl_model, path_n)
    path_tree = get_arithmetic_avg(path_tree)
    path_tree = get_payoff(path_tree)
    path_tree = backward_induction(path_tree, vol)
    paths_dir = output_root / "paths"

    for value_col in ["C", "S_bar", "S", "V"]:
        outputs[f"{value_col}_paths"] = plot_all_paths(
            path_tree,
            path_n,
            value_col=value_col,
            output_dir=paths_dir,
        )

    full_tree = get_binomial_tree(aapl_model, tree_n)
    full_tree = get_arithmetic_avg(full_tree)
    full_tree = normalise(full_tree, "C")
    full_tree = get_payoff(full_tree)

    distributions_dir = output_root / "distributions"
    for value_col in ["C", "S_bar", "S", "V", "normalised_C"]:
        outputs[f"{value_col}_terminal_distribution"] = plot_terminal_distribution(
            full_tree,
            tree_n,
            value_col=value_col,
            output_dir=distributions_dir,
        )

    outputs["V_terminal_distribution_excl_zero"] = plot_terminal_distribution(
        full_tree,
        tree_n,
        value_col="V",
        exclude_zero=True,
        output_dir=distributions_dir,
    )

    full_tree = backward_induction(full_tree, vol)
    outputs["value_by_time"] = plot_option_value_by_time(
        full_tree,
        tree_n,
        output_dir=output_root / "value_vs_S_by_time",
    )

    values_by_n = different_n(values_stop)

    outputs["values_by_n"] = plot_values_by_n(
        values_by_n,
        output_dir=output_root,
        filename="values_by_n.png",
        y_label="Value",
    )

    return outputs

def plot_weighted_dn_distribution_normal(
    tree: pl.DataFrame,
    q: float,
    mu: float,
    var: float,
    n: int,
    bins: int = 100,
    show_raw_mean: bool = False,
    output_dir: str | Path = "graphs",
    filename: str | Path = "Dn_risk_neutral_normal_approximation.png",
) -> str:
    """
    Plot the risk-neutral terminal distribution of Dn with the normal approximation.

    This is the correct graph for the normal approximation exercise:
    - terminal Dn values are weighted by risk-neutral path probabilities;
    - the fitted normal density N(mu, var) is overlaid.
    """
    required_cols = {"t", "history", "Dn"}
    missing = required_cols - set(tree.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if var <= 0:
        raise ValueError("Variance must be strictly positive.")

    tree = tree.filter(pl.col("t") <= n)

    if tree.is_empty():
        raise ValueError(f"No rows found for t <= {n}.")

    final_t = int(tree.select(pl.col("t").max()).item())

    terminal = (
        tree.filter(pl.col("t") == final_t)
        .with_columns(
            pl.col("history").str.count_matches("U").alias("n_up"),
            pl.col("history").str.count_matches("D").alias("n_down"),
        )
        .with_columns(
            (
                (pl.lit(q) ** pl.col("n_up"))
                * (pl.lit(1 - q) ** pl.col("n_down"))
            ).alias("q_weight")
        )
        .select(["Dn", "q_weight"])
        .drop_nulls()
    )

    if terminal.is_empty():
        raise ValueError("No terminal Dn values found.")

    dn = np.asarray(terminal["Dn"].to_list(), dtype=float)
    weights = np.asarray(terminal["q_weight"].to_list(), dtype=float)

    weight_sum = float(np.sum(weights))

    if not np.isclose(weight_sum, 1.0, atol=1e-8):
        weights = weights / weight_sum

    sigma = var**0.5
    raw_mean = float(np.mean(dn))

    x_min = min(float(np.min(dn)), mu - 4 * sigma)
    x_max = max(float(np.max(dn)), mu + 4 * sigma)
    x = np.linspace(x_min, x_max, 1_000)

    normal_density = (1 / (sigma * (2 * np.pi) ** 0.5)) * np.exp(
        -((x - mu) ** 2) / (2 * var)
    )

    output_path = _prepare_output_path(output_dir, filename)

    fig, ax = _new_figure(figsize=(6.8, 3.9))

    ax.hist(
        dn,
        bins=bins,
        weights=weights,
        density=True,
        color=COLORS["blue"],
        alpha=0.60,
        edgecolor=COLORS["paper"],
        linewidth=0.35,
        label="Risk-neutral binomial distribution",
    )

    ax.plot(
        x,
        normal_density,
        color=COLORS["purple"],
        linewidth=1.75,
        label=fr"Normal approximation $N({mu:.3f}, {var:.3f})$",
    )

    ax.axvline(
        mu,
        color=COLORS["orange"],
        linewidth=1.35,
        linestyle=(0, (4, 3)),
        label=fr"Risk-neutral mean = {_format_value(mu, 3)}",
    )

    if show_raw_mean:
        ax.axvline(
            raw_mean,
            color=COLORS["muted"],
            linewidth=1.15,
            linestyle=(0, (1, 2)),
            label=fr"Raw path mean = {_format_value(raw_mean, 3)}",
        )

    _style_axes(
        ax,
        title="",
        xlabel=r"$D_n = S_n - \bar{S}_n$",
        ylabel="Risk-neutral density",
    )
    _style_legend(ax)

    return _save_figure(fig, output_path)

def plot_dn_positive_payoff_density(
    mu: float,
    var: float,
    r_: float = 0.01,
    output_dir: str | Path = "graphs",
    filename: str | Path = "Dn_positive_payoff_density.png",
) -> str:
    """
    Plot x^+ times the normal density of Dn.

    The area under this curve equals E[(Dn)^+] under the normal approximation.
    Discounting this area gives the approximate option price.
    """
    if var <= 0:
        raise ValueError("Variance must be strictly positive.")

    sigma = var**0.5
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1_000)

    normal_density = (1 / (sigma * (2 * np.pi) ** 0.5)) * np.exp(
        -((x - mu) ** 2) / (2 * var)
    )

    positive_payoff_density = np.maximum(x, 0) * normal_density
    expected_positive_payoff = np.trapezoid(positive_payoff_density, x)
    discounted_value = np.exp(-r_ * 0.5) * expected_positive_payoff

    output_path = _prepare_output_path(output_dir, filename)

    fig, ax = _new_figure(figsize=(6.8, 3.9))

    ax.plot(
        x,
        positive_payoff_density,
        color=COLORS["green"],
        linewidth=1.75,
        label=fr"$x^+ f_{{D_n}}(x)$",
    )

    ax.fill_between(
        x,
        positive_payoff_density,
        where=x >= 0,
        color=COLORS["green"],
        alpha=0.18,
    )

    ax.axvline(
        0,
        color=COLORS["muted"],
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        label="$D_n = 0$",
    )

    ax.text(
        0.98,
        0.92,
        fr"$V_0^N \approx {_format_value(discounted_value, 4)}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLORS["ink"],
    )

    _style_axes(
        ax,
        title="",
        xlabel=r"$D_n = S_n - \bar{S}_n$",
        ylabel=r"$x^+ f_{D_n}(x)$",
    )
    _style_legend(ax)

    return _save_figure(fig, output_path)

if __name__ == "__main__":
    Bi, tree = get_v0(25, return_tree=True)

    vol = get_vol(
        get_AAPL_timeseries("AAPL", dt.date(2020, 1, 1), dt.date(2026, 4, 28))
    )

    q = get_q(tree, vol, r=0.01)

    mu_Dn, var_Dn = get_dn_moments(tree, q)
    Dn_price = get_v0_with_Dn(mu_Dn, var_Dn)

    plot_dn_positive_payoff_density(
        mu=mu_Dn,
        var=var_Dn,
        r_=0.01,
        output_dir="graphs/distributions",
    )
