from pathlib import Path

import datetime as dt
from math import comb, e, exp, pi, sqrt

import matplotlib.pyplot as plt
import numpy as np

from helpers import get_AAPL_timeseries, get_vol


COLORS = {
    "paper": "#ffffff",
    "ink": "#111827",
    "muted": "#4b5563",
    "grid": "#d8dee9",
    "blue": "#2563eb",
    "blue_soft": "#dbeafe",
    "orange": "#d97706",
    "purple": "#7c3aed",
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
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 360,
}


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    return (1 / sqrt(2 * pi * sigma**2)) * exp(-((x - mu) ** 2) / (2 * sigma**2))


<<<<<<< feat/juju-changes-needed
x_cont = np.linspace(0, n, 500)
pdf_q = [normal_pdf(x, mu_q, sigma_q) for x in x_cont]

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

COLOR_Q    = "#4C72B0"   # bleu
COLOR_MEAN = "#E07B39"   # orange

ax.plot(x_cont, pdf_q, color=COLOR_Q, lw=2, label=f"Risk-neutral probability  $q = {q:.3f}$")
ax.fill_between(x_cont, pdf_q, alpha=0.12, color=COLOR_Q)
ax.axvline(mu_q, color=COLOR_MEAN, linestyle="--", lw=1.5, label=f"Mean $= {mu_q:.2f}$")

# Grille en arrière-plan
ax.set_axisbelow(True)
ax.grid(True, color="lightgrey", linestyle="-", linewidth=0.7, alpha=0.8)

stats_text = (
    f"Skewness $= {skew_q:.3f}$\n"
    f"Excess kurtosis $= {kurt_q:.3f}$"
)

ax.text(
    0.97, 0.95, stats_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="square,pad=0.4",
        facecolor="white",
        edgecolor="#333333",
        alpha=1
    ),
    color="#1a1a1a",
)

ax.set_title(
    "Risk-Neutral Distribution of Up Moves  ($n = 25$ steps, AAPL)",
    fontsize=12, pad=12,
)
ax.set_xlabel("Number of up moves", fontsize=10)
ax.set_ylabel("Probability density", fontsize=10)
ax.legend(frameon=True, framealpha=1, edgecolor="#cccccc", fontsize=9)

plt.tight_layout()
plt.savefig("distribution_q.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.show()
=======
def binomial_pmf(k: int, n: int, q: float) -> float:
    return float(comb(n, k) * (q**k) * ((1 - q) ** (n - k)))


def plot_risk_neutral_up_moves_distribution(
    output_dir: str | Path | None = None,
    filename: str = "distribution_q",
    n: int = 25,
) -> dict[str, str]:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "hand_in" / "graphs"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aapl = get_AAPL_timeseries(
        start=dt.date(2020, 1, 1),
        end=dt.date(2026, 4, 28),
    )
    vol = get_vol(aapl)

    delta_t = 0.5 / n
    u = e ** (vol * delta_t**0.5)
    d = e ** (-vol * delta_t**0.5)
    q = (e ** (0.01 * delta_t) - d) / (u - d)

    mu_q = n * q
    sigma_q = sqrt(n * q * (1 - q))
    skew_q = (1 - 2 * q) / sqrt(n * q * (1 - q))
    kurt_q = (1 - 6 * q * (1 - q)) / (n * q * (1 - q))

    k_values = np.arange(0, n + 1)
    pmf_q = np.asarray([binomial_pmf(int(k), n, q) for k in k_values])

    x_cont = np.linspace(0, n, 500)
    pdf_q = np.asarray([normal_pdf(float(x), mu_q, sigma_q) for x in x_cont])

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(6.8, 3.9), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["paper"])
    ax.set_facecolor(COLORS["paper"])

    ax.bar(
        k_values,
        pmf_q,
        width=0.82,
        color=COLORS["blue"],
        alpha=0.62,
        edgecolor=COLORS["paper"],
        linewidth=0.35,
        label="Binomial risk-neutral probabilities",
    )
    ax.plot(
        x_cont,
        pdf_q,
        color=COLORS["purple"],
        linewidth=1.75,
        label="Normal approximation",
    )
    ax.axvline(
        mu_q,
        color=COLORS["orange"],
        linestyle=(0, (4, 3)),
        linewidth=1.35,
        label=fr"Mean = {mu_q:.2f}",
    )

    stats_text = (
        fr"$q = {q:.3f}$" "\n"
        fr"Skewness = {skew_q:.3f}" "\n"
        fr"Excess kurtosis = {kurt_q:.3f}"
    )
    ax.text(
        0.98,
        0.94,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        color=COLORS["ink"],
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": COLORS["blue_soft"],
            "edgecolor": COLORS["grid"],
            "alpha": 0.96,
        },
    )

    ax.set_xlabel("Number of up moves")
    ax.set_ylabel("Probability")
    ax.grid(True, color=COLORS["grid"], linewidth=0.65, alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color(COLORS["grid"])
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(axis="both", colors=COLORS["muted"], length=3, width=0.8)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor=COLORS["paper"],
        edgecolor=COLORS["grid"],
        framealpha=0.96,
    )
    for text in legend.get_texts():
        text.set_color(COLORS["ink"])

    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    fig.savefig(png_path, dpi=360, bbox_inches="tight", facecolor=COLORS["paper"])
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=COLORS["paper"])
    plt.close(fig)

    print(f"vol      = {vol:.4f}")
    print(f"q        = {q:.4f}")
    print(f"Mean     = {mu_q:.2f}")
    print(f"Sigma    = {sigma_q:.2f}")
    print(f"Skewness = {skew_q:.4f}")
    print(f"Kurtosis = {kurt_q:.4f}")

    return {"png": str(png_path), "pdf": str(pdf_path)}


if __name__ == "__main__":
    outputs = plot_risk_neutral_up_moves_distribution()
    print(outputs["png"])
    print(outputs["pdf"])
>>>>>>> main
