from math import e, exp, pi, sqrt
import datetime as dt

from helpers import get_AAPL_timeseries, get_vol
import matplotlib.pyplot as plt
import numpy as np


n = 25
aapl = get_AAPL_timeseries(start=dt.date(2020, 1, 1), end=dt.date(2026, 4, 28))
vol = get_vol(aapl)

delta_t = 0.5 / n
u = e ** (vol * delta_t**0.5)
d = e ** (-vol * delta_t**0.5)
q = (e ** (0.01 * delta_t) - d) / (u - d)

mu_q = n * q
sigma_q = sqrt(n * q * (1 - q))
skew_q = (1 - 2 * q) / sqrt(n * q * (1 - q))
kurt_q = (1 - 6 * q * (1 - q)) / (n * q * (1 - q))

print(f"vol      = {vol:.4f}")
print(f"q        = {q:.4f}")
print(f"Mean     = {mu_q:.2f}")
print(f"Sigma    = {sigma_q:.2f}")
print(f"Skewness = {skew_q:.4f}")
print(f"Kurtosis = {kurt_q:.4f}")


def normal_pdf(x, mu, sigma):
    return (1 / sqrt(2 * pi * sigma**2)) * exp(-((x - mu) ** 2) / (2 * sigma**2))


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