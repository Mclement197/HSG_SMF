from helpers import get_AAPL_timeseries, get_vol, get_log_returns
import polars as pl
from math import e, pi, sqrt, exp
import numpy as np
import matplotlib.pyplot as plt

def get_p():
    aapl = get_AAPL_timeseries()
    log_returns = get_log_returns(aapl)
    returns = log_returns.select(pl.col("log_returns")).to_series().to_list()
    hausses = sum(1 for r in returns if r > 0)
    return hausses / len(returns)

def get_q(n):
    delta_t = 0.5 / n
    r = 0.01
    vol = get_vol(get_AAPL_timeseries())
    u = e ** (vol * (delta_t**0.5))
    d = e ** (-vol * (delta_t**0.5))
    return (e ** (r * delta_t) - d) / (u - d)

def normal_pdf(x, mu, sigma):
    return (1 / sqrt(2 * pi * sigma**2)) * exp(-((x - mu)**2) / (2 * sigma**2))

n = 25
p = get_p()
q = get_q(n)

mu_p = n * p
sigma_p = sqrt(n * p * (1 - p))
mu_q = n * q
sigma_q = sqrt(n * q * (1 - q))

print(f"p  = {p:.4f}  →  mean = {mu_p:.2f}, sigma = {sigma_p:.2f}")
print(f"q  = {q:.4f}  →  mean = {mu_q:.2f}, sigma = {sigma_q:.2f}")

x_cont = np.linspace(0, n, 500)
pdf_p = [normal_pdf(x, mu_p, sigma_p) for x in x_cont]
pdf_q = [normal_pdf(x, mu_q, sigma_q) for x in x_cont]

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(x_cont, pdf_p, color="#ff6b6b", lw=2, label=f"p = {p:.3f}")
ax.fill_between(x_cont, pdf_p, alpha=0.15, color="#ff6b6b")
ax.plot(x_cont, pdf_q, color="#4a9eff", lw=2, label=f"q = {q:.3f}")
ax.fill_between(x_cont, pdf_q, alpha=0.15, color="#4a9eff")

ax.axvline(mu_p, color="#ff6b6b", linestyle="--", lw=1, label=f"Mean p = {mu_p:.2f}")
ax.axvline(mu_q, color="#4a9eff", linestyle="--", lw=1, label=f"Mean q = {mu_q:.2f}")

ax.set_title("Distribution of p vs q (n=25)", fontweight="bold")
ax.set_xlabel("Number of up moves")
ax.set_ylabel("Probability density")
ax.legend()

plt.tight_layout()
plt.savefig("distribution_pq.png", dpi=150)
plt.show()