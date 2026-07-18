# Volatility Cascade — Mathematical Treatment

This document is the formal mathematical treatment of the volatility
cascade. It is referenced from `src/volcascade/__init__.py` and
`src/volcascade/cascade.py`.

The locked design decisions, pre-registered parameter set, and the four
pre-registered hypotheses (H1–H4) are documented in
**`docs/DESIGN_MEMO.md`**. The vol-peak mechanism and the empirical
findings are documented in **`results/MECHANISM.md`**.

## 1. Cascade construction

Let $r_t$ denote the log return at time $t$. The volatility cascade at
differentiation order $k$ is defined recursively.

Order 1 is realized volatility over a rolling window of length
$w_{\text{inner}}$:

    sigma^(1)_t = sqrt(sum_{i=0..w_inner-1} r^2_{t-i})

For $k \geq 2$, order-$k$ volatility is the rolling sample standard
deviation of order-$(k-1)$ volatility, with the same window length:

    sigma^(k)_t = std(sigma^(k-1)_{t-w_inner+1:t})

We use $K = 4$ throughout, per the pre-registered design (Gatheral,
Jaisson, Rosenbaum 2018; orders $> 4$ are noise-dominated). The warmup
period is $(K-1) \times w_{\text{inner}}$ observations.

## 2. Z-scoring and cascade slope

Each order is z-scored against its own trailing history with a lookback
$L_z$:

    z^(k)_t = (sigma^(k)_t - mu^(k)_t) / s^(k)_t

where $\mu^{(k)}_t$ and $s^{(k)}_t$ are the rolling mean and standard
deviation computed on the trailing window $[t - L_z, t - 1]$ (strictly
past data, no look-ahead bias).

The cascade slope is the OLS regression slope of differentiation order
index against the z-scored order-specific volatilities at time $t$:

    beta_t = sum_k (k - k_bar)(z^(k)_t - z_bar_t) / sum_k (k - k_bar)^2

where $\bar{k} = 2.5$ and $\bar{z}_t$ is the cross-order mean z-score.

$\beta_t > 0$ indicates steepening (higher orders more elevated than
lower orders); $\beta_t < 0$ indicates inversion (higher orders less
elevated).

## 3. Auxiliary statistics

The Shannon entropy of the $|z|$-weighted order distribution is
computed as a non-linear robustness check:

    H_t = -sum_k p_k log p_k,    p_k = |z^(k)_t| / sum_k |z^(k)_t|

with the convention $0 \log 0 = 0$. Entropy is bounded above by
$\log K$; high entropy indicates a flat cascade (orders evenly
weighted), low entropy indicates concentration in one order.

## 4. Cross-sectional decoupling

For a stock and its sector, the decoupling order $k^*$ is the smallest
order at which the joint distribution of $(z^{(k)}_{\text{stock}},
z^{(k)}_{\text{sector}})$ rejects the null of equal conditional
distributions via a Chow test. The Chow F-statistic at breakpoint $k$
with window length $L$ is:

    F = ((RSS_pooled - RSS_unpooled) / q) / (RSS_unpooled / (2L - 2q))

where $q = 2$ is the number of parameters per regression and
$RSS_{\text{unpooled}} = RSS_{\text{before}} + RSS_{\text{after}}$ over the
equal-sized before/after windows.

Low $k^*$ (decoupling at order 1–2) predicts idiosyncratic events; high
$k^*$ or no-decoupling predicts systemic events.

Two API functions are provided. `chow_decoupling` runs a single-midpoint
Chow test on a single z-series (the "flat cascade baseline" used in
early experiments). `chow_decoupling_cascade` runs a sliding-window
Chow test on each order's z-series independently, returning the
smallest $k^*$ where any window rejects. The latter is the formalization
of the per-order Chow test used in the H3 v3–v5 experiment scripts.

## 5. Pre-registered parameters

| Parameter        | Pre-reg value | Sensitivity range     |
|------------------|---------------|-----------------------|
| `orders`         | (1, 2, 3, 4)  | fixed                 |
| `inner_window`   | 10 days       | {5, 10, 20, 40}      |
| `zscore_lookback`| 120 days      | {60, 120, 252}        |
| `forward_days`   | 5 days        | {1, 2, 3, 5, 10, 20} |
| `n_orders`       | 4             | {3, 4}                |

The pre-registered adversarial test parameters: 1000 universes of 5000
iid $\mathcal{N}(0, \sigma^2)$ returns, PASS criterion $|\rho| < 0.05$
in 95%+ of universes; 1000 universes of AR(1)-GARCH(1,1) with
Student-$t$ innovations, mean Spearman $\approx 0$ expected.

## 6. Cross-references

- `docs/DESIGN_MEMO.md` — locked design decisions, pre-registered
  hypotheses, comparison battery, sample scope, multiple testing.
- `results/MECHANISM.md` — the vol-peak mechanism, the GARCH
  relationship, frontier markets, Granger causality.
- `results/h3_v11_summary.md` — H3 v11 model results and the
  days-since-last-earnings calendar caveat.
- `results/reframed_results.md` — the reframed findings for the paper.
- `tests/` — 26 unit tests for the cascade, baselines, and decoupling.

## 7. References

Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough.
*Quantitative Finance*, 18(6), 933–949.

Hamilton, J. D. (1989). A new approach to the economic analysis of
nonstationary time series and the business cycle. *Econometrica*, 57(2),
357–384.

Chow, G. C. (1960). Tests of equality between sets of coefficients in
two linear regressions. *Econometrica*, 28(3), 591–605.

Brunnermeier, M. K., & Pedersen, L. H. (2009). Market liquidity and
funding liquidity. *Review of Financial Studies*, 22(6), 2201–2238.
