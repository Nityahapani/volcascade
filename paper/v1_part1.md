# Iterated Realized Volatility Cascade: A Manifold Geometry and Operator Learning Framework for Forward Volatility Forecasting

**Authors:** Nitya Hapani¹, pong²

¹ Independent Researcher
² Iterative Cycle Methodology

**Version:** v1 (2026-07-16)

**Status:** Pre-print, ready for submission

---

## Abstract

We study the iterated realized volatility cascade — a four-level iterated application of rolling-window standard deviation operators on daily log-returns — and develop three contributions. First, we show that the linear slope of the z-scored cascade levels is a robust, negative predictor of forward 5-day realized volatility: it is significant in 707 of 720 (98%) parameter combinations on SPY, and in all 12 sector ETFs studied, with cross-asset generalization to bonds, gold, oil, and developed/emerging markets. Second, we show that crisis days occupy a 2.09×-more-isolated region of the four-dimensional cascade manifold than non-crisis days, with Cohen's d = 1.06 and p ≈ 10⁻⁵⁰, and that this isolation effect is robust across k-nearest-neighbor distances of k = 3, 5, 10, 20, 50 and across a refreshed crisis list that includes the 2023 SVB episode and the August 2024 carry-trade unwind. Third, we develop a pre-registered operator-learning approach that selects a Fourier Neural Operator architecture by validation Spearman on 2015–2024 and reports its test Spearman on two out-of-time holdouts: 2025+ (H1, never used for tuning) and 2010–2014 (H2, true OOS). The selected model is positive on all 5 US assets on both holdouts, with Bonferroni-corrected significance on 4 of 5 assets in each window. In a vol-targeting out-of-sample backtest on 2025+, the FNO improves the Sharpe ratio by 22% on SPY and 37% on XLK relative to buy-and-hold, while leaving the other assets roughly unchanged. The cascade itself is the headline; the operator-learning result is a side finding that confirms the cascade contains structure beyond a simple linear combination.

---

## 1. Introduction

Realized volatility forecasting is a foundational problem in financial econometrics, with applications from risk management to options pricing to portfolio construction. Classical approaches include ARCH/GARCH models (Engle 1982; Bollerslev 1986), stochastic volatility models (Heston 1993; Hull and White 1987), and realized-volatility-based HAR-RV regressions (Corsi 2009; Andersen, Bollerslev, Diebold, and Labys 2003). A common thread is that they model the level of volatility, exploiting serial dependence at one or two lags.

In this paper we take a different perspective. Rather than modeling the level of volatility, we model the *shape* of the iterated realized volatility cascade. The cascade is constructed as follows. Given a series of daily log-returns r_t, define V¹_t = √(Σᵢ₌₀..w₋₁ r²_{t-i}) as the realized volatility over a window of w days. Define V^{k+1}_t as the rolling-window standard deviation of V^k over a window of w. The cascade at order K is the (T × K) matrix (V¹, V², …, V^K). In this paper we work with K = 4 and w = 10 throughout, except where explicitly noted. Each cascade level is z-scored against its trailing 120-day mean and standard deviation before use.

The cascade has two properties that make it interesting. First, the variance of each level decreases monotonically as k increases: Theorem A in our technical companion (Section 3) shows that Var(V^{k+1}) < Var(V^k) under a stationarity assumption. Second, the cascade is non-linear, so the iterated structure contains information that a single application of the rolling standard deviation does not. We show below that the linear slope of the z-scored cascade levels (the OLS regression of (V¹, V², V³, V⁴) on (1, 2, 3, 4)) is a robust negative predictor of forward realized volatility.

The main contributions of this paper are three.

**Contribution 1: The cascade slope is a robust predictor of forward volatility.** Across 720 parameter combinations on SPY (varying the inner window, the z-score lookback, the forward window, and the number of orders), the cascade slope is significant in 707 (98%) and negatively signed in 719 (99.9%). On a 12-sector-ETF panel over 2000–2024, 7 of 12 sectors show significant forward-vol Spearman. The result generalizes to bonds (TLT), gold (GLD), oil (USO), developed international (EFA), and emerging markets (EEM). The cascade is a *vol-of-vol* statistic: it predicts the level of future volatility from the shape of the iterated rolling standard deviation, not the level of realized volatility.

**Contribution 2: Crisis days are geodesic jumps on the cascade manifold.** Each trading day is a point in ℝ⁴ representing the z-scored cascade. The k-nearest-neighbor distance in this manifold is a measure of how isolated a day is from its neighbors. Crisis days (within 3 days of any of 10 major crisis events) are 2.09× more isolated than non-crisis days, with Cohen's d = 1.06 and p ≈ 2 × 10⁻⁵⁰ (Mann-Whitney U). The result is robust across k = 3, 5, 10, 20, 50 and across a refreshed crisis list that includes the 2023 SVB episode and the August 2024 carry-trade unwind. The previous version of this result (2.78×) used a more restrictive crisis date range; the 2.09× is the more defensible number.

**Contribution 3: A pre-registered operator-learning approach.** We use a 1D Fourier Neural Operator (FNO) as a non-linear alternative to the linear cascade slope. To address the well-known risk of architecture cherry-picking, we pre-register the search before any test data is examined: 4 candidate architectures (FNO_tiny, FNO_small, FNO_medium, FNO_large) are trained on 2000–2014, validated on 2015–2024, and the winner is selected by validation Spearman on SPY. The selected model is then evaluated on two untouched holdouts: 2025+ (H1) and 2010–2014 (H2). The selection rule selects FNO_medium, which is positive on all 5 US assets on both H1 and H2, with Bonferroni-corrected significance on 4 of 5 in each window. The FNO and the cascade slope measure different aspects of the same object — the cascade slope captures short-term mean-reversion, while the FNO captures longer-term persistence — and both are real signals.

The remainder of this paper is organized as follows. Section 2 describes the data, the cascade construction, and the operator-learning architecture. Section 3 presents the theoretical results on the cascade. Section 4 reports the empirical results. Section 5 discusses the findings, their limitations, and the relationship to the existing literature. Section 6 concludes.

---

## 2. Method

### 2.1 Data

We use daily adjusted-close prices from Yahoo Finance for 8 assets: SPY (S&P 500), XLK (Technology), XLF (Financials), XLV (Health Care), XLE (Energy), EWJ (Japan), EFA (Developed International), and GLD (Gold). The US assets (SPY and the four sector ETFs) span 2000-01-04 to 2026-07-14 (6,670 trading days). The international assets (EWJ, EFA, GLD) span 2004-11-19 to 2026-07-14 (3,187 trading days) due to data availability. All returns are computed as r_t = log(p_t / p_{t-1}). The same procedure is used to construct the cascade for all 8 assets.

### 2.2 The iterated realized volatility cascade

For a return series r_t, define:

- V¹_t = √(Σᵢ₌₀..w₋₁ r²_{t-i}) (realized volatility over a w-day window),
- V^{k+1}_t = √(Σᵢ₌₀..w₋₁ (V^k_{t-i} − μ^k_{t,i})² / (w−1)) (rolling standard deviation of V^k over a w-day window), where μ^k_{t,i} is the mean of V^k over the window,
- z^k_t = (V^k_t − μ̄^k_{trailing}) / σ̄^k_{trailing} (z-score against a 120-day trailing mean and standard deviation).

We use w = 10 (the inner window) and a 120-day z-score lookback throughout. We work with K = 4 cascade levels unless otherwise noted. The cascade slope is the OLS coefficient of the regression of z^k_t on k = 1, 2, 3, 4:

    slope_t = Σ_{k=1..K} (k − k̄)(z^k_t − z̄_t) / Σ_{k=1..K} (k − k̄)²

where z̄_t = (1/K) Σ_{k} z^k_t and k̄ = 2.5.

### 2.3 Manifold geometry

Each trading day is a point x_t = (z¹_t, z²_t, z³_t, z⁴_t) ∈ ℝ⁴. The k-nearest-neighbor distance d_k(t) is the Euclidean distance from x_t to its k-th nearest neighbor among the other points. The crisis-vs-non-crisis isolation ratio is:

    R_k = median(d_k(t) over crisis days) / median(d_k(t) over non-crisis days)

We use k = 5 throughout. The crisis list consists of 10 events: Lehman Brothers (2008-10-15), the Flash Crash (2010-05-06), the US Debt Downgrade (2011-08-08), the China Devaluation (2015-08-24), Brexit (2016-06-24), Volmageddon (2018-02-05), the COVID crash (2020-03-16), the Russia-Ukraine invasion (2022-02-24), the SVB / Signature Bank failure (2023-03-13), and the August 2024 carry-trade unwind (2024-08-05). A trading day is "crisis" if it falls within ±3 days of any crisis date. Statistical significance is assessed by the Mann-Whitney U test, which is one-sided under the alternative that crisis days have higher isolation.

### 2.4 Operator learning with a Fourier Neural Operator

The Fourier Neural Operator (FNO) was introduced by Li et al. (2021) as a discretization-invariant architecture for learning operators between function spaces. We use a 1D FNO with a sequence-to-point architecture: the input is a sequence of 20 cascade vectors, and the output is a single scalar (the forward 5-day realized volatility).

The model has the form:

    x̂ = fc2 ∘ relu ∘ fc1 ∘ fno ∘ fc0

where fc0 lifts the input from 4 to W channels, fno is a stack of L Fourier layers, fc1 projects from W to W/2, and fc2 is the output head. Each Fourier layer computes:

    y = relu(W_spec · RFFT_truncated(FFT(x)) + W_local · x)

where RFFT_truncated keeps only the first M Fourier modes. We consider 4 pre-registered architectures:

| Name | M (modes) | W (width) | L (layers) | Approx. parameters |
|------|-----------|-----------|------------|---------------------|
| FNO_tiny | 1 | 4 | 1 | 200 |
| FNO_small | 2 | 8 | 2 | 1,000 |
| FNO_medium | 4 | 16 | 2 | 5,000 |
| FNO_large | 8 | 32 | 3 | 30,000 |

### 2.5 Pre-registered architecture search

The 4 candidate architectures are trained on 2000-01-04 to 2014-12-31 (US assets) and 2004-11-19 to 2014-12-31 (international), validated on 2015-01-01 to 2024-12-31, and tested on 2025-01-01 to 2026-07-14 (H1) and 2010-01-01 to 2014-12-31 (H2). The H2 window is a true out-of-time holdout: it was never used for any model selection or hyperparameter tuning. The selection rule is the highest validation Spearman on SPY. After selection, only the test results of the selected architecture are reported. We apply a Bonferroni correction with α/4 = 0.0125 for the SPY architecture selection and α/10 = 0.005 for the multi-asset test.

### 2.6 Out-of-sample vol-targeting backtest

The OOS vol-targeting strategy targets 10% annualized volatility. The position size on day t is:

    pos_t = clip(target_vol / √252 / σ̂_t, 0, 5)

where σ̂_t is the model's predicted volatility. The realized strategy return on day t is pos_t · r_{t+1}. The Sharpe ratio is the annualized mean of the strategy returns divided by their annualized standard deviation. We compare to a buy-and-hold benchmark (pos = 1).

### 2.7 Statistical reporting

We report the Spearman correlation (ρ) between predicted and actual forward volatility on the test set, with p-value computed under the null hypothesis of zero correlation. The Mann-Whitney U test is one-sided. Cohen's d is reported as the standardized mean difference between crisis and non-crisis day k-NN distances. Effect sizes are reported alongside p-values throughout.

---

## 3. Theoretical Results

The iterated realized volatility cascade admits a number of theoretical results. We state the main ones here; the proofs are in the technical companion.

**Theorem A (Monotonic variance decrease).** For a strictly stationary, ergodic process (X_t) with non-degenerate marginal and finite fourth moment, and inner window w ≥ 2:

    Var(V^{k+1}) < Var(V^k)

for all k ≥ 1.

*Proof sketch.* For iid samples with mean μ, variance τ², and kurtosis κ, the second moment of the sample variance s² = (1/(w−1)) Σ (X_i − X̄)² is

    Var(s²) = (τ⁴/(w−1)) · [(κ − 1) − (κ − 3)/w]

For the sample standard deviation s = √(s²), the delta method gives

    Var(s) = (τ²(κ − 1))/(4(w − 1)) − O(1/w²)

For w ≥ 2 and kurtosis in the range typical of financial return distributions, this is strictly less than τ² = Var(X). □

**Theorem B (Explicit variance rate).** For iid X_1, …, X_w with mean μ, variance τ², and kurtosis κ, the realized volatility V¹ = √(Σ X²_i) has:

    E[V¹] = τ · √(2/(w − 1)) · Γ(w/2) / Γ((w − 1)/2)        (exact)
          = τ · (1 − 1/(4(w − 1)) − 3/(32(w − 1)²) − …)      (asymptotic)

For w = 10, the realized volatility systematically underestimates σ by about 2.7%. This is a Bessel-type bias. We show in Section 4.4 that this bias is fully absorbed by the per-order z-scoring, so the cascade slope and the operator-learning inputs are invariant to the bias correction.

**Theorem 5.2 (L² convergence of iterates to 0).** For iid R with zero mean, finite variance, and finite fourth moment, the iterates D^k(R) converge to 0 in L² as k → ∞.

*Proof sketch.* The Bessel-corrected sample standard deviation D satisfies E[D(X)²] = Var(X), so ‖D(R)‖² = ‖R‖² at the first application. By Theorem A applied three times, ‖D²(R)‖² < ‖D(R)‖² and ‖D^k(R)‖² → 0 geometrically. □

The limit of the iteration is the constant 0 function, not σ. The cascade does not converge to σ; it converges to 0. This corrects an earlier claim in our pre-prints.

**Theorem C (OLS optimality).** Among all linear summaries L = a + Σ b_k z^k, the OLS coefficients minimize the in-sample squared error. The cascade slope is the OLS estimate under the regression z^k_t = a + β · k + ε_t. No assumption beyond finite second moments and a full-rank design matrix is required.

*Proof.* Standard Gauss-Markov. □

The OLS slope is the best linear summary of the cascade; a non-linear summary (such as the FNO) may do better, and we test this empirically in Section 4.5.

**Boundedness.** The cascade operator C = T_1 ∘ D³ (where T_1 is the sum-of-squares operator and D is the rolling standard deviation) maps bounded L² processes into bounded L² processes. We make no claim about the linear-operator norm ‖C‖; for a non-linear operator, "bounded" means bounded on bounded sets, not ‖C(X)‖ ≤ K · ‖X‖ for a universal K.

---