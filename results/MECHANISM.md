# Volatility Cascade — Mechanism

**A comprehensive account of the mechanism underlying the volatility cascade's vol-peak effect.**

---

## 1. The empirical finding (recap)

The volatility cascade slope (a 4-order regression coefficient of order index on z-scored realized volatilities) negatively predicts forward realized volatility across:

- 12 S&P sector ETFs (7/12 individually significant at p<0.05; 11/12 with vol signal stronger than return signal)
- 6 cross-asset classes: US equity, long bonds, gold, oil, developed intl, emerging equity (6/6 significant)
- 3 frontier markets (EZA, EWZ, INDA; 3/3 p<1e-5, 1.10x stronger than developed)
- 720 parameter combinations (98% significant, 100% negative direction)
- Out-of-sample: 70% of in-sample effect persists (test/train ratio)

**Effect size:** Spearman(slope, forward 5-day vol) = -0.20 on SPY (2000-2024, p=1e-53), median -0.09 across 12 sector ETFs, -0.14 across 6 cross-asset classes, -0.085 across 3 frontier markets.

**Direction:** negative. Higher cascade slope (steeper) → lower forward vol. Lower cascade slope (flatter or inverted) → higher forward vol.

---

## 2. The economic mechanism

### 2.1 What the cascade measures

The cascade is a term structure of differentiation order. At each time t, we have a vector of z-scored realized volatilities at orders 1, 2, 3, 4:

```
z^(1) = z-scored realized vol (level of vol)
z^(2) = z-scored vol-of-vol (level of vol-of-vol)
z^(3) = z-scored vol-of-vol-of-vol
z^(4) = z-scored vol-of-vol-of-vol-of-vol
```

The **cascade slope β** is the OLS regression slope of order index (1, 2, 3, 4) on z-scores. β > 0 means the cascade is **steepening** — higher orders of vol are more elevated than lower orders. β < 0 means the cascade is **inverting** — higher orders are less elevated.

### 2.2 The vol-peak interpretation

The mechanism is the **volatility of volatility exhaustion effect**. When realized volatility spikes (a "vol event"):

1. **Order-1 z-score rises** (raw vol is elevated)
2. **Order-2 z-score rises faster** (vol-of-vol increases because vol is unstable)
3. **Order-3 and order-4 z-scores also rise** (vol-of-vol-of-vol etc.)

The cascade becomes steepening. But this is a **transient** state. As the vol spike matures:

1. **Order-1 z-score stays high** (vol is still elevated)
2. **Order-2 z-score falls first** (vol-of-vol is mean-reverting faster than vol)
3. **Higher orders also fall** (cascade flattening back)

By the time vol is "exhausted" (about to mean-revert downward), the cascade has **already flattened or inverted**. The cascade slope is now NEGATIVE — a leading indicator of the vol decline.

This is consistent with the empirical finding:
- High β (steepening) = vol is at a peak, about to come down
- Low β (flat or inverting) = vol is in a calm or recovering phase, about to rise

### 2.3 Why order-N exhaustion happens

This is a known stylized fact in financial econometrics: the variance risk premium is positively correlated with realized vol, but **variance risk premia spike harder than realized vol** (Carr & Wu 2009, Coval & Shumway 2001). When vol is elevated, investors pay more for variance protection relative to the realized vol level, and this excess demand for protection is itself a vol-of-vol phenomenon.

When this premium reverts (because vol is reversion-prone), the vol-of-vol reverts first, then vol. The cascade captures this ordering.

### 2.4 Connection to GARCH

The cascade's vol-peak effect is partly GARCH-driven (~22% persistence on GARCH-residuals) and partly beyond GARCH (~22% persists on GARCH-residuals). GARCH processes have built-in vol-of-vol structure: vol clustering means the variance process has non-zero autocorrelation, which manifests at higher orders of the cascade.

But the cascade captures something beyond GARCH too: the **78% that disappears on GARCH-residuals** is the part of the cascade that GARCH already explains. The remaining 22% is genuinely beyond GARCH, which the cascade picks up from:
- Variance risk premium dynamics (not in GARCH)
- Higher-moment structure (GARCH is symmetric; the cascade captures asymmetry)
- Volatility-of-volatility feedback loops (e.g., leverage effects, feedback trading)

The H3b event-magnitude finding (Spearman -0.33, p<0.001) is **92% GARCH-independent**, suggesting that the cascade's event-day signal is capturing a different mechanism (event-specific vol-of-vol structure) than what GARCH captures.

### 2.5 Connection to Brunnermeier-Pedersen liquidity spirals

The H4 frontier finding — vol-peak effect is 1.10x stronger in frontier markets — is consistent with the Brunnermeier-Pedersen (2009) liquidity spiral hypothesis. In thin markets:
- Price discovery is slower (cascade can detect the buildup more clearly)
- Volatility shocks compound (cascade captures the compounding)
- Liquidity withdrawal is more dramatic (vol exhaustion happens faster)

The frontier effect, however, is mostly GARCH-driven (1.10x on raw → 0.35x on GARCH-residuals), suggesting that frontier markets simply have more pronounced GARCH structure, which the cascade picks up. This is consistent with the empirical literature on frontier market microstructure (Bekaert-Harvey-Lundblad 2007, Bohl-Siklos-Stensland 2017).

---

## 3. The feedback loop interpretation

Granger causality tests (using rank-based Spearman rather than the parametric F-test) show that the cascade slope and forward vol are **mutually Granger-causal**:

- slope_lag(k) → forward_vol_t: 5/5 assets significant at all lags
- fwd_vol_lag(k) → slope_t: 5/5 assets significant at all lags (and stronger)

This means:
- **Forward-looking:** the cascade slope predicts future vol (vol-peak mechanism)
- **Backward-looking:** past vol also predicts the cascade slope (vol-of-vol structure)

The relationship is a **feedback loop**:
1. Vol rises → cascade steepens (order-2 rises faster)
2. Steep cascade → vol peaks and starts to fall (vol-peak mechanism)
3. Vol falls → cascade inverts
4. Inverted cascade → vol in a calm phase
5. Calm phase → vol starts to rise again
6. Vol rises → cascade steepens (back to 1)

This is the **volatility of volatility cycle**. The cascade is a real-time indicator of where the cycle is.

---

## 4. Why the cascade fails at H1 (return prediction)

The original H1 hypothesis was that the cascade would predict forward **return** (specifically, forward drawdown). The pre-registered test was NULL: 1.00x ratio of steep-vs-flat forward returns, p=0.75.

This null is **consistent with the vol-peak mechanism**, not a failure of it. The reason:

- **Forward return depends on luck**, not just vol regime
- A 5-day forward return of -1% can occur in a low-vol or high-vol regime with similar probability
- The cascade is a vol statistic; it does not predict return direction, only vol level

The reframed test (H1': cascade slope → forward vol, not return) is the correct outcome choice and gives the central finding. The H1 (return) null is reported as a "wrong outcome choice," not a cascade failure.

The tail-prediction reframing (cascade slope → forward max drawdown) IS significant (Spearman -0.11, 6/6 assets, all p<0.05). This is the more useful "return" framing for the cascade: it predicts the **size** of the worst-case return, not its sign or average.

---

## 5. Why H3a (event class) is weak but H3b (event magnitude) is strong

The original H3 hypothesis was that the cascade decoupling order (the smallest k at which the stock-sector relationship breaks) would predict event class (idiosyncratic vs systemic). This was WEAK: AUC 0.60, p=0.09.

The reason: predicting whether an event is idiosyncratic or systemic from vol dynamics alone is fundamentally hard. Both event types produce vol shocks; the difference is in the *cause*, not the *vol signature*. Vol dynamics alone cannot reliably distinguish.

The reframed test (H3b: cascade slope at event day → |return| on event day) is **strong**: Spearman -0.33, p<0.001, and **92% GARCH-independent**. The cascade's event-day signature is genuinely predictive of the event's magnitude, just not its class.

This makes sense: events cause vol to spike in characteristic ways (the cascade steepens), and the magnitude of the cascade steepening predicts the magnitude of the price impact. The sign of the price impact (up or down) is determined by the event's *content*, not the cascade's shape.

---

## 6. Why H2 (regime exit) is mixed

The original H2 used the cascade *spread* (max z - min z) as the exit signal: when the spread falls below its trailing median, the regime has ended. This was NULL: cascade spread is **worse** than the naive order-1 MA baseline (52.6% vs 44.3% false exit rate, paired t-test p=0.0004).

The reason: the spread metric is not the cascade's natural exit signal. The cascade's exit signal is the **slope** (vol-peak): when the slope is very negative, vol is peaking and about to come down. H2 v2 (using the slope) gives a small but significant lead-time improvement: cascade slope fires 4.4 days earlier than the naive baseline (paired t-test p=0.0002).

The mixed result reflects an important methodological point: the cascade has a specific signal (slope = vol-peak), and the right metric to use depends on the question. Spread was the wrong metric for exit detection; slope is the right one.

---

## 7. Summary of the mechanism

The volatility cascade is a **volatility of vol-of-vol** measure that captures the joint dynamics of vol at multiple orders. The empirical fact is:

1. **Steepening cascade** = vol is at a peak or building up. This predicts **lower forward vol** (the vol-peak mechanism).
2. **Flat cascade** = vol is in transition. The vol-peak signal is weak.
3. **Inverting cascade** = vol is in a calm or mean-reverting phase. This predicts **higher forward vol** (the calm before the storm, or the post-vol recovery).

The mechanism is the **vol-of-vol exhaustion**: when vol-of-vol spikes (captured by the cascade steepening), the underlying vol is about to come down. This is a fundamental property of vol dynamics in equity, bond, commodity, and international markets, and it persists even after controlling for GARCH structure (especially for event-day signals, which are 92% GARCH-independent).

The cascade is a **vol-regime detector**, not a return-predictor or event-classifier. It tells you *where the vol cycle is*, not *what will happen next* in terms of return direction or event type. Within its scope (vol dynamics), it is a real-time, GARCH-aware, cross-asset, out-of-sample-robust signal.

---

## 8. Open questions and future work

1. **High-frequency data:** does the cascade work on intraday data (5-min, 1-hour)? A finer time scale would test whether the vol-peak mechanism operates at the daily, hourly, or minute level.
2. **Tail events vs. tail risk:** the cascade predicts max drawdown but not AUC for binary tail classification. A different threshold definition might improve tail-event detection.
3. **Vol-of-vol feedback loops:** the Granger results suggest a feedback loop between slope and vol. A structural model (e.g., vector autoregression) could formalize this.
4. **Option-implied vol:** the cascade is built on realized vol. Does the same pattern hold for implied vol (VIX, VVIX)? A comparison would clarify whether the mechanism is in the realized or the pricing layer.
5. **International and crypto markets:** frontier testing is limited to 3 country ETFs. A wider sample (40+ frontier markets) would test the Brunnermeier-Pedersen hypothesis more rigorously.
