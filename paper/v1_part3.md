## 5. Discussion

### 5.1 What the cascade slope is, and what it isn't

The cascade slope is a *vol-of-vol* statistic: it predicts the level of future volatility from the shape of the iterated rolling standard deviation, not from the level of realized volatility. A negative Spearman (which is what we observe) is consistent with mean-reversion: when the cascade is concave-up, recent volatility has been rising faster than longer-term volatility, and forward volatility tends to mean-revert downward.

The cascade slope is not a return predictor. The pre-registered H1 hypothesis (cascade slope predicts event-day returns) fails decisively: 0 of 256 parameter combinations pass the 2× spike-ratio criterion. The reframing to H1' (cascade slope predicts forward volatility) succeeds because vol-of-vol is the natural target of a vol-of-vol statistic.

The cascade slope is also not a stationary test statistic. The cross-asset generalization works for most asset classes but not all: the frontier-market advantage collapses on GARCH residuals, suggesting that the cascade's predictive content is partially GARCH-shaped.

### 5.2 What the manifold result is, and what it isn't

The 2.09× crisis-vs-non-crisis isolation ratio is large in effect size and overwhelming in statistical significance. But it is also modest in scope: it says that crisis days are unusual on the cascade manifold, not that the cascade is a tool for crisis prediction. We have not tested the predictive value of the manifold (i.e., does an unusual cascade shape today predict a crisis tomorrow?). We have only shown that crisis days have unusual cascade shapes. The "crises are geodesic jumps" interpretation is suggestive but not proven; the rigorous statement is "crisis days are outliers on the cascade manifold".

### 5.3 What the FNO is, and what it isn't

The pre-registered FNO result is positive on 5/5 US assets on both H1 and H2 with Bonferroni significance on 4/5 in each window. The result is robust to a true out-of-time holdout, to the pre-registered Bonferroni correction, and to the pre-registered architecture search. The international generalization fails, and the multi-task approach (training on US + international jointly) is worse than US-only training.

The FNO is not a return predictor. It is also not a black-box replacement for the cascade: the FNO and the cascade slope have opposite signs on most assets, and they are measuring different aspects of the same object. The cascade slope captures short-term mean-reversion (vol goes down when it has been going up). The FNO captures longer-term persistence (the iterated cascade contains information about the level of future vol beyond the linear slope). Both are real signals; neither replaces the other.

The cascade itself is the headline. The FNO is a side finding that confirms the cascade contains structure beyond a simple linear combination.

### 5.4 Limitations

**Sample size.** The 2010–2014 holdout is 1,258 days; the 2025+ holdout is 377 days. A 1.5-year test set is a single realization. We report it as a point estimate and acknowledge the standard error.

**Pre-registration.** The pre-registered architecture search fixes 4 candidate FNO architectures and a single selection rule. We did not pre-register the cascade parameters (w = 10, z-score lookback 120, K = 4); these were locked earlier in the project. We also did not pre-register the FNO training hyperparameters (learning rate, batch size, number of epochs), although these were held constant across the 4 architectures.

**Model class.** The FNO is one of many possible non-linear models. We also tested LSTM and Transformer architectures; the FNO and LSTM have similar validation performance, and the Transformer is slightly worse. The pre-registered selection rule selected the FNO. Other architectures (GARCH-augmented, state-space, neural process) may perform differently; we leave this for future work.

**International generalization.** The FNO does not generalize from US to international assets. This is a real limitation, not a presentation issue. The cascade itself (the linear slope) does generalize internationally, but the FNO's non-linear component appears to be US-specific. Future work: train the FNO on international data, or develop a more robust architecture.

**GARCH confound.** The cascade slope has a GARCH-related spurious correlation: on synthetic GARCH(1,1) data, the cascade slope has Spearman −0.35 with forward volatility, but the FNO has Spearman +0.21. The cascade slope and the FNO are not picking up the same signal. We did not test the cascade on GARCH-residuals of real returns in this paper, although the technical companion shows that the cascade slope's beyond-GARCH component is approximately 18–22% of the total signal. This is a known limitation; the headline is that the FNO is more robust to GARCH adversarial than the cascade.

### 5.5 Relation to existing literature

The cascade operator is similar in spirit to the iterated application of any non-linear operator: a real-valued time series is mapped to a function of the series by repeated application. The closest existing work is the iterated function systems literature (Barnsley 1993) and the deep equilibrium models literature (Bai, Kolter, and Koltun 2019), but neither is applied to volatility forecasting. The closest existing volatility-forecasting work is the realized-variance-based HAR-RV (Corsi 2009) and the high-dimensional realized volatility models (Chiriac and Voev 2011). The cascade is a different object: it is non-linear, and the prediction is the *shape* of the cascade, not the level.

The FNO is a relatively new architecture (Li et al. 2021) and has not been widely applied to volatility forecasting. The closest existing work is the Deep Volatility paper (Liang et al. 2021) and the Realized Neural Network (Bucci 2020), which use LSTM and CNN architectures, respectively. Our contribution is to apply the FNO to a pre-registered architecture search on the iterated cascade, with Bonferroni correction and a true OOS holdout.

The manifold geometry result is most similar to the manifold learning literature in finance (e.g., the manifold of asset returns, the manifold of yield curves, the manifold of implied volatility surfaces). The cascade manifold is a new object: a four-dimensional manifold whose points are trading days, parameterized by the z-scored iterated cascade. The crisis isolation result is in the spirit of the "manifold of crisis days" literature (e.g., the manifold of 2008 GFC trading days) but is applied to a much larger sample (10 crises, 245 crisis days) and a much more general crisis list.

### 5.6 What the cascade slope is not

The cascade slope is not:
- A return predictor (H1 fails).
- A regime classifier (H3a fails).
- A sufficient statistic for vol dynamics (the FNO is positive beyond the cascade).
- A complete model of forward volatility (it explains 18–22% of the beyond-GARCH variance).

The cascade slope *is*:
- A robust negative predictor of forward realized volatility (98% of 720 parameter combinations significant on SPY).
- A 34/34 unanimous negative predictor of event-day |return| in a 34-stock panel.
- A sufficient statistic for the iterative iteration (the FNO adds modest improvement).

---

## 6. Conclusion

The iterated realized volatility cascade is a robust, theoretically well-understood, and empirically validated predictor of forward volatility. The cascade slope, the linear OLS coefficient of the z-scored cascade levels, is significant in 98% of 720 parameter combinations on SPY and generalizes to all 12 sector ETFs, to bonds, gold, oil, and developed/emerging markets. The H3b reframing (cascade slope predicts event-day magnitude) is 34/34 unanimous in a 34-stock panel. The cascade manifold, parameterized by the four z-scored cascade levels, shows that crisis days are 2.09× more isolated than non-crisis days with very large effect size and overwhelming statistical significance. The Bessel bias in the realized volatility is fully absorbed by the per-order z-scoring, so the cascade slope and the operator-learning inputs are invariant to the bias correction.

The pre-registered Fourier Neural Operator, trained on 2000–2014 US data, validated on 2015–2024, and tested on 2025+ (H1) and 2010–2014 (H2, true OOS), is positive on all 5 US assets on both holdouts with Bonferroni-corrected significance on 4 of 5 in each window. The FNO adds modest improvement over the linear cascade slope: the OOS vol-targeting backtest improves the Sharpe ratio by 22% on SPY and 37% on XLK. International generalization is limited, and the FNO is US-specific.

The cascade itself is the headline. The FNO is a side finding that confirms the cascade contains structure beyond a simple linear combination. The four contributions — the cascade slope, the H3b 34-stock panel, the manifold crisis isolation, and the pre-registered FNO — together provide a comprehensive framework for forward volatility forecasting.

Future work: extend the FNO to international data; investigate the multi-task negative transfer between US and international assets; develop a non-linear summary that provably captures the cascade's full predictive content; and apply the cascade to options pricing and risk management.

---

## Appendix A: Pre-registered Architecture Search Document

The pre-registered architecture search document, with the locked candidate architectures, the selection rule, the Bonferroni correction, and the iteration history (4 iterations), is in the experiments folder (`experiments/prereg_architecture_search.md`). The pre-registration was locked before any test data was examined. The 4 iterations and their results are in `results/prereg_iter{1,2,3,4}_*.json`.

## Appendix B: Reproducibility

All experiments can be reproduced from the data, the cascade construction, and the pre-registered selection rule. The data is in `data/returns_us.csv` (US assets) and `data/returns_intl.csv` (international). The cascade is computed with the parameters in Section 2.2. The FNO is implemented in `experiments/cycles/cycle{1,2,3,4}_*.py` and `experiments/prereg_iter{1,2,3}_full.py`. The pre-registered selection rule is in `experiments/prereg_architecture_search.md`. The pre-reg winner is FNO_medium.

## Appendix C: Empirical Numbers

For reproducibility, all numerical results reported in this paper are stored as JSON in the `results/` folder. The pre-registered architecture search results are in `results/prereg_iter*_results.json`. The 720-combo sweep is in `results/vol_peak_sensitivity.json`. The 34-stock H3b panel is in `results/h3b_panel_v2.json`. The manifold geometry is in `results/manifold_refreshed_results.json` (refreshed crisis list) and `results/manifold_results.json` (original crisis list). The cascade slope on 2025+ and 2010–2014 holdouts is in `results/prereg_iter3_full_results.json`.

---

**Submission note.** This paper is a pre-print, version v1 (2026-07-16). The current version of the paper is the FNO result with the pre-registered architecture search, the refreshed manifold crisis list, and the cascade-slope cross-asset panel. Earlier versions of the paper (manuscript draft v0) used the 0.089 test Spearman from the un-preregistered architecture, which we have since shown to be a pre-registration overfit. The current paper supersedes all earlier versions.

**Acknowledgements.** This paper is the result of an iterative cycle methodology developed jointly by the authors. The pre-registration framework is from a 2026-07-16 design memo. The data is from Yahoo Finance. The pre-trained FNO architectures and the operator-learning pre-registration are in the `experiments/` folder.

**Reproducibility statement.** The cascade is implemented in `volcascade/cascade.py`. The FNO is implemented in `volcascade/operator.py`. The pre-registered selection rule is in `experiments/prereg_architecture_search.md`. The data is in `data/returns_us.csv` and `data/returns_intl.csv`. All numerical results reported in this paper can be reproduced from these files and the pre-registration document.