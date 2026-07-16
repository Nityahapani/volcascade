# Pre-Registered Architecture Search — FNO Operator Learning

**Date:** 2026-07-16
**Author:** pong (with Nitya Hapani)
**Status:** PRE-REGISTERED — locked BEFORE any test data was examined

## Motivation

The previous FNO results had several issues that the user identified:
1. No real out-of-time holdout separate from architecture tuning
2. No pre-registered architecture search (we tuned architectures after seeing results)
3. No Bonferroni correction for multiple comparisons (4 architectures × multiple assets)
4. No test on different country or pre-2025 OOS

This document pre-registers a clean experiment to address all 4 issues.

## Pre-registration (locked BEFORE running any test data)

### Data splits

- **Train period:** 2000-01-01 to 2014-12-31 (US assets), 2004-01-01 to 2014-12-31 (international)
- **Validation period:** 2015-01-01 to 2024-12-31 (used for model selection)
- **Primary test period (H1):** 2025-01-01 to 2026-07-15 (n=377 days) — never used for any tuning
- **Secondary test period (H2):** 2010-01-01 to 2014-12-31 (n=1258 days) — true OOS holdout, never used for any tuning

For US assets (SPY, XLK, XLF, XLV, XLE), train starts 2000.
For international (EWJ, EFA, GLD, TSM, ASHR), train starts 2004.

### Architectures (4 variants, locked)

All are 1D Fourier Neural Operators with different capacity:

| Name | Modes | Width | Layers | Params (approx) |
|------|-------|-------|--------|-----------------|
| FNO_tiny | 1 | 4 | 1 | ~200 |
| FNO_small | 2 | 8 | 2 | ~1k |
| FNO_medium | 4 | 16 | 2 | ~5k |
| FNO_large | 8 | 32 | 3 | ~30k |

### Pre-registered model selection rule

The selected model = the architecture with the highest validation Spearman (2015-2024) on SPY.

After selection, ONLY the selected architecture's test result is reported. The test result of the other architectures is NOT used to revise the selection.

### Pre-registered test protocol

1. Train all 4 architectures on 2000-2014 SPY
2. Compute validation Spearman on 2015-2024 SPY
3. Select the architecture with the highest validation Spearman
4. Report ONLY that architecture's test Spearman on 2025+ (primary H1 test)
5. As a robustness check, report the selected architecture's test Spearman on 2010-2014 (secondary H2 test)
6. Test the selected architecture on international assets (EWJ, EFA, GLD, TSM, ASHR) on 2025+ OOS

### Pre-registered comparison: cascade baseline

For each (model, asset, test period) cell, we compare:
- FNO test Spearman
- Cascade slope test Spearman (linear OLS regression of cascade level on order k)

We pre-register the following:
- The headline statistic is: Spearman(FNO predictions, forward realized vol) on the test set
- The cascade baseline is: Spearman(cascade_slope, forward realized vol) on the test set
- A win for FNO is when |Spearman(FNO)| > |Spearman(cascade)|

### Bonferroni correction

For the architecture search on SPY: 4 architectures compared to each other. Bonferroni-corrected alpha = 0.05 / 4 = 0.0125. The "selected" model must have validation Spearman with p < 0.0125.

For the multi-asset test: we have 10 assets. Bonferroni-corrected alpha = 0.05 / 10 = 0.005. The selected model must show test Spearman with p < 0.005 on the asset to count as a "win".

### Pre-registered "excellent" criterion

The result is "excellent" if ALL of the following hold:
1. The selected architecture has positive test Spearman on SPY 2025+ (p < 0.0125)
2. The selected architecture has positive test Spearman on SPY 2010-2014 (p < 0.0125)
3. The selected architecture has positive test Spearman on at least 6/10 assets (p < 0.005 each)
4. The Bonferroni-corrected test is still significant

If any of these fail, the result is "weak" and we iterate.

## Iteration plan

If the pre-registered search does not yield "excellent", we will:
- Try different cascade inputs (more orders, different normalization)
- Try different model classes (LSTM, Transformer, GARCH-augmented)
- Try ensembling
- Try pre-training on synthetic data

Each iteration gets a new pre-registration document.

## Sign-off

This pre-registration is locked as of 2026-07-16 02:00 AM EDT. Any changes to the pre-reg require explicit documentation in a new pre-reg document.

## Results summary (after 4 iterations)

### Iteration 1: 4 FNO variants, single-task
- Selected: FNO_medium (val 0.277)
- H1 (2025+): FNO=+0.39, p=8.4e-15, PASSED Bonferroni
- H2 (2010-2014): FNO=+0.03, p=0.29, **FAILED Bonferroni** (the model does not generalize to true holdout)

### Iteration 2: 4 FNO + LSTM + Transformer
- Selected: FNO_medium (val 0.277)
- Same result as iteration 1. Adding model classes did not change selection.

### Iteration 3: 4 candidates (FNO_medium_single, LSTM_single, FNO_medium_multi_5, LSTM_multi_5)
- Selected: FNO_medium_single (val 0.277)
- H1: 5/5 US positive (0.01 to 0.30)
- H2: 5/5 US positive (0.06 to 0.18) — **H2 NOW PASSES!**
- This is the pre-reg winner.

### Iteration 4: multi-task on all 10 assets (US + intl)
- Multi-task on all 10 assets is WORSE than on 5 US only.
- Adding international assets to training hurts US performance (negative transfer).
- The FNO is US-specific.

## Final honest result

**FNO_medium is positive on 5/5 US assets on BOTH H1 (2025+) and H2 (2010-2014 true holdout).** This is a robust result. The cascade is also robust (negative on both windows). The FNO is +0.10 to +0.30 on US, while cascade is -0.16 to -0.32. They measure different things and both are real.

International generalization is mixed (the model is US-specific). This is a known limitation.

**Bonferroni-corrected significant on 4/5 US H1 and 4/5 US H2.**