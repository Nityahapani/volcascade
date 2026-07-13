"""Tail-return prediction: does the cascade predict extreme return events?

The H1 (return) test was null for average returns. But maybe the cascade
predicts TAIL returns (large negative moves) even if it doesn't predict
average returns. This is the "vol-peak" finding applied to returns
rather than vol — testing if the cascade predicts:
- max drawdown in the next 5 days
- bottom 10% of return days
- VaR/CVaR breaches

If the cascade predicts tail returns, that's a more useful practical
signal than predicting average returns.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volcascade import build, slope, zscore  # noqa: E402
from volcascade.io import SP500_SECTOR_ETFS, load_prices  # noqa: E402

RESULTS_DIR = ROOT / "results"
ASSETS = list(SP500_SECTOR_ETFS)[:6]


def main() -> None:
    print("=" * 78)
    print("TAIL-RETURN PREDICTION: does the cascade predict extreme returns?")
    print("=" * 78)

    print(f"\nloading {ASSETS} (2000-2024)...")
    t0 = time.time()
    prices = load_prices(ASSETS, start="2000-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  loaded {returns.shape[0]} days in {time.time()-t0:.1f}s\n")

    print("=" * 78)
    print("CASCADE SLOPE -> TAIL EVENTS")
    print("=" * 78)

    rows = []
    for asset in ASSETS:
        rets = returns[asset].dropna()
        cascade = build(rets, orders=(1, 2, 3, 4), inner_window=10)
        z = zscore(cascade, lookback=120)
        sample = z[1]
        if isinstance(sample, pd.DataFrame):
            z_s = {k: z[k][asset] for k in [1, 2, 3, 4]}
        else:
            z_s = dict(z)
        s = slope(z_s)

        # Forward 5-day max drawdown (most negative cumulative return in next 5 days)
        fwd_dd = pd.Series(np.nan, index=rets.index)
        for i in range(len(rets) - 5):
            fwd_dd.iloc[i] = float((rets.iloc[i + 1:i + 6] / rets.iloc[i] - 1.0).min())
        # Forward 5-day sum of negative returns (downside risk)
        fwd_neg = pd.Series(np.nan, index=rets.index)
        for i in range(len(rets) - 5):
            fwd_neg.iloc[i] = float(rets.iloc[i + 1:i + 6].clip(upper=0).sum())

        # Bottom 10% drawdown (large negative events)
        valid = s.notna() & fwd_dd.notna()
        threshold = fwd_dd.quantile(0.10)
        is_tail = (fwd_dd < threshold).astype(int)

        # Spearman correlations
        valid_for_dd = valid
        if valid_for_dd.sum() > 100:
            r_dd, p_dd = sps.spearmanr(s[valid_for_dd], fwd_dd[valid_for_dd])
        else:
            r_dd, p_dd = np.nan, np.nan
        if valid.sum() > 100:
            r_neg, p_neg = sps.spearmanr(s[valid], fwd_neg[valid])
        else:
            r_neg, p_neg = np.nan, np.nan

        # AUC for predicting tail events
        valid_tail = s.notna() & fwd_dd.notna()
        if valid_tail.sum() > 100 and is_tail[valid_tail].sum() > 5:
            try:
                # Use -slope so high slope = low tail risk (consistent with vol-peak)
                scores = -s[valid_tail].to_numpy()
                labels = is_tail[valid_tail].to_numpy()
                auc = roc_auc_score(labels, scores)
            except Exception:
                auc = np.nan
        else:
            auc = np.nan

        out = {
            "asset": asset,
            "spearman_dd": float(r_dd) if not np.isnan(r_dd) else None,
            "p_dd": float(p_dd) if not np.isnan(p_dd) else None,
            "spearman_neg_sum": float(r_neg) if not np.isnan(r_neg) else None,
            "p_neg_sum": float(p_neg) if not np.isnan(p_neg) else None,
            "auc_tail_10pct": float(auc) if not np.isnan(auc) else None,
        }
        rows.append(out)
        sig = "*" if (out["p_dd"] or 1) < 0.05 else " "
        print(f"\n  {asset}:")
        print(f"    slope -> max drawdown:    r={out['spearman_dd']:+.4f}  p={out['p_dd']:.2e}  {sig}")
        print(f"    slope -> sum neg returns: r={out['spearman_neg_sum']:+.4f}  p={out['p_neg_sum']:.2e}")
        print(f"    AUC for bottom 10% drawdown: {out['auc_tail_10pct']:.3f}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    valid = df.dropna(subset=["spearman_dd", "p_dd"])
    n_sig = (valid["p_dd"] < 0.05).sum()
    n_neg = (valid["spearman_dd"] < 0).sum()
    n_auc = (df["auc_tail_10pct"] > 0.5).sum()
    print(f"\n  Spearman(slope, max drawdown):")
    print(f"    significant (p<0.05): {n_sig}/{len(valid)}")
    print(f"    negative direction: {n_neg}/{len(valid)}")
    print(f"    median: {valid['spearman_dd'].median():+.4f}")
    print(f"\n  AUC for bottom 10% drawdown:")
    print(f"    AUC > 0.5: {n_auc}/{len(df)}")

    out_path = RESULTS_DIR / "tail_return_prediction.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_asset": rows,
            "summary": {
                "n_assets": int(len(df)),
                "n_dd_significant": int(n_sig),
                "n_dd_negative": int(n_neg),
                "n_auc_above_05": int(n_auc),
                "median_spearman_dd": float(valid["spearman_dd"].median()) if len(valid) > 0 else None,
            },
        }, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
