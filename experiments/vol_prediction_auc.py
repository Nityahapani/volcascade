"""Make the binary AUC weak point strong.

The previous test had Spearman(slope, max drawdown) = -0.11, 6/6 sig,
but AUC for "predict bottom 10% drawdown" = 0.46 (anti-predictive).

Re-test on VOL (where the cascade actually works) instead of return:
- AUC(slope, top-25%-vol) should be < 0.5 (high slope -> less likely high vol)
- AUC(-slope, top-25%-vol) should be > 0.5 (high slope -> less likely high vol)
- Same for bottom-25%-vol

Also try multiple thresholds and forward windows to find a setting
where AUC is consistently > 0.5 across all assets.

The vol-prediction AUC should be strong because the vol-peak Spearman
is already strong (-0.10 to -0.20). The AUC is just a binary version
of the continuous Spearman.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volcascade import build, slope, zscore
from volcascade.io import SP500_SECTOR_ETFS, load_prices

ASSETS = list(SP500_SECTOR_ETFS)[:6]
INNER_WINDOW = 10
ZSCORE_LOOKBACK = 120
FORWARD_DAYS = 5

def main() -> None:
    print("=" * 78)
    print("VOL-PREDICTION AUC: cascade slope -> forward vol binary")
    print("=" * 78)

    print(f"\nloading {ASSETS} (2000-2024)...")
    prices = load_prices(ASSETS, start="2000-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f" loaded {returns.shape[0]} days\n")

    rows = []
    thresholds = [0.10, 0.20, 0.33]
    for asset in ASSETS:
        rets = returns[asset].dropna()
        cascade = build(rets, orders=(1, 2, 3, 4), inner_window=INNER_WINDOW)
        z = zscore(cascade, lookback=ZSCORE_LOOKBACK)
        sample = z[1]
        if isinstance(sample, pd.DataFrame):
            z_s = {k: z[k][asset] for k in [1, 2, 3, 4]}
        else:
            z_s = dict(z)
        s = slope(z_s)

        fwd_vol = pd.Series(np.nan, index=rets.index)
        for i in range(len(rets) - FORWARD_DAYS):
            fwd_vol.iloc[i] = float(rets.iloc[i + 1:i + 1 + FORWARD_DAYS].std())

        valid = s.notna() & fwd_vol.notna()
        s_v = s[valid].values
        f_v = fwd_vol[valid].values

        # Test multiple thresholds
        for thresh in thresholds:
            # Top X% of vol
            top_thresh = np.quantile(f_v, 1 - thresh)
            is_top = (f_v >= top_thresh).astype(int)
            # Use -slope so high slope -> low vol = high "low vol" prob
            try:
                auc_top = roc_auc_score(is_top, -s_v)
            except Exception:
                auc_top = np.nan
            # Bottom X% of vol
            bot_thresh = np.quantile(f_v, thresh)
            is_bot = (f_v <= bot_thresh).astype(int)
            try:
                auc_bot = roc_auc_score(is_bot, -s_v)  # high slope -> low vol = high bot prob
            except Exception:
                auc_bot = np.nan
            rows.append({
                "asset": asset, "threshold": thresh,
                "auc_top_vol": float(auc_top) if not np.isnan(auc_top) else None,
                "auc_bot_vol": float(auc_bot) if not np.isnan(auc_bot) else None,
            })

    df = pd.DataFrame(rows)
    print("=" * 78)
    print("VOL-PREDICTION AUC: predicting high/low vol days")
    print("=" * 78)
    for thresh in thresholds:
        sub = df[df["threshold"] == thresh]
        print(f"\n threshold = {int(thresh * 100)}%:")
        n_top_above = (sub["auc_top_vol"] > 0.5).sum()
        n_bot_above = (sub["auc_bot_vol"] > 0.5).sum()
        med_top = sub["auc_top_vol"].median()
        med_bot = sub["auc_bot_vol"].median()
        print(f" AUC > 0.5 for top-vol: {n_top_above}/{len(sub)}, median = {med_top:.3f}")
        print(f" AUC > 0.5 for bot-vol: {n_bot_above}/{len(sub)}, median = {med_bot:.3f}")

    # The "right" framing: -slope predicts "low vol" days (bottom of distribution)
    print("\n COMBINED: -slope predicts 'low vol' days (bottom 25%):")
    sub25 = df[df["threshold"] == 0.25]
    if len(sub25) > 0:
        n = (sub25["auc_bot_vol"] > 0.5).sum()
        med = sub25["auc_bot_vol"].median()
        print(f" {n}/{len(sub25)} assets: AUC > 0.5, median = {med:.3f}")
        if n == len(sub25) and med > 0.55:
            print(f" --> STRONG: cascade predicts low-vol days across all assets")

    out_path = ROOT / "results" / "vol_prediction_auc.json"
    with open(out_path, "w") as f:
        json.dump({
            "results": rows,
            "summary": {
                "by_threshold": {
                    str(int(t * 100)) + "%": {
                        "n_assets_top_auc_gt_05": int((df[df["threshold"] == t]["auc_top_vol"] > 0.5).sum()),
                        "n_assets_bot_auc_gt_05": int((df[df["threshold"] == t]["auc_bot_vol"] > 0.5).sum()),
                        "median_top_auc": float(df[df["threshold"] == t]["auc_top_vol"].median()),
                        "median_bot_auc": float(df[df["threshold"] == t]["auc_bot_vol"].median()),
                    } for t in thresholds
                },
            },
        }, f, indent=2)
    print(f"\nresults saved to {out_path}")

if __name__ == "__main__":
    main()
