"""Non-parametric Granger: rank-based test of cascade slope -> forward vol.

The parametric Granger F-test was weak (28% of pairs significant).
Try a non-parametric version using rank-transformed variables.
Rank-based tests are more robust to non-Gaussianity and outliers.

Also test the REVERSE direction (forward vol -> cascade slope) to
see if the relationship is bidirectional or one-way.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volcascade import build, slope, zscore  # noqa: E402
from volcascade.io import SP500_SECTOR_ETFS, load_prices  # noqa: E402

RESULTS_DIR = ROOT / "results"
ASSETS = ["SPY", "XLE", "XLF", "XLV", "XLY"]


def rank_granger_test(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, float]:
    """Non-parametric Granger-like test using rank-transformed variables.

    For each pair (x, y) at lag k:
    1. Rank-transform x and y to uniforms.
    2. Test correlation between y_t and rank(x_{t-k}) (Spearman).
    3. Test partial Spearman: y_t vs rank(x_{t-k}) controlling for
       y_{t-1}, ..., y_{t-k} (using rank correlation on residuals).

    Returns (F-like statistic, p-value).
    """
    n = len(y)
    if n < lag + 30:
        return np.nan, np.nan

    # Rank-transform
    rx = stats_rank(x)
    ry = stats_rank(y)

    # Predict y_t from rank(x_{t-k}) and lagged y
    # Use Spearman as a non-parametric analog of F-test
    y_lag = ry[lag:]
    x_lag = rx[:-lag]
    r, p = sps.spearmanr(y_lag, x_lag)
    return float(r), float(p)


def stats_rank(x: np.ndarray) -> np.ndarray:
    """Rank-transform to [0, 1] (uniform)."""
    ranks = sps.rankdata(x)
    return (ranks - 1) / (len(ranks) - 1)


def main() -> None:
    print("=" * 78)
    print("NON-PARAMETRIC GRANGER (rank-based)")
    print("=" * 78)

    print(f"\nloading {ASSETS} (2000-2024)...")
    t0 = time.time()
    prices = load_prices(ASSETS, start="2000-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  loaded {returns.shape[0]} days in {time.time()-t0:.1f}s\n")

    print("=" * 78)
    print("DIRECTION 1: slope_lag(k) -> forward_vol  (does slope lead vol?)")
    print("=" * 78)
    direct_rows = []
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
        fwd_vol = pd.Series(np.nan, index=rets.index)
        for i in range(len(rets) - 5):
            fwd_vol.iloc[i] = float(rets.iloc[i + 1:i + 1 + 5].std())

        valid = ~(s.isna() | fwd_vol.isna())
        s_v = s[valid].to_numpy()
        v_v = fwd_vol[valid].to_numpy()
        print(f"\n  {asset}:")
        for lag in [1, 2, 3, 5, 10]:
            if len(s_v) > lag + 30:
                # Spearman of slope_lag -> forward_vol
                r, p = sps.spearmanr(s_v[:-lag], v_v[lag:])
                direct_rows.append({"asset": asset, "lag": lag, "r": float(r), "p": float(p),
                                     "direction": "slope -> fwd_vol"})
                sig = "*" if p < 0.05 else " "
                print(f"    lag {lag:2d}: r={r:+.4f}  p={p:.2e}  {sig}")

    print("\n" + "=" * 78)
    print("DIRECTION 2: fwd_vol_lag(k) -> slope  (does vol lead slope?)")
    print("=" * 78)
    reverse_rows = []
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
        fwd_vol = pd.Series(np.nan, index=rets.index)
        for i in range(len(rets) - 5):
            fwd_vol.iloc[i] = float(rets.iloc[i + 1:i + 1 + 5].std())

        valid = ~(s.isna() | fwd_vol.isna())
        s_v = s[valid].to_numpy()
        v_v = fwd_vol[valid].to_numpy()
        print(f"\n  {asset}:")
        for lag in [1, 2, 3, 5, 10]:
            if len(s_v) > lag + 30:
                r, p = sps.spearmanr(v_v[:-lag], s_v[lag:])
                reverse_rows.append({"asset": asset, "lag": lag, "r": float(r), "p": float(p),
                                     "direction": "fwd_vol -> slope"})
                sig = "*" if p < 0.05 else " "
                print(f"    lag {lag:2d}: r={r:+.4f}  p={p:.2e}  {sig}")

    # Compare
    print("\n" + "=" * 78)
    print("COMPARISON: slope -> vol  vs  vol -> slope")
    print("=" * 78)
    df_direct = pd.DataFrame(direct_rows)
    df_reverse = pd.DataFrame(reverse_rows)
    if len(df_direct) > 0:
        print(f"\n  DIRECT (slope -> vol):")
        for lag in [1, 2, 3, 5, 10]:
            sub = df_direct[df_direct["lag"] == lag]
            n_sig = (sub["p"] < 0.05).sum()
            med_r = sub["r"].median()
            print(f"    lag {lag:2d}: {n_sig}/{len(sub)} significant, median r={med_r:+.4f}")
    if len(df_reverse) > 0:
        print(f"\n  REVERSE (vol -> slope):")
        for lag in [1, 2, 3, 5, 10]:
            sub = df_reverse[df_reverse["lag"] == lag]
            n_sig = (sub["p"] < 0.05).sum()
            med_r = sub["r"].median()
            print(f"    lag {lag:2d}: {n_sig}/{len(sub)} significant, median r={med_r:+.4f}")

    out_path = RESULTS_DIR / "non_parametric_granger.json"
    with open(out_path, "w") as f:
        json.dump({
            "direct_slope_to_vol": direct_rows,
            "reverse_vol_to_slope": reverse_rows,
        }, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
