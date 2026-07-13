"""H2 regime exit: does cascade convergence mark regime exit?

The methodology: cascade convergence (higher orders decaying faster than
order-1) should mark regime exit EARLIER OR MORE RELIABLY than the naive
order-1-MA baseline.

Definitions:
  - "Regime" = a period where forward 5-day realized vol > 1.5x the
    trailing 60-day median vol.
  - "Regime exit" = the day when forward vol first falls below the
    threshold.
  - Cascade convergence: $C_t$ = rolling 20-day mean of (max_k z^(k) -
    min_k z^(k)). Exit flagged when $C_t$ falls below its 60-day median.
  - Naive baseline: order-1-MA exit = order-1 falling below 60-day MA.

Compare:
  - False exit rate: fraction of exits where a new regime begins within
    next 20 days
  - Time-to-correctly-flag: average lead/lag vs ground-truth exit
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


def main() -> None:
    print("=" * 78)
    print("H2: cascade convergence vs naive order-1-MA for regime exit detection")
    print("=" * 78)

    print("\nloading SPY + 5 sector ETFs (2000-2024)...")
    t0 = time.time()
    prices = load_prices(list(SP500_SECTOR_ETFS)[:6], start="2000-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  loaded {returns.shape[0]} days in {time.time()-t0:.1f}s\n")

    # Define regime as: forward 5-day realized vol > 1.5x trailing 60-day median vol
    fwd_vol = returns.rolling(5, min_periods=1).std().shift(-5)
    trail_med = returns.rolling(60, min_periods=30).median()
    # This is the regime condition at each time t: fwd_vol[t] > 1.5 * median[fwd_vol[t-60:t-1]]
    fwd_vol_trail_med = fwd_vol.rolling(60, min_periods=30).median().shift(1)
    is_regime = (fwd_vol > 1.5 * fwd_vol_trail_med)

    # For each asset, compute cascade and run the H2 test
    results = []
    for asset in returns.columns:
        rets = returns[asset].dropna()
        cascade = build(rets, orders=(1, 2, 3, 4), inner_window=10)
        z = zscore(cascade, lookback=120)
        sample = z[1]
        if isinstance(sample, pd.DataFrame):
            z_s = {k: z[k][asset] for k in [1, 2, 3, 4]}
        else:
            z_s = dict(z)

        # Cascade convergence: rolling 20-day mean of (max_k z - min_k z)
        zmat = np.stack([z_s[k].to_numpy() for k in [1, 2, 3, 4]], axis=-1)
        spread = np.nanmax(zmat, axis=-1) - np.nanmin(zmat, axis=-1)
        spread_series = pd.Series(spread, index=rets.index)
        C = spread_series.rolling(20, min_periods=5).mean()
        C_trail_med = C.rolling(60, min_periods=30).median()
        cascade_exit = C < C_trail_med

        # Naive order-1-MA baseline
        order1 = z_s[1]
        o1_ma = order1.rolling(60, min_periods=30).mean()
        naive_exit = order1 < o1_ma

        # Find regime transitions
        regime = is_regime[asset]
        # A regime is a contiguous block of True in `regime`
        regime_change = regime.astype(int).diff().fillna(0)
        regime_starts = regime_change[regime_change == 1].index  # entering
        regime_ends = regime_change[regime_change == -1].index    # exiting

        # For each regime, find the ground-truth exit and compare
        # signals' lead time
        ground_truth_exits = regime_ends
        n_regimes = len(ground_truth_exits)

        # Skip if there are too few regimes
        if n_regimes < 10:
            print(f"  {asset}: only {n_regimes} regimes, skipping")
            continue

        # Cascade signal: when does cascade_exit fire after each regime entry?
        cascade_lead_times = []
        for start in regime_starts:
            # Find first cascade_exit in the next 60 days
            end_window = start + pd.Timedelta(days=60)
            exits = cascade_exit.loc[start:end_window]
            exits = exits[exits]
            if len(exits) > 0:
                first_exit = exits.index[0]
                lead = (first_exit - start).days
                cascade_lead_times.append(lead)
        # Naive signal lead times
        naive_lead_times = []
        for start in regime_starts:
            end_window = start + pd.Timedelta(days=60)
            exits = naive_exit.loc[start:end_window]
            exits = exits[exits]
            if len(exits) > 0:
                first_exit = exits.index[0]
                lead = (first_exit - start).days
                naive_lead_times.append(lead)

        # False exit rate: fraction of cascade exits that are followed by
        # a new regime within 20 days
        all_cascade_exits = cascade_exit[cascade_exit].index
        n_false_cascade = 0
        for ex in all_cascade_exits:
            future_window = ex + pd.Timedelta(days=20)
            if regime.loc[ex:future_window].any():
                n_false_cascade += 1
        false_rate_cascade = n_false_cascade / len(all_cascade_exits) if len(all_cascade_exits) > 0 else np.nan

        all_naive_exits = naive_exit[naive_exit].index
        n_false_naive = 0
        for ex in all_naive_exits:
            future_window = ex + pd.Timedelta(days=20)
            if regime.loc[ex:future_window].any():
                n_false_naive += 1
        false_rate_naive = n_false_naive / len(all_naive_exits) if len(all_naive_exits) > 0 else np.nan

        out = {
            "asset": asset,
            "n_regimes": n_regimes,
            "n_cascade_exits": int(len(all_cascade_exits)),
            "n_naive_exits": int(len(all_naive_exits)),
            "false_rate_cascade": float(false_rate_cascade) if not np.isnan(false_rate_cascade) else None,
            "false_rate_naive": float(false_rate_naive) if not np.isnan(false_rate_naive) else None,
            "cascade_mean_lead_days": float(np.mean(cascade_lead_times)) if cascade_lead_times else None,
            "naive_mean_lead_days": float(np.mean(naive_lead_times)) if naive_lead_times else None,
            "cascade_pct_with_exit_signal": float(len(cascade_lead_times) / n_regimes) if n_regimes > 0 else None,
            "naive_pct_with_exit_signal": float(len(naive_lead_times) / n_regimes) if n_regimes > 0 else None,
        }
        results.append(out)
        print(f"\n  {asset}: n_regimes={n_regimes}, "
              f"cascade false={out['false_rate_cascade']:.2%}, "
              f"naive false={out['false_rate_naive']:.2%}")
        if out['cascade_mean_lead_days'] is not None:
            print(f"    mean lead days: cascade={out['cascade_mean_lead_days']:.1f}, naive={out['naive_mean_lead_days']:.1f}")

    df = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    valid = df.dropna(subset=["false_rate_cascade", "false_rate_naive"])
    if len(valid) > 0:
        print(f"\n  mean false exit rate:")
        print(f"    cascade: {valid['false_rate_cascade'].mean():.2%}")
        print(f"    naive:   {valid['false_rate_naive'].mean():.2%}")
        # Paired t-test: cascade vs naive
        t, p = sps.ttest_rel(valid["false_rate_cascade"], valid["false_rate_naive"])
        print(f"    paired t-test: t={t:.3f}, p={p:.4f}")
        if valid["cascade_mean_lead_days"].notna().any():
            print(f"\n  mean lead time (days):")
            print(f"    cascade: {valid['cascade_mean_lead_days'].mean():.1f}")
            print(f"    naive:   {valid['naive_mean_lead_days'].mean():.1f}")

    out_path = RESULTS_DIR / "h2_regime_exit.json"
    with open(out_path, "w") as f:
        json.dump({"per_asset": [dict(r) for r in results],
                   "summary": {
                       "n_assets": int(len(df)),
                       "mean_false_rate_cascade": float(valid["false_rate_cascade"].mean()) if len(valid) > 0 else None,
                       "mean_false_rate_naive": float(valid["false_rate_naive"].mean()) if len(valid) > 0 else None,
                   }}, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
