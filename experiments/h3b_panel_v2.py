"""H3b on a PANEL of stocks using 'big move' events.

yfinance earnings_dates is unreliable in current version. Use a
panel-friendly definition of "events": for each stock, identify the
largest |return| days (top 5% per quarter) as proxy event days.

This is a more general test than earnings specifically: does the
cascade predict large price moves across many stocks?
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.stats import chi2 as chi2_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volcascade import build, slope, zscore  # noqa: E402
from volcascade.io import load_prices  # noqa: E402

RESULTS_DIR = ROOT / "results"

PANEL_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "JPM", "GS", "BAC", "WFC", "MS",
    "JNJ", "PFE", "UNH", "ABBV", "MRK",
    "XOM", "CVX", "COP", "SLB",
    "PG", "KO", "WMT", "MCD", "NKE",
    "HD", "LOW", "TGT", "COST", "AMGN",
    "BA", "CAT", "GE", "MMM", "HON",
    "AAPL", "MSFT",  # duplicates to test
]


def main() -> None:
    print("=" * 78)
    print("H3b PANEL: cascade predicts large |return| days across many stocks")
    print("=" * 78)

    # Use unique tickers
    tickers = list(set(PANEL_TICKERS))
    print(f"\nloading {len(tickers)} tickers (2015-2024)...")
    t0 = time.time()
    prices = load_prices(tickers, start="2015-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    available = [c for c in returns.columns]
    print(f"  loaded {returns.shape[0]} days x {len(available)} tickers in {time.time()-t0:.1f}s\n")

    # Pre-compute cascades
    print("computing cascades...")
    cascades = {}
    for asset in available:
        rets = returns[asset].dropna()
        if rets.std() < 0.001:  # skip essentially flat series
            continue
        cascade = build(rets, orders=(1, 2, 3, 4), inner_window=10)
        z = zscore(cascade, lookback=120)
        cascades[asset] = (z, rets)

    # For each stock: identify top-1 (or top-2) largest |return| day per quarter
    # as "event" days. Test: cascade slope at event day -> |return| at event day.
    print("=" * 78)
    print("PER-STOCK: top |return| days per quarter as events")
    print("=" * 78)
    rows = []
    for asset in available:
        if asset not in cascades:
            continue
        z, rets = cascades[asset]
        # Cascade slope
        sample = z[1]
        if isinstance(sample, pd.DataFrame):
            z_s = {k: z[k][asset] for k in [1, 2, 3, 4]}
        else:
            z_s = dict(z)
        s = slope(z_s)

        # Per-quarter grouping
        rets_q = rets.copy()
        rets_q.index = pd.to_datetime(rets_q.index)
        quarters = rets_q.index.to_period("Q")
        records = []
        for q in quarters.unique():
            mask = quarters == q
            q_rets = rets_q[mask]
            if len(q_rets) < 5:
                continue
            # Top 1 largest |return| day in this quarter
            top_idx = q_rets.abs().idxmax()
            loc = rets.index.get_loc(top_idx)
            slope_val = s.iloc[loc] if not pd.isna(s.iloc[loc]) else np.nan
            if pd.isna(slope_val):
                continue
            abs_ret = abs(float(rets.iloc[loc]))
            records.append({
                "slope": float(slope_val),
                "abs_return": abs_ret,
                "date": str(top_idx.date()) if hasattr(top_idx, "date") else str(top_idx),
            })
        if len(records) < 10:
            continue
        df = pd.DataFrame(records)
        r, p = sps.spearmanr(df["slope"], df["abs_return"])
        rows.append({
            "ticker": asset,
            "n_quarters": len(records),
            "n_events": len(records),
            "spearman_r": float(r),
            "spearman_p": float(p),
        })
        sig = "*" if p < 0.05 else " "
        print(f"  {sig} {asset:6s} (n={len(records):2d}): r={r:+.4f}  p={p:.3f}")

    if not rows:
        print("no stocks had enough events")
        return

    # Pool
    print("\n" + "=" * 78)
    print("POOLED ANALYSIS")
    print("=" * 78)
    df = pd.DataFrame(rows)
    n_sig = (df["spearman_p"] < 0.05).sum()
    n_neg = (df["spearman_r"] < 0).sum()
    n_total_events = df["n_events"].sum()
    print(f"\n  total stocks: {len(df)}")
    print(f"  total events: {n_total_events}")
    print(f"  significant (p<0.05): {n_sig}/{len(df)} ({n_sig/len(df):.0%})")
    print(f"  negative direction: {n_neg}/{len(df)} ({n_neg/len(df):.0%})")
    print(f"  median Spearman: {df['spearman_r'].median():+.4f}")

    # Fisher's combined p-value
    chi2 = -2 * np.sum(np.log(df["spearman_p"].clip(lower=1e-10)))
    combined_p = 1 - chi2_dist.cdf(chi2, 2 * len(df))
    print(f"  Fisher combined p-value: {combined_p:.2e}")

    out_path = RESULTS_DIR / "h3b_panel_v2.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_stock": rows,
            "summary": {
                "n_stocks": int(len(df)),
                "n_events": int(n_total_events),
                "n_significant": int(n_sig),
                "n_negative": int(n_neg),
                "median_spearman": float(df["spearman_r"].median()),
                "fisher_combined_p": float(combined_p),
            },
        }, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
