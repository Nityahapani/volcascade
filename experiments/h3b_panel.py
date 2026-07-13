"""H3b on a PANEL of S&P 500 stocks — does the event-magnitude finding generalize?

The H3b result (cascade slope at event day -> |return|, Spearman -0.33)
was on AAPL only (39 earnings dates). Test on a panel of 20-30 S&P 500
stocks. If the result holds across many names, much stronger.

Uses earnings dates from yfinance. For each stock:
1. Get historical earnings dates
2. Compute cascade
3. At each earnings date, test slope -> |return|

Pool the results across stocks.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volcascade import build, slope, zscore  # noqa: E402
from volcascade.io import load_prices  # noqa: E402

RESULTS_DIR = ROOT / "results"

# 20 large S&P 500 stocks across sectors for a broad test
PANEL_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",  # Tech
    "JPM", "GS", "BAC", "WFC", "MS",          # Financials
    "JNJ", "PFE", "UNH", "ABBV", "MRK",        # Healthcare
    "XOM", "CVX", "COP", "SLB",               # Energy
    "PG", "KO", "WMT", "MCD", "NKE",          # Consumer
]


def get_earnings_dates(ticker: str) -> list[str]:
    """Get historical earnings dates for a ticker using yfinance."""
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        # earnings_dates is a DataFrame with earnings dates
        edf = t.earnings_dates
        if edf is None or len(edf) == 0:
            return []
        # edf index is the earnings date
        dates = []
        for d in edf.index:
            if hasattr(d, "date"):
                dates.append(d.date().isoformat())
            elif isinstance(d, str):
                dates.append(d)
        return sorted(dates)
    except Exception as e:
        return []


def main() -> None:
    print("=" * 78)
    print("H3b on PANEL of S&P 500 stocks")
    print("=" * 78)

    print(f"\nloading prices for {len(PANEL_TICKERS)} tickers (2015-2024)...")
    t0 = time.time()
    prices = load_prices(PANEL_TICKERS, start="2015-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  loaded {returns.shape[0]} days in {time.time()-t0:.1f}s\n")

    print("fetching earnings dates for each ticker (yfinance)...")
    earnings_by_ticker = {}
    t0 = time.time()
    for ticker in PANEL_TICKERS:
        dates = get_earnings_dates(ticker)
        # Filter to dates in our price window
        in_window = [d for d in dates if "2015-01-01" <= d <= "2024-12-31"]
        earnings_by_ticker[ticker] = in_window
        print(f"  {ticker}: {len(in_window)} earnings dates ({time.time()-t0:.1f}s)")

    # Pre-compute cascades
    print("\ncomputing cascades...")
    cascades = {}
    for asset in PANEL_TICKERS:
        if asset not in returns.columns:
            continue
        rets = returns[asset].dropna()
        cascade = build(rets, orders=(1, 2, 3, 4), inner_window=10)
        z = zscore(cascade, lookback=120)
        cascades[asset] = (z, rets)

    # For each stock, at each earnings date, test slope -> |return|
    print("\n" + "=" * 78)
    print("PER-STOCK ANALYSIS")
    print("=" * 78)
    rows = []
    for ticker in PANEL_TICKERS:
        if ticker not in cascades:
            continue
        dates = earnings_by_ticker.get(ticker, [])
        if len(dates) < 5:
            print(f"  {ticker}: only {len(dates)} dates, skipping")
            continue
        z, rets = cascades[ticker]
        # Cascade slope
        sample = z[1]
        if isinstance(sample, pd.DataFrame):
            z_s = {k: z[k][ticker] for k in [1, 2, 3, 4]}
        else:
            z_s = dict(z)
        s = slope(z_s)

        records = []
        for d in dates:
            ts = pd.Timestamp(d)
            if ts not in rets.index:
                continue
            loc = rets.index.get_loc(ts)
            if loc + 5 >= len(rets):
                continue
            slope_val = s.iloc[loc] if not pd.isna(s.iloc[loc]) else np.nan
            if pd.isna(slope_val):
                continue
            abs_ret = abs(float(rets.iloc[loc]))
            records.append({"slope": float(slope_val), "abs_return": abs_ret})

        if len(records) < 5:
            continue
        df = pd.DataFrame(records)
        r, p = sps.spearmanr(df["slope"], df["abs_return"])
        n_neg = (df["slope"] < 0).sum()
        n_pos = (df["slope"] > 0).sum()
        rows.append({
            "ticker": ticker,
            "n_earnings": len(records),
            "spearman_r": float(r),
            "spearman_p": float(p),
            "negative_slope": int(n_neg),
            "positive_slope": int(n_pos),
        })
        sig = "*" if p < 0.05 else " "
        print(f"  {sig} {ticker:6s} (n={len(records):2d}): r={r:+.4f}  p={p:.3f}")

    if not rows:
        print("no stocks had enough earnings dates")
        return

    # Pool
    print("\n" + "=" * 78)
    print("POOLED ANALYSIS")
    print("=" * 78)
    df = pd.DataFrame(rows)
    n_sig = (df["spearman_p"] < 0.05).sum()
    n_neg = (df["spearman_r"] < 0).sum()
    print(f"\n  total stocks: {len(df)}")
    print(f"  significant (p<0.05): {n_sig}/{len(df)} ({n_sig/len(df):.0%})")
    print(f"  negative direction: {n_neg}/{len(df)} ({n_neg/len(df):.0%})")
    print(f"  median Spearman: {df['spearman_r'].median():+.4f}")
    # Meta: Fisher's combined p-value across stocks
    chi2 = -2 * np.sum(np.log(df["spearman_p"].clip(lower=1e-10)))
    from scipy.stats import chi2 as chi2_dist
    dfree = 2 * len(df)
    combined_p = 1 - chi2_dist.cdf(chi2, dfree)
    print(f"  Fisher combined p-value: {combined_p:.2e}")

    out_path = RESULTS_DIR / "h3b_panel.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_stock": rows,
            "summary": {
                "n_stocks": int(len(df)),
                "n_significant": int(n_sig),
                "n_negative": int(n_neg),
                "median_spearman": float(df["spearman_r"].median()),
                "fisher_combined_p": float(combined_p),
            },
        }, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
