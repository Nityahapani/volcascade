"""GARCH-residual test for H4 frontier: is the 1.10x frontier effect GARCH-driven?

The H4 frontier extension showed the vol-peak is 1.10x stronger in
frontier markets. But the vol-peak itself is 22% GARCH-driven. We need
to check whether the FRONTIER EFFECT is also GARCH-driven (in which
case it's not a frontier-specific phenomenon) or if it persists
beyond GARCH (in which case frontier markets have a genuine
beyond-GARCH vol-peak advantage).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volcascade import build, slope, zscore  # noqa: E402
from volcascade.io import load_prices  # noqa: E402

RESULTS_DIR = ROOT / "results"

DEVELOPED = ["SPY"]
FRONTIER = ["EZA", "EWZ", "INDA"]


def garch_residuals(rets: pd.Series) -> pd.Series:
    am = arch_model(rets * 100, mean="Constant", vol="GARCH", p=1, q=1,
                    dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 50})
    return res.std_resid.dropna()


def analyze(rets: pd.Series, name: str) -> dict:
    cascade = build(rets, orders=(1, 2, 3, 4), inner_window=10)
    z = zscore(cascade, lookback=120)
    sample = z[1]
    if isinstance(sample, pd.DataFrame):
        z_s = {k: z[k][name] for k in [1, 2, 3, 4]}
    else:
        z_s = dict(z)
    s = slope(z_s)
    fwd_vol = pd.Series(np.nan, index=rets.index)
    for i in range(len(rets) - 5):
        fwd_vol.iloc[i] = float(rets.iloc[i + 1:i + 1 + 5].std())
    valid = s.notna() & fwd_vol.notna()
    if valid.sum() < 50:
        return None
    r, p = sps.spearmanr(s[valid], fwd_vol[valid])
    return {"r": float(r), "p": float(p)}


def main() -> None:
    print("=" * 78)
    print("H4 GARCH-RESIDUAL TEST: is the frontier effect GARCH-driven?")
    print("=" * 78)

    tickers = DEVELOPED + FRONTIER
    print(f"\nloading {tickers} (2007-2024)...")
    t0 = time.time()
    prices = load_prices(tickers, start="2007-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  loaded {returns.shape[0]} days in {time.time()-t0:.1f}s\n")

    # Compute GARCH residuals
    print("computing GARCH residuals...")
    garch_resids = {}
    for asset in tickers:
        rets = returns[asset].dropna()
        garch_resids[asset] = garch_residuals(rets)
        print(f"  {asset} done")

    print("\n" + "=" * 78)
    print("RAW RETURNS")
    print("=" * 78)
    raw_results = {}
    for asset in tickers:
        rets = returns[asset].dropna()
        out = analyze(rets, asset)
        if out:
            raw_results[asset] = out
            print(f"  {asset:5s} ({'developed' if asset == 'SPY' else 'frontier':>9s}): rho={out['r']:+.4f}  p={out['p']:.2e}")

    print("\n" + "=" * 78)
    print("GARCH-RESIDUALS")
    print("=" * 78)
    res_results = {}
    for asset in tickers:
        rets = garch_resids[asset]
        out = analyze(rets, asset)
        if out:
            res_results[asset] = out
            print(f"  {asset:5s} ({'developed' if asset == 'SPY' else 'frontier':>9s}): rho={out['r']:+.4f}  p={out['p']:.2e}")

    # Compare developed vs frontier on raw and residuals
    print("\n" + "=" * 78)
    print("DEVELOPED vs FRONTIER")
    print("=" * 78)
    dev_raw = abs(raw_results["SPY"]["r"])
    fr_raw = np.mean([abs(raw_results[a]["r"]) for a in FRONTIER])
    dev_res = abs(res_results["SPY"]["r"])
    fr_res = np.mean([abs(res_results[a]["r"]) for a in FRONTIER])

    print(f"\n  RAW:       |developed| = {dev_raw:.4f}, |frontier| mean = {fr_raw:.4f}, ratio = {fr_raw/dev_raw:.2f}x")
    print(f"  GARCH-RES: |developed| = {dev_res:.4f}, |frontier| mean = {fr_res:.4f}, ratio = {fr_res/dev_res:.2f}x")

    if fr_res / dev_res > fr_raw / dev_raw * 0.8:
        print(f"  --> frontier effect PERSISTS on GARCH-residuals ({(fr_res/dev_res) / (fr_raw/dev_raw) * 100:.0f}% of raw effect)")
    else:
        print(f"  --> frontier effect is mostly GARCH-driven")

    out_path = RESULTS_DIR / "h4_garch_residual_test.json"
    with open(out_path, "w") as f:
        json.dump({
            "raw": {k: v for k, v in raw_results.items()},
            "garch_residual": {k: v for k, v in res_results.items()},
            "summary": {
                "raw_dev_rho": dev_raw,
                "raw_frontier_rho": fr_raw,
                "garch_residual_dev_rho": dev_res,
                "garch_residual_frontier_rho": fr_res,
                "raw_frontier_ratio": fr_raw / dev_raw,
                "garch_residual_frontier_ratio": fr_res / dev_res,
            },
        }, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
