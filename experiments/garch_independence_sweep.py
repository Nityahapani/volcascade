"""Cascade configuration sweep: GARCH-independence (FASTER version).

Reduced sweep to fit in time limit. Tests 3 metrics (slope, entropy,
max_abs) x 3 inner_windows (10, 20, 80) x 2 zscore_lookbacks (60, 252)
x 1 n_orders (4) = 18 configurations per asset x 3 assets = 54 total.
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

from volcascade import build, entropy as cascade_entropy, slope, zscore  # noqa: E402
from volcascade.io import load_prices  # noqa: E402

RESULTS_DIR = ROOT / "results"
ASSETS = ["SPY", "XLE", "XLY"]  # 3 representative assets


def cascade_max_abs(z_cascade):
    orders = sorted(z_cascade.keys())
    sample = z_cascade[orders[0]]
    abs_stack = np.stack([np.abs(z_cascade[k].to_numpy() if hasattr(z_cascade[k], "to_numpy")
                               else z_cascade[k]) for k in orders], axis=-1)
    if isinstance(sample, pd.DataFrame):
        return pd.DataFrame(abs_stack.max(axis=-1), index=sample.index, columns=sample.columns)
    return pd.Series(abs_stack.max(axis=-1), index=sample.index)


def garch_residuals(rets):
    am = arch_model(rets * 100, mean="Constant", vol="GARCH", p=1, q=1,
                    dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 50})
    return res.std_resid.dropna()


def analyze(rets, name, inner_window, zscore_lookback, metric):
    orders = (1, 2, 3, 4)
    cascade = build(rets, orders=orders, inner_window=inner_window)
    z = zscore(cascade, lookback=zscore_lookback)
    sample = z[1]
    if isinstance(sample, pd.DataFrame):
        z_s = {k: z[k][name] for k in orders}
    else:
        z_s = dict(z)
    if metric == "slope":
        m = slope(z_s)
    elif metric == "entropy":
        m = cascade_entropy(z_s)
    elif metric == "max_abs":
        m = cascade_max_abs(z_s)
    else:
        return None
    fwd_vol = pd.Series(np.nan, index=rets.index)
    for i in range(len(rets) - 5):
        fwd_vol.iloc[i] = float(rets.iloc[i + 1:i + 1 + 5].std())
    valid = m.notna() & fwd_vol.notna()
    if valid.sum() < 100:
        return None
    r, p = sps.spearmanr(m[valid], fwd_vol[valid])
    return {"r": float(r), "p": float(p)}


def main() -> None:
    print("=" * 78)
    print("Cascade configuration sweep: GARCH-independence (FASTER)")
    print("=" * 78)

    print(f"\nloading {ASSETS} (2000-2024)...")
    t0 = time.time()
    prices = load_prices(ASSETS, start="2000-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  loaded {returns.shape[0]} days in {time.time()-t0:.1f}s\n")

    # Pre-compute GARCH residuals
    print("computing GARCH residuals...")
    t0 = time.time()
    garch_resids = {}
    for asset in ASSETS:
        rets = returns[asset].dropna()
        garch_resids[asset] = garch_residuals(rets)
        print(f"  {asset} GARCH done ({time.time()-t0:.1f}s)")

    # Reduced sweep
    inner_windows = [10, 20, 80]
    zscore_lookbacks = [60, 252]
    metrics = ["slope", "entropy", "max_abs"]
    # 3 x 2 x 3 = 18 configs per asset

    rows = []
    t0 = time.time()
    for asset in ASSETS:
        rets = returns[asset].dropna()
        g_res = garch_resids[asset]
        for iw in inner_windows:
            for zl in zscore_lookbacks:
                for met in metrics:
                    raw = analyze(rets, asset, iw, zl, met)
                    res = analyze(g_res, asset, iw, zl, met)
                    if raw is None or res is None:
                        continue
                    ratio = (res["r"] / raw["r"]) if abs(raw["r"]) > 0.01 else None
                    rows.append({
                        "asset": asset, "inner_window": iw, "zscore_lookback": zl,
                        "metric": met,
                        "raw_r": raw["r"], "raw_p": raw["p"],
                        "res_r": res["r"], "res_p": res["p"],
                        "ratio": ratio,
                    })
    print(f"\n{len(rows)} configurations tested ({time.time()-t0:.1f}s)\n")

    df = pd.DataFrame(rows)
    print("=" * 78)
    print("GARCH-INDEPENDENCE BY METRIC (median ratio = residual_r / raw_r)")
    print("=" * 78)
    for met in metrics:
        sub = df[df["metric"] == met].dropna(subset=["ratio"])
        if len(sub) == 0:
            continue
        print(f"\n  {met}:")
        print(f"    median raw_r: {sub['raw_r'].median():+.4f}")
        print(f"    median residual_r: {sub['res_r'].median():+.4f}")
        print(f"    median ratio: {sub['ratio'].median():+.3f}")

    # Best by metric
    print("\n" + "=" * 78)
    print("BEST PER METRIC (highest |ratio|)")
    print("=" * 78)
    for met in metrics:
        sub = df[df["metric"] == met].dropna(subset=["ratio"]).copy()
        if len(sub) == 0:
            continue
        sub["abs_ratio"] = sub["ratio"].abs()
        best = sub.nlargest(3, "abs_ratio")
        print(f"\n  {met}:")
        for _, r in best.iterrows():
            print(f"    {r['asset']} iw={int(r['inner_window'])} zl={int(r['zscore_lookback'])}: "
                  f"raw={r['raw_r']:+.4f}  res={r['res_r']:+.4f}  ratio={r['ratio']:+.3f}")

    out_csv = RESULTS_DIR / "garch_independence_sweep.csv"
    df.to_csv(out_csv, index=False)
    out_json = RESULTS_DIR / "garch_independence_sweep.json"
    with open(out_json, "w") as f:
        json.dump({
            "by_metric": {
                met: {
                    "median_raw_r": float(df[df["metric"] == met]["raw_r"].median()),
                    "median_residual_r": float(df[df["metric"] == met]["res_r"].median()),
                    "median_ratio": float(df[df["metric"] == met]["ratio"].median()) if df[df["metric"] == met]["ratio"].notna().any() else None,
                } for met in metrics
            },
        }, f, indent=2, default=str)
    print(f"\nresults saved to {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
