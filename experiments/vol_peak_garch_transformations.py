"""Vol-peak GARCH-independence: try cascade on |r|, r^2, log(r^2).

The vol-peak on returns is 22% GARCH-independent. Try alternative
return transformations to see if any give a higher GARCH-independence
fraction:

- |r| (absolute returns): proxy for vol with different GARCH properties
- r^2 (squared returns): natural GARCH input
- log(r^2 + epsilon) (log-realized-variance): more stationary

If any transformation gives > 50% GARCH-independent, that's a
stronger vol-peak finding (most of the effect is genuinely beyond
GARCH, not just a smoothed version of it).
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
ASSETS = ["SPY", "XLE", "XLY"]


def transform(returns: pd.Series, kind: str) -> pd.Series:
    if kind == "raw":
        return returns
    elif kind == "abs":
        return returns.abs()
    elif kind == "sq":
        return returns.pow(2)
    elif kind == "log_sq":
        # log(r^2 + epsilon) where epsilon is small to handle 0
        return np.log(returns.pow(2) + 1e-6)
    else:
        raise ValueError(f"unknown transform: {kind}")


def garch_residuals(rets: pd.Series) -> pd.Series:
    am = arch_model(rets * 100, mean="Constant", vol="GARCH", p=1, q=1,
                    dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 50})
    return res.std_resid.dropna()


def analyze(rets: pd.Series, name: str, inner_window: int = 10,
            zscore_lookback: int = 120, forward_days: int = 5) -> dict:
    cascade = build(rets, orders=(1, 2, 3, 4), inner_window=inner_window)
    z = zscore(cascade, lookback=zscore_lookback)
    sample = z[1]
    if isinstance(sample, pd.DataFrame):
        z_s = {k: z[k][name] for k in [1, 2, 3, 4]}
    else:
        z_s = dict(z)
    s = slope(z_s)
    fwd_vol = pd.Series(np.nan, index=rets.index)
    for i in range(len(rets) - forward_days):
        fwd_vol.iloc[i] = float(rets.iloc[i + 1:i + 1 + forward_days].std())
    valid = s.notna() & fwd_vol.notna()
    if valid.sum() < 100:
        return None
    r, p = sps.spearmanr(s[valid], fwd_vol[valid])
    return {"r": float(r), "p": float(p)}


def main() -> None:
    print("=" * 78)
    print("VOL-PEAK GARCH-INDEPENDENCE: alternative return transformations")
    print("=" * 78)

    print(f"\nloading {ASSETS} (2000-2024)...")
    prices = load_prices(ASSETS, start="2000-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()

    print("computing GARCH residuals...")
    garch_resids = {}
    for asset in ASSETS:
        garch_resids[asset] = garch_residuals(returns[asset].dropna())
        print(f"  {asset} done")

    transforms = ["raw", "abs", "sq", "log_sq"]
    print("\n" + "=" * 78)
    print("GARCH-INDEPENDENCE by transformation (median across 3 assets)")
    print("=" * 78)

    rows = []
    for kind in transforms:
        print(f"\n  Transform = {kind}:")
        for asset in ASSETS:
            rets = returns[asset].dropna()
            g_res = garch_resids[asset]
            try:
                rets_t = transform(rets, kind)
                g_res_t = transform(g_res, kind)
                raw = analyze(rets_t, asset)
                res = analyze(g_res_t, asset)
                if raw is None or res is None:
                    continue
                ratio = res["r"] / raw["r"] if abs(raw["r"]) > 0.01 else None
                rows.append({"asset": asset, "transform": kind,
                             "raw_r": raw["r"], "raw_p": raw["p"],
                             "res_r": res["r"], "res_p": res["p"],
                             "ratio": ratio})
                print(f"    {asset}: raw_r={raw['r']:+.4f}  res_r={res['r']:+.4f}  "
                      f"ratio={ratio:+.3f}" if ratio is not None else f"    {asset}: ratio=n/a")
            except Exception as e:
                print(f"    {asset}: failed ({e})")

    # Aggregate
    print("\n" + "=" * 78)
    print("AGGREGATE BY TRANSFORM")
    print("=" * 78)
    df = pd.DataFrame(rows)
    for kind in transforms:
        sub = df[df["transform"] == kind].dropna(subset=["ratio"])
        if len(sub) == 0:
            continue
        med_raw = sub["raw_r"].median()
        med_res = sub["res_r"].median()
        med_ratio = sub["ratio"].median() if sub["ratio"].notna().any() else None
        print(f"  {kind}: median raw_r={med_raw:+.4f}  res_r={med_res:+.4f}  "
              f"ratio={med_ratio:+.3f}" if med_ratio is not None else f"  {kind}: median raw_r={med_raw:+.4f}  res_r={med_res:+.4f}")

    out_path = RESULTS_DIR / "vol_peak_garch_transformations.json"
    with open(out_path, "w") as f:
        json.dump({"results": rows,
                   "by_transform": {
                       k: {
                           "median_raw": float(df[df["transform"] == k]["raw_r"].median()),
                           "median_res": float(df[df["transform"] == k]["res_r"].median()),
                           "median_ratio": float(df[df["transform"] == k]["ratio"].median())
                                if df[df["transform"] == k]["ratio"].notna().any() else None,
                       } for k in transforms
                   }}, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
