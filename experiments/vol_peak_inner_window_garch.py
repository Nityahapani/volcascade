"""Vol-peak on GARCH-residuals with longer inner_window.

The vol-peak is 22% GARCH-driven with default inner_window=10. Try
longer inner_windows (20, 40, 80) which should produce a SMOOTHER
cascade less reactive to short-term GARCH dynamics. The hypothesis:
a longer-window cascade is more GARCH-independent.

This would strengthen the vol-peak finding by showing that the
beyond-GARCH component can be boosted with the right parameter choice.
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
ASSETS = ["SPY", "XLE", "XLY"]  # 3 representative


def garch_residuals(rets):
    am = arch_model(rets * 100, mean="Constant", vol="GARCH", p=1, q=1,
                    dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 50})
    return res.std_resid.dropna()


def analyze(rets, name, inner_window, zscore_lookback=120, forward_days=5):
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
    if valid.sum() < 50:
        return None
    r, p = sps.spearmanr(s[valid], fwd_vol[valid])
    return {"r": float(r), "p": float(p)}


def main() -> None:
    print("=" * 78)
    print("VOL-PEAK GARCH-INDEPENDENCE: longer inner_window")
    print("=" * 78)

    print(f"\nloading {ASSETS} (2000-2024)...")
    prices = load_prices(ASSETS, start="2000-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()

    print("computing GARCH residuals...")
    garch_resids = {}
    for asset in ASSETS:
        garch_resids[asset] = garch_residuals(returns[asset].dropna())
        print(f"  {asset} done")

    print("\n" + "=" * 78)
    print("GARCH-INDEPENDENCE by inner_window (median across 3 assets)")
    print("=" * 78)

    for iw in [5, 10, 20, 40, 80]:
        raw_rs, res_rs = [], []
        for asset in ASSETS:
            rets = returns[asset].dropna()
            g_res = garch_resids[asset]
            raw = analyze(rets, asset, iw)
            res = analyze(g_res, asset, iw)
            if raw and res:
                raw_rs.append(raw["r"])
                res_rs.append(res["r"])
        if raw_rs:
            med_raw = np.median(raw_rs)
            med_res = np.median(res_rs)
            ratio = med_res / med_raw if abs(med_raw) > 0.01 else None
            print(f"  inner_window={iw:3d}: raw={med_raw:+.4f}  residual={med_res:+.4f}  "
                  f"ratio={ratio:+.3f}" if ratio else f"  inner_window={iw:3d}: raw={med_raw:+.4f}  residual={med_res:+.4f}")

    out_path = RESULTS_DIR / "vol_peak_garch_independence_inner_window.json"
    rows = []
    for iw in [5, 10, 20, 40, 80]:
        for asset in ASSETS:
            rets = returns[asset].dropna()
            g_res = garch_resids[asset]
            raw = analyze(rets, asset, iw)
            res = analyze(g_res, asset, iw)
            if raw and res:
                ratio = res["r"] / raw["r"] if abs(raw["r"]) > 0.01 else None
                rows.append({"asset": asset, "inner_window": iw,
                             "raw_r": raw["r"], "raw_p": raw["p"],
                             "res_r": res["r"], "res_p": res["p"],
                             "ratio": ratio})
    with open(out_path, "w") as f:
        json.dump({"results": rows,
                   "by_inner_window": {
                       int(iw): {
                           "median_raw": float(np.median([r["raw_r"] for r in rows if r["inner_window"] == iw])),
                           "median_res": float(np.median([r["res_r"] for r in rows if r["inner_window"] == iw])),
                       } for iw in [5, 10, 20, 40, 80]
                   }}, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
