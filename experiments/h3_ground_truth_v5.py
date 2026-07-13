"""H3 v5: predict event MAGNITUDE instead of class.

The H3 class prediction (idiosyncratic vs systemic) was weak (best
AUC=0.60, p=0.088). Try predicting the MAGNITUDE of the event-day
return — a continuous outcome that may give more statistical power.

For each event, predict:
- |return on event day|  (absolute return = magnitude)
- max(|return|) in [event_day-1, event_day+1]  (peak impact)
- |return| in [event_day+1, event_day+5]  (post-event drift)

Predictors: cascade slope, F-statistic at each order (H3 v3 best),
cross-correlation at each order (H3 v4 best).
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
sys.path.insert(0, str(ROOT / "experiments"))

from volcascade import build, slope, zscore  # noqa: E402
from volcascade.decoupling import chow_statistic  # noqa: E402
from volcascade.io import load_prices  # noqa: E402
from h3_ground_truth import aapl_earnings_dates, fomc_dates  # noqa: E402

RESULTS_DIR = ROOT / "results"


def main() -> None:
    print("=" * 78)
    print("H3 v5: predict event magnitude from cascade shape")
    print("=" * 78)

    events = aapl_earnings_dates() + fomc_dates()
    print(f"\n{len(events)} curated events")

    print("\nloading AAPL, XLK, SPY (2015-2024)...")
    t0 = time.time()
    prices = load_prices(["AAPL", "XLK", "SPY"], start="2015-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  loaded {returns.shape[0]} days in {time.time()-t0:.1f}s\n")

    aapl_cascade = build(returns["AAPL"], orders=(1, 2, 3, 4), inner_window=10)
    aapl_z = zscore(aapl_cascade, lookback=120)
    xlk_cascade = build(returns["XLK"], orders=(1, 2, 3, 4), inner_window=10)
    xlk_z = zscore(xlk_cascade, lookback=120)
    spy_cascade = build(returns["SPY"], orders=(1, 2, 3, 4), inner_window=10)
    spy_z = zscore(spy_cascade, lookback=120)

    records = []
    t0 = time.time()
    for i, ev in enumerate(events):
        if (i + 1) % 30 == 0:
            print(f"  event {i+1}/{len(events)}  ({time.time()-t0:.1f}s)")
        d = pd.Timestamp(ev["date"])
        if d not in returns.index:
            continue

        end_loc = returns.index.get_loc(d)
        if end_loc < 30 or end_loc + 5 >= len(returns):
            continue

        # Outcomes
        ret_event = float(returns["AAPL"].iloc[end_loc])  # event-day return
        abs_event = abs(ret_event)
        # Peak impact in [-1, +1]
        window = returns["AAPL"].iloc[end_loc - 1:end_loc + 2]
        peak_impact = float(window.abs().max())
        # Post-event drift in [+1, +5]
        post_drift = float(returns["AAPL"].iloc[end_loc + 1:end_loc + 6].abs().mean())

        # Predictors: cascade shape at event
        # Slope at event day
        slope_event = float(slope({k: aapl_z[k] for k in [1, 2, 3, 4]}).iloc[end_loc]) \
            if end_loc < len(aapl_z[1]) else np.nan
        if np.isnan(slope_event):
            continue

        # F-statistic at each order for AAPL-XLK decoupling
        f_per_order = {}
        for k in [1, 2, 3, 4]:
            aapl_s = aapl_z[k].iloc[end_loc - 30:end_loc + 30].dropna().to_numpy()
            xlk_s = xlk_z[k].iloc[end_loc - 30:end_loc + 30].dropna().to_numpy()
            min_len = min(len(aapl_s), len(xlk_s))
            if min_len < 30:
                continue
            aapl_s = aapl_s[-min_len:]
            xlk_s = xlk_s[-min_len:]
            f, p, _ = chow_statistic(aapl_s, xlk_s, k=min_len // 2, lookback=15)
            f_per_order[k] = float(f) if not np.isnan(f) else None

        # Cross-correlation at each order
        corr_per_order = {}
        for k in [1, 2, 3, 4]:
            aapl_s = aapl_z[k].iloc[end_loc - 15:end_loc + 5].dropna().to_numpy()
            xlk_s = xlk_z[k].iloc[end_loc - 15:end_loc + 5].dropna().to_numpy()
            min_len = min(len(aapl_s), len(xlk_s))
            if min_len < 5:
                continue
            r, _ = sps.pearsonr(aapl_s[-min_len:], xlk_s[-min_len:])
            corr_per_order[k] = float(r)

        # Spread of cascade at event (max abs z - min abs z)
        z_at_event = {k: aapl_z[k].iloc[end_loc] for k in [1, 2, 3, 4]}
        z_abs = [abs(z_at_event[k]) for k in [1, 2, 3, 4] if not pd.isna(z_at_event[k])]
        z_spread = max(z_abs) - min(z_abs) if z_abs else np.nan

        records.append({
            "date": ev["date"],
            "label": ev["label"],
            "class": ev["class"],
            "asset": ev["asset"],
            "ret_event": ret_event,
            "abs_event": abs_event,
            "peak_impact": peak_impact,
            "post_drift": post_drift,
            "slope_event": slope_event,
            "z_spread": z_spread,
            "f1": f_per_order.get(1), "f2": f_per_order.get(2),
            "f3": f_per_order.get(3), "f4": f_per_order.get(4),
            "corr_1": corr_per_order.get(1), "corr_2": corr_per_order.get(2),
            "corr_3": corr_per_order.get(3), "corr_4": corr_per_order.get(4),
        })

    df = pd.DataFrame(records)
    print(f"\nanalyzed {len(df)} events")

    # Test: do predictors correlate with event magnitude?
    print("\n" + "=" * 78)
    print("CORRELATIONS WITH EVENT MAGNITUDE")
    print("=" * 78)
    outcomes = ["abs_event", "peak_impact", "post_drift"]
    predictors = ["slope_event", "z_spread", "f1", "f2", "f3", "f4", "corr_1", "corr_2"]

    print(f"\n{'predictor':15s} | " + " | ".join(f"{o:>12s}" for o in outcomes))
    print("-" * 60)
    for pred in predictors:
        row = f"{pred:15s} | "
        for out in outcomes:
            valid = df[[pred, out]].dropna()
            if len(valid) > 10:
                r, p = sps.spearmanr(valid[pred], valid[out])
                sig = "*" if p < 0.05 else " "
                row += f" {r:+.3f}{sig} (p={p:.2f}) | "
            else:
                row += f"   n/a       | "
        print(row)

    # By event class
    print("\n" + "=" * 78)
    print("BY EVENT CLASS")
    print("=" * 78)
    for cls in ["idiosyncratic", "systemic"]:
        sub = df[df["class"] == cls]
        if len(sub) < 5:
            continue
        print(f"\n  {cls} (n={len(sub)}):")
        print(f"    median |return|: {sub['abs_event'].median():.4f}")
        print(f"    median peak impact: {sub['peak_impact'].median():.4f}")
        print(f"    median post drift: {sub['post_drift'].median():.4f}")
        # Best predictor of magnitude
        for pred in ["slope_event", "f2", "corr_1"]:
            valid = sub[[pred, "abs_event"]].dropna()
            if len(valid) > 5:
                r, p = sps.spearmanr(valid[pred], valid["abs_event"])
                print(f"    Spearman({pred}, |return|) = {r:+.4f}  p={p:.3f}")

    out_path = RESULTS_DIR / "h3_ground_truth_v5_magnitude.json"
    with open(out_path, "w") as f:
        json.dump({"events": records,
                   "summary": {
                       "n_total": int(len(df)),
                       "n_idiosyncratic": int((df["class"] == "idiosyncratic").sum()),
                       "n_systemic": int((df["class"] == "systemic").sum()),
                   }}, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
