"""H3 ground-truth v2: find the FIRST order of decoupling at each event.

The Chow test was too sensitive in v1 (detected decoupling at every event).
The real signal is in WHICH ORDER decoupling first appears:
- Idiosyncratic events (AAPL earnings): relationship breaks at order 1
  (raw vol diverges between stock and sector)
- Systemic events (FOMC): relationship breaks at HIGHER orders
  (raw vol co-moves, but vol-of-vol diverges)

For each event, run Chow tests at orders 1, 2, 3, 4. The decoupling
order k* is the smallest k where the test rejects. Test whether the
distribution of k* differs between event classes.
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

from volcascade import build, zscore  # noqa: E402
from volcascade.decoupling import chow_statistic  # noqa: E402
from volcascade.io import load_prices  # noqa: E402

RESULTS_DIR = ROOT / "results"

# Reuse the curated event lists
sys.path.insert(0, str(ROOT / "experiments"))
from h3_ground_truth import aapl_earnings_dates, fomc_dates  # noqa: E402


def main() -> None:
    print("=" * 78)
    print("H3 ground-truth v2: first order of decoupling at each event")
    print("=" * 78)

    events = aapl_earnings_dates() + fomc_dates()
    print(f"\n{len(events)} curated events")

    print("\nloading AAPL, XLK (2015-2024)...")
    t0 = time.time()
    prices = load_prices(["AAPL", "XLK"], start="2015-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  loaded {returns.shape[0]} days in {time.time()-t0:.1f}s\n")

    aapl_cascade = build(returns["AAPL"], orders=(1, 2, 3, 4), inner_window=10)
    aapl_z = zscore(aapl_cascade, lookback=120)
    xlk_cascade = build(returns["XLK"], orders=(1, 2, 3, 4), inner_window=10)
    xlk_z = zscore(xlk_cascade, lookback=120)

    # For each event, find decoupling order
    records = []
    t0 = time.time()
    for i, ev in enumerate(events):
        if (i + 1) % 20 == 0:
            print(f"  event {i+1}/{len(events)}  ({time.time()-t0:.1f}s)")
        d = pd.Timestamp(ev["date"])
        if d not in returns.index:
            continue

        end_loc = returns.index.get_loc(d)
        # Use a 60-day window centered on the event. Event is at the MIDDLE
        # of the window so the Chow test can have equal-sized before/after
        # windows (15 days each).
        window_size = 60
        half = window_size // 2
        lookback = 15  # size of each before/after window for Chow
        if end_loc < half or end_loc + half >= len(returns):
            continue

        # For each cascade order, run Chow test
        # k* = smallest order where Chow rejects at p < alpha
        k_star = None
        f_per_order = {}
        for k in [1, 2, 3, 4]:
            aapl_s = aapl_z[k].iloc[end_loc - half:end_loc + half].dropna().to_numpy()
            xlk_s = xlk_z[k].iloc[end_loc - half:end_loc + half].dropna().to_numpy()
            min_len = min(len(aapl_s), len(xlk_s))
            if min_len < 2 * lookback:
                continue
            aapl_s = aapl_s[-min_len:]
            xlk_s = xlk_s[-min_len:]
            # Break is at the middle of the window (index = half)
            # Use the actual midpoint of the trimmed data
            mid = min_len // 2
            f, p, _ = chow_statistic(aapl_s, xlk_s, k=mid, lookback=lookback)
            f_per_order[k] = (float(f) if not np.isnan(f) else None,
                              float(p) if not np.isnan(p) else None)
            if k_star is None and not np.isnan(p) and p < 0.05:
                k_star = k

        records.append({
            "date": ev["date"],
            "label": ev["label"],
            "class": ev["class"],
            "asset": ev["asset"],
            "k_star": k_star,
            "f_per_order": f_per_order,
        })

    df = pd.DataFrame(records)
    df_with_kstar = df.dropna(subset=["k_star"])
    print(f"\nanalyzed {len(df)} events, {len(df_with_kstar)} have non-NaN k_star")

    print("\n" + "=" * 78)
    print("RESULTS: distribution of decoupling order k* by event class")
    print("=" * 78)
    for cls in ["idiosyncratic", "systemic"]:
        sub = df_with_kstar[df_with_kstar["class"] == cls]
        if len(sub) == 0:
            continue
        ks = sub["k_star"]
        print(f"\n  {cls} (n={len(sub)}):")
        for k in [1, 2, 3, 4]:
            count = (ks == k).sum()
            print(f"    k* = {k}: {int(count)} events ({count / len(ks):.1%})")
        no_decoupling = (ks.isna() | (ks == 0)).sum()
        print(f"    no decoupling: {no_decoupling}")
        print(f"    mean k*: {ks.mean():.2f}, median k*: {ks.median():.1f}")

    # Statistical test: Mann-Whitney on k* by class
    idio_ks = df_with_kstar[df_with_kstar["class"] == "idiosyncratic"]["k_star"]
    sys_ks = df_with_kstar[df_with_kstar["class"] == "systemic"]["k_star"]
    if len(idio_ks) > 5 and len(sys_ks) > 5:
        u, p = sps.mannwhitneyu(idio_ks, sys_ks, alternative="two-sided")
        print(f"\n  Mann-Whitney on k*: U={u:.0f}, p={p:.4f}")
        if p < 0.05:
            direction = "lower" if idio_ks.median() < sys_ks.median() else "higher"
            print(f"  --> idiosyncratic events have {direction} k* than systemic (p<0.05)")

    # AUC: predict class from k* (binary: idiosyncratic=1, systemic=0)
    if len(idio_ks) > 5 and len(sys_ks) > 5:
        # AUC of (low k* -> idiosyncratic)
        labels = np.concatenate([np.ones(len(idio_ks)), np.zeros(len(sys_ks))])
        scores = np.concatenate([-idio_ks.to_numpy(), -sys_ks.to_numpy()])  # negate so high score = idiosyncratic
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(labels, scores)
            print(f"\n  AUC (low k* -> idiosyncratic): {auc:.3f}")
        except Exception as e:
            print(f"  AUC failed: {e}")

    out_path = RESULTS_DIR / "h3_ground_truth_v2.json"
    with open(out_path, "w") as f:
        json.dump({
            "events": records,
            "summary": {
                "n_total": int(len(df)),
                "n_with_kstar": int(len(df_with_kstar)),
                "idio_mean_kstar": float(idio_ks.mean()) if len(idio_ks) > 0 else None,
                "sys_mean_kstar": float(sys_ks.mean()) if len(sys_ks) > 0 else None,
            },
        }, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
