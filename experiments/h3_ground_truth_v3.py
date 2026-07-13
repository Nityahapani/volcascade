"""H3 ground-truth v3: use the decoupling F-statistic MAGNITUDE at order 1.

In v2, decoupling order k* didn't differentiate well (both event classes
have k*=1 most of the time). Try a different metric:

The F-statistic at order 1 measures HOW STRONGLY the stock-sector
relationship breaks. Higher F = stronger decoupling. The hypothesis:
idiosyncratic events have stronger decoupling (the relationship truly
breaks) than systemic events (where the relationship is just noisy
because everything is moving).

Also test: the relative F at order 1 vs order 4. If the relationship
breaks MORE at order 1 than at order 4, the event is idiosyncratic.
If it breaks at all orders equally, the event is systemic.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from volcascade import build, zscore  # noqa: E402
from volcascade.decoupling import chow_statistic  # noqa: E402
from volcascade.io import load_prices  # noqa: E402
from h3_ground_truth import aapl_earnings_dates, fomc_dates  # noqa: E402

RESULTS_DIR = ROOT / "results"


def main() -> None:
    print("=" * 78)
    print("H3 v3: F-statistic magnitude at order 1 (and ratio across orders)")
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

    records = []
    t0 = time.time()
    for i, ev in enumerate(events):
        if (i + 1) % 30 == 0:
            print(f"  event {i+1}/{len(events)}  ({time.time()-t0:.1f}s)")
        d = pd.Timestamp(ev["date"])
        if d not in returns.index:
            continue

        end_loc = returns.index.get_loc(d)
        half = 30
        lookback = 15
        if end_loc < half or end_loc + half >= len(returns):
            continue

        # F-statistic at each cascade order
        f_at_order = {}
        for k in [1, 2, 3, 4]:
            aapl_s = aapl_z[k].iloc[end_loc - half:end_loc + half].dropna().to_numpy()
            xlk_s = xlk_z[k].iloc[end_loc - half:end_loc + half].dropna().to_numpy()
            min_len = min(len(aapl_s), len(xlk_s))
            if min_len < 2 * lookback:
                f_at_order[k] = None
                continue
            aapl_s = aapl_s[-min_len:]
            xlk_s = xlk_s[-min_len:]
            mid = min_len // 2
            f, p, _ = chow_statistic(aapl_s, xlk_s, k=mid, lookback=lookback)
            f_at_order[k] = float(f) if not np.isnan(f) else None

        # Derived metrics
        f1 = f_at_order.get(1)
        f2 = f_at_order.get(2)
        f3 = f_at_order.get(3)
        f4 = f_at_order.get(4)
        # Ratio of order-1 to order-4 F (high = decoupling at low orders = idiosyncratic)
        ratio_1_4 = (f1 / f4) if (f1 is not None and f4 is not None and f4 > 0) else None
        # Mean F across orders
        valid_fs = [f for f in [f1, f2, f3, f4] if f is not None]
        mean_f = float(np.mean(valid_fs)) if valid_fs else None
        # Order-1 F (the primary metric)
        max_f = max(valid_fs) if valid_fs else None

        records.append({
            "date": ev["date"],
            "label": ev["label"],
            "class": ev["class"],
            "asset": ev["asset"],
            "f1": f1, "f2": f2, "f3": f3, "f4": f4,
            "ratio_1_4": ratio_1_4,
            "mean_f": mean_f,
            "max_f": max_f,
        })

    df = pd.DataFrame(records)
    print(f"\nanalyzed {len(df)} events")

    print("\n" + "=" * 78)
    print("RESULTS: F-statistic metrics by event class")
    print("=" * 78)
    for cls in ["idiosyncratic", "systemic"]:
        sub = df[df["class"] == cls]
        if len(sub) == 0:
            continue
        print(f"\n  {cls} (n={len(sub)}):")
        for col in ["f1", "f2", "f3", "f4", "ratio_1_4"]:
            valid = sub[col].dropna()
            if len(valid) > 0:
                print(f"    {col}: mean={valid.mean():.3f}, median={valid.median():.3f}")

    # Test each metric for class differentiation
    print("\n" + "=" * 78)
    print("CLASS DIFFERENTIATION TESTS")
    print("=" * 78)
    for col in ["f1", "f2", "f3", "f4", "ratio_1_4", "mean_f", "max_f"]:
        idio = df[df["class"] == "idiosyncratic"][col].dropna()
        sys_ = df[df["class"] == "systemic"][col].dropna()
        if len(idio) > 5 and len(sys_) > 5:
            u, p = sps.mannwhitneyu(idio, sys_, alternative="two-sided")
            print(f"\n  {col}: idio median={idio.median():.3f}, sys median={sys_.median():.3f}, p={p:.4f}")
            # AUC
            labels = np.concatenate([np.ones(len(idio)), np.zeros(len(sys_))])
            scores = np.concatenate([idio.to_numpy(), sys_.to_numpy()])
            try:
                auc = roc_auc_score(labels, scores)
                print(f"    AUC (high {col} -> idiosyncratic) = {auc:.3f}")
            except Exception:
                pass

    out_path = RESULTS_DIR / "h3_ground_truth_v3.json"
    with open(out_path, "w") as f:
        json.dump({"events": [r for r in records],
                   "summary": {
                       "n_total": int(len(df)),
                       "n_idiosyncratic": int((df["class"] == "idiosyncratic").sum()),
                       "n_systemic": int((df["class"] == "systemic").sum()),
                   }}, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
