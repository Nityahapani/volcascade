"""Can the cascade predict DIRECTION of return?

The cascade currently predicts:
- Forward vol (vol-peak): strong
- Magnitude of |return| at event days (H3b): strong
- Direction of return (sign): UNTESTED — was assumed null because the
  cascade is a vol-of-vol statistic, not a return predictor.

Test multiple framings of direction prediction:
1. Direct: cascade slope -> sign of forward 1/5/20-day return (binary)
2. Event-day: cascade slope at event -> sign of event-day return
3. Largest move: cascade slope -> sign of max abs return in next 5 days
4. Cumulative: cascade slope -> sign of cumulative 5-day return

If any of these show AUC > 0.5 consistently across assets, the cascade
DOES predict direction. This would convert the H1 (return) null into a
positive finding.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

ROOT = Path("/opt/data/volcascade")
sys.path.insert(0, str(ROOT / "src"))

from volcascade import build, slope, zscore
from volcascade.io import load_prices

INNER_WINDOW = 10
ZSCORE_LOOKBACK = 120


def main() -> None:
    print("=" * 78)
    print("DIRECTION PREDICTION: cascade slope -> sign of return")
    print("=" * 78)

    ASSETS = ["SPY", "XLE", "XLF", "XLV", "XLY"]
    print(f"\nloading {ASSETS} (2000-2024)...")
    prices = load_prices(ASSETS, start="2000-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  loaded {returns.shape[0]} days\n")

    rows = []
    for asset in ASSETS:
        rets = returns[asset].dropna()
        cascade = build(rets, orders=(1, 2, 3, 4), inner_window=INNER_WINDOW)
        z = zscore(cascade, lookback=ZSCORE_LOOKBACK)
        sample = z[1]
        if isinstance(sample, pd.DataFrame):
            z_s = {k: z[k][asset] for k in [1, 2, 3, 4]}
        else:
            z_s = dict(z)
        s = slope(z_s)

        out = {"asset": asset}

        for h in [1, 5, 20]:
            fwd_ret = pd.Series(np.nan, index=rets.index)
            for i in range(len(rets) - h):
                fwd_ret.iloc[i] = float(rets.iloc[i + h] / rets.iloc[i] - 1.0)
            valid = s.notna() & fwd_ret.notna()
            if valid.sum() < 100:
                continue
            f = fwd_ret[valid]
            sv = s[valid]
            # Spearman of slope vs fwd_return (sign test)
            r, p = sps.spearmanr(sv, f)
            out[f"spear_h{h}"] = float(r)
            out[f"spear_p_h{h}"] = float(p)
            # AUC of slope -> positive fwd_return
            is_pos = (f > 0).astype(int)
            try:
                auc = roc_auc_score(is_pos, sv)
            except Exception:
                auc = np.nan
            out[f"auc_pos_h{h}"] = float(auc) if not np.isnan(auc) else None

        # Event-day: at top-1 |return| day per quarter, does cascade predict SIGN?
        rets_q = rets.copy()
        rets_q.index = pd.to_datetime(rets_q.index)
        quarters = rets_q.index.to_period("Q")
        records = []
        for q in quarters.unique():
            mask = quarters == q
            q_rets = rets_q[mask]
            if len(q_rets) < 5:
                continue
            top_idx = q_rets.abs().idxmax()
            loc = rets.index.get_loc(top_idx)
            slope_val = s.iloc[loc] if not pd.isna(s.iloc[loc]) else np.nan
            if pd.isna(slope_val):
                continue
            ret = float(rets.iloc[loc])
            records.append({"slope": float(slope_val), "ret": ret, "abs_ret": abs(ret)})
        if len(records) >= 10:
            df = pd.DataFrame(records)
            # Spearman slope vs ret (sign test)
            r_event, p_event = sps.spearmanr(df["slope"], df["ret"])
            out["event_spear_ret"] = float(r_event)
            out["event_spear_p"] = float(p_event)
            # AUC slope -> positive ret
            is_pos_event = (df["ret"] > 0).astype(int)
            try:
                auc_event = roc_auc_score(is_pos_event, df["slope"])
            except Exception:
                auc_event = np.nan
            out["event_auc_pos"] = float(auc_event) if not np.isnan(auc_event) else None
            # % positive returns at event
            out["event_pct_pos"] = float(is_pos_event.mean())

        rows.append(out)

    df = pd.DataFrame(rows)
    print("=" * 78)
    print("PER-ASSET RESULTS")
    print("=" * 78)
    print(f"\n  {'asset':6s} | {'spear_h1':>9s} | {'spear_h5':>9s} | {'spear_h20':>9s} | {'auc_h1':>7s} | {'auc_h5':>7s} | {'auc_event':>9s}")
    for _, r in df.iterrows():
        print(f"  {r['asset']:6s} | {r.get('spear_h1', np.nan):+.4f}   | {r.get('spear_h5', np.nan):+.4f}   | "
              f"{r.get('spear_h20', np.nan):+.4f}   | {r.get('auc_pos_h1', np.nan):.3f}  | "
              f"{r.get('auc_pos_h5', np.nan):.3f}  | {r.get('event_auc_pos', np.nan):.3f}")

    # Aggregate
    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    for col, name in [("spear_h1", "Spearman h=1"), ("spear_h5", "Spearman h=5"),
                       ("spear_h20", "Spearman h=20"), ("auc_pos_h1", "AUC pos h=1"),
                       ("auc_pos_h5", "AUC pos h=5"), ("event_spear_ret", "Event Spearman ret"),
                       ("event_auc_pos", "Event AUC pos")]:
        if col in df.columns:
            valid = df[col].dropna()
            n_above = (valid > 0.5).sum() if "auc" in col else (valid > 0).sum()
            med = valid.median()
            sign = "ABOVE 0.5" if "auc" in col else "POSITIVE"
            print(f"  {name:20s}: n={n_above}/{len(valid)} {sign}, median={med:+.4f}")

    # Event-day: % positive vs predicted
    if "event_pct_pos" in df.columns:
        avg_pct = df["event_pct_pos"].mean()
        print(f"\n  At event days, fraction of positive returns: {avg_pct:.1%} (vs 50% baseline)")

    out_path = ROOT / "results" / "direction_prediction.json"
    with open(out_path, "w") as f:
        json.dump({"per_asset": [dict(r) for r in rows],
                   "summary": {
                       "median_spear_h1": float(df["spear_h1"].median()) if "spear_h1" in df else None,
                       "median_spear_h5": float(df["spear_h5"].median()) if "spear_h5" in df else None,
                       "median_spear_h20": float(df["spear_h20"].median()) if "spear_h20" in df else None,
                       "median_auc_pos_h1": float(df["auc_pos_h1"].median()) if "auc_pos_h1" in df else None,
                       "median_auc_pos_h5": float(df["auc_pos_h5"].median()) if "auc_pos_h5" in df else None,
                       "median_event_auc_pos": float(df["event_auc_pos"].median()) if "event_auc_pos" in df else None,
                   }}, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
