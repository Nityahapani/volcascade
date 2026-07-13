"""Improve direction prediction.

The current direction signal is modest (AUC 5/5 > 0.5, median 0.524).
Try to make it strong via:

1. CONDITIONAL: only predict direction on high-vol days (top quartile)
   where the cascade is more informative.

2. CUMULATIVE: predict direction of 5-day and 20-day cumulative return
   rather than 1-day. The cascade might predict the recovery PERIOD
   better than individual days.

3. COMBINED: combine cascade slope with recent return momentum to
   predict direction. The cascade captures vol, momentum captures
   trend; together they predict direction.

4. EVENT-DAY: predict sign of the BIGGEST move in next 5 days
   (the event-day framing of direction).

5. STRESS-ONLY: at high-vol days, the cascade's signal might be
   stronger. Test AUC conditional on vol regime.
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
    print("IMPROVING DIRECTION PREDICTION")
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

        # Forward 5-day cumulative return
        cum_5d = pd.Series(np.nan, index=rets.index)
        for i in range(len(rets) - 5):
            cum_5d.iloc[i] = float(np.log(rets.iloc[i + 5] / rets.iloc[i]))
        cum_20d = pd.Series(np.nan, index=rets.index)
        for i in range(len(rets) - 20):
            cum_20d.iloc[i] = float(np.log(rets.iloc[i + 20] / rets.iloc[i]))

        # 1-day forward return
        fwd_1d = pd.Series(np.nan, index=rets.index)
        for i in range(len(rets) - 1):
            fwd_1d.iloc[i] = float(rets.iloc[i + 1] - rets.iloc[i])

        # 20-day rolling realized vol
        vol_20d = rets.rolling(20, min_periods=10).std() * np.sqrt(252)

        # ==== Method 1: cumulative 5-day return direction ====
        valid = s.notna() & cum_5d.notna()
        if valid.sum() > 100:
            f = cum_5d[valid]
            sv = s[valid]
            r, p = sps.spearmanr(sv, f)
            out["spear_cum5"] = float(r)
            is_pos = (f > 0).astype(int)
            try:
                out["auc_cum5"] = float(roc_auc_score(is_pos, sv))
            except Exception:
                out["auc_cum5"] = None

        # ==== Method 2: cumulative 20-day return direction ====
        valid = s.notna() & cum_20d.notna()
        if valid.sum() > 100:
            f = cum_20d[valid]
            sv = s[valid]
            r, p = sps.spearmanr(sv, f)
            out["spear_cum20"] = float(r)
            is_pos = (f > 0).astype(int)
            try:
                out["auc_cum20"] = float(roc_auc_score(is_pos, sv))
            except Exception:
                out["auc_cum20"] = None

        # ==== Method 3: conditional on high-vol ====
        valid = s.notna() & fwd_1d.notna() & vol_20d.notna()
        if valid.sum() > 100:
            vol_thresh = vol_20d[valid].quantile(0.75)
            is_high_vol = vol_20d[valid] > vol_thresh
            f = fwd_1d[valid]
            sv = s[valid]
            # AUC among high-vol days only
            if is_high_vol.sum() > 30:
                is_pos_hv = (f[is_high_vol] > 0).astype(int)
                try:
                    out["auc_hv_h1"] = float(roc_auc_score(is_pos_hv, sv[is_high_vol]))
                except Exception:
                    out["auc_hv_h1"] = None
                out["n_hv"] = int(is_high_vol.sum())
            # AUC among low-vol days only
            is_low_vol = ~is_high_vol
            if is_low_vol.sum() > 30:
                is_pos_lv = (f[is_low_vol] > 0).astype(int)
                try:
                    out["auc_lv_h1"] = float(roc_auc_score(is_pos_lv, sv[is_low_vol]))
                except Exception:
                    out["auc_lv_h1"] = None
                out["n_lv"] = int(is_low_vol.sum())

        # ==== Method 4: combined signal (cascade + momentum) ====
        # 5-day momentum
        mom_5d = rets.rolling(5, min_periods=3).sum()
        valid = s.notna() & fwd_1d.notna() & mom_5d.notna()
        if valid.sum() > 100:
            f = fwd_1d[valid]
            sv = s[valid]
            mv = mom_5d[valid]
            # Standardize both
            sv_z = (sv - sv.mean()) / sv.std()
            mv_z = (mv - mv.mean()) / mv.std()
            # Combined: cascade + (negated) momentum
            # Negative momentum -> vol often increases -> cascade rises
            # So use: signal = -cascade + momentum? Or cascade - momentum?
            # Try several combinations
            is_pos = (f > 0).astype(int)
            for combo_name, combo_signal in [
                ("cascade_only", sv_z),
                ("momentum_only", mv_z),
                ("cascade_plus_momentum", sv_z + mv_z),
                ("cascade_minus_momentum", sv_z - mv_z),
                ("cascade_times_momentum", sv_z * mv_z),
            ]:
                try:
                    out[f"auc_{combo_name}"] = float(roc_auc_score(is_pos, combo_signal))
                except Exception:
                    out[f"auc_{combo_name}"] = None

        rows.append(out)

    df = pd.DataFrame(rows)
    print("=" * 78)
    print("PER-ASSET: 5-day cumulative direction")
    print("=" * 78)
    for _, r in df.iterrows():
        print(f"  {r['asset']:6s}: AUC_cum5={r.get('auc_cum5', np.nan):.3f}  "
              f"AUC_cum20={r.get('auc_cum20', np.nan):.3f}")

    print("\n" + "=" * 78)
    print("PER-ASSET: conditional on high-vol (top quartile)")
    print("=" * 78)
    for _, r in df.iterrows():
        n_hv = r.get("n_hv", 0)
        n_lv = r.get("n_lv", 0)
        auc_hv = r.get("auc_hv_h1", np.nan)
        auc_lv = r.get("auc_lv_h1", np.nan)
        print(f"  {r['asset']:6s}: high-vol (n={n_hv}) AUC={auc_hv:.3f}  "
              f"low-vol (n={n_lv}) AUC={auc_lv:.3f}")

    print("\n" + "=" * 78)
    print("PER-ASSET: combined signals (cascade + momentum)")
    print("=" * 78)
    for _, r in df.iterrows():
        print(f"  {r['asset']:6s}: cascade={r.get('auc_cascade_only', np.nan):.3f}  "
              f"mom={r.get('auc_momentum_only', np.nan):.3f}  "
              f"c+m={r.get('auc_cascade_plus_momentum', np.nan):.3f}  "
              f"c-m={r.get('auc_cascade_minus_momentum', np.nan):.3f}  "
              f"c*m={r.get('auc_cascade_times_momentum', np.nan):.3f}")

    # Aggregate
    print("\n" + "=" * 78)
    print("AGGREGATE: best methods")
    print("=" * 78)
    for col, name in [("auc_cum5", "5-day cumulative"),
                       ("auc_cum20", "20-day cumulative"),
                       ("auc_hv_h1", "high-vol only (h=1)"),
                       ("auc_lv_h1", "low-vol only (h=1)"),
                       ("auc_cascade_only", "cascade only"),
                       ("auc_momentum_only", "momentum only"),
                       ("auc_cascade_plus_momentum", "cascade + momentum"),
                       ("auc_cascade_minus_momentum", "cascade - momentum"),
                       ("auc_cascade_times_momentum", "cascade * momentum")]:
        if col in df.columns:
            valid = df[col].dropna()
            n_above = (valid > 0.5).sum()
            med = valid.median()
            sign_str = f"  --> {n_above}/{len(valid)} above 0.5" if n_above > 0 else ""
            print(f"  {name:35s}: median={med:+.3f}{sign_str}")

    # Find the best
    best_col = None
    best_med = 0
    for col in df.columns:
        if col.startswith("auc_") and col in df:
            med = df[col].median()
            if med > best_med and not pd.isna(med):
                best_med = med
                best_col = col
    print(f"\n  BEST METHOD: {best_col} with median AUC = {best_med:.3f}")

    out_path = ROOT / "results" / "improved_direction.json"
    with open(out_path, "w") as f:
        json.dump({"per_asset": [dict(r) for r in rows],
                   "best_method": best_col,
                   "best_median_auc": float(best_med) if not pd.isna(best_med) else None,
                   "summary": {col: float(df[col].median()) for col in df.columns if col.startswith("auc_") and col in df}}, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
