"""GARCH-residual test: is the vol-peak effect BEYOND GARCH dynamics?

The GARCH adversarial showed that the cascade has a spurious -0.087
correlation with forward vol on GARCH noise. The real-data effect is
-0.20, ~2.3x the GARCH null.

This is a critical test: if the vol-peak effect PERSISTS on
GARCH-RESIDUALIZED returns (returns divided by their GARCH-conditional
volatility), then the cascade is detecting something beyond GARCH
dynamics. If it DISAPPEARS, the effect is fully GARCH-driven.

Procedure:
1. Fit a GARCH(1,1) to each return series (SPY + sector ETFs).
2. Compute residuals = returns / conditional_std.
3. Apply the cascade to residuals.
4. Test Spearman(slope, forward vol of RESIDUALS).
5. Compare to the original effect on raw returns.

If the effect PERSISTS on residuals, the vol-peak is real (beyond GARCH).
If it DISAPPEARS, the effect is GARCH-driven and the paper should
reframe as "the cascade captures GARCH structure, not new information."
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
from volcascade.io import SP500_SECTOR_ETFS, load_prices  # noqa: E402

RESULTS_DIR = ROOT / "results"


def garch_residuals(rets: pd.Series) -> pd.Series:
    """Fit GARCH(1,1) with Student-t innovations, return standardized residuals."""
    am = arch_model(rets * 100, mean="AR", lags=1, vol="GARCH", p=1, q=1,
                    dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    # Standardized residuals = (returns - mean) / conditional_std
    std_res = res.std_resid
    # Drop the first few observations (GARCH warmup)
    return std_res.dropna()


def analyze(rets: pd.Series, name: str, inner_window=10, zscore_lookback=120, forward_days=5):
    """Compute Spearman(slope, forward vol) for raw and GARCH-residualized returns."""
    out = {"asset": name, "n_days": int(len(rets))}

    # Raw returns
    cascade_raw = build(rets, orders=(1, 2, 3, 4), inner_window=inner_window)
    z_raw = zscore(cascade_raw, lookback=zscore_lookback)
    sample = z_raw[1]
    if isinstance(sample, pd.DataFrame):
        z_s = {k: z_raw[k][name] for k in [1, 2, 3, 4]}
    else:
        z_s = dict(z_raw)
    s_raw = slope(z_s)

    fwd_vol_raw = pd.Series(np.nan, index=rets.index)
    for i in range(len(rets) - forward_days):
        fwd_vol_raw.iloc[i] = float(rets.iloc[i + 1:i + 1 + forward_days].std())

    valid = s_raw.notna() & fwd_vol_raw.notna()
    if valid.sum() > 100:
        r, p = sps.spearmanr(s_raw[valid], fwd_vol_raw[valid])
        out["raw_spearman_r"] = float(r)
        out["raw_spearman_p"] = float(p)
        out["raw_n"] = int(valid.sum())

    # GARCH residuals
    try:
        resids = garch_residuals(rets)
        cascade_res = build(resids, orders=(1, 2, 3, 4), inner_window=inner_window)
        z_res = zscore(cascade_res, lookback=zscore_lookback)
        s_res = slope(dict(z_res))

        fwd_vol_res = pd.Series(np.nan, index=resids.index)
        for i in range(len(resids) - forward_days):
            fwd_vol_res.iloc[i] = float(resids.iloc[i + 1:i + 1 + forward_days].std())

        valid_res = s_res.notna() & fwd_vol_res.notna()
        if valid_res.sum() > 100:
            r, p = sps.spearmanr(s_res[valid_res], fwd_vol_res[valid_res])
            out["garch_residual_spearman_r"] = float(r)
            out["garch_residual_spearman_p"] = float(p)
            out["garch_residual_n"] = int(valid_res.sum())
    except Exception as e:
        out["garch_error"] = str(e)

    return out


def main() -> None:
    print("=" * 78)
    print("GARCH-residual test: is vol-peak BEYOND GARCH?")
    print("=" * 78)

    print("\nloading SPY + 5 sector ETFs (2000-2024)...")
    t0 = time.time()
    prices = load_prices(list(SP500_SECTOR_ETFS)[:6], start="2000-01-01", end="2024-12-31")
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  loaded {returns.shape[0]} days in {time.time()-t0:.1f}s\n")

    results = []
    for asset in returns.columns:
        rets = returns[asset].dropna()
        out = analyze(rets, asset)
        results.append(out)
        raw_r = out.get("raw_spearman_r")
        res_r = out.get("garch_residual_spearman_r")
        print(f"  {asset}: raw rho={raw_r:+.4f}  garch-residual rho={res_r:+.4f}")

    df = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print("HEADLINE COMPARISON")
    print("=" * 78)
    print(f"\n  Spearman on RAW returns (the vol-peak effect):")
    print(f"    median: {df['raw_spearman_r'].median():+.4f}")
    print(f"    significant (p<0.05): {(df['raw_spearman_p'] < 0.05).sum()}/{len(df)}")
    print(f"\n  Spearman on GARCH-RESIDUALIZED returns (beyond GARCH?):")
    print(f"    median: {df['garch_residual_spearman_r'].median():+.4f}")
    print(f"    significant (p<0.05): {(df['garch_residual_spearman_p'] < 0.05).sum()}/{len(df)}")

    # Paired test: is the residual effect significantly different from 0?
    valid = df.dropna(subset=["garch_residual_spearman_r"])
    if len(valid) > 0:
        t, p = sps.ttest_1samp(valid["garch_residual_spearman_r"], 0)
        print(f"\n  one-sample t-test (garch-residual vs 0): t={t:+.3f}  p={p:.4f}")

    # Ratio: residual / raw
    valid_both = df.dropna(subset=["raw_spearman_r", "garch_residual_spearman_r"])
    if len(valid_both) > 0:
        valid_both = valid_both.copy()
        valid_both["ratio"] = valid_both["garch_residual_spearman_r"] / valid_both["raw_spearman_r"]
        print(f"\n  Ratio of residual to raw effect (per asset):")
        for _, r in valid_both.iterrows():
            print(f"    {r['asset']:6s}: ratio={r['ratio']:+.3f}")
        print(f"  median ratio: {valid_both['ratio'].median():.3f}")
        print("  (ratio near 0 = vol-peak is mostly GARCH; ratio near 1 = mostly beyond GARCH)")

    out_path = RESULTS_DIR / "garch_residual_test.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_asset": [dict(r) for r in results],
            "summary": {
                "n_assets": int(len(df)),
                "raw_median_spearman": float(df["raw_spearman_r"].median()),
                "garch_residual_median_spearman": float(df["garch_residual_spearman_r"].median()),
                "garch_residual_n_significant": int((df["garch_residual_spearman_p"] < 0.05).sum()),
            },
        }, f, indent=2)
    print(f"\nresults saved to {out_path}")


if __name__ == "__main__":
    main()
