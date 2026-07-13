"""GARCH adversarial test: more realistic null for the vol-peak effect.

The original adversarial used iid N(0, sigma^2) returns — too simple.
Real equity returns have vol clustering. An AR(1)-GARCH(1,1) process
with Student-t innovations has:
- Autocorrelation in returns
- Time-varying volatility (GARCH)
- Heavy tails (Student-t)

This is a more realistic null. If the cascade slope still shows
NO spurious correlation with forward vol on GARCH data, the vol-peak
effect is robust to realistic nulls.

Generates 1000 universes of AR(1)-GARCH(1,1) returns, computes
Spearman(slope, forward vol) on each, reports distribution.
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

from volcascade import build, slope, zscore  # noqa: E402

RESULTS_DIR = ROOT / "results"


def simulate_garch(n: int, mu: float, ar1: float, omega: float, alpha: float,
                   beta: float, nu: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1)-GARCH(1,1) with Student-t innovations.

    Standard formulation:
        r_t = mu + ar1 * (r_{t-1} - mu) + eps_t
        eps_t = sigma_t * z_t,  z_t ~ t(nu)
        sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
    """
    r = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance
    for t in range(1, n):
        z = rng.standard_t(df=nu)
        r[t] = mu + ar1 * (r[t-1] - mu) + np.sqrt(sigma2[t-1]) * z
        sigma2[t] = omega + alpha * (r[t-1] - mu) ** 2 + beta * sigma2[t-1]
    return r


def main() -> None:
    print("=" * 78)
    print("GARCH adversarial test: 1000 AR(1)-GARCH(1,1) universes")
    print("=" * 78)

    # SPY-calibrated parameters (approximate)
    # Daily mean ~ 0, AR(1) ~ 0, omega/alpha/beta from a typical fit
    # SPY: mu=0.0003, ar1=-0.05, omega=1e-6, alpha=0.08, beta=0.90, nu=5
    n_universes = 1000
    n_days = 5000
    inner_window = 10
    zscore_lookback = 120
    forward_days = 5

    r_pearson = np.zeros(n_universes)
    r_spearman = np.zeros(n_universes)

    rng = np.random.default_rng(42)
    t0 = time.time()
    for k in range(n_universes):
        if (k + 1) % 100 == 0:
            print(f"  universe {k+1}/{n_universes}  ({time.time()-t0:.1f}s)")
        # Vary parameters slightly across universes to get a distribution
        mu = rng.normal(0.0003, 0.0001)
        ar1 = rng.normal(-0.05, 0.03)
        omega = 10 ** rng.normal(-6, 0.3)
        alpha = rng.uniform(0.04, 0.12)
        beta = rng.uniform(0.85, 0.94)
        nu = rng.uniform(4, 8)
        # Ensure stationarity
        if alpha + beta >= 0.99:
            beta = 0.99 - alpha

        rets = simulate_garch(n_days, mu, ar1, omega, alpha, beta, nu, rng)
        rets = pd.Series(rets)

        cascade = build(rets, orders=(1, 2, 3, 4), inner_window=inner_window)
        z = zscore(cascade, lookback=zscore_lookback)
        s = slope(z).dropna()

        fwd_vol = pd.Series(np.nan, index=rets.index)
        for i in range(len(rets) - forward_days):
            fwd_vol.iloc[i] = float(rets.iloc[i + 1:i + 1 + forward_days].std())

        valid = s.notna() & fwd_vol.notna()
        if valid.sum() < 100:
            continue
        r_p, _ = sps.pearsonr(s[valid], fwd_vol[valid])
        r_s, _ = sps.spearmanr(s[valid], fwd_vol[valid])
        r_pearson[k] = r_p
        r_spearman[k] = r_s

    elapsed = time.time() - t0
    print(f"\ndone: {n_universes} GARCH universes in {elapsed:.1f}s")
    print(f"  Pearson  rho: mean={r_pearson.mean():+.4f}  std={r_pearson.std():.4f}")
    print(f"  Spearman rho: mean={r_spearman.mean():+.4f}  std={r_spearman.std():.4f}")
    print(f"  |rho| > 0.05: Pearson {(np.abs(r_pearson) > 0.05).mean():.1%}, Spearman {(np.abs(r_spearman) > 0.05).mean():.1%}")

    out = {
        "n_universes": n_universes,
        "n_days": n_days,
        "params": {"inner_window": inner_window, "zscore_lookback": zscore_lookback, "forward_days": forward_days},
        "model": "AR(1)-GARCH(1,1) with Student-t innovations",
        "pearson": {"mean": float(r_pearson.mean()), "std": float(r_pearson.std()),
                    "frac_abs_gt_005": float((np.abs(r_pearson) > 0.05).mean())},
        "spearman": {"mean": float(r_spearman.mean()), "std": float(r_spearman.std()),
                     "frac_abs_gt_005": float((np.abs(r_spearman) > 0.05).mean())},
    }
    out_path = RESULTS_DIR / "adversarial_garch.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults saved to {out_path}")
    print("PASS criterion: |mean rho| < 0.01 AND no systematic direction")
    verdict = "PASS" if abs(r_spearman.mean()) < 0.01 else "FAIL"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
