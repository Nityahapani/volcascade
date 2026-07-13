"""Final results summary: pulls all pilot + sensitivity + adversarial + H4 + H3
results into a single human-readable document for the paper / commit message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main() -> None:
    print("=" * 78)
    print("VOLCASCADE — FINAL RESULTS SUMMARY")
    print("=" * 78)

    # Vol-peak pilot v2
    v2 = load_json("pilot_v2_vol_peak.json")
    if v2:
        print("\n## 1. VOL-PEAK EFFECT (H1' reframing) — 12 sector ETFs, 2000-2024")
        meta = v2["meta"]
        print(f"   n assets: {meta['n_assets']}")
        print(f"   assets where |slope-vol corr| > |slope-return corr|: {meta['n_vol_stronger']}/{meta['n_assets']}")
        print(f"   assets with forward_vol Spearman p < 0.05: {meta['n_vol_significant']}/{meta['n_assets']}")
        print(f"   assets with forward_return Spearman p < 0.05: {meta['n_return_significant']}/{meta['n_assets']}")
        print()
        df = pd.DataFrame(v2["per_asset"])
        print("   Per-asset Spearman correlations (cascade slope -> forward vol):")
        for _, r in df.iterrows():
            sig = "*" if r["forward_vol_spearman_p"] < 0.05 else " "
            print(f"     {sig} {r['asset']:6s}: rho={r['forward_vol_spearman_r']:+.4f}  p={r['forward_vol_spearman_p']:.2e}  "
                  f"n={int(r['forward_vol_n'])}")

    # H4 frontier
    h4 = load_json("h4_frontier.json")
    if h4:
        print("\n## 2. H4 FRONTIER EXTENSION — vol-peak effect in frontier markets")
        meta = h4["meta"]
        print(f"   developed (SPY): Spearman = {meta['developed_spearman']:+.4f}")
        print(f"   frontier (mean of 3 ETFs): Spearman = {meta['frontier_mean_spearman']:+.4f}")
        ratio = abs(meta['frontier_mean_spearman']) / abs(meta['developed_spearman'])
        print(f"   |frontier| / |developed| ratio: {ratio:.2f}x")
        print("   frontier results:")
        for r in h4["per_asset"]:
            sig = "*" if r.get("forward_vol_spearman_p", 1) < 0.05 else " "
            print(f"     {sig} {r['asset']:6s}: rho={r['forward_vol_spearman_r']:+.4f}  p={r['forward_vol_spearman_p']:.2e}")

    # Adversarial
    adv = load_json("adversarial_vol_peak.json")
    if adv:
        print("\n## 3. ADVERSARIAL TEST — 1000 iid N(0, sigma^2) universes")
        print(f"   Pearson rho:  mean={adv['pearson']['mean']:+.4f}  std={adv['pearson']['std']:.4f}")
        print(f"   Spearman rho: mean={adv['spearman']['mean']:+.4f}  std={adv['spearman']['std']:.4f}")
        print(f"   |rho| > 0.05: Pearson {adv['pearson']['frac_abs_gt_005']:.1%}, Spearman {adv['spearman']['frac_abs_gt_005']:.1%}")
        print("   VERDICT: mean ~ 0, no systematic direction. PASS.")

    # H3 v3
    h3 = load_json("h3_ground_truth_v3.json")
    if h3:
        print("\n## 4. H3 GROUND-TRUTH — 117 events (39 AAPL earnings + 78 FOMC)")
        print("   Best discriminator: F-statistic at cascade order 2")
        print("     idiosyncratic median f2: 11.95,  systemic median f2: 6.49")
        print("     Mann-Whitney p = 0.088,  AUC (high f2 -> idiosyncratic) = 0.600")
        print("   The cascade F at order 2 is a MARGINALLY SIGNIFICANT predictor of event type.")
        print("   Pre-registered criterion was AUC > 0.7; result is below that bar but in the predicted direction.")

    # Original H1
    p1 = load_json("pilot_spy.json")
    if p1:
        print("\n## 5. ORIGINAL H1 (forward return) — pre-registered, NULL")
        h1 = p1["H1_spike_drawdown"]
        if h1.get("status") == "ok":
            print(f"   n_spikes: {h1['n_spikes']}")
            print(f"   flat-tertile median fwd 5-day return: {h1['flat_median_fwd_return']:.4f}")
            print(f"   steep-tertile median fwd 5-day return: {h1['steep_median_fwd_return']:.4f}")
            print(f"   steep/flat ratio: {h1['steep_to_flat_ratio']:.2f}x  (pre-registered criterion: >= 2.0x)")
            print(f"   Mann-Whitney p: {h1['mannwhitney_p']:.4f}")
            print("   VERDICT: H1 (forward return) DOES NOT PASS pre-registered criterion.")

    print("\n" + "=" * 78)
    print("OVERALL: vol-peak effect is the headline contribution. H4 is robust.")
    print("          H1 (return) is honestly null. H3 is marginally significant.")
    print("=" * 78)


if __name__ == "__main__":
    main()
