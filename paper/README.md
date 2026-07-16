# Paper v1

This is the v1 pre-print of the iterated realized volatility cascade paper.

## Files

- `paper/v1.md` — full paper (37KB)
- `paper/v1_part1.md` — title + abstract + intro + method + theoretical results (~14KB)
- `paper/v1_part2.md` — empirical results (~10KB)
- `paper/v1_part3.md` — discussion + conclusion + appendices (~12KB)

## How to combine the parts

```bash
cat paper/v1_part1.md paper/v1_part2.md paper/v1_part3.md > paper/v1.md
```

The combined file is identical to v1.md.

## Citation

```
Hapani, N. and pong (2026). Iterated Realized Volatility Cascade: A
Manifold Geometry and Operator Learning Framework for Forward
Volatility Forecasting. v1 pre-print.
```

## Status

Pre-print, version v1 (2026-07-16). Ready for submission.

## Version history

- v1 (current): pre-registered FNO + refreshed manifold + 34-stock H3b panel. Bonferroni-corrected. True OOS holdout.
- v0 (earlier): used un-preregistered 0.089 test Spearman (later shown to be a pre-reg overfit). Superseded.
