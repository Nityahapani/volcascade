# Theory (FINAL) — Iterated Realized Volatility Cascade

**Compiled PDF (9 pages, 258KB):** https://backend.composio.dev/api/v3/sl/fapS0g1S45

This is the **final, journal-ready** version of the volcascade theory. It is in a NEW folder `theory_final/` (not `theory/`) so there are no merge conflicts with the existing `theory/` folder.

## 4 theorems (rigorous proofs)

| Theorem | Statement |
|---------|-----------|
| T1 Variance Contraction | $\V(V^{(k)}) \leq \rho \V(V^{(k-1)}) + O(w^{-2})$, $\V(V^{(k)}) = O(\rho^k)$ |
| T2 Convergence | $\V(V^{(k)}) \to 0$, $V^{(k)} \to \mu$ in $L^2$ |
| T3 Asymptotic Inference | $\bar{\beta}_T \to_p \beta$, $\sqrt{T}(\bar{\beta}_T - \beta) \to_d \N(0, \Omega)$ |
| T4 Affine Invariance | $\tilde{V}^{(k)} = a V^{(k)}$, $\tilde{z}^{(k)} = z^{(k)}$, $\tilde{\beta} = \beta$ |

## Comprehensive empirical

- Forecast encompassing (HEADLINE): Cascade vs HAR $p = 0.0055$
- Clark-West: $p = 0.019$
- Forecast combination: HAR + Cascade vs HAR wins on MSE, MAE, QLIKE
- CER: Vol-timing 0.284 vs B&H 0.146, improvement +0.138
- Forecast horizon robustness: $\rho(h)$ negative for all $h \in \{1, 2, 3, 5, 10, 20\}$

## How to compile

```bash
cd theory_final
python3 build_theorems.py
pdflatex theorems.tex
```

## Files

- `theorems_part1.md` — preamble + Assumptions 1-3 + T1-T4
- `theorems_part2.md` — empirical analysis
- `theorems_part3.md` — discussion + bibliography
- `build_theorems.py` — assembles 3 parts into one theorems.tex

## This PR replaces

- PR #19 (v1), #20 (v2), #21 (v3), #22 (v4), #23 (v5), #25 (v6), #26 (v7)

The earlier PRs had merge conflicts because they all modified the same files in `theory/`. This PR is in `theory_final/` so it should merge cleanly.

## The story

The cascade contributes information not contained in HAR. Forecast encompassing, Clark-West, nested regression, forecast combination, residual ACF, Ljung-Box, and rolling-origin CV all confirm. The combined model (HAR + Cascade) significantly outperforms HAR on MSE, MAE, and QLIKE. The cascade is the contribution.
