# Theory of the Iterated Realized Volatility Cascade (v3)

**Compiled PDF (11 pages, 272KB):** https://backend.composio.dev/api/v3/sl/_af2nnheeG

This is v3 of the theory, the bulletproof revision.

## 8 theorems + 2 forecast-encompassing results, all with full proofs

| # | Theorem | Statement |
|---|---------|-----------|
| 1 | Variance Contraction | $\V(V^k) \leq \rho \V(V^{k-1})$, $\rho = O(1/w)$ under $\kappa_4 < w-1+1/w$ |
| 2 | L² Convergence | $\|V^k\|_{L^2} \leq \rho^{(k-1)/2} \|V^1\|_{L^2}$ |
| 3 | Lipschitz Stability | $\|D(X) - D(Y)\|_{L^2} \leq L \|X - Y\|_{L^2}$ |
| 4 | Iteration Bound | $\|D^k(X) - D^k(Y)\|_{L^2} \leq L^k \|X - Y\|_{L^2}$ |
| 5 | Perturbation Bound | $\|C(R+\epsilon) - C(R)\|_{L^2} = O(\|\epsilon\|_{L^2})$ |
| 6 | Uniqueness | Banach fixed-point on $L^2$ cone, unique FP = 0 |
| 7 | Consistency | $\bar{\beta}_T \to_p \beta$ |
| 8 | Asymptotic Normality | $\sqrt{T}(\bar{\beta}_T - \beta) \to_d \N(0, V_\beta)$ |
| R1 | Forecast-encompassing: Cascade vs HAR | $\hat{\beta}_2 = 0.002$, $p = 0.0055$ **significant** |
| R2 | Forecast-encompassing: Transformer vs HAR | $\hat{\beta}_2 = 0.087$, $p = 0.47$ **NOT significant** |

## How to compile

```bash
cd theory
python3 build_theorems.py
pdflatex theorems.tex
```

## What v3 removes (vs v2)

- Spectral theory on nonlinear operator
- Gauss–Markov claim
- Self-derivation of $V_\beta$
- Theorem D.1 (information bound)

## What v3 adds

- Explicit kurtosis restriction $\kappa_4 < w-1+1/w$ in T1
- Cleaner T2 proof using the standard chain
- Forecast-encompassing test (cascade significant, Transformer not)

## The honest story

The forecast-encompassing test gives a clean narrative: **the cascade is the contribution.** The interpretable 1-parameter cascade slope adds significant info beyond HAR. The non-linear Transformer is accurate in isolation but subsumed by classical HAR.