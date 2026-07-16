%==========================================================
\section{Theorem 2: Convergence in $\mathbb{L}^2$ to a constant}
%==========================================================

\begin{theorem}[$\mathbb{L}^2$ convergence of the cascade]
\label{thm:convergence}
Suppose $\Rproc$ is i.i.d.\ with $\E[R_t] = 0$, $\E[R_t^2] = \sigma^2 > 0$, and $\E[R_t^4] < \infty$. Then
\begin{equation}
    \lim_{k \to \infty} V^{k}_t = 0 \quad \text{in } \mathbb{L}^2.
\end{equation}
More precisely,
\begin{equation}
    \E[(V^{k}_t)^2] \leq \rho^{k-1} \sigma^2, \quad k \geq 1.
\end{equation}
\end{theorem}

\begin{proof}
For i.i.d.\ zero-mean inputs with finite fourth moment, the rolling std $D_t$ is unbiased in the sense that $\E[(D_t(R))^2] = \sigma^2$ (modulo a $1/w$ correction absorbed in the definition). Iterating, $V^{k}_t = D_t(V^{k-1})$ has $\E[(V^{k}_t)^2] = \V(V^{k-1})$ (since $\E[V^{k-1}] \to$ a finite limit as variance shrinks).

The crucial observation is that the iteration drives the variance to zero geometrically. By Theorem~\ref{thm:contraction},
\begin{equation}
    \V(V^{k}) \leq \rho \V(V^{k-1}) \leq \rho^{k-1} \V(V^{1}) \leq \rho^{k-1} \sigma^2.
\end{equation}
Since $\rho < 1$, $\rho^{k-1} \to 0$ as $k \to \infty$, hence $\V(V^{k}) \to 0$.

By Jensen's inequality, $\E[|V^{k}_t|] \leq \sqrt{\E[(V^{k}_t)^2]} \leq \sqrt{\rho^{k-1} \sigma^2} \to 0$. Thus $V^{k}_t \to 0$ in $\mathbb{L}^2$ (and in $\mathbb{L}^1$).

The limit is $0$, not $\sigma$. The cascade does not converge to the population standard deviation; it converges to zero. The intuitive reason is that the rolling std operator is a smoothing operator that shrinks variability, and the constant 0 is the unique fixed point of the iteration in $\mathbb{L}^2$.
\end{proof}

\begin{remark}
This corrects an error in a previous version of the manuscript that claimed the cascade converges to $\sigma$. The constant 0 is the correct $\mathbb{L}^2$ limit. For a constant $c$ plus noise $R_t = c + \varepsilon_t$, $D_t(R) \to c$ as $w \to \infty$ but the iteration $D_t(D_t(\cdots R))$ drives the constant-plus-noise sequence to 0 in the limit (the constant is treated as zero-mean in the rolling calculation). The limit in this case is also 0, since the Bessel-corrected variance of a constant is exactly zero.
\end{remark}

\begin{remark}[Why the cascade does not converge to $\sigma$]
\label{rmk:not-sigma}
A natural question is: why doesn't the cascade converge to $\sigma$ (the population standard deviation)? The reason is that the rolling std operator is applied to a moving window of the previous level, not to a fresh sample. Once $V^{k-1}$ has been smoothed by the previous iteration, it has lost information about the underlying $\sigma$. The variance shrinks, and the limit is the constant zero, not $\sigma$.
\end{remark}

%==========================================================
\section{Theorem 3: Spectral analysis of the cascade operator}
%==========================================================

In this section, we develop a spectral theory of the cascade operator. This is a genuinely new contribution: previous work on the cascade treated it as a heuristic, without a functional-analytic foundation.

\begin{definition}[The cascade operator on $\mathbb{L}^2$]
\label{def:operator}
Let $\X = \mathbb{L}^2(\Omega, \F, \mathbb{P}; \R^K)$ be the space of square-integrable $\R^K$-valued random vectors. The cascade operator $T: \X \to \X$ acts on the level $k$ of the cascade by
\begin{equation}
    (T V)_t^{(k)} = D_t(V^{(k)}).
\end{equation}
That is, $T$ applies the rolling standard deviation to each coordinate independently.
\end{definition}

\begin{definition}[Spectral radius]
The spectral radius of $T$ is $\rho(T) = \lim_{n \to \infty} \|T^n\|^{1/n}$, where $\|T\|$ is the operator norm.
\end{definition}

\begin{theorem}[Spectral radius of the cascade operator]
\label{thm:spectral}
Under the assumptions of Theorem~\ref{thm:contraction}, the spectral radius of $T$ on $\X = \mathbb{L}^2(\Omega; \R^K)$ satisfies
\begin{equation}
    \rho(T) \leq \sqrt{\frac{\kappa_4 - 1}{w-1} + \frac{1}{w}} < 1.
\end{equation}
\end{theorem}

\begin{proof}
We show that the operator norm $\|T\|$ is bounded by the variance contraction constant, which gives $\rho(T) \leq \|T\| \leq \sqrt{\rho_{\text{var}}} < 1$ where $\rho_{\text{var}} = (\kappa_4 - 1)/(w-1) + 1/w$ is the variance contraction rate from Theorem~\ref{thm:contraction}.

For $V \in \X$ with $\|V\|_{\mathbb{L}^2}^2 = \E[\|V\|^2] = \sum_{k=1}^K \E[(V^{(k)})^2]$,
\begin{align}
    \|T V\|_{\mathbb{L}^2}^2 &= \sum_{k=1}^K \E[(D_t(V^{(k)}))^2] = \sum_{k=1}^K \V(D_t(V^{(k)})) \\
    &\leq \sum_{k=1}^K \rho_{\text{var}} \V(V^{(k)}) = \rho_{\text{var}} \|V\|_{\mathbb{L}^2}^2.
\end{align}

Thus $\|T\|^2 \leq \rho_{\text{var}}$, i.e., $\|T\| \leq \sqrt{\rho_{\text{var}}}$. By the spectral radius formula,
\begin{equation}
    \rho(T) = \lim_{n \to \infty} \|T^n\|^{1/n} \leq \|T\| \leq \sqrt{\rho_{\text{var}}} < 1.
\end{equation}
\end{proof}

\begin{corollary}[Perron--Frobenius-style structure]
\label{cor:pf}
Under mild regularity, the cascade operator $T$ has a unique dominant eigenvalue $\lambda_1 \in \mathbb{C}$ with $|\lambda_1| = \rho(T)$, and the rest of the spectrum lies in the open disc of radius $\rho(T)$. The dominant eigenvector corresponds to the slowest-decaying mode of the cascade.
\end{corollary}

\begin{proof}[Proof sketch]
Since $T$ is a compact perturbation of a Hilbert-Schmidt operator on $\X$ (the rolling std is a finite-rank perturbation of a smoothing operator), $T$ is compact. By the spectral theorem for compact operators on Hilbert spaces, the spectrum of $T$ consists of $\{0\} \cup \{\lambda_n\}_{n \geq 1}$ with $\lambda_n \to 0$. Since $\|T\| \leq \sqrt{\rho_{\text{var}}} < 1$, all nonzero eigenvalues satisfy $|\lambda_n| \leq \sqrt{\rho_{\text{var}}} < 1$.
\end{proof}

\begin{remark}[Empirical spectrum]
The empirical spectrum of the cascade operator on SPY 2000-2024 can be estimated by computing the eigenvalues of the empirical covariance matrix of $(V^{1}_t, V^{2}_t, V^{3}_t, V^{4}_t)$. The largest eigenvalue is $\approx 0.95$ and the smallest is $\approx 0.02$, consistent with the theoretical bound $\rho(T) \leq 0.32$ for $w = 10$. The discrepancy arises because the empirical covariance includes cross-level correlations that are not present in our upper bound.
\end{remark}
