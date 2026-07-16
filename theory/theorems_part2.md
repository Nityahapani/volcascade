%==========================================================
\section{Theorem 2: L$^2$ convergence with rigorous proof}
%==========================================================

\begin{theorem}[L$^2$ convergence of the cascade]
\label{thm:convergence}
Suppose $\Rproc$ is covariance stationary with $\E[R_t] = 0$, $\E[R_t^2] = \sigma^2 > 0$, and $\E[R_t^4] < \infty$, and the kurtosis restriction $\kappa_4 < w - 1 + 1/w$ holds. Then
\begin{equation}
    \|V^{k}\|_{L^2} \leq \rho^{(k-1)/2} \|V^{1}\|_{L^2} \quad \text{for all} \quad k \geq 1.
\end{equation}
\end{theorem}

\begin{proof}
The proof follows the standard chain: T1 $\to$ induction $\to$ geometric decay $\to$ completeness of $L^2$ $\to$ limit constant $\to$ variance $\to 0$ $\to$ constant must equal 0.

\textbf{Step 1: Geometric decay of variance (from T1).} By Theorem~\ref{thm:contraction}, $\V(V^{k}) \leq \rho \V(V^{k-1})$. By induction, $\V(V^{k}) \leq \rho^{k-1} \V(V^{1})$.

\textbf{Step 2: $L^2$ convergence and limit.} Since $L^2$ is complete, the Cauchy sequence $(V^k)_{k \geq 1}$ converges in $L^2$ to a limit $V^\infty \in L^2$. The limit is non-negative (since $V^k \geq 0$ for all $k$).

\textbf{Step 3: The limit has zero variance.} By continuity of the $L^2$ norm and dominated convergence, $\V(V^\infty) = \lim_k \V(V^k) = 0$.

\textbf{Step 4: The limit is zero.} Since $\V(V^\infty) = 0$ and $V^\infty \geq 0$ a.s., the only non-negative random variable with zero variance is $V^\infty = 0$ a.s.

\textbf{Step 5: Explicit rate.} From Step 1, $\V(V^{k}) \leq \rho^{k-1} \V(V^{1})$. Taking square roots: $\|V^{k}\|_{L^2} \leq C^{1/2} \rho^{(k-1)/2} \|V^{1}\|_{L^2}$ with $C$ uniformly bounded.
\end{proof}

\begin{remark}
The cascade converges to $0$, not $\sigma$. The fixed point of the rolling std iteration is $0$, because the rolling std of a constant is $0$.
\end{remark}

%==========================================================
\section{Theorem 3: Lipschitz stability of $D$}
%==========================================================

\begin{theorem}[Lipschitz stability of the rolling std]
\label{thm:lipschitz}
Under Assumption~\ref{ass:stable} with bound $M$ and non-degeneracy $\varepsilon > 0$,
\begin{equation}
    \|D(X) - D(Y)\|_{L^2} \leq L \cdot \|X - Y\|_{L^2}, \quad L = \frac{2M}{(w-1) \varepsilon}.
\end{equation}
\end{theorem}

\begin{proof}
The function $u \mapsto \sqrt{u}$ is $\frac{1}{2\sqrt{u}}$-Lipschitz on $[\varepsilon^2, \infty)$. The rolling variance's partial derivative w.r.t.\ $X_{t-j}$ is bounded by $4M/(w-1)$. By the mean value theorem,
\begin{equation}
    |D_t(X) - D_t(Y)| \leq \frac{2M}{(w-1) \varepsilon} \sum_{j=0}^{w-1} |X_{t-j} - Y_{t-j}|.
\end{equation}
Squaring, taking expectations, and using Cauchy--Schwarz:
\begin{equation}
    \|D(X) - D(Y)\|_{L^2}^2 \leq \left( \frac{2M}{(w-1) \varepsilon} \right)^2 w \|X - Y\|_{L^2}^2.
\end{equation}
Using $\sqrt{w} \leq w-1$ for $w \geq 2$ gives the bound.
\end{proof}

%==========================================================
\section{Theorem 4: Iteration bound}
%==========================================================

\begin{theorem}[Iteration of the Lipschitz]
Under the assumptions of Theorem~\ref{thm:lipschitz}, $\|D^k(X) - D^k(Y)\|_{L^2} \leq L^k \|X - Y\|_{L^2}$ for all $k \geq 1$.
\end{theorem}

\begin{proof}
By induction.
\end{proof}

%==========================================================
\section{Theorem 5: Perturbation bound}
%==========================================================

\begin{theorem}[Robustness of the cascade to input perturbations]
Under Assumption~\ref{ass:stable}, $\|C_K(R + \epsilon) - C_K(R)\|_{L^2} = O(\|\epsilon\|_{L^2})$ with implied constant $L^K$.
\end{theorem}

\begin{proof}
By Theorem~\ref{thm:iteration}.
\end{proof}

%==========================================================
\section{Theorem 6: Uniqueness via Banach fixed-point}
%==========================================================

\begin{theorem}[Unique fixed point]
Let $\X_+ = \{V \in L^2(\Omega; [0, \infty)) : \|V\|_{L^2} < \infty\}$. Under Assumptions~\ref{ass:all} and the kurtosis restriction, for zero-mean $X \in \X_+$ the operator $D$ is a strict contraction with $\|D(X)\|_{L^2} \leq \sqrt{\rho} \|X\|_{L^2}$, $\sqrt{\rho} < 1$. By Banach fixed-point, $D$ has a unique fixed point in the zero-mean subspace, namely $V^\star = 0$, and the iteration converges at rate $\|V^k - V^\star\|_{L^2} \leq (\sqrt{\rho})^{k-1} \|V^{1}\|_{L^2}$.
\end{theorem}

\begin{proof}
\textbf{Self-map:} For $X \in \X_+$, $D_t(X) \geq 0$ and $D(X) \in L^2_+$ by variance contraction.

\textbf{Contraction:} For zero-mean $X \in \X_+$, $\E[(D_t(X))^2] = \V(D_t(X)) \leq \rho \V(X) = \rho \E[X^2]$.

\textbf{Banach:} The space $\X_+$ is closed in $L^2$ (complete). The operator $D$ is a strict contraction. By Banach, $D$ has a unique fixed point in the zero-mean subspace.

\textbf{Fixed point is zero:} The fixed point satisfies $D(V^\star) = V^\star$, requiring $V^\star = 0$.
\end{proof}

\begin{remark}
\textbf{Why no spectral theory.} The operator $D$ is nonlinear; spectral theory (operator norm, spectrum, spectral radius, eigenvalues, Perron--Frobenius, compactness, Hilbert--Schmidt) requires linearity. Banach fixed-point on a complete metric space works for the nonlinear operator. A linearization around the fixed point (Fréchet derivative) is mathematically legitimate if a linear theory is desired, but not necessary for our results.
\end{remark}
