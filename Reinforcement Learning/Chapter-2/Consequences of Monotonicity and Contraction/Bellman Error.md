# Error Bounds of Bellman Equation
#rl/bellman-equation/residual #rl/bellman-equation/error
By [[Uniqueness of Fixed Point]], we know that $V = T^{\pi}V$ $(\text{or } V=T^{*}V)$ implies $V=V^{\pi}$ $(\text{or } V=V^{*})$.
But given $V \approx T^{\pi}V$ $(\text{or } V \approx T^{*}V)$, how close is the approximate solution $V$ to $V^{\pi}$ $(\text{or } V^{*})$.

---
## Sources of Error
- **Computation Error**: Finiteness of computational budget prevents our algorithms to be ran forever.
- **Representational Error**: `Function approximator` are used to represent the [[Value Function]].
  This may introduce an error on how well we approximate the true value function.
- **Statistical Errors**

---
## Error Bound based on Bellman Error
For any $V \in \mathcal{B}(\mathcal{S})$ or $Q \in \mathcal{B}(\mathcal{S} \times \mathcal{A})$, we have
$$
\boxed{ \ ||V - V^{*}||_{\infty} \leq 
\frac{|| V- T^{*}V ||_{\infty}}{1 - \gamma} \ }
\ , \quad
\boxed{ \ ||Q - Q^{*}||_{\infty} \leq 
\frac{|| Q- T^{*}Q ||_{\infty}}{1 - \gamma} \ }
$$
where
- the quantity $\boxed{BR(V) \triangleq T^{\pi}V - V}$ and $\boxed{BR^{*}(V) = T^{*}V - V}$ are `Bellman Residuals`
- their [[Norm|norms]] are called `Bellman Errors`

---
### Proof
We want to start by upper bounding $||V - V^{*}||_{\infty}$.
$$
\begin{align}
V - V^{*} &= V - T^{*}V + T^{*}V - V^{*} \\[8pt]
\implies ||V - V^{*}||_{\infty}
&= ||V - T^{*}V + T^{*}V - V^{*}||_{\infty} \\[8pt]
&\leq ||V - T^{*}V||_{\infty} 
+ ||T^{*}V - V^{*}||_{\infty}
\end{align}
$$
Focusing on the term $||T^{*}V - V^{*}||_{\infty}$, we can make $2$ observations:
- $V^{*} = T^{*}V^{*}$
- The [[Bellman Optimality Operator|Bellman optimality operator]] is a $\gamma$-[[Contraction of Bellman Operator|contraction]] $w.r.t$ the [[Infinity Norm|supremum norm]].

Thus,
$$
\begin{align}
||T^{*}V - V^{*}||_{\infty}
&= ||T^{*}V - T^{*}V^{*}||_{\infty} \\[6pt]
&\leq \gamma \ ||V - V^{*}||_{\infty}
\end{align}
$$
Substituting it back in, we get
$$
\begin{align}
&||V - V^{*}||_{\infty}
\leq ||V - T^{*}V||_{\infty} 
+ \gamma||V - V^{*}||_{\infty}\\[8pt]

\implies &(1 - \gamma) \ ||V - V^{*}||_{\infty}
\leq ||V - T^{*}V||_{\infty} \\[8pt]

\implies &||V - V^{*}||_{\infty}
\leq \frac{||V - T^{*}V||_{\infty}}{ 1 - \gamma}
\\ & & \blacksquare
\end{align}
$$

---
## Error Bound for Policy Evaluation
For any $V \in \mathcal{B}(\mathcal{S})$ or $Q \in \mathcal{B}(\mathcal{S} \times \mathcal{A})$, and any $\pi \in \Pi$, we have
$$
\boxed{ \ ||V - V^{\pi}||_{\infty} \leq 
\frac{|| V- T^{\pi}V ||_{\infty}}{1 - \gamma} \ }
\ , \quad
\boxed{ \ ||Q - Q^{\pi}||_{\infty} \leq 
\frac{|| Q- T^{\pi}Q ||_{\infty}}{1 - \gamma} \ }
$$

### Proof
This result can be proven in the same way as above.
But we will prove it differently.

Using the fact that $V^{\pi}$ is the [[Fixed Point|fixed point]] of the [[Bellman Operator]] $(V^{\pi} = T^{\pi}V^{\pi})$,

$$
\begin{align}
V - V^{\pi}
&= V - T^{\pi}V + T^{\pi}V - V^{\pi} \\[6pt]
&= (V - T^{\pi}V) + (T^{\pi}V - T^{\pi}V^{\pi}) \\[6pt]
&= (V - T^{\pi}V) + \gamma \ \mathcal{P}^{\pi}(V  
- V^{\pi}) \\[6pt]
\implies (\mathbf{I} - \gamma \mathcal{P}^{\pi})
(V - V^{\pi}) &= V - T^{\pi}V
\end{align}
$$

As $||\gamma \ \mathcal{P}^{\pi}||_{\infty} = \gamma$ is less than $1$, we can use some lemma to conclude that $\mathbf{I} - \gamma \mathcal{P}^{\pi}$ is `non-singular`. 
Hence, it is invertible. Therefore,
$$
V - V^{\pi} = (\mathbf{I} - \gamma
\mathcal{P}^{\pi})^{-1} \ (V - T^{\pi}V)
$$

By taking the [[Infinity Norm|supremum norm]] of both sides, we get
$$
\begin{align}
||V - V^{\pi}||_{\infty}
&= ||(\mathbf{I} - \gamma \mathcal{P}^{\pi})^{-1}
\ (V - T^{\pi}V)||_{\infty} \\[6pt]
&\leq ||(\mathbf{I} - \gamma  
\mathcal{P}^{\pi})^{-1}||_{\infty}
\cdot \ ||V - T^{\pi}V||_{\infty} \\[6pt]
\end{align}
$$
We can bound $||(\mathbf{I} - \gamma \mathcal{P}^{\pi})^{-1}||_{\infty}$ as
$$
\begin{align}
||(\mathbf{I} - \gamma  
\mathcal{P}^{\pi})^{-1}||_{\infty}
&\leq \frac{1}{1 - ||\gamma \ \mathcal{P}^{\pi}||_{\infty}} \\[6pt]
&= \frac{1}{1 - \gamma}
\end{align}
$$

Substituting it back in, 
$$
\begin{align}
||V - V^{\pi}||_{\infty}
\leq \frac{||V - T^{\pi}V||_{\infty}}{ 1 - \gamma}
\\ & & \blacksquare
\end{align}
$$

---
## Error Bound Tightness Coefficient
We can define `error bound tightness coefficient` as
the ratio of the upper bound in these results 
to actual value of quantity of interest for particular $V$.

$$
\text{EB-subopt}^{\pi}(V)
= \frac{\frac{||V - T^{\pi}V||_{\infty}}
{1 - \gamma}}
{||V - V^{\pi}||_{\infty}}
$$

We can also define
$$
\text{EB-supopt}^{\pi}
= \sup_{\mathcal{V} \in \mathcal{B}(\mathcal{S})}
\text{EB-supopt}^{\pi}(V)
$$

---
## See Also
- [[Monotonicity of Bellman Operator]]
- [[Contraction of Bellman Operator]]
- [[Uniqueness of Fixed Point]]