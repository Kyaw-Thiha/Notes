# Fixed Point Theorem (FPT)
#numerical-methods/non-linear/fpt 

`Theorem`
If there is an interval $[a, b]$ s.t.
1. $g(x) \in [a, b], \forall x \in [a, b]$
2. $|| g'(x) || \leq L < 1, \forall x \in [a, b]$

then $g(x)$ has a unique fixed point in $[a, b]$

---
`Proof`

Start with an arbitrary initial guess $\hat{x}_{0} \in [a, b]$, and iterate.
$\hat{x}_{k+1} = g(\hat{x}_{k})$, $k= 0, 1, 2, \dots$
Then, all $\hat{x}_{k} \in [a, b]$ since $g(x) \in [a, b]$

Furthermore by `Mean Value Theorem (MVT)`,
$$
\begin{align}
x_{k+1} - x_{k}
&= g(x_{k}) - g(x_{k+1}) \\[6pt]
&= g'(n_{k}) (x_{k} - x_{k+1})
\end{align}
$$
where $n_{k} \in [x_{k-1}, x_{k}] \subset [a,b]$

Therefore, 
$$
\begin{align}
|x_{k} - x_{k+1}|
&= |g'(n_{k})(x_{k} - x_{k-1})| \\[6pt]
&\leq |g'(n_{k})| \ |x_{k} - x_{k-1}| \\[6pt]
&= L \ |x_{k} - x_{k-1}| \\[6pt]
\end{align}
$$

Then, $|x_{k} - x_{k-1}| \leq \dots \leq L^k \ |x_{1} - x_{0}|$.
Since we know that $L<1$, $|x_{k} - x_{k+1}| \to 0$ as $k \to \infty$.

This means that $x_{k}$ converges to some point $\tilde{x} \in [a, b]$.

To complete the proof we have to show 2 things:
1. $\tilde{x}$ is a `Fixed Point`. I.e. $\tilde{x} = g(\tilde{x})$
2. $\tilde{x}$ is unique


---
## See Also
- [[Non-Linear Methods]]
- [[Fixed Point Methods (FPM)]]
- [[Fixed Point Iteration (FPI)]]
