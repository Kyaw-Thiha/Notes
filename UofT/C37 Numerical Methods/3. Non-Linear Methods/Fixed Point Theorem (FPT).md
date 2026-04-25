# Fixed Point Theorem (FPT)
#numerical-methods/non-linear/fpt 

`Theorem`
If there is an interval $[a, b]$ s.t.
1. $g(x)$ is continuous on $[a, b]$
2. $g(x) \in [a, b], \forall x \in [a, b]$
3. $|| g'(x) || \leq L < 1, \forall x \in [a, b]$

then $g(x)$ has a unique fixed point in $[a, b]$

This is also known as `Functional Convergence Theorem (FCT)`.

---
`Proof`

Start with an arbitrary initial guess $\hat{x}_{0} \in [a, b]$, and iterate.
$\hat{x}_{k+1} = g(\hat{x}_{k})$, $k= 0, 1, 2, \dots$
Then, all $\hat{x}_{k} \in [a, b]$ since $g(x) \in [a, b]$

Furthermore by `Mean Value Theorem (MVT)`,
$$
\begin{align}
x_{k+1} - x_{k}
&= g(x_{k}) - g(x_{k-1}) \\[6pt]
&= g'(n_{k}) (x_{k} - x_{k-1})
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
`1. Proof of Fixed Point`

`WTS`: $\tilde{x}$ is a `Fixed Point`. I.e. $\tilde{x} = g(\tilde{x})$

Note that
- $x_{k} \to \tilde{x}$ as $k \to \infty$
- $x_{k+1} = g(x_{k})$ for all $k$
- $g(x)$ is differentiable on $[a,b] \Rightarrow g$ is continuous on $[a,b]$

`LHS`: By `continuity` of $g(x)$,
$$
\lim_{ k \to \infty } g(x_{k}) = g(\lim_{ k \to \infty } x_{k}) = g(\tilde{x})
$$

`RHS`: By $g(x_{k}) = x_{k+1}$ ,
$$
\lim_{ k \to \infty } g(x_{k})
= \lim_{ k \to \infty } x_{k+1}
= \tilde{x}
$$

Combining `LHS` and `RHS`,
$$
g(\tilde{x}) = \tilde{x}
$$

Hence, $\tilde{x}$ is indeed a `Fixed Point` of $g(x)$.

---
`2. Proof of Uniqueness`

Proof by `Contradiction`.

For the sake of contradiction, suppose there are two fixed points in $[a, b]$.
$$
\tilde{x} \in [a, b] \text{ and } \tilde{y} \in [a, b] \text{ , where } \tilde{x} \neq \tilde{y}
$$
such that 
$$
\tilde{x} = g(\tilde{x}) \text{ and } \tilde{y} = g(\tilde{y})
$$

Then,
$$
\begin{align}
|\tilde{x} - \tilde{y}| 
&= |g(\tilde{x}) - g(\tilde{y})| \\[6pt]
&= |g'(c) (\tilde{x} - \tilde{y})| 
&\text{ by MVT} \\[6pt]
&= |g'(c)| |\tilde{x} - \tilde{y}| \\[6pt]
&\leq L|\tilde{x} - \tilde{y}| 
& \text{since } |g'(x)| \leq L \text{ on } [a, b] \\[6pt]
& & \text{and } c \in [a, b] \text{ by MVT}
\end{align}
$$
Continuing on,
$$
\begin{align}
|\tilde{x} - \tilde{y}| &\leq L|\tilde{x} - \tilde{y}| \\[6pt]

|\tilde{x} - \tilde{y}| - L|\tilde{x} - \tilde{y}| &\leq 0\\[6pt]

(1 - L)|\tilde{x} - \tilde{y}| &\leq 0\\[6pt]
\end{align}
$$
But:
- $L < 1 \implies 1-L > 0$
- $|\tilde{x} - \tilde{y}| \ge 0$ always, and it’s actually $> 0$ if $\tilde{x} \neq \tilde{y}$

Hence, this is a `contradiction`.

$\therefore$ $\tilde{x} = \tilde{y}$. The `fixed point` is `unique`.

---
## See Also
- [[Non-Linear Methods]]
- [[Fixed Point Methods (FPM)]]
- [[Fixed Point Iteration (FPI)]]
