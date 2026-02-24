# Learning Stability
A `learning rule` is unstable if a small change in the input makes a big change in the output.

> Stable Rules do not overfit.

Let
- $A$ be the `learning algorithm`.
- $S$, of $m$ examples be the `training set`.
- $A(S)$ be the output of $A$

Note that the algorithm overfits if $L_{\mathcal{D}}( \ A(S) \ ) - L_{S}( \ A(S) \ )$.
But we will focus on
$$
\mathbb{E}_{S}[ \ L_{\mathcal{D}}(A (S)) - L_{S}(A(S))\ ]
$$

Let us define stability in terms of dataset changes.
Define an example $z'$ and a dataset 
$$
S^{(i)} = (z_{1}, \ z_{2}, \ \dots, \ z_{i-1}, \ z', \ z_{i+1}, \ \dots, \ z_{m})
$$
where $z'$ is swapped in for $z_{i}$.

---
## Theorem
> Stable Rules do not overfit.

**Claim**
Let 
- $\mathcal{D}$ be a `distribution`
- $S = (z_{1}, \ \dots, \ z_{m})$ be an $i.i.d$ sequence of examples
- $S^{(i)} = (z_{1}, \ z_{2}, \ \dots, \ z_{i-1}, \ z', \ z_{i+1}, \ \dots, \ z_{m})$
- $z'$ be another $i.i.d$ example
- $U(m)$ be an `uniform distribution` over $[m]$.

Then for any `learning algorithm`,
$$
\begin{align}
&\mathbb{E}_{S \sim \mathcal{D}^m}
[ \ L_{\mathcal{D}}(A(S)) - L_{S}(A(S)) \ ] \\[6pt]
&= \mathbb{E}_{(S, z') \sim \mathcal{D}^{m+1},  
\ i \sim U(m)} [ \ \ell(A(S^{(i)}), \ z_{i}) \  
- \ell(A(S), \ z_{i})]
\end{align}
$$

**Proof**
$S$ and $\mathbf{z}'$ are both $i.i.d$ from $\mathcal{D}$.
Therefore for each $i$,
$$
\begin{align}
&\mathbb{E}_{S} [ \ L_{\mathcal{D}}(A(S)) \ ] \\[6pt]

&= \mathbb{E}_{S, \ z'} [ \ \ell(A(S), z') \ ]
&\text{by defn of true risk} \\[6pt]

&= \mathbb{E}_{S, \ z'} [ \ \ell(A(S^{(i)}), \ z') \ ]
&\text{by i.i.d samples}
\end{align}
$$
and
$$
\begin{align}
&\mathbb{E}_{S} [ \ L_{\mathcal{S}}(A(S)) \ ] \\[6pt]
&= \mathbb{E}_{S, \ i} [ \ \ell(A(S), \ z_{i}) \ ]
&\text{by defn of empirical risk}
\end{align}
$$

Distribute expectation in the `RHS` of the claim.
Then, note that $p(i) = \frac{1}{m}$ and $\mathbb{E}[x] = x$.
Hence, we have proven the claim.

---
## On-Average-Replace-One Stable
Let $\epsilon: \mathbb{N} \to \mathbb{R}$ be a `monotonically decreasing function`.

We say a learning algorithm $A$ is `on-average-replace-one-stable` with rate $\epsilon(m)$ if for every distribution $\mathcal{D}$:
$$
\mathbb{E}_{(S, \ z') \sim \mathcal{D}^{m+1}, \ i \sim U(m)}
[ \ \ell(A(S^{(i)}) , \ z_{i}) 
- \ell(A(S), \ z_{i}) \ ] \leq \epsilon(m)
$$
We can conclude from the [[#Theorem|last result]] that if the above holds, 
then the algorithm is not overfit $\iff$ it is `on-average-replace-one stable`.

---
## Controlling Fitting-Stability Trade-Off
Consider the [[Empirical Risk|expected risk]]:
$$
\begin{align}
&\mathbb{E}_{S} [ \ L_{\mathcal{D}}(A(S)) \ ] \\[6pt]
&= \mathbb{E}_{S} [ \ L_{S}(A(S)) \ ]
+ \mathbb{E}_{S}[ \ L_{\mathcal{D}}(A(S))  
- L_{S}(A(S)) \ ]
\end{align}
$$
we can use knowledge of the [[Loss Function|loss functions]] to bound the expected risk and make learnability guarantees.

---
