# Surrogate Loss Function

`Surrogate loss function` is a substitute loss function used in place of the true loss, typically because the true loss is difficult to optimize.

![Surrogate Loss Function|400](https://www.researchgate.net/profile/Qijia-He-5/publication/384704291/figure/fig1/AS:11431281290155728@1731504730380/Convex-surrogate-loss-functions.png)

This `surrogate function` 
- is convex
- upper bounds the original loss function 
  $\ell_{surr}(\mathbf{w}, \ (\mathbf{x}, \mathbf{y})) \geq \ell_{orig}(\mathbf{w}, \ (\mathbf{x}, \mathbf{y}))$

---
## Motivation

With the preceding constraints of [[Convex-Lipschitz-Bounded Learning Problem (CLB)|CLB]] and [[Convex-Smooth-Bounded Learning Problem (CSB)|CSB]], we can assert that the [[Convex Learning Problems|problems]] are learnable.
However, they depend on the loss function being convex, which is not always going to be the case.

For example, consider classification with [[Halfspace|halfspaces]]:
$$
\begin{align}
&\ell^{0-1}(\mathbf{w}, (\mathbf{x}, \mathbf{y}))
\\[6pt]
&= \mathbb{1}(y \neq \text{sign}(\mathbf{w} \cdot \mathbf{x})) \\[6pt]
&= \mathbb{1}(y(\mathbf{w} \cdot \mathbf{x}) \leq 0)
\end{align}
$$
The `0-1 loss function` is not [[Convex Function|convex]] $w.r.t$ $\mathbf{w}$.
And there are many possible local minima in the [[Empirical Risk]].

Hence, we solve it using a `surrogate loss function` that
- is convex
- upper bounds the original loss function 
  $\ell_{surr}(\mathbf{w}, \ (\mathbf{x}, \mathbf{y})) \geq \ell_{orig}(\mathbf{w}, \ (\mathbf{x}, \mathbf{y}))$

---
## Hinge Loss: A Surrogate for 0-1 Loss

Note that `Hinge loss` is defined as
$$
\ell^{hinge}(\mathbf{w}, (\mathbf{x}, \mathbf{y}))
\triangleq \max\{ 0, \ 1-y(\mathbf{w} \cdot \mathbf{x}) \}
$$

![Hinge Loss|300](https://www.researchgate.net/profile/Michael-Jordan-3/publication/220485537/figure/fig1/AS:669532263833601@1536640339061/llustrations-of-the-0-1-loss-function-and-three-surrogate-loss-functions-hinge-loss.png)

Trivially, `hinge` upper bounds `0-1 loss`.
And it is [[Convex Function|convex]] since it is the max of two convex functions.

Now, we can learn $w.r.t$ the `surrogate loss`:
$$
\begin{align}
L_{\mathcal{D}}^{hinge} ( \ A(S) \ )
&\leq \min_{w \in \mathcal{H}}  
L_{\mathcal{D}}^{hinge}(\mathbf{w}) + \epsilon  
\\[6pt]

\implies L_{\mathcal{D}}^{0-1} ( \ A(S) \ )
&\leq \min_{w \in \mathcal{H}} 
L_{\mathcal{D}}^{hinge}(\mathbf{w}) + \epsilon  
\end{align}
$$

This can be rewritten as
$$
L_{\mathcal{D}}^{0-1} ( \ A(S) \ )
\leq \min_{w \in \mathcal{H}} L_{\mathcal{D}}^{0-1}(\mathbf{w})
+ \left( \min_{w \in \mathcal{H}} L_{\mathcal{D}}^{hinge}(\mathbf{w}) 
- \min_{w \in \mathcal{H}}L_{\mathcal{D}}^{0-1}(\mathbf{w}) \right) 
+ \epsilon
$$

---
## Risk Composition for Surrogate Loss

$$
L_{\mathcal{D}}^{0-1} ( \ A(S) \ )
\leq \underbrace{\min_{w \in \mathcal{H}} L_{\mathcal{D}}^{0-1}(\mathbf{w})}
_{\text{Approximation Error}}

+ \underbrace{\left( \min_{w \in \mathcal{H}} L_{\mathcal{D}}^{hinge}(\mathbf{w}) 
- \min_{w \in \mathcal{H}}L_{\mathcal{D}}^{0-1}(\mathbf{w}) \right)}_{\text{Optimization Error}}

+ \underbrace{\epsilon}_{\text{Estimation Error}}
$$

where
- `Approximation error`
  The ability of the hypothesis class to perform on $\mathcal{D}$.
- `Optimization error`
  The performance error because we are optimizing our algorithm $w.r.t$ the surrogate and not the real loss function.
  Depends on $\mathcal{D}$ and the choice of the surrogate loss.
- `Estimation error`
  The error in estimating due to training on a dataset and not $\mathcal{D}$

---
## See Also
- [[Loss Function]]
- [[Convex Learning Problems]]
  