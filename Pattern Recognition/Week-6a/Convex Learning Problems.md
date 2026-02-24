# Convex Learning Problem
A learning problem $(\mathcal{H}, \mathcal{Z}, \mathcal{l})$ is `convex` if
- the hypothesis class $\mathcal{H}$ is a [[Convex Set|convex set]], and
- the loss function $\mathcal{l}(\cdot, z)$ is a [[Convex Function|convex]]

![Convex Learning Problem|400](https://miro.medium.com/v2/resize:fit:1352/0*mQBoDp77GAoKnBEE.jpg)

Example: Consider `least squared regression`.
Then, 
- The `hypothesis class` is $\mathcal{H}=\{ \mathbf{x} \to \langle \mathbf{w}, \mathbf{x}\rangle: \mathbf{w} \in \mathbb{R}^d \}$.
- and the `loss function` is $l(h, (\mathbf{x},y)) = (\langle \mathbf{w}, \mathbf{x} \rangle - y)^2$


---
## Empirical Risk Minimization
We can model the problem as $\mathcal{H}=\mathbb{R}^d$ and $\mathcal{Z} = \mathcal{X} \times \mathcal{Y} = \mathbb{R}^d \times \mathbb{R} = \mathbb{R}$.
Then, $\mathcal{H}$ is obviously `convex` and $l$ is `convex` $w.r.t$ $\mathbf{w}$.

From prior results, we can deduce that $l$ is a [[Convex Function|convex loss function]] and $\mathcal{H}$ is a [[Convex Set|convex set]].
Then, finding [[Empirical Risk Minimization (ERM)|empirical risk minimization]] $\text{ERM}_{\mathcal{H}}$ is a `convex optimization function`.

Recall that
$$
\begin{align}
\text{ERM}_{\mathcal{H}}
&= \arg \min_{w \in \mathcal{H}} L_{S}(\mathbf{w})  \\[6pt]
&= \arg \min_{w \in \mathcal{H}} \frac{1}{m}
\sum^m_{i=1} l(\mathbf{w}, \mathbf{z}_{i})
\end{align}
$$
since a weighted sum of convex functions is a convex function.

---
## Adding Constraints
In order to prove learnability, we need additional constraints on the `learning problem`.
We can do this by defining two classes of learning problem:
1. **Convex-Lipschitz-Bounded Learning Problem (CLB)**
   - Bound the norm of hypothesis function $\mathbf{w} \in \mathcal{H}$.
   - Assert [[Lipschitzness]] on the loss function $\ell$.
   [[Convex-Lipschitz-Bounded Learning Problem (CLB)|Read More]]
   
2. **Convex-Smooth-Bounded Learning Problem**
   - Bound the norm of hypothesis function $\mathbf{w} \in \mathcal{H}$.
   - Assert [[Lipschitz Smoothness]] on the loss function $\ell$.
   [[Convex-Smooth-Bounded Learning Problem (CSB)|Read More]]

---
## Surrogate Loss Function
Even with constraints like [[Convex-Lipschitz-Bounded Learning Problem (CLB)|CLB]] and [[Convex-Smooth-Bounded Learning Problem (CSB)|CSB]], we still need to assert [[Convex Function|convexity]].

In order to learn such non-convex loss functions, we replace them with a `surrogate loss function` that 
- is convex
- upper bounds the original loss function 
  $\ell_{surr}(\mathbf{w}, \ (\mathbf{x}, \mathbf{y})) \geq \ell_{orig}(\mathbf{w}, \ (\mathbf{x}, \mathbf{y}))$

[[Surrogate Loss Function|Read More]]

---
## See Also
- [[Convex Set]]
- [[Convex Function]]
- [[Lipschitzness]]
- [[Lipschitz Smoothness]]
- [[Convex-Lipschitz-Bounded Learning Problem (CLB)]]
- [[Convex-Smooth-Bounded Learning Problem (CSB)]]
- [[Surrogate Loss Function]]