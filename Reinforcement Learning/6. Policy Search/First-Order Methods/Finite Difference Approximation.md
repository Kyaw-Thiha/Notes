# Finite Difference Approximation of Policy Gradient

Recall that given a function $f: \mathbb{R} \to \mathbb{R}$, the [[Finite Difference Approximation|finite difference approximation]] of the derivative $f'(x)=\frac{df}{dx}(x)$ is
$$
f'_{\text{FD}}(x)
= \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$
where $\Delta x$ is a small number.

This is called `forward difference approximation`.

---
### Bounding the Error
#### Taylor's Approximation
By [[First-Order Taylor Expansion|Taylor's theorem]], assuming twice differentiability, we have
$$
f(x + \Delta x) 
= f(x) + f'(x) \Delta x
+ f''(z) \ |_{x < z < x + \Delta x}
\frac{(\Delta x)^{2}}{2!}
$$
Therefore,
$$
f'(x) 
= \frac{f(x + \Delta x) - f(x)}{\Delta x}
- f''(z) \ |_{x < z < x + \Delta x}
\frac{(\Delta x)^{2}}{2!}
$$

The error between the FD approximation and $f'(x)$ is
$$
\left| f''(z)\mid_{ x < z < x + \Delta x} 
\frac{(\Delta x)^{2}}{2!} \right|
$$
which is $O((\Delta x)^{2})$.

---
#### Central Difference Approximation
The central difference approximation is
$$
f'_{FD}(x)
= \frac{f(x + \Delta x) - f(x - \Delta x)}{2 \Delta x}
$$
Using it, the error is $O((\Delta x)^{3})$.

---
## Computing the Gradient
To compute the [[Gradient Descent|gradient]] of $J_{\rho}(\pi_{\theta})$ $w.r.t$ $\theta \in \mathbb{R}^{p}$, we need to compute $2p$ evaluations of $J_{\rho}$:
![image|500](https://notes-media.kthiha.com/Finite-Difference-Approximation/40a71e29d114661af525ea03d899a9f2.png)
where $e_{i}$ is a unit vector along dimension $i$ of $\mathbb{R}^{p}$.

---
## Notes
- We cannot directly compute $J_{\rho}(\pi_{\theta})$.
- We can only compute $\hat{J}_{n}(\pi_{\theta})$ using rollouts.
- Replace each $J_{\rho}$ above with their corresponding $\hat{J}_{n}$.
- This requires $2pm$ rollouts in total.
- Given the approximated gradient, it has an error caused by both the [[Finite Difference Approximation|FD approximation]] and using $\hat{J}_{n}$ instead of $J_{\rho}$.
  Using this approximation, we can use [[Gradient Descent|gradient ascent]] to move towards higher value of $J_{\rho}(\pi_{\theta})$:

$$
\theta_{k+1}
\leftarrow \theta_{k} + \alpha_{k} 
\nabla_{\theta}^{(FD)} \ \hat{J}_{n}(\pi_{\theta_{k}})
$$

---
