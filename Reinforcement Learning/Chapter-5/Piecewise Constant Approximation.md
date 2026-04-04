# Piecewise Constant Approximation
#rl/vfa/piecewise

We can define a function as piecewise function of multiple [[Linear Function Approximation]].

![Piecewise Constant Approximation|500](https://notes-media.kthiha.com/Piecewise-Constant-Approximation/6e0a07df6ab787bf4a6bf9ed95e709c8.png)

Assume that the domain is $[-b, +b]$.
We can define $\phi_{i}$ (for $i=0, 1, \dots, \left[ \frac{2b}{\epsilon} \right]$) as
$$
\phi_{i}(s)
= \mathbb{I}\{ s \in [-b + i\epsilon, \ -b + (i+1)\epsilon] \}
$$
Any function $V$ can be approximated by a 
$$
\hat{V}(s)
= \hat{V}(s;w)
= \phi(s)^{T}w
$$
with $w \in \mathbb{R}^{[2b/\epsilon] + 1}$.
We can denote such function space as $\mathcal{F}_{\epsilon}$.

---
## Approximation Error
The approximation quality depends on the regularity or structure of the [[Value Function|value function]] $V$. If we allow $V$ to change arbitrary, we cannot hope to have a good approximation.

But if the [[Value Function|value function]] has some regularities, we can make more definitive statements.

For any [[Value Function|value function]] $V$ that is [[Lipschitzness|L-Lipschitz]], we have
$$
\inf_{\hat{V} \in \mathcal{F}_{\epsilon}}
||\hat{V} - V||_{\infty}
\leq L \epsilon
$$
This is called the `approximation error` or `bias`.

The approximation error depends on
- **structure of function approximator**
  $\text{e.g:}$ piecewise constant, piecewise linear, etc
- **class of functions being approximated**
  $\text{e.g:}$ $L$-Lipschitz functions

---
## Curse of Dimensionality in the no. of parameters
If the domain was $\mathcal{S} = [-1, +1]^{d}$ for $d\geq 1$, we would need
$$
O\left( \frac{1}{\epsilon^{d}} \right)
$$
parameters to describe such a function.
Note that the term $\epsilon^{d}$ causes it to grow exponentially fast as a function of $d$.

This exponential growth of the number of parameters required to represent a high-dimensional function is an instance of the [[Curse of Dimensionality]].

---
## Estimation Error
We also need to pay attention to the statistical aspect of estimating a function within this function space using a finite number of data points.

The estimation accuracy depends on complexity of $\mathcal{F}$.
There are different ways of quantifying this:
- VC Dimension
- Covering Number / Metric Entropy
- Rademacher Complexity

Briefly speaking, the statistical error behaves as
$$
O\left( \sqrt{ \frac{\log |\mathcal{F}|}{n} } \right)
$$
where $n$ is the number of data points used in the estimation.

- Recall that the variance of [[Sample Average Estimator]] $m_{t}$ with $t$ data points is $\frac{\sigma^{2}}{t}$.
  This means that
$$
\mathbb{E}[|m_{t} - \mu|^{2}]
= \mathbb{E}[|m_{t} - \mathbb{E}[m_{t}]|^{2}]
= \frac{\sigma^{2}}{t}
$$
	This implies that
$$
\mathbb{E}[|m_{t} - \mu|] 
\leq \sqrt{ \mathbb{E}[|m_{t} - \mu|^{2}] }
= \frac{\sigma}{\sqrt{ t }}
$$

- Comparing with $O\left( \sqrt{ \frac{\log |\mathcal{F}|}{n} } \right)$ here, for the [[Sample Average Estimator|mean estimation problem]], $|\mathcal{F}|$ is effectively equal to $1$.

---
## See Also 
- [[Linear Function Approximation]]