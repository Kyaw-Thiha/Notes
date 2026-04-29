# Online Learning of Mean of Random Variable
#rl/online-learning

> Consider a simple problem of estimating the mean of a random variable, given samples from it.

Concretely, assume we are given $n$ real-valued $r.v$ $Z_{1}, \ \dots, \ Z_{n}$ all drawn $i.i.d$ from a distribution $v$.
How can we estimate expectation $m = \mathbb{E}\ [\mathbb{Z}]$ with $\mathbb{Z} \sim v$?

We could use the [[Sample Average Estimator|sample average estimator]]:
$$
m_{t} \triangleq \frac{1}{t} \sum^t_{i=1} Z_{t}
$$
Under mild conditions by the `Law of Large Numbers`, $m_{t} \to m$.

---
## Online Estimator
The above naive implementation require us to store all $Z_{1}, \ \dots, \ Z_{t}$. 
This is unfeasible when $t$ is very large.

Instead, we could do it online:
$$
\begin{align}
m_{t+1}
&= \frac{1}{t+1} \sum^{t+1}_{i=1} Z_{i} \\[6pt]
&= \frac{1}{t+1} \left[ \sum^{t}_{i=1} Z_{i}  
+ Z_{t+1} \right] \\[6pt]
&= \frac{1}{t+1} [t \ m_{t} + Z_{t+1}] \\[6pt]
&= \left( 1 - \frac{1}{t+1} \right) \ m_{t}
+ \frac{1}{t+1} Z_{t+1}
\end{align}
$$

Let $\alpha_{t} = \frac{1}{t+1}$. Then, we can rewrite this as
$$
m_{t+1} = (1 - \alpha_{t}) \ m_{t} 
+ \alpha_{t} Z_{t+1}
$$
The variable $\alpha_{t}$ is called the `learning rate` or the `step size`.

With this choice of $\alpha_{t}$, the estimate $m_{t}$ converges to $m$ as $t\to \infty$.
Since this is computing [[Sample Average Estimator|empirical mean]], the convergence is guaranteed by [[Weak Law of Large Numbers (LLN)]].

---
## Stochastic Approximation
The equation above is a form of [[Stochastic Approximation(SA)|stochastic approximation]].
For a more detailed analysis of mean and variance with a fixed value of $\alpha_{t}$, you can read it [[Stochastic Approximation (SA) Analysis|here]].

---
## See Also
- [[Sample Average Estimator]]
- [[Weak Law of Large Numbers (LLN)]]
- [[Online Learning of the Reward Function]]