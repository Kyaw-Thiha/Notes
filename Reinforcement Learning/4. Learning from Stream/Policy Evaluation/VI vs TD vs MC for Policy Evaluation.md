# VI vs TD vs MC
[[Value Iteration]], [[Monte Carlo Estimation for Policy Evaluation|Monte Carlo estimation]], and [[Temporal Difference Learning for Policy Evaluation(TD)|Temporal Difference learning]] can all be see as procedures to estimate the [[Value Function]] $V^{\pi}$.

Hence, it is helpful to know the type of approximations they are doing:
- In [[Monte Carlo Estimation for Policy Evaluation|Monte Carlo estimation]], we use $G^{\pi}$ as the target value.
  - As $V^{\pi}(s) = \mathbb{E}[G^{\pi} \mid S=s]$, we have a noisy but unbiased estimate of $V^{\pi}$. 
  - [[Stochastic Approximation(SA)|SA]] allows us to converge to its mean.
- In [[Value Iteration|VI]], we update $V_{k+1} \leftarrow (T^{\pi}V_{k})(s) = \mathbb{E}[R + \gamma V_{k}(S') \mid S=s]$.
  - Here we do not know $V^{\pi}$, but use the current approximation $V_{k}$ instead.
  - Because of the [[Contraction of Bellman Operator|contraction property of the Bellman operator]], this converges to $V^{\pi}$.
- In [[Temporal Difference Learning for Policy Evaluation(TD)|TD learning]], the target is $R + \gamma V_{k}(S')$.
  This has two sources of approximation:
  - we use $V_{k}$ instead of $V^{\pi}$.
  - we use a sample to estimate an expectation.

---
## See Also
- [[Value Iteration]]
- [[Monte Carlo Estimation for Policy Evaluation]]
- [[Temporal Difference Learning for Policy Evaluation(TD)]]
- [[Value Function]]
- [[Contraction of Bellman Operator]]
- [[Stochastic Approximation(SA)]]