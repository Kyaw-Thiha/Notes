# Policy Gradient Theorem
#rl/policy/gradient #math 

Recall that for [[Policy Gradient Methods]], the `Objective Function` is 
$$
J(\theta) = \sum_{\tau} P(\tau; \theta).R(\tau)
$$

So, differentiating get us
$$
\begin{align}
\nabla J(\theta)  &= \nabla _{\theta} \sum_{\tau} P(\tau; \theta).R(\tau) \\


&=  \sum_{\tau} \nabla _{\theta} [ P(\tau; \theta).R(\tau) ] \\
&=  \sum_{\tau} \nabla _{\theta}  P(\tau; \theta).R(\tau) \quad  \text{since } R(\tau) \text{ is not dependent on } \tau
  \\

\end{align}
$$

Multiplying both sides by $\frac{P(\tau; \theta)}{P(\tau; \theta)}$, we get

$$
\begin{align}
\nabla J(\theta)
&= \sum_{\tau} \frac{P(\tau; \theta)}{P(\tau; \theta)} \nabla_{\theta} P(\tau; \theta) R(\tau) \\
&= \sum_{\tau} P(\tau; \theta) \frac{\nabla_{\theta} P(\tau; \theta)}{P(\tau; \theta)} R(\tau)
\end{align}
$$

Now we can apply the `log-derivative trick`, which states that for any differentiable function $f(x)$,

$$
\nabla_x \log f(x) = \frac{\nabla_x f(x)}{f(x)}.
$$

Using this, we replace the fraction with the gradient of the log:

$$
\begin{align}
\nabla J(\theta)
&= \sum_{\tau} P(\tau; \theta) \nabla_{\theta} \log P(\tau; \theta) R(\tau)
\end{align}
$$

This is the `likelihood ratio policy gradient`, also known as the `REINFORCE` form.

Since summing over all trajectories is infeasible, we approximate the expectation by sampling $m$ trajectories $\tau^{(1)}, \dots, \tau^{(m)}$ from the policy. The Monte Carlo estimate becomes

$$
\begin{align}
\nabla_{\theta} J(\theta)
&\approx \hat{g}
= \frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} \log P(\tau^{(i)}; \theta) R(\tau^{(i)}).
\end{align}
$$

Next, we expand $P(\tau^{(i)}; \theta)$ using the definition of a trajectory probability:

$$
P(\tau^{(i)}; \theta)
= \mu(s_0) \prod_{t=0}^{H} P(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)}) \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)}),
$$

where 
- $\mu(s_0)$ is the initial state distribution,  
- $P(s_{t+1} \mid s_t, a_t)$ is the environment transition dynamics,  
- $\pi_{\theta}(a_t \mid s_t)$ is the policy parameterized by $\theta$.

Taking the log of the product gives

$$
\begin{align}
\log P(\tau^{(i)}; \theta)
&= \log \mu(s_0)
+ \sum_{t=0}^{H} \log P(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)})
+ \sum_{t=0}^{H} \log \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)}).
\end{align}
$$

Differentiating with respect to $\theta$:

$$
\begin{align}
\nabla_{\theta} \log P(\tau^{(i)}; \theta)
&= \nabla_{\theta} \log \mu(s_0)
+ \sum_{t=0}^{H} \nabla_{\theta} \log P(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)})
+ \sum_{t=0}^{H} \nabla_{\theta} \log \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)}).
\end{align}
$$

Since both the initial state distribution and the transition probabilities do not depend on $\theta$, their gradients are zero:

$$
\nabla_{\theta} \log \mu(s_0) = 0, \quad
\nabla_{\theta} \log P(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)}) = 0.
$$

This leaves only the policy term:

$$
\begin{align}
\nabla_{\theta} \log P(\tau^{(i)}; \theta)
&= \sum_{t=0}^{H} \nabla_{\theta} \log \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)}).
\end{align}
$$

Substituting this back into the gradient estimator, we get the final `REINFORCE` expression:

$$
\begin{align}
\nabla_{\theta} J(\theta)
&\approx \hat{g}
= \frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{H}
\nabla_{\theta} \log \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)}) \, R(\tau^{(i)}).
\end{align}
$$

This gives us a sample-based estimate of the policy gradient that can be computed directly from trajectory rollouts.


## See Also
- [[Policy-Based Methods]]
- [[Policy Gradient Theorem]]
