# Policy Gradient Methods
#rl/policy/gradient 

This is a form of [[Policy-Based Methods]], which uses `gradient ascent` to directly optimize the `policy function` (parameter $\theta$).

The `objective function` is
$$
J(\theta) = E_{\tau ~ \pi} [R(\tau)]
$$
where 
- $R(\tau) = r_{t+1} + \gamma.r_{t+2} + \gamma^2.r_{t+3} + \dots$
- $\tau$ is the `trajectory` (sequence of `states` & `actions`)
- $\gamma$ is the `discount rate`

Hence, it can be expressed as
$$
J(\theta) = \sum_{\tau} P(\tau; \theta).R(\tau)
$$
where
- $P(\tau; \theta)$ is the probability of choosing the `trajectory` based on the `policy` $\theta$
- $R(\tau)$ is the `reward` from choosing the `trajectory`

The `probability` of choosing the `trajectory` can be further expanded as
$$
P(\tau; \theta) = \Pi_{t=0} P(s_{t+1} | s_{t}, a_{t}) . \pi_{\theta}(a_{t} | s_{t})
$$

To optimize this `objective function`, we use gradient ascent.
$$
\theta = \theta + \alpha.\nabla_{\theta}.J(\theta)
$$

However, note that we cannot directly differentiate $J(\theta)$. 
So, we need to apply `Policy Gradient Theorem`
