# Policy-Based Methods
#rl/policy 

Compared to [[Value-Based Methods]], `Policy-Based Methods` aim to directly parameterize the `policy function`, and learn from it.

$$
\pi_{\theta}(s) = P[A | s; \theta]
$$

![Policy-Based Methods](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_based.png)

There are ways to indirectly optimize the parameter $\theta$ indirectly by maximizing the local approximation of `objective function` such as
- `Hill Climbing`
- `Simulated Annealing`
- `Evolution Strategies`

To directly optmize the parameter $\theta$, we can use [[Policy Gradient Methods]] which optimize by performing `gradient ascent` on the objective function $J(\theta)$.

## See Also
- [[Value-Based Methods]]
- [[Policy Gradient Methods]]
