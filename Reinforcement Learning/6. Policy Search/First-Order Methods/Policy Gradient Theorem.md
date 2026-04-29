# Policy Gradient Theorem
Assume that $\pi_{\theta}$ is differentiable $w.r.t$ $\theta \in \Theta$.
We then have
$$
\begin{align}
&\nabla_{\theta} \ J_{\rho}(\pi_{\theta}) \\[6pt]
&= \sum_{k \geq 0} \gamma^{k} \int d\rho(s) 
\mathcal{P}^{\pi_{\theta}}(ds' \mid s;k)
\int \nabla_{\theta} \pi_{\theta}(a' \mid s')
Q^{\pi_{\theta}}(s', a') \ da' \\[6pt]

&= \frac{1}{1 - \gamma} \int  
\rho^{\pi_{\theta}}_{\gamma}(ds)
\int \nabla_{\theta} \pi_{\theta}(a \mid s)
\ Q^{\pi_{\theta}}(s,a) ds \\[6pt]

&= \frac{1}{1 - \gamma} \mathbb{E}_{S \sim 
\rho^{\pi_{\theta}}_{\gamma}, \ A \sim \pi_{\theta}  
(\cdot \mid S)} [ \nabla_{\theta} \log \pi_{\theta} 
(A \mid S) \ Q^{\pi_{\theta}}(S, A)]
\end{align}
$$

---
