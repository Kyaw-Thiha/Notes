# First-Order Methods
The gradient of $J_{\rho}(\pi_{\theta})$ $w.r.t$ $\theta$ allows us to design first-order optimization methods.

Repeatedly compute:
$$
\theta_{k+1} \leftarrow \theta_{k}
+ \alpha_{k} \nabla_{\theta} \ J_{\rho}
(\pi_{\theta_{k}})
$$
with $\alpha_{k}$ as the [[Learning Rate|learning rate]].

This is potentially more efficient in finding optimum of [[Policy Search Performance Measure|performance]] than [[Zero-Order Methods for Policy Search|zero-order methods]].

---