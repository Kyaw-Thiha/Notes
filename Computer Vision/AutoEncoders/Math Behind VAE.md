# Math Behind VAE

The [[Variational AutoEncoders]] aim to learn the `marginal likelihood` of the data 

$$
max_{\phi, \theta} \ E_{q \phi (z|x)} \  [\log p_{\theta} (x|z)]
$$

where 
- $\phi$ represent the parameters of `VAE Encoders`
- $\theta$ represent the parameters of `VAE Decoders`

We can rewrite this as
$$
\begin{align}
\log p_{\theta}(x) 
&= \log \int p_{\theta} (x,z) \ dz \\[6pt]

&= \log \int q_{\phi} (z|x) \frac{p_{\theta}(x, z)}{q_{\phi} (z|x)} \ dz \\[6pt]

&= \log E_{z \sim q_{\phi}(z|x)} \left[ \frac{ p_{\theta}(x,z) }{ q_{\phi} (z|x) } \right]
& \text{ since } E_{z \sim q_{\phi}(z|x)} = \int q_{\phi} (z|x).f(z) dz \\[6pt]

&\geq E_{z \sim q _{\phi} (z|x)} \left[ \log \frac{ p_{\theta}(x,z) }{ q_{\phi} (z|x) } \right]
& \text{by Jensen's Inequality since log is concave} \\[6pt]

&= L(\theta, \phi \ ; x) 
& \text{defining Evidence Lower Bound (ELBO)}
\end{align}
$$

This means to optimize $\log p_{\theta}(x)$, we need to minimize `ELBO`.

## ELBO

Hence, we can rewrite `ELBO` as
$$
\begin{align}
L(\theta, \phi; x)
&= E_{q_{\phi} (z|x)} \left[ \log \frac{p_{\theta} (x, z)}{q_{\phi} (z|x)} \right] \\[6pt]

&= E_{q_{\phi} (z|x)} \left[ \log p_{\theta}(x, z) - \log q_{\phi}(z|x)  \right] \\[6pt]

&= E_{q_{\phi} (z|x)} \left[ \log p_{\theta}(x|z) + \log p_{\theta}(z) - \log q_{\phi}(z|x)  \right] 
& \text{by } p(x, z) = p(x|z).p(z) \\[6pt]

&= E_{q_{\phi}} \left[ \log p_{\theta}(x|z) \right] + E_{q_{\phi}} \left[ \log p_{\theta}(z) \right] - E_{q_{\phi}} \left[ \log q_{\phi}(z|x) \right]   
\end{align}
$$

Recall that [[KL Divergence]] is
$$
KL(q_{\phi}(z|x) \ || \ p(x)) 
= E_{q_{\phi}} [\log q_{\phi}(z | x)] - E_{q_{\phi}} [\log p(z)] 
$$

Hence,
- $E_{z \sim q_{\phi}} \left[ \log p_{\theta}(x|z) \right]$ is the `reconstruction loss`
- $E_{z \sim q_{\phi}} \left[ \log p_{\theta}(z) \right] - E_{z \sim q_{\phi}} \left[ \log q_{\phi}(z|x) \right]$ is the `negative KL Divergence`

## Backpropagation Problem
Lets look back into the `reconstruction loss` $E_{z \sim q_{\phi}} \left[ \log p_{\theta}(x | z) \right]$ 
$$
\begin{align}
E_{z \sim q_{\phi}} \left[ \log p_{\theta}(x | z) \right]
&= \int q_{\phi} (z|x) . \log p_{\theta} (x|z) \ dz \\[6pt]

\nabla_{\phi} \ E_{z \sim q_{\phi}} \left[ \log p_{\theta}(x | z) \right]
&= \nabla_{\phi} \int q_{\phi} (z|x) . \log p_{\theta} (x|z) \ dz \\[6pt]
\end{align}
$$
Note that $q_{\phi}(z|x)$ depends on the $\phi$ inside the integral, hence making gradient non-trivial.

Specifically, `backpropagation` works if all operations are deterministic functions 
But here, $z$ is drawn randomly from a distribution which depends on $\phi$.
$$
z \sim q_{\phi}(z|x) = N(\mu(x), \sigma^2_{\phi} \ (x))
$$

## Reparameterization Trick
To fix the stochasticity inside `backpropagation`, we rewrite random variable $z$ as deterministic function of $\phi$ and independent noise source $\epsilon$.
$$
z = \mu_{\phi}(x) + \sigma_{\phi}(x) . \epsilon
$$
This is done by approximating posterior $p_{\phi}(z|x)$ as
$$
p_{\phi}(z|x) = N(z; \mu_{\phi}(x), \ diag(\ \sigma^2_{\phi}(x_{j}) \ ))
$$
and $p(z) \sim N(0, I)$

Applying this `reparameterization trick`, we get
$$
\begin{align}
&\nabla_{\phi} E_{q_{\phi}(z|x)} \left[ \log p_{\theta}(x|z) \right] \\[6pt]
&= \nabla_{\phi} E_{\epsilon \sim N(0, I)} \left[ \log p_{\theta}(x \ | \ \mu_{\phi}(x) + \sigma_{\phi}(x).\epsilon) \right]
\end{align}
$$

## Backpropagation
Differentiating the `objective function`,
$$
\begin{align}
&\nabla_{\phi} E_{q_{\phi}(z|x)} \left[ \log p_{\theta}(x|z) \right] \\[6pt]

&= \nabla_{\phi} E_{\epsilon \sim N(0, I)} \left[ \log p_{\theta}(x \ | \ \mu_{\phi}(x) + \sigma_{\phi}(x).\epsilon) \right] \\[6pt]

&=  E_{\epsilon \sim N(0, I)} \left[ \nabla_{\phi} \log p_{\theta}(x \ | \ \mu_{\phi}(x) + \sigma_{\phi}(x).\epsilon) \right] \\[6pt]
\end{align}
$$

Applying `Chain Rule` to sampled $z$
$$
\nabla_{\phi} \log p_{\theta}(x \mid z)
= 
\underbrace{
\frac{\partial \log p_{\theta}(x \mid z)}{\partial z}
}_{\text{decoder gradient}}
\cdot
\underbrace{
\frac{\partial z}{\partial \phi}
}_{\text{encoder gradient}}.
$$

Analyzing the first term,
$$
\frac{\partial \log p_{\theta}(x \mid z)}{\partial z}
$$
it is differentiable via `decoder` $p_{\theta}(x|z)$, and the gradient flows from `reconstruction loss`.

Analyzing the second term,
$$
\frac{\partial z}{\partial \phi}
= \frac{\partial \mu_{\phi}(x) }{\partial \phi} + \epsilon. \frac{\partial \sigma_{\phi} (x)}{\partial \phi}
$$
it is differentiable since $\mu$ and $\sigma$ network outputs.

Hence, 
$$
\nabla_{\phi} \, \mathbb{E}_{\varepsilon}
\!\left[
\log p_{\theta}\!\left(x \mid \mu_{\phi}(x) + \sigma_{\phi}(x) \odot \varepsilon\right)
\right]
=
\mathbb{E}_{\varepsilon}
\!\left[
\left(
\frac{\partial \log p_{\theta}(x \mid z)}{\partial z}
\right)
\left(
\frac{\partial \mu_{\phi}(x)}{\partial \phi}
+ \varepsilon \odot \frac{\partial \sigma_{\phi}(x)}{\partial \phi}
\right)
\right].
$$

## Differentiation Flow

`Sample`: $\epsilon \sim N(0, I)$, $z = \mu_{\phi}(x) + \sigma_{\phi}(x).\epsilon$
`Forward`: $\log p_{\theta}(x | z)$
`Backward`:
$$
\frac{\partial \log p_{\theta}(x \mid z)}{\partial \phi}
=
\frac{\partial \log p_{\theta}(x \mid z)}{\partial z}
\left(
\frac{\partial \mu_{\phi}(x)}{\partial \phi}
+ \varepsilon \odot \frac{\partial \sigma_{\phi}(x)}{\partial \phi}
\right).
$$

