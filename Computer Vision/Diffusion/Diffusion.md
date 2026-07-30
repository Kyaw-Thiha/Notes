# Diffusion
[[Diffusion|Diffusion models]] create new data by learning to reverse a gradual noising process.

![Diffusion|300](https://media.geeksforgeeks.org/wp-content/uploads/20250804190118579985/diffusion_model.webp)

- **[[#Forward Diffusion Process|Forward Process]]:** Gradually adds [[Gaussian Distribution|gaussian noise]] to training data over many steps.
- **[[#Reverse Diffusion Process|Reverse Process]]:** Uses a [[Neural Network|neural network]] to predict and remove the added noise step by step.
- **Inference:** Starts from pure random noise and applies the learned reverse steps iteratively.

---
## Forward Diffusion Process
**Preliminary**:
Given data point sampled from real data distribution,

$$
x_{0} \sim q(x)
$$
We add small amount of Gaussian noise in $T$ steps, producing a sequence of noisy samples 

$$
x_{1}, \dots, x_{T}
$$
Where the step sizes are controlled by a variance schedule 

$$
\{ \beta_{t} \in (0,1) \}^{T}_{t=1}
$$

**Definition**:
We can thus define the [[Diffusion|forward diffusion process]] as
$$
q(x_{t} \mid x_{t-1})
= \mathcal{N}(x_{t}; \sqrt{ 1-\beta_{t} } \ x_{t-1}, \beta_{t} \ \mathbf{I})
$$
which after $T$ steps is
$$
q(x_{1:T} \mid x_{0})
= \prod^{T}_{t=1} q(x_{t} \mid x_{t-1})
$$
The data sample $x_{0}$ gradually loses its distinguishable features as step $t$ becomes larger.

When $T\to \infty$, $x_{T}$ is equivalent to an [[Gaussian Distribution|isotropic Gaussian distribution]].

---
### Closed Form using Reparameterization Trick
Using [[Maths Behind VAE|reparameterization trick]]([external](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick)), we can sample $x_{t}$ at any arbitrary time step $t$ in a closed form.

**Merging Gaussian Distribution**:
Suppose we merge two [[Gaussian Distribution|Gaussian distributions]] with different variances of
$$
\mathcal{N}(0, \ \sigma^{2}_{1}\mathbf{I})
\ \quad \text{and} \quad  \ 
\mathcal{N}(0, \ \sigma^{2}_{2}\mathbf{I})
$$
Then, the new distribution is
$$
\mathcal{N}(0, \ (\sigma^{2}_{1} + \sigma^{2}_{2}) \mathbf{I})
$$

Hence given variances $\sigma_{1} = (1 - \alpha_{t})$ and $\sigma_{2} = \alpha_{t}(1 - \alpha_{t-1})$, we get
$$
(1 - \alpha_{t}) + \alpha_{t}(1 - \alpha_{t-1})
= 1 - \alpha_{t}\alpha_{t-1}
$$
with a standard deviation of
$$
\sqrt{ 1 - \alpha_{t}\alpha_{t-1} }
$$

---
**Reparameterization Trick**:
Let $\alpha_{t} = 1 - \beta_{t}$ and $\bar{\alpha}_{t} = \prod^{t}_{i=1} \alpha_{i}$.

Given a mean of $\sqrt{ \alpha_{t} } = \sqrt{ 1 - \beta_{t} }$ and variance of noise scaled by $\sqrt{ 1 - \alpha_{t} } = \sqrt{ \beta_{t} }$, we can use the [[Maths Behind VAE|reparameterization trick]] as
$$
x_{t} = \sqrt{ \alpha_{t} } \ x_{t-1}
+ \sqrt{ 1 - \alpha_{t} } \ \epsilon_{t-1}
$$
where $\epsilon_{t-1} \sim \mathcal{N}(0, \mathbf{I})$.

---
**Closed Form**:
Now extending this into multiple $t$ steps, we get
$$
\begin{align}
x_{t}
&= \sqrt{ \alpha_{t} } \ x_{t-1}
+ \sqrt{ 1 - \alpha_{t} } \ \epsilon_{t-1} \\[6pt]
&= \sqrt{ \alpha_{t} \alpha_{t-1} } \ x_{t-2}
+ \sqrt{ 1 - \alpha_{t}\alpha_{t-1} } \  
\bar{\epsilon}_{t-2}  
&\text{by } (*)\\[6pt]
&= \dots \\[6pt]
&= \sqrt{ \bar{a}_{t} } \ x_{0} + \sqrt{ 1 - \bar{\alpha}_{t} } \ \mathbf{\epsilon}
\end{align}
$$
Hence, this results in
$$
\boxed{ \ 
q(x_{t} \mid x_{0}) = \mathcal{N}(x_{t}; \ 
\sqrt{ \bar{a}_{t} } \ x_{0}, \ (1 - \bar{a}_{t}) \ \mathbf{I})
\ }
$$

$(*)$: 
- Mean of two gaussians $\sqrt{ \alpha_{1} }$ and $\sqrt{ \alpha_{2} }$ is $\sqrt{ \alpha_{1}\alpha_{2} }$. (commutative)
- Standard Deviation is $\sqrt{ 1 - \alpha_{t}\alpha_{t-1} }$ as derived before.

---
Usually we can afford larger update steps when the sample gets noisier.

Since $\beta_{1} < \beta_{2} < \dots < \beta_{T}$, we get $\bar{\alpha}_{1} > \dots > \bar{\alpha}_{T}$.

---
## Reverse Diffusion Process
Reversing the [[#Forward Diffusion Process|forward process]] and sampling from $q(x_{t-1} \mid x_{t})$ will allow us to recreate the true sample from Gaussian noise input $x_{T} \sim \mathcal{N}(0, \mathbf{I})$.

![image|350](https://notes-media.kthiha.com/Diffusion/eb02fee9fd90e4d1d75470ec18c70966.png)

---
### Reverse Diffusion Model
We will be learning a model $p_{\theta}$ to approximate the conditional probabilities of $q(x_{t-1} \mid x_{t})$ as 
$$
p_{\theta}(x_{\theta:T})
= p(x_{T}) \prod^{T}_{t=1} 
p_{\theta}(x_{t-1} \mid x_{t})
$$
where
$$
p_{\theta}(x_{t-1} \mid x_{t})
= \mathcal{N}(x_{t-1}; \ \mu_{\theta}(x_{t}, t), \ \Sigma_{\theta}(x_{t}, t))
$$
with 
- $\mu_{\theta}(x_{t}, t)$: a network predicting the mean $\mu$ matrix
- $\Sigma_{\theta}(x_{t}, t)$: a network predicting the variance $\Sigma$ matrix

---
### Loss Function
#### ELBO Bound
Similar to [[Variational AutoEncoders (VAE)|VAE]], we use [[Maths Behind VAE|variational lower bound]] to optimize the [[Negative Log Likelihood|negative log likelihood]] as follows:
$$
\begin{align}
L_{VLB}
&= \mathbb{E}_{q(x_{0:T})}
\left[ \log \frac{q(x_{1:T} \mid x_{0})}{p_{\theta}(x_{0:T})} \right]
\end{align}
$$
which acts as an upper bound over
$$
- \mathbb{E}_{q(x_{0})} \log(p_{\theta}(x_{\theta}))
$$

---
#### Loss Function Decomposition:
This can be decomposed into
$$
L_{VLB}
= L_{T} + L_{T-1} + \dots + L_{0}
$$
where
- $L_{T} = D_{KL}(q(x_{T} \mid x_{0}) \mid\mid p_{\theta}(x_{T}))$
- $L_{t} = D_{KL}(q(x_{t} \mid x_{t+1}, x_{0}) \mid\mid p_{\theta}(x_{t} \mid x_{t+1}))$ for $1 \leq t \leq T-1$
- $L_{0} = -\log p_{\theta}(x_{0} \mid x_{1})$

---
#### Parameterization of $L_{t}$
We can parameterize $L_{t}$ to minimize the difference from $\tilde{\mu}$:
$$
\mathbb{E}_{x_{0}, \epsilon} 
\left[ \tau \mid\mid \epsilon_{t} - 
\epsilon_{\theta} \sqrt{ \bar{\alpha}_{t} } x_{0} 
+ \sqrt{ 1 - \bar{\alpha}_{t} } \epsilon_{t}, \ t \mid\mid^{2} \right]
$$
where $\tau = \frac{(1-\alpha_{t})^{2}}{2 \alpha_{t} (1 - \bar{\alpha}_{t}) \mid\mid \Sigma_{\theta} \mid\mid^{2}_{2}}$ is the weighing parameter.

---
#### Simplification
Empirical research found that training works better with a simplified objective that ignores weighing term:
$$
\begin{align}
L_{t}^{\text{simple}}
&= \mathbb{E}_{t \sim [1,T], \ x_{0},  \\
\ \epsilon_{t}}
\left[ \mid\mid \epsilon_{t} - \epsilon_{\theta} \\
(x_{t}, t) \mid\mid^{2} \right] \\[6pt]

&= \mathbb{E}_{t \sim [1,T], \ x_{0},  \\
\ \epsilon_{t}}
\left[ \mid\mid \epsilon_{t} - \epsilon_{\theta} \\
(\sqrt{ \bar{\alpha}_{t}} x_{0} + \sqrt{ 1 - \bar{\alpha}_{t} } \epsilon_{t}, \ t) \mid\mid^{2} \right] \\[6pt]
\end{align}
$$

Hence, the final simple objective is 
$$
L_{\text{simple}}
= L^{\text{simple}}_{t} + C
$$

---
## See Also
- [Very Good Blog (This note is based on this)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [DDPM Paper](https://www.alphaxiv.org/abs/2006.11239)