# Generative Adversarial Network(GAN)
[[Generative Adversarial Network(GAN)|GAN]] are [[Neural Network|generative models]] consisting of two networks:
- [[Generative Adversarial Network(GAN)|Generator Network]]: Create fake data from random noise.
- [[Generative Adversarial Network(GAN)|Discriminator Network]]: Classify between real data(from training set) and fake data(from generator)

![300](https://www.presidio.com/wp-content/uploads/2024/05/1_gPaBNqy4YPpiQx_liRJ9Mw.webp)

---
## Training Loop
- Sample a minibatch of real data $x$ and noise $z$.
- Update the [[Generative Adversarial Network(GAN)|discriminator network]].
	- Compute the [[Generative Adversarial Network(GAN)|discriminator loss]] $\mathcal{L}_{D}$.
	- Update discriminator weights via [[Gradient Descent|gradient ascent]](or [[Gradient Descent|descent]] on $-\mathcal{L}_{D}$).
- Sample a fresh minibatch of noise $z$.
- Update the [[Generative Adversarial Network(GAN)|generator network]].
	- Compute the [[Generative Adversarial Network(GAN)|generator loss]] $\mathcal{L}_{G}$.
	- Update generator weights via [[Gradient Descent|gradient descent]].
- Repeat

---
## Loss Function
The [[Generative Adversarial Network(GAN)|generator]] and [[Generative Adversarial Network(GAN)|discriminator]] are trained in a minimax game.

### Minimax Objective Function
[[Generative Adversarial Network(GAN)|GAN training]] can be formulated as two-player minimax game with the [[Value Function|value function]] of
$$
\min_{G} \max_{D} \ V(D,G)
= \mathbb{E}_{x \sim p_{data}(x)} [ \ \log D(x) \ ]
+ \mathbb{E}_{z \sim p_{z}(z)} [ \ \log(1 - D(G(z)) \ )]
$$
where
- $D(x)$: [[Generative Adversarial Network(GAN)|discrimantor]]'s prediction that real sample $x$ is real.
- $G(z)$: [[Generative Adversarial Network(GAN)|generator]]'s output given noise $z$.
- $D(G(z))$: [[Generative Adversarial Network(GAN)|discriminator]]'s prediction that generated sample is real.

The discriminator wants to maximize $V$:
- push $D(x) \to 1$ for real data
- push $D(G(z)) \to 0$ for fake data

The generator wants to minimize $V$:
- push $D(G(z)) \to 1$ to fool the discriminator

---
### Spliting Loss Functions
In practice, the two networks are trained as separate optimization steps with separate losses.

#### Discriminator Loss
$$
\mathcal{L}_{D} =
- \mathbb{E}_{x \sim p_{data}(x)} [ \ \log D(x) \ ]
- \mathbb{E}_{z \sim p_{z}(z)} [ \ \log(1 - D(G(z)) \ )]
$$
This is [[Cross-Entropy|binary cross entropy]] where real samples have label $1$ and fake samples have label $0$.

---
#### Generator Loss
$$
\mathcal{L}_{G} = \mathbb{E}_{z \sim p_{z}}
[ \ \log(1 - D(G(z))) \ ]
$$

Note that early in training, the discriminator confidently reject samples produced leading to
$$
D(G(z)) \approx 0
$$
This creates small gradient with weak learning signal. 
#### Non-Saturating Gradient Loss
To fix this, generator instead minimizes:
$$
\mathcal{L}_{G}
= - \mathbb{E}_{z \sim p_{z}}
[ \ \log D(G(z)) \ ]
$$
which leads to much stronger gradients early on.

---
## Weaknesses of GAN
- **Vanishing Gradient**:
  If discriminator is too good, 
	- small changes in generator's weights won't change discriminator output.
	- hence, there is no gradients learning signal for generator.
- **Mode Collapse**:
	- Nothing in [[#Loss Function|objective]] explicitly rewards diversity.
	- Generator can minimize its loss by producing a narrow set of highly convincing samples.
- **Failing to Converge**:
  Takes a long time to train before we see progress. To train [[Generative Adversarial Network(GAN)|GAN]]s faster, we used
	- [[Leaky ReLU]] instead of [[ReLU]]
	- [[Batch Normalization]]
	- [[Regularization|Regularizing]] discriminator weights. Adding noise to discriminator inputs.

---
## See Also
- [[Variational AutoEncoders (VAE)]]
- [[Conditional Variational AutoEncoders (CVAE)]]
- [[Diffusion]]