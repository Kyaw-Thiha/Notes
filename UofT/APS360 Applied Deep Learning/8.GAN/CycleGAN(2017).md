# CycleGAN
[[CycleGAN(2017)]] performs unpaired image-to-image translation by jointly training two [[Generative Adversarial Network(GAN)|GANs]] with inverse generator mappings.

![400](https://miro.medium.com/v2/resize:fit:1400/0*OzNvh6pDJ4evvDJB.png)

Its [[Generative Adversarial Network(GAN)|adversarial losses]] of generators $G:X \to Y$ and $F:Y\to X$ are regularized by [[CycleGAN(2017)|cycle-consistency loss]]:
$$
|| F(G(x)) - x ||_{1} + || G(F(y)) - y ||_{1}
$$

---
## Step-By-Step
For each training step,
- Start with real image from domain $X$.
- Feed into Generator $G: X\to Y$ to get $Y'$.
- Compute [[Generative Adversarial Network(GAN)|GAN loss]] with Discriminator $D_{Y}$.
- Feed fake $Y'$ into Generator $F:Y\to X$ to get $X'$.
- Compute [[Loss Function|pixel reconstruction loss]] between $X'$ and $X$.
- Repeat in opposite direction, starting with domain $Y$.
- Update the [[Neural Network|networks]].

Given that algorithm runs bidirectional, the losses are summed.

---
## See Also
- [[Generative Adversarial Network(GAN)]]