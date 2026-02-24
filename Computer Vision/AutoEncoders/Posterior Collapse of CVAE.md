# Posterior Collapse of CVAE

`Posterior Collapse` in [[Conditional Variational AutoEncoders (CVAE)|CVAE]] happens when the model ignores the latent variable $z$, and generates output $y$ purely based on conditions.

![image|500](https://notes-media.kthiha.com/Posterior-Collapse-of-CVAE/0cfe5aa954c1ad52fada4c358684c95b.png)

### Properties of Posterior Collapse
- Decoder learns to generate outputs using only conditioning variable $c$, and completely ignoring the latent variable $z$.
- The [[KL Divergence]] in the [[Maths Behind VAE|ELBO loss]] drops to near $0$.
- Sampling different $z$ values produces nearly identical outputs.

---
## Causes of Posterior Collapse
Recall from [[Maths Behind VAE]] that `CVAE` aims to minimize the `Evidence Lower Bound (ELBO)`
$$
\begin{align}
L_{\text{ELBO}}
&= E_{p(X)} [ \ \underbrace{E_{q_{\varphi}(z \mid x)}
[\log p_{\theta}(x \mid z)]} 
_{\text{Reconstruction term}}
\underbrace{- D_{KL}[q_{\varphi}(z \mid x) \mid\mid p(z)]}_{\text{Negative KL Div Term}} \ ]
\end{align}
$$

In other words, [[Variational AutoEncoders (VAE)|VAE]] is trained by jointly
- maximizing the `reconstruction term`
- minimizing the `KL Divergence term`

Note that there is a contradiction in this `ELBO` optimization.

> - The `reconstruction term` encourages the latent variables to transmit more information about input data $x$.
> - The `KL term` limits the amount of information that can be transmitted.

Additionally, during early stages of training, 
- The `latent variable` $z$ contain little information about the input data $x$.
- Hence when equipped with a strong decoder like [[Transformer]], `VAE` tends to give up difficult latent variables and rely entirely on the decoder for generation.

---
## Methods on solving Posterior Collapse
### 1. Weaken the Decoder
Weaken the decoder in order to force the model to rely on latent variables for generation.

- [Hybrid CVAE 2017](https://arxiv.org/abs/1702.02390) and [VAE with Dilated Convolutions (2017)](https://arxiv.org/abs/1702.08139) implement the decoder with [[Convolutional Neural Network (CNN)|CNN]] instead of an `auto-regressive model`
- [Bowman et. al (2016)](https://arxiv.org/abs/1511.06349) randomly removed some fraction of the conditioned variable in order to weaken the decoder.
- [Petit & Corro (2021)](https://arxiv.org/abs/2110.14945) uses `Fraternal dropout` regularization.
  1. Take the same input sentence
  2. Apply dropout twice with different random masks.
     Hence, creating two different versions
  3. Process both versions through the LSTM decoder
  4. Add a penalty if the hidden states are too different

### 2. Alleviate the contradiction
Alleviate the contradiction between the `reconstruction term` and the `KL term`.
Or use a looser constraint on the `upper bound` of the information transmitted by latent variables.

- [KL Annealing (2016)](https://arxiv.org/abs/1511.06349) 
  Gradually increases the weight of KL term during training
- [Cyclic KL Annealing (2019)](https://arxiv.org/abs/1903.10145) 
  Repeats annealing cycles to progressively learn more meaningful latent codes
- [Beta-VAE (2016)](moz-extension://cc26e639-2c95-4a42-92c2-4edab8306cd7/pdfjs/web/viewer.html?file=https%3A%2F%2Fwww.cs.toronto.edu%2F~bonner%2Fcourses%2F2022s%2Fcsc2547%2Fpapers%2Fgenerative%2Fdisentangled-representations%2Fbeta-vae%2C-higgins%2C-iclr2017.pdf) 
  Re-weights the KL term using a hyperparameter $\beta$.
- 

### 3. Adding MI-based term
Add `Mutual Information-based` terms to the objective.
This enforce the relation between latent variables $z$ and input data $x$.

### 4. Reduce difficulty of learning latent variables
Reduce the difficulty of exploiting latent variables, especially in the initial stages of training.

---
## See Also
- [Main Reference Paper](https://www.semanticscholar.org/paper/Scale-VAE%3A-Preventing-Posterior-Collapse-in-Song-Sun/ea7d13bde25a306d515664fbaf2df107370fcc32)