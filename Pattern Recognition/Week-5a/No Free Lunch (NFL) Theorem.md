# No Free Lunch (NFL) Theorem

>  `TLDR`
>  You cannot find a single algorithm $A$ and a fixed training size $m$ that will succeed on all possible distributions $\mathcal{D}$.

---
## Formal Theorem

Given a
- Specific `learning task` specified by distribution $\mathcal{D}$ over $\mathcal{X} \times \mathcal{Y}$
- `Learner` $A(S)$ whose goal is to produce hypotheses $h:\mathcal{X \to Y}$ with a risk $L_{\mathcal{D}}(h) \leq \epsilon$.

there does not exist a learner $A(S)$ and a training size $m$ such that for all distributions $\mathcal{D}$,
it will output a predictor $h$ with low risk with high probability.

In other words, it is not [[PAC Learning|PAC Learnable]].

---
## Avoiding NFL by introducing Prior Knowledge
We can avoid the [[No Free Lunch (NFL) Theorem]] by constraining our hypothesis class $\mathcal{H}$ using prior knowledge about specific learning tasks.
This helps us avoid getting tricked by certain distributions.

We can implement this by testing different classes of models.

These models are differentiated by things that `parameterize` an algorithm: 
- $\text{Number of Neurons}$
- $\text{Style of feature embeddings}$
- $\text{Dimensionality of input}$
- $\text{etc.}$

or the `style of algorithm`:
- $\text{Decision Tree}$
- $\text{Neural Networks}$
- $\text{Kernal Density Estimators}$

---
