# Sequence Model-Based Global Optimization
`SMBO` uses a `surrogate model` to approximate the true function.

![SMBO|400](https://www.researchgate.net/publication/379765735/figure/fig1/AS:11431281243953274@1715785702802/Schematic-of-sequential-model-based-optimization-SMBO.tif)

---
## Algorithm
- We optimize the surrogate
- Find a guess at an optimal set of hyper-parameters
- Test these set of hyperparameters
- Improve the model
- Repeat until we find a good setting of $\theta$.

Note that
- The surrogate model can be provided by a number of ML algorithms.
- Different optimizers use different approximation of $f_{true}$.

---
## Pseudocode

**Input**: True function $f_{true}$
**Variables**:
- Surrogate function $f_{sur}$
- Model $M_{0}$
- Sampling budget $T$
- Hyperparameter space $\Theta$

**Loop**
$Z \leftarrow \emptyset$
**for** $t=1, \dots, T$ do:
- $\theta^* \leftarrow \arg \min_{\theta \in \Theta}$ $f_{sur}(\theta, M_{t-1})$
- Evaluate $f_{true}(\theta^*)$
- $Z \leftarrow Z \cup (\theta^*, f_{true}(\theta^*))$
- Fit new model, $M_{t}$ to $Z$ ()

**endfor**
return $Z$

---
