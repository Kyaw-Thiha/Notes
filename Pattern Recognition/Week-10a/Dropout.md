# Dropout

 [[Dropout|Dropout]] is a [[Regularization|regularization technique]] that randomly deactivates a fraction of [[Neural Network|neurons]] in a layer at each forward pass.

![Dropout|300](https://preview.redd.it/dropout-in-neural-networks-what-it-is-and-how-it-works-v0-gtz7f6j8rgm91.png?auto=webp&s=8b4df21fbb6b6bfb18c5e1a403ffd03953c478eb)

- **Training**: Drop [[Activation Function|activations]](by setting to $0$) with prob $p$
- **Inference**: Multiply [[Activation Function|pre-activation]] weights by $(1-p)$ to keep the same distribution.

Since we trained with $(1-p)$ neurons on average, during inference when all neurons are active, the [[Activation Function|activation]] is $\frac{1}{1-p}$ larger than during training. Hence, we need to scale them down by $(1-p)$ to match the expected magnitude.

In practice, we use [[Dropout|inverted dropout]] by scaling up $\frac{1}{1-p}$ during the training so inference requires zero modification.

---
## Benefits
- Forces network to not rely on a single neuron
- Simulates a small ensemble of smaller networks
- Better loss than without dropout

---
## See Also
- [[Regularization]]
- [[Early Stopping]]
- [[Weight Decay]]