# Skip Connections
`Skip Connections` skip one or more layers.
This allow information to flow directly from earlier to later layers in a [[Neural Network]].

![Skip Connections|400](https://theaisummer.com/static/8d19d048cd68d6dce362e025cf3b635a/1ac66/skip-connection.png)

---
## Purpose

1. `Gradient flow` 
   Enable [[Backpropagation|gradients]] to flow backward through the network without vanishing
2. `Identity preservation` 
   Allow network to learn identity function easily
3. `Feature reuse` 
   Combine low-level and high-level features

---
## Variants

### Additive (ResNet style)
Element-wise `addition`

$$
y = F(x) + x
$$

Input and output must have same dimensions (or use projection)

---
### Concatenation (DenseNet, U-Net style)
`Concatenate` features along channel dimension
$$
y = \text{Concat}(F(x) + x)
$$
Allows different dimensions

---
### Gated (Highway Networks)
Data-dependent gates control information flow
$$
y = T(x) \cdot H(x) + (1-T(x)) \cdot x
$$
Adds parameters

---
## Architectures Using Skip Connections
- [[ResNet]] - additive, identity shortcuts
- U-Net - concatenation, for encoder-decoder
- DenseNet - concatenation, connects all layers
- [[Transformer]] - additive, in attention and FFN blocks
- Highway Networks - gated shortcuts

---
## Key Insight from ResNet Paper
Identity shortcuts (parameter-free) are sufficient for addressing degradation problem. Projection shortcuts only needed when matching dimensions.