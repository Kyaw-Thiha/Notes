# ResNet 
`ResNet` learn residual mappings $F(x) = H(x) - x$ via [[Skip Connection]].

![ResNet|500](https://notes-media.kthiha.com/ResNet/b5d42d4e7230e47a78d8b4172adf09bb.png)

This enable successful training of networks with $100 \text{+ layers}$.

---
## Core Innovation
**Problem Solved** 
[[Degradation Problem]]: As the network depth increases, accuracy gets saturated then degrades rapidly. 

**Solution** 
Learn residual mappings $F(x) = H(x) - x$ instead of $H(x)$ directly.
- `Original mapping`: $H(x) = F(x) + x$
- `Hypothesis` 
  Easier to optimize the residual mapping than the original unreferenced mapping
- `Extreme case` 
  If identity mapping is optimal, easier to push residual to zero than to fit identity mapping with stack of nonlinear layers

---
## Key Mechanism 

### Residual Block

$$
y = F(x, \ W_{i}) + x
$$

- $F(x, \ W_{i})$: `residual mapping` to be learned 
  (stacked conv layers)
- $x$: `input`, passed through identity shortcut connection
- $+$: `element-wise addition`
- Second `nonlinearity` is applied after addition: $\sigma(y)$

### Shortcut Connections
Two types used in the paper:

1. **Identity shortcuts** (parameter-free)
$$
y = F(x, \ W_{i}) + x
$$
   - Used when input/output dimensions match
   - Simply performs identity mapping
   - No extra parameters or computational complexity

2. **Projection shortcuts** 
$$
y = F(x, \ W_{i}) + W_{s}x
$$
   - Used when dimensions increase (dotted lines in architecture)
   - Implemented with `1×1 convolutions`
   - Paper shows identity shortcuts sufficient for addressing degradation


---
## Why It Works
From the paper's analysis:

1. **Optimization perspective**
   If added layers can be constructed as identity mappings, deeper model should have $\text{training error} \leq \text{shallower counterpart}$.
   Residual learning helps solvers find these solutions.

2. **Empirical evidence** 
   Learned residual functions generally have small responses.
   This suggest identity mappings provide reasonable preconditioning.

3. **Gradient flow**
   Neither forward nor backward signals vanish. 

---
## Architecture Overview
See [[ResNet Variants]] for detailed architectures.

---
## See Also
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [[Skip Connections]] - general architectural pattern
- [[Degradation Problem]] - the problem ResNet solves
- [[ResNet Variants]] - specific architectures
