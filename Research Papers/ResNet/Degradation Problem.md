# Degradation Problem

`Degradation problem` is the phenomenon where deeper neural networks have **worse training accuracy** than shallower ones.

![Degradation Problem|400](https://guandi1995.github.io/images/ResNet/degradation_problem.PNG)

This is despite the fact that deeper network's solution space contains the shallower network as a subspace.

---
## Key Characteristics

**NOT caused by overfitting**
- Deeper networks show higher **training error** 
  (not just test error)
- If it were overfitting, training error would be lower but test error higher

**Counter-intuitive**
- Deeper network's solution space contains the shallower network as a subspace.
- Theoretically, the deeper model could learn identity mappings for extra layers and match the shallower model.
- But in practice, solvers cannot find solutions that are comparably good.

---

## Why It Happens?

From the [[ResNet|ResNet paper's]] analysis:

**Not vanishing gradients**
- Plain networks trained with Batch Normalization
- Forward propagated signals have non-zero variances
- Backward propagated gradients exhibit healthy norms
- Both forward and backward signals verified to not vanish

**Likely cause: Optimization difficulty**
- Deep plain nets may have exponentially low convergence rates
- Current solvers have difficulty approximating identity mappings by multiple nonlinear layers

---

## Solution

[[ResNet]] addresses this through **residual learning**:
- Explicitly reformulate layers to learn residual functions $F(x) = H(x) - x$
- If identity mappings are optimal, easier to push residual to zero
- Shortcuts allow easier optimization of deep networks

---

## See Also
- [[ResNet]]: The solution to this problem
- [[Skip Connections]]: Architectural pattern that helps
- [[Vanishing Gradients]]: Related but different problem