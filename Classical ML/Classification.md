# Classification
There are two types of classification problems
- Binary Classification
  $y \in \{0, 1\} \text{ or } y \in \{-1, 1\}$
- Multi-Class Classification
  $y \in {c_{1}, c_{2}, \dots, c_{n}}$

## Steps to compute f(x)
1. Learn a decision $g_{c}(x)$, which output a score representing how confident the model is
2. $f(x) = argmax_{c}(g_{c}(x))$
