# Moment Generating Function (MGF)
The [[Moment Generating Function(MGF)|MGF]] is defined as
$$
M_{X}(t) = E[e^{tx}]
$$
where
- $E[X^{k}] = \frac{d^{k}}{dt^{k}} M_{x}(t)$
- Uniquely identifies a PDF.

---
## Independance
If $x \ \& \ y$ are independent, 
$$
M_{x+y}(t) = M_{x}(t) * M_{y}(t)
$$

For $x_{i} \sim \mathbb{N}(\mu_{i}, \ \sigma^{2}_{i})$,
$$
M_{x_{i}}(t) = e^{\mu_{i}t + 1/2 \ 
\sigma^{2}_{i}t^{2}}
$$

---
