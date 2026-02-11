# Perceptron Convergence Theorem

> This theorem guarantees that when the [[Perceptron|Perceptron Algorithm]] stops, all the points will be correctly classified.

---
## Theorem
- Assume that $((x_{1}, y_{1}), \ \dots, \ (x_{m}, y_{m}) )$ is a separate `dataset`.
- Let $B = \min\{ ||w||: \forall i \in [m], \ y_{i} \langle \mathbf{w}, \mathbf{x} \rangle \geq 1 \}$
- Let $R = \max_{i} ||x_{i}||$

Then, the [[Perceptron|Perceptron Algorithm]] stops after at most $(RB)^2$ iterations.
And when it stops, it holds that
$$
\boxed{ \ \forall i \in [m], \ y_{i} 
\langle \mathbf{w}, \mathbf{x_{i}} \rangle \geq 0 \ }
$$

In other words, when it stops, it has a $100\%$ classification accuracy.

---
## Proof

It suffices to show that
- The algorithm stops after $T$ iterations
- and $T \leq (RB)^2$

### Proof Plan
Let $\mathbf{w}^*$ be a vector that minimizes $B$.
Of all vectors that gets $100\%$ classification, $\mathbf{w}^*$ has the `minimal norm`.

> Show that after $T$ iterations,the cosine angle between $\mathbf{w}$ and $\mathbf{w^*}$ is at least $\frac{\sqrt{ T }}{RB}$.

In other words,
$$
\frac{\langle \mathbf{w}^*, \mathbf{w}^{(T+1)}
\rangle}{||\mathbf{w}^*|| \ ||\mathbf{w}^{(T+1)}||}
\geq \frac{\sqrt{ T }}{RB}
$$
