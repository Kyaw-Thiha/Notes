# Halfspace
Consider an infinite plane
$$
(x, y) \ \in \  ] -\infty, \infty[ \ \times \ 
] -\infty, \infty[
$$

Our `weight vectors` $w$ from [[Linear Predictor]] define the `normal` of a line that splits the Euclidean space of our input domain in half.


---
## Half-Space Classification

We can determine which side of a plane a point is on by taking the dot product with its `normal vector`.

Suppose a `plane` is defined as
$$
a_{1}\ x_{1} + a_{2} \ x_{2} + \dots + a_{d} \ x_{d} 
= c
$$
Then, its `normal vector` can be defined by
$$
\mathbf{n} = (a_{1}, a_{2}, \dots, a_{d})
$$

A point can be labelled which side it is on by comparing the `dot product` of the normal vector and the point to the constant $c$:
$$
\mathbf{n} \cdot \mathbf{x} \geq c
\iff
\mathbf{n} \cdot \mathbf{x} - c \geq 0
$$

---
## Halfspace Example
Consider the `halfspace` defined by a `hyperplane` below.

![Halfspace|400](https://media.geeksforgeeks.org/wp-content/uploads/20240627122545/Hyperplane.png)

- A point is considered `positive (blue in the fig)` if it is `above` the plane: $\boxed{\ \mathbf{n} \cdot \mathbf{x} - c \geq 0 \ }$
- And it is considered `negative (red in the fig)` if it is `below` the plane: $\boxed{\ \mathbf{n} \cdot \mathbf{x} - c \leq 0 \ }$

This can be formulated in the definition of the `class of halfspaces` as
$$
\begin{align}
\text{HS}_{d}
&= \text{sign} \circ L_{d} \\[6pt]
&= \{ x \mapsto \text{sign}( \ h_{w,b}(x) \ ): h_{w,b} \in L_{d} \}
\end{align}
$$

---
