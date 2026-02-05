# Gradient Descent
To minimize the [[Empirical Risk]] for a [[Linear Predictor]], we can use a generalized loss function:
$$
\arg \min L_{S}(h)
= \arg \min_{w \in \mathbb{R}^d}
\ \frac{1}{m} \sum^m_{i=1} 
\ \mathcal{l}(h, (x_{i}, y_{i}))
$$

To get the direction that `increases` $L_{S}(h)$ by changing $w$, find the `gradient` of $L_{S}(h)$ $w.r.t$ $w$.
By negation, this maximises the `decrease` in [[Empirical Risk]].
[[Gradient Descent|Read More]]

---
## Gradients of Arbitrary Loss Functions
To get the `gradient` of the [[Loss Function|loss function]], we can carry out [[Gradient Descent|gradient descent]].

$$
\begin{align}
&\nabla_{w} \ \mathcal{l}(h, (x_{i}, y_{i})) \\[6pt]

&= l'(h, (x_{i}, y_{i})) \ \nabla_{w}
\varphi( \langle w, x_{i} \rangle ) \\[6pt]

&= l'(h, (x_{i}, y_{i})) \ \varphi'(w^T x_{i}) 
\ \nabla_{w} \langle w,x \rangle \\[6pt]

&= l'(h, (x_{i}, y_{i})) \ \varphi'(w^T x_{i}) 
\ \nabla_{w} x_{i}^T \\[6pt]
\end{align}
$$

To minimize the `loss function`, we find a new $w^{(t+1)}$ by taking some small step $\eta > 0$ opposite the direction of maximum increase.

$$
w^{(t+1)}
= w^{(t)} - \eta \nabla_{\mathbf{w}}
\ l(h, (x_{i}, y_{i}))
$$
This is done for every data points $x_{i}$.

Note that this also works for $\varphi(x)$.

---
