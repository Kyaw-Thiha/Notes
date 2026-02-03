# Linear Regression Review
Let's consider [[Linear Regression]] in context of [[Empirical Risk Minimization (ERM)]].

We can define the hypothesis class as
$$
\mathcal{H}_{reg} = L_{d}
= \{ x \mapsto \langle w,x \rangle + b: \quad w \in \mathbb{R}^d , \ b \in \mathbb{R} \}
$$

We will be using the `squared error` [[Loss Function]]: 
$$
l(h, \ (x,y)) = (h(x) - y)^2
$$
---
`Empirical Risk`
This results in the [[Empirical Risk]] of
$$
L_{S}(h) = \frac{1}{m} \sum^m_{i=1} (h(x_{i}) - y_{i})^2
$$


We can then express the empirical risk in vector form of
$$
L_{S}(h) = \frac{1}{m} ||Xw - y||^2
$$
Hence, we can solve the value of $w$ that minimizes $||Xw - y||^2$.
This will be $X^+y$.

---