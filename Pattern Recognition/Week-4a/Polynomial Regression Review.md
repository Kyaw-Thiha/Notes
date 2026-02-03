# Polynomial Regression 
In [[Polynomial Regression]], instead of running the output of a linear classifier through a function, we run inputs through
$$
\begin{align}
\mathcal{H}^n_{poly}
&= L_{d} \ \circ \varphi_{n}(x) \\[6pt]
&= \{ x \mapsto h_{w, b}( \ \varphi_{n}(x) \ ): h_{w,b} \in L_{d} \}
\end{align}
$$

where
- $\mathcal{H}^n_{poly}$ is the `hypothesis class`
- $L_{d} = \{ x \mapsto \langle w,x \rangle + b: \ w \in \mathbb{R}^d , \ b \in \mathbb{R} \}$ is class of `linear predictors` in $d$ dimensions
- $\circ$ is the `function composition`: $(f\circ g)(x) = f(g(x))$
- $\varphi_{n}(x) = (1, \ x, \ x^2, \ \dots, \  x^n)$ is the `basis function`

`Polynomial regression` can be thought of as applying `feature embeddings`.

---
## See Also
- [[Linear Predictor]]
- [[Linear Regression Review]]
- [[Polynomial Regression]]