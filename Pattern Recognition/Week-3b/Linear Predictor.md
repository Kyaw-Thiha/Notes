# Linear Predictor
#ml/linear-predictor
`Linear predictors` are hypotheses that belong to a class.
$$
L_{d}
= \{ h_{w,b}: w \in \mathbb{R}^d, \ b \in \mathbb{R} \}
$$
where $h_{w,b}: \mathcal{X} \to \mathcal{Y}$ is a `linear hypothesis function`:
$$
h_{\mathbf{w}, b}(x)
= \langle \mathbf{w}, \mathbf{x} \rangle + b
= \left( \sum_{i=1}^d w_{i}x_{i} + b \right)
$$

For sake of compactness, we can rewrite this as
$$
h_{\mathbf{w}, b}(x)
= h_{\mathbf{w'}}(x')
= \langle w', x' \rangle
$$
where
- $w' = (b, w_{1}, w_{2}, \dots, w_{d})$ 
- $x' = (1, x_{1}, x_{2}, \dots, x_{d})$

---
## Classes of Linear Predictors
There are different kinds of `linear predictors` such as
- `Classifiers`: $h: \mathcal{X} \to \{ -1, 1 \}$
- `Regression`: $h: \mathcal{X} \to \mathbb{R}$
- `Classifiers`: $h: \mathcal{X} \to (\ \mathcal{Y} \to [0, 1] \ )$

We can express the space of linear predictors as
$$
L_{d} 
= \{ \ x \mapsto \langle w,x \rangle + b: \quad 
w\in \mathbb{R}^d, \ b \in \mathbb{R} \ \}
$$
Adding in a non-linear `activation`/`transfer` function, we get
$$
\begin{align}
&\mathcal{H}_{\varphi, d}
= \varphi \circ L_{d} \\[6pt]
\implies &h_{\varphi, d}
= \varphi(h_{w,b}(x))
\end{align}
$$

---
## See Also
- [[Linear Regression]]
- [[Linear Algebra Review]]