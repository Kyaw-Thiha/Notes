# Logistic Regression
[[Logistic Regression]] is learning a family of functions $h:\mathbb{R}^d \to [0,1]$ for [[Classification|classification tasks]].

$$
\begin{align}
\mathcal{H}_{\text{logistic}} 
&= \varphi_{sig}  \ \circ \ L_{d} \\[6pt]
&= \{ x \mapsto \varphi_{sig}( \langle w,x \rangle ): w \in \mathbb{R}^d \}
\end{align}
$$
where
- $\varphi_{sig}: \mathbb{R} \to [0,1]$ is the [[Sigmoid Function]]

For `logistic regression`, 
$$
\varphi_{sig}(z)
= \frac{1}{1 + e^{-z}}
$$

---
## Link to Probability
We can interpret $h(x)$ as probability that the label of $x$ is $1$.
$$
h(x_{i}) = p(y_{i} = 1 \ | \ x)
$$

---

`Analysis w.r.t decision boundary`
As $\lvert \langle w,x \rangle \rvert \to \infty$ $\text{(points getting further from decision boundary)}$,
- $\lvert sign(\langle w,x \rangle) \rvert \to 1$
- $\lvert \varphi_{sig}(\langle w, x \rangle) \rvert \to 1$

As $\lvert \langle w,x \rangle \rvert \to 0$ $\text{(points getting closer from decision boundary)}$,
- $\lvert sign(\langle w,x \rangle) \rvert \to \pm 1$
- $\lvert \varphi_{sig}(\langle w, x \rangle) \rvert \to 0.5$

---
## Loss Function
We want to maximize the [[Log Likelihood|likelihood function]] by minimizing the [[Negative Log Likelihood]].

Since [[Empirical Risk Minimization (ERM)|ERM]] wants a central tendency of performance, we can use the Geometric mean:
$$
\begin{align}
&p(S \mid \mathbf{w})^{1/|S|} \\[6pt]
&= \left( \prod^m_{i=1} h(y_{i} x_{i}) \right)^{1/m} \\[6pt]
&= \left( \prod^m_{i=1}  
\frac{1}{1 + \exp(-y_{i} \langle w, x_{i} \rangle)} 
\right)^{1/m}
\end{align}
$$
Taking the negative log, 
$$
-\log p(S \mid \mathbf{w})^{1/|S|}
= \frac{1}{m} \sum^m_{i=1}
\log(1 + \exp(-y_{i} \langle w, x_{i} \rangle ) \ )
$$

---
## Solving Logistic Regression
Unlike [[Linear Predictor]], we can't use [[Solving with Matrices|linear programming]] , [[Perceptron|perceptron algorithm]] nor `least squares minimization`.

Since [[Empirical Risk Minimization (ERM)|ERM objective]] of linear regression is convex function, we can use standard optimization methods like [[Newton's Method]] as well as [[Gradient Descent]].

---
## Gradients of Arbitrary Loss Functions
We need `gradient` of [[Loss Function|loss function]] before we can do [[Gradient Descent|gradient descent]].

$$
\begin{align}
&\nabla_{w} \  \mathcal{l}(\ h(x_{i}, y_{i}) \ ) \\[6pt]

&= \mathcal{l}'(h, (x_{i}, y_{i})) \  
\nabla_{w} \ \varphi( \langle w, x_{i} \rangle) \\[6pt]

&= \mathcal{l}'(h, (x_{i}, y_{i})) \  
\varphi'(w^T x_{i})  
\ \nabla_{w} \langle w,x_{i} \rangle \\[6pt]

&= \mathcal{l}'(h, (x_{i}, y_{i})) \  
\varphi'(w^T x_{i}) \ x_{i}^T
\end{align}
$$

To decrease the `loss function`, we find $w^{(t+1)}$ by 
- taking some small steps $\eta>0$ 
- opposite the direction of maximum increase

This is also called [[Gradient Descent]]:
$$
w^{(t+1)}
= w^{(t)} - \eta \ \nabla_{w} \ l(h, (x_{i}, y_{i}))
$$

---
## See Also
- [[Logistic Regression]]
- [[Gradient Descent]]
- [[Empirical Risk Minimization (ERM)]]
