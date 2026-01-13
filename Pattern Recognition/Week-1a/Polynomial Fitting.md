Define a `function estimator`

$$
\begin{align}
h(x, \theta)  
&= \theta_{0} + \theta_{1}x + \theta_{2} x^2 + \dots + \theta_{M} \ x^M \\[6pt]
&= \sum^M_{m=1} \theta_{m} \ x^m \\[6pt]
&= \theta x; \quad x= (1, x, x^2, \dots, x^M)
\end{align}
$$

We can evaluate function estimator's performance as
$$
E[h(\cdot; \theta)]
= \frac{1}{2} 
\sum^N_{n=1} (h(x_{n}; \theta) - y_{n})^2
$$
And then optimize $\theta$ to improve performance

### How to find theta?
Define matrices $X, y$ s.t
$$
X = \begin{bmatrix}
1 & x_{1} & x_{1}^2 & \dots & x_{1}^M \\
1 & x_{2} & x_{2}^2 & \dots & x_{2}^M \\
\vdots & \vdots & \vdots & & \vdots  \\
1 & x_{n} & x_{n}^2 & & x_{n}^M
\end{bmatrix}
, \text{ and } 
t = \begin{bmatrix}
y_{1} \\ y_{2} \\ \vdots y_{n}
\end{bmatrix}
$$

Then, $\hat{f}(X; \ \theta) = X \theta$ and $E[\ h(x; \ \theta)\ ] = \frac{1}{2} || y - X\theta ||^2_{2}$ 
Hence,
$$
E[ \ h(x; \ \theta) \ ] = 0
\implies X\theta = y
$$

---
`Probability Distributions`

`Rules of Discrete Probability`
- Sum Rule: 
- Product Rule:
- Independance: 


