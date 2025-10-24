# Expected Test Error

Due to multiple values of $y$ for a given $x$, we have [[Expected Target Value]].
We can compute [[Expected Test Error Given Dataset D]] by finding difference between predicted value $\hat{y}_{D}$ for dataset $D$, and the `Expected Target Value`.

![[Expected Test Error.png]]

To extrapolate the concept for the entire dataset, we can combine [[Expected Test Error Given Dataset D]] with [[Expected Test Error Given Dataset D#Expected Predictor|Expected Predictor]].

$$
\begin{align}
E_{X, Y, D}[ \ L(\hat{y}_{D}(x), y) \ ]
&= E_{X, Y, D} [ \ (\hat{y}_{D}(x) - y)^2 \ ] \\[6pt]
&= \int_{D} \int_{x} \int_{y} (\hat{y}_{D}(x) - y)^2 \ p(x, y) \ p(D) \ dy \ dx \ dD
\end{align}
$$

