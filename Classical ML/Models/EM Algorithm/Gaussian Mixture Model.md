# Gaussian Mixture Model
#ml/classic-models/mixture-model/gaussian  

This is a [[Mixture Model]] where the data distributions are modelled as [[Gaussian Distribution]].

![GMM|400](https://towardsdatascience.com/wp-content/uploads/2023/01/1GWkzcCKBqQV7GgwTaa4l4g.gif)

---
`1D Case`

Let $\{ y_{i} \}^N_{i=1}$ be the `dataset`.

`Case-1`: $K=1$ cluster distribution
$$
m_{1} = 1 
\ , \quad \ \mu = \frac{1}{N}\sum^N_{i=1} y_{i}
\ , \quad \ \sigma^2 = \frac{1}{N} \sum^N_{i=1} (y_{i} - \mu)^2
$$

`Case-2`: $K>1$ cluster distributions

Let $l_{i,j}$ be the `cluster assignment variable`.
$$
l_{i,j} = 
\begin{cases}
1, & \text{if } y_{i}  
\text{ is assigned to cluster } j \\[6pt]
0, & \text{otherwise}
\end{cases}
$$
Then,
$$
m_{j} = \frac{\sum^N_{i=1} l_{i,j}}{N}
\ , \quad \ 
\mu_{j} = \frac{\sum^N_{i=1} l_{i,j} \ y_{i}}{\sum^N_{i=1} l_{i,j}}
\ , \quad \ 
\sigma_{j}^2 = \frac{\sum^N_{i=1} l_{i,j} (y_{i} - \mu_{j})^2}{\sum^N_{i=1} l_{i,j}}
$$

`Probabilistic Distribution`

Using [[Bayes Rule in ML]], we can use the `posterior probabilistic distribution` over assignment variable $l_{i,j}$
$$
p(l = j \ | \ y_{i}) = \frac{p(l=j) \ p(y_{i} | l=j)}{p(y_{i})}
$$


---
## Relation to RBF Regression
Compared to [[RBF Regression]], 
- `GMM` has weights constraint
  $\begin{cases}m_{j} \in [0,1] \\ \sum^K_{j=1} m_{j} = 1 \end{cases}$ 
- `GMM` has [[Loss Function]] of [[Log Likelihood|Negative Log Loss of Data Likelihood]] while `RBF Regression` uses [[Error Metrics|Squared Error]]

---
## See Also
- [[Mixture Model]]
- [[Maths Behind GMM]]
- [[EM Algorithm]]