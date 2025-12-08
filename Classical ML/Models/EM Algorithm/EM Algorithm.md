# EM Algorithm
#ml/classic-models/mixture-model/em-algorithm  

`EM Algorithm` is an algorithm used to optimize [[Gaussian Mixture Model]].

![EM Algorithm](https://upload.wikimedia.org/wikipedia/commons/6/69/EM_Clustering_of_Old_Faithful_data.gif)

---
`Algorithm`

Initialize `cluster center` $\gamma_{i,j}$ and its `parameters` $\theta = \{ m_{1:K},\ \mu_{1:K},\ C_{1:K} \}$ 

`E-Step`
Fix $\theta$, and update $\gamma_{i,j}$.

For each $i \in \{ 1,\ \dots,\ N \}$,
$$
\begin{align}
\gamma_{i,j}
&= p(l=j \ | \ y_{i}, \theta) \\[6pt]

&= \frac{p(l=j \ | \ \theta) \  
p(y_{i} \ | \ l=j, \theta)}{p(y_{i} \ | \ \theta)}  
\\[6pt]

&= \frac{m_{j} \  p(y_{i} \ | \ l=j, \theta)} 
{\sum^K_{h=1} m_{h} \ p(y_{i} \ | \ l=h, \theta)}  
\\[6pt]
\end{align}
$$

`M-Step`
Fix $\gamma_{i,j}$ and update $\theta$.

For each cluster $j$, 

$$
\begin{align}
m_{j}  
&= \frac{\sum_{i} \gamma_{i,j}}{N} \\[6pt]

\mu_{j}
&= \frac{\sum_{i} (\gamma_{i,j} \ y_{i})} 
{\sum_{i} \gamma_{i,j}} \\[6pt]

C_{j} 
&= \frac{\sum_{i} \gamma_{i,j}  
(y_{i} - \mu_{j})(y_{i} - \mu_{j})^T} 
{\sum_{i} \gamma_{i,j}}
\end{align}
$$

[[Maths Behind EM Algorithm|Read More about how EM Algorithm is derived]]

---
## Relation to K-Means
Note that `EM Algorithm` reduces to [[K-Means]] when
- `Mixing Probabilities` $m_{j} = 1$ $,\forall j$
- `Gaussian components` are spherical with identical variances
  $C_{j} = \sigma^2 I$ $, \forall j$
- `Gaussian variances` are infinitesimal $(\sigma^2 \to 0)$
  $\lim_{ \sigma^2 \to 0 } P(l=j | y_{i}) = 1$
  Hence, they become hard binary assignments.

---
## See Also 
- [[Maths Behind EM Algorithm]]
- [[Gaussian Mixture Model]]
- [[K-Means]]