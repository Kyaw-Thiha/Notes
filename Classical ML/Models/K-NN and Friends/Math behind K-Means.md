# Math behind K-Means
#ml/classic-models/k-means

`K-Means` is a clustering algorithm.

![K-Means|500](https://media.licdn.com/dms/image/v2/D5622AQHTq9ExiUVJNA/feedshare-shrink_800/B56ZaUs7MAGkAk-/0/1746251527641?e=2147483647&v=beta&t=knpGXF8Sd_45NZGmpcUseYYnlMvTyYgRew0hPtEdy8U)

---
`Problem Representation`

Let $\{ y_{i} \}^N_{i=1}, \ y_{i} \in R^d$ be the `dataset`.
Define $K$ to be the `number of clusters`.

Let $\{ c_{j} \}^K_{j=1}, \ c_{j} \in R^d$ be the `centers` of clusters.
Let $L \in \{ 0, 1 \}^{N \times K}$ be the binary `cluster assignment matrix` with elements $l_{i,j}$, where
$$
l_{i,j} = \begin{cases}
1 \quad \text{if } y_{i} \text{ is assigned to cluster } j \\
0 \quad \text{otherwise}
\end{cases}
$$
---

`Cluster Assignment Matrix`
This means that for clusters
- $c_{1} = \{ y_{1}, y_{3} \}$
- $c_{2} = \{ y_{2}, y_{5} \}$
- $c_{3} = \{ y_{4} \}$

we will have assignment matrix $L$ of
$$
L = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$
where
- Row sum of $L$ is 
$$
\sum^K_{j=1} l_{i,j} = 1
$$
- Column sum of $L$ is 
$$
\sum^N_{i=1} l_{i,j} = \text{No. of points assigned to cluster } j
$$
- Sum of all elements in $L$ is
$$
\sum^N_{i=1} \sum^K_{j=1} l_{i,j}
= \sum^K_{j=1} \sum^N_{i=1} l_{i,j}
= N
$$
- Sum of values of all points in cluster $j$
$$
\sum^N_{i=1} l_{i,j} \ y_{i}
= \text{sum of } y_{i} \text{ in cluster } j
$$

---
## Proof of Objective Function

`WTS: ` $c_{j} = argmin_{c_{j}}E(L, c_{j})$
`Objective Function`
$$
\begin{align}
E(L, c_{j})  
&= \sum^N_{i=1} l_{i,j}.||y_{i} - c_{i}||^2  \\
&= \sum^N_{i=1} l_{i,j}.(y_{i} - c_{j})^T . (y_{i} - c_{i}) \\
&= \sum^N_{i=1} l_{i,j}.(y_{i}^T.y_{i} - 2.y_{i}^T.c_{j} + c_{j}^T.c_{j}) \\
\end{align}
$$
`Minimizing Objective Function`
$$
\begin{align}
\frac{\partial E}{\partial c_{j}} &= 0 \\[4pt] 

\sum^N_{i=1}.l_{i, j}.(-2y_{i} + 2c_{j}) &= 0 \\[4pt] 

\sum^N_{i=1}.l_{i, j}.y_{i} &= \sum^N_{i=1}.l_{i, j}.c_{j}  \\
\\[4pt]

\sum^N_{i=1}.l_{i, j}.y_{i} &= .c_{j}.\sum^N_{i=1}.l_{i, j}  \\
\\[4pt]

c_{j} &= \frac{\sum^N_{i=1}.l_{i, j}.y_{i}}{\sum^N_{i=1}.l_{i, j}}
\end{align}
$$

---
`Speedup Computation`

In order to speed up computation in labelling, we can `precompute` & `loopup` values.
$$
\begin{align}
||y_{i} - c_{j}||^2
&= (y_{i} - c_{j})^T(y_{i} - c_{j}) \\[6pt]
&= (y_{i}^T y_{i} - 2c_{j}^T y_{i} + c_{j}^T c_{j}) 
\\[6pt]
&= ||y_{i}||^2 + ||c_{j}||^2 - 2 c_{j}^T y_{i}
\end{align}
$$
where we can precompute
- $||y_{i}||^2$ outside the loop
- $||c_{j}||^2$ in each loop

We can also `vectorize` in order to use matrix multiplication.
$$
Y = \begin{bmatrix}
| & | & | \\
y_{1} & \dots & y_{N} \\
| & | & |
\end{bmatrix}_{d \times N}
, \quad
C = \begin{bmatrix}
| & | & | \\
c_{1} & \dots & c_{k} \\
| & | & |
\end{bmatrix}_{d \times K}
$$

---
## See Also
- [[K-Means]]
