# Weighted K-NN
#ml/classic-models/k-nn/weighted

![Weighted KNN](https://scikit-learn.org/stable/_images/sphx_glr_plot_classification_001.png)

[[K-NN]] with weighing factor to give more influence to closer points.


$$
y = \frac{\sum_{i \in N_k(x)} w(x_i) \, y_i}{\sum_{i \in N_k(x)} w(x_i)}, 
\quad w(x_i) = e^{-\|x_i - x\|^2 / (2\sigma^2)}
$$
where $\alpha^2$ is weight factor

## Binary Classification
$$
y = \operatorname{sgn} \!\left( \sum_{i \in N_k(x)} w(x_i)\, y_i \right),
\quad w(x_i) = e^{-\|x_i - x\|^2 / (2\sigma^2)}
$$

where 
$$
\operatorname{sgn}(z) =
\begin{cases}
+1, & z > 0, \\
0, & z = 0, \\
-1, & z < 0
\end{cases}
$$

## Multi-Class Classification
$$
y_{\text{new}} = \arg\max_{c \in \mathcal{C}} 
\sum_{i \in N_k(x)} w(x_i)\, \mathbf{1}\{y_i = c\}
$$
where
$$
\mathbf{1}\{y_i = c\} =
\begin{cases}
1, & \text{if } y_i = c, \\
0, & \text{if } y_i \neq c
\end{cases}

$$

## See Also
- [[K-NN]]
