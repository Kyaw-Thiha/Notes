# K-NN
#ml/classic-models/k-nn

![KNN](https://towardsdatascience.com/wp-content/uploads/2021/11/13SwcOCUyVdGauhHrHvOaLA.png)


1. Choose the $k$ hyper-parameter
2. Calculate the distance between the point and all the points (e.g. using Euclidean distance $|| x - x_{i}||^2$)
3. Based on distance, find $k$-nearest neighbour in the training data set
4. (a) For classification, the data point is assigned to category most appeared in the k-nearest neighbours
   
   (b) For regression, mean value is taken over the k-nearest neighbours

### Hyperparameters
- $k$
- $\alpha$ (if weighted)
- Similarity Measure

## Regression
![K-NN Regression](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/KNN-diagram.png?ssl=1)
Prediction $y$ is average of training outputs of k-nearest neighbours.
$$
y = \frac{1}{k}.\sum_{i \in N_{k}(x)} y_{i}
$$
## Binary Classification
For class labels $\{-1, 1\}$
$$
y_{\text{new}} = \operatorname{sgn} \!\left( \sum_{i \in N_k(x)} y_i \right)
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
\sum_{i \in N_k(x)} \mathbf{1}\{y_i = c\}
$$

where
$$
\mathbf{1}\{y_i = c\} =
\begin{cases}
1, & \text{if } y_i = c, \\
0, & \text{if } y_i \neq c
\end{cases}

$$

---

## Optimization
K-NN implicitly locally optimize the squared error.

### Proof

$$
\begin{align}
&E(\hat y) = \sum_{i \in N_k(x)} (y_i - \hat y)^2 \\[6pt]
&\frac{\partial E}{\partial \hat y} = -2 \sum_{i \in N_k(x)} (y_i - \hat y) = 0 \\[6pt]
&\implies \quad k \hat y = \sum_{i \in N_k(x)} y_i \\[6pt]
&\implies \quad \hat y = \frac{1}{k} \sum_{i \in N_k(x)} y_i
\end{align}
$$


Note that if we fit K-NN based on median instead of mean, it will be optimizing for absolute error.

---
## Metric Choices
### ℓ-p norm (Minkowski distance)
For two vectors $x, y \in \mathbb{R}^d$, the Minkowski (ℓ-p norm) distance is:

$$
d_p(x,y) = \left( \sum_{j=1}^d |x_j - y_j|^p \right)^{1/p}
$$

- \(p = 1\) → **Manhattan distance** (L1 norm, city-block).  
- \(p = 2\) → **Euclidean distance** (most common).  
- \(p \to \infty\) → **Chebyshev distance**:  
  $$
  d_\infty(x,y) = \max_j |x_j - y_j|
  $$

### scikit-learn settings
- `metric="minkowski", p=2` → Euclidean (default).  
- `metric="minkowski", p=1` → Manhattan.  
- `metric="minkowski", p=3` → cubic distance (less common).  
- `metric="chebyshev"` → max-difference.  
- `metric="manhattan"` or `"l1"` → same as \(p=1\).  
- `metric="euclidean"` → same as \(p=2\).  

---
## Effect of K on Error
### Overfit
At the start of the graph, training RMSE has much lower error while testing RMSE has much higher error.
This is since K-NN is trying to fit to the noise.

![[k-nn_overfit.png]]

### Underfit
For larger values of k, RMSE for both training and testing increase.
Note that training RMSE becomes higher than that of testing RMSE.
![[k-nn_underfit.png]]

## Heuristic Choice of K
$$
K < \sqrt{ N }
$$
where $N$ is the number of data points

## Problems with KNN
- **Large amount of data**
  Need to compute distances between all of the dataset
- **Curse of Dimensionality**
  In high-dimensional space, points can be far apart.
  If closest point is as far as average point, it has low prediction power.
  [[Curse of Dimensionality|Read More]]
  ![Curse of Dimensionality](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/c2/cursefigure.png)

---
## Weakness
`K-NN` is essentially an interpolation method.
This means that `K-NN` cannot be used properly to predict extrapolated data
