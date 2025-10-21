# Distance Measure
## 1. Cosine Similarity
Let $\vec{x_{i}}, \vec{x_{i}}, \dots \vec{x_{i}} \in R$
Then, 
$$
\cos(\theta) = \frac{\vec{x_{1} . \vec{x_{2}}}}{||\vec{x_{1}}|| . ||\vec{x_{2}}||}
$$

## 2. Manhalanobis Distance
$$
d_{M}(\vec{x_{1}}, \vec{x_{2}}) = \sqrt{ (\vec{x_{1}} - \vec{x_{2}}) . S^{-1} . (\vec{x_{1}} - \vec{x_{2}}) }
$$
where
- $S$ is the Covariance Matrix

## 3. Hamming Distance
$$
d_{H}(\vec{x_{1}}, \vec{x_{2}}) = \sum^D_{i=1}[(\vec{x_{1}})_{i} \neq (\vec{x_{2}})_{i}]
$$

