# Feature Selection
Consider a feature space $\mathcal{X} \in \mathbb{R}^d$.
Note that $d$ affects the computational complexity of our learning algorithm.

We can also do `feature selection`.

## Exhaustive Evaluation
Task is to select $k<$a features from $\mathcal{X}$ such that we preserve algorithm performance, but reduce the dimensionality.

If dimension $d$ is a small number or training our algorithm is cheap, 
then we can exhaustively search all possible $\begin{pmatrix}d \\ k\end{pmatrix}$ combinations.

Having a good validation/test set is important.

---
## Filtering Methods
Using a performance metric, access each feature individually and
- choose $k$ best scoring features
- or choose $k$ based on scores of the features 
  (fixed no. of features)

Define $v = (x_{1,j}, \ \dots, \ x_{m,j})$ as the $j^{th} \text{ feature}$ 
from our dataset $S = ( \ (x_{1}, y_{1}), \ \dots, \ (x_{m}, y_{m}) \ )$.

To access the utility of the feature, we use the [[Empirical Risk Minimization (ERM)|ERM]] 
$$
\min_{a,b \in \mathbb{R}} \frac{1}{m}
|| av + b - y ||^2
$$

---
## Filtering by Correlation
Select features by minimizing the [[Empirical Risk Minimization (ERM)|ERM]] of a [[Linear Predictor]].
This is equivalent to selecting features based on absolute value of the `Pearson`'s correlation coeff