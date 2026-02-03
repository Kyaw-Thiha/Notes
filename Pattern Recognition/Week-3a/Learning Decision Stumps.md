# Learning Decision Stumps
#ml/models/classic/decision-tree/decision-stump 
Learning [[Decision Stump|decision stumps]] over $\mathcal{X}=\mathbb{R}^d$ from the `hypothesis class` can be defined as
$$
\mathcal{H}_{DS}
= \{ x \mapsto \text{sign}(\theta - x_{j}) \cdot b
\quad : \theta \in \mathbb{R}, \ j \in [d], \ 
b \in \{ \pm 1 \} \}
$$
where
- $\mathcal{X} = \mathbb{R}^d$ is the `input space` with $d$ features
- $\mathcal{H}_{DS}$ is the `hypothesis class` of all decision stumps over $\mathbb{R}^d$
- $j \in [d]$ is the `feature index` which can be any integer in $[d] = \{ 1, 2, \dots, d \}$
- $\theta \in \mathbb{R}$ is the `threshold` used to split along feature $j$
- $b \in \{ \pm 1 \}$ is a `polarity/flip bit` that can invert prediction
- $\text{sign}(\cdot)$ is the `sign function` such that
$$
\text{sign}(t)
= \begin{cases}
+1 & , \ t>0 \\
-1 & , \ t<0 \\
0 & , \ t=0
\end{cases}
$$

---
## Defining Empirical Risk
To define [[Empirical Risk]] for learning `decision stumps`, let
- $b=1$ for sake of simplicity
- $S = ((x_{1}, y_{1}, \ \dots, \ (x_{m}, y_{m}))$ be our `dataset`
- $\mathbf{D} \in R^m_{+}$ be the `factor distribution` indicating the relative importance of our sample points

Implementing a [[Decision Stump|decision stump]] minimizing the [[Empirical Risk|empirical risk]] $L_{S}(h)$ $w.r.t$ distribution $\mathbf{D}$, we get
$$
L_{D}(h)
= \sum^m_{i=1} D_{i} \ \mathbb{1}(h(x) \neq y_{i})
$$

Hence, minimizing $L_{D}(h)$ becomes
$$
F(\theta)
= \min_{j\in[D]} \min_{\theta \in \mathbb{R}}
\left( 
\sum_{i:y_{i} = 1} D_{i} \ \mathbb{1}(x_{i,j} > \theta) 
+ \sum_{i:y_{i}=-1} D_{i} \ \mathbb{1}(x_{i,j} \leq \theta)
\right)
$$
Note that the order in which we minimize $j \in [D]$ and $\theta \in \mathbb{R}$ does not matter.

---
## ERM for Decision Stumps
Fix the number of features $j \in [d]$ .
Sort the data points such that $x_{1,j} \leq x_{2,j} \leq \dots \leq x_{m,j}$.

Define
$$
\Theta_{j}
= \left\{  \frac{x_{i,j} + x_{i+1,j}}{2}: i \in [m-1]  \right\}
\cup 
\{ (x_{1,j} - 1), (x_{m,j} + 1) \}
$$
where
- $\left\{  \frac{x_{i,j} + x_{i+1,j}}{2}: i \in [m-1]  \right\}$ is a set of `midpoints` between adjacent datasets $x_{i,j}$ and $x_{i+1,j}$
- $x_{i,j}-1$ and $x_{m,j}+1$ are `extreme thresholds` on either side of the set

So essentially, $\Theta_{j}$ can be thought of as a set of all candidate thresholds for a `decision stump`.

Note that for any $\theta \in \mathbb{R}$, $\exists \ \theta' \in \Theta_{j}$ that results in same predictions.

Therefore, we can minimize over the `finite set` $\Theta_{j}$.

---
`Updating`
Since we have to compute a sum of $m$ elements for $d$ dimensions and $m$ elements in $\Theta_{j}$, this process has complexity of $O(dm^2)$.

Note that when we loop through elements of $\Theta_{j}$, we are changing the label of $1$ element from positive to negative.

Hence, we can update the score as
$$
\begin{align}
&F(\theta') \\[6pt]
&= F(\theta) - D_{i} \ \mathbb{1}(y_{i}=1)
+ D_{i} \ \mathbb{1}(y_{i} = -1) \\[6pt]
&= F(\theta) - y_{i}D_{i}
\end{align}
$$

---
## See Also
- [[Decision Stump]]
- [[Decision Tree]]