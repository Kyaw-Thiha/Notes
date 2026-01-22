# Learning Decision Stumps
#ml/models/classic/decision-tree/decision-stump 
Learning [[Decision Stump|decision stumps]] over $\mathcal{X}=\mathbb{R}^d$ from the `hypothesis class` can be defined as
$$
\mathcal{H}_{DS}
= \{ x \to \text{sign}(\theta - x_{j}) \cdot b
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
