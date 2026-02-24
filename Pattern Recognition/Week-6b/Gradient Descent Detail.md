# Gradient Descent
The `gradient` of a function $\nabla f(\mathbf{w})$ gives the direction to move in the input space $\mathbf{w}$ that most effectively increase the value of the function $f: \mathbb{R}^d \to \mathbb{R}$.

We define the `gradient` as:
$$
\nabla f(\mathbf{w})
= \left( \frac{\partial f(\mathbf{w})}{\partial w_{1}} 
\ , \dots \ , \  \frac{\partial f(\mathbf{w})}{\partial w_{d}}\right)
$$
---
## Gradient Descent Algorithm
If we start from some initial point $\mathbf{w}^{(1)}$, then we can minimize $f(\mathbf{w})$ by updating the input as follows:
$$
\mathbf{w}^{(t+1)}
= \mathbf{w}^{(t)} - \eta \ \nabla f(\mathbf{w^{(t)}})
$$
where $\eta > 0$.

The output of this process after $T$ steps could be
- the last vector $\mathbf{w}^{(T)}$
- an average vector $\frac{1}{T} \sum^T_{t=1} \mathbf{w}^{(t)}$
- the best vector $\arg \min_{t\in [T]} f(\mathbf{w}^{(t)})$

---
## The Taylor Series Expansion
The [[First-Order Taylor Expansion]] of $f(u) \approx f(\mathbf{w}) + \langle \mathbf{u} - \mathbf{w}, \nabla f(\mathbf{w}) \rangle$.

But if $f$ is [[Convex Function|convex]], then `Taylor series` approximation lower-bounds the function:
$$
f(\mathbf{u}) \geq
f(\mathbf{w}) + \langle \mathbf{u} - \mathbf{w}, 
\ \nabla f(\mathbf{w}) \rangle
$$
meaning that we won't accidentally increase $f$. 
The approximation will be close as long as the candidate point $\mathbf{w}$ is near the current value $\mathbf{w}^{(t)}$.

**Objective Function**
In order to keep our selected point close to $\mathbf{w}^{(t)}$, we can define an objective function:
$$
\mathbf{w}^{(t+1)}
= \arg\min_{\mathbf{w}} \frac{1}{2} 
||\mathbf{w} - \mathbf{w}^{(t)}||^{2}_{2}
+ \eta \ \left( \ f(\mathbf{w}^{(t)}) 
+ \langle \mathbf{w} - \mathbf{w}^{(t)}, 
\ \nabla f(\mathbf{w}^{(t)}) \rangle \ \right)
$$
where $\eta$ trades off the relative importance of minimizing $f$ and staying clase to $\mathbf{w}^{(t)}$.

Taking the derivative of the above $w.r.t$ $\mathbf{w}$ yields the same equation as derived from directly describing the gradient descent.

---
## Optimizing Non-differentiatable Functions
Recall that we replaced our non-convex loss function with a [[Surrogate Loss Function|surrogate loss function]] that we could use during our [[Empirical Risk Minimization (ERM)|ERM procedure]].

With a [[Convex Function|convex loss function]] we can use gradient methods to minimize the risk, provided the loss function is differentiable everywhere.

But not all loss functions are differentiable, regardless of whether they are convex or not.
For example, 
- $\ell^{0-1}(h, \ (\mathbf{x}, \mathbf{y})) = \mathbb{1}(h(\mathbf{x} \neq \mathbf{y}))$
- Absolute error: $|h(\mathbf{x}) - \mathbf{y}|$

In order to optimize them, we employ [[Subgradient|subgradients]].

---
## Sensitivity to Initialization
Our optimization task is unlikely to really be convex, so weight initialization matters.

![Weight Initialization Effect|400](https://miro.medium.com/v2/1*yd9YMy2pkzCY-9jrsWFesA.gif)

The `Glorot initialization` is often used:
$$
w_{i,j} \sim \mathcal{U}
\left( -\sqrt{ \frac{6}{m+n} }, 
\ \sqrt{ \frac{6}{m+n} } \right)
$$
where $w_{i,j}$ is an element of $W \in \mathbb{R}^{m \times n}$.

Also `random restarts` are an important tool in optimization problems.

---
## Epochs & Mini-Batches
[[Gradient Descent]] is technically computed over all the data in $S = (z_{1}, \ \dots, \ z_{m})$ and the function we are minimizing is the [[Empirical Risk|empirical risk]] over the entire dataset:

$$
\begin{align}
&\mathbf{w}^{(t+1)} \\[6pt]

&= \mathbf{w}^{(t)} - \eta \ \nabla  
L_{S}(\mathbf{w}^{(t)}) \\[6pt]

&= \mathbf{w}^{(t)} - \eta \ \frac{1}{m} 
\sum^m_{i=1} \nabla \ell(\mathbf{w}^{(t)},  
\ (\mathbf{x}_{i}, \mathbf{y}_{i})) \\[6pt]
\end{align}
$$
Note that to evaluate this, we need to 
- shuffle the dataset $S$
- next, lock the weights in place
- then, compute a gradient at every data point in $S$
- average them together
- and compute the next weights

We do this once per epoch $(t)$,  and then repeat the process.

For large datasets, it is impractical to keep in memory.
So, we break them up into mini-batches.
`Mini-batches` are random subsamplings of $S$ without replacement.
