# Linear System Policy Evaluation
#rl/planning/linear-system 

We take advantage of the recursive nature of the [[Value Function]] $V^{\pi}$ as represented by [[Bellman Equation for Value Function|Bellman Equation]] $V^{\pi} = T^{\pi}V^{\pi}$, by using a `linear system of equations` to solve it.

---
## Linear System
In the discrete state-action case, the [[Bellman Equation for Value Function|Bellman Equation]] 
- defines a set of $n = |\mathcal{S}|$ equations
- with $|\mathcal{S}|$ unknowns $( \ V(s_{1}), \ \dots, \ V(s_{n}) \ )$
- and for each $s \in \mathcal{S}$, it has the form of

$$
V(x) = r^{\pi}(s) + \gamma \sum_{s \in \mathcal{S}}
\mathcal{P}^{\pi}(s' \mid s) \ V(s')
$$

As the solution of [[Bellman Equation for Quality Function|Bellman Equation]] is unique (as proven [[Uniqueness of Fixed Point|here]]), the solution $V$ of these equations would be same as $V^{\pi}$.

Since the [[Bellman Equation for Value Function|Bellman equation]] for a policy $\pi$ is a linear system of equations, we can use any standard solver to compute $V = V^{\pi}$.

---
## Showing linearity
To clearly see the linearity, rearrange the equation to get
$$
V(x) - \gamma \sum_{s' \in \mathcal{S}}
\mathcal{P}^{\pi}(s' \mid s) \ V(s')
= r^{\pi}(s)
$$

We can also vectorize it more compactly as
$$
(\mathbf{I} - \gamma \mathcal{P}^{\pi}) \ V = r^{\pi}
$$
where
- $\mathbf{I}$ is an $n \times n$ identity matrix
- $\mathcal{P}^{\pi}$ denote $n \times n$ stochastic matrix with $[\mathcal{P}^{\pi}]_{i,j} = \mathcal{P}^{\pi}(s_{j} \mid s_{i})$.

Note that this is the same form as 
$$
A_{n \times n} \ s_{n \times 1}
= b_{n \times 1}
$$
which represent a linear system of equations.

If the size of state-space $n$ is not very large, we can solve the linear system by computing the inverse of $A$.
In other words, we can solve $V = (\mathbf{I}  - \gamma \mathcal{P}^{\pi})^{-1} \ r^{\pi}$.

---
## Limitations
- When the state space is very large, computing the matrix inverse does not scale very well.
- To solve the `control problem` of finding $V^{*}$, we need to solve $V = T^{*}V$ whose unique solution is $V^{*}$.
  This would be $n$ equations of
$$
V(s) = \max_{a \in \mathcal{A}} 
\left\{  r(s,a) + \gamma \ \sum_{s' \in \mathcal{S}}  \mathcal{P}(s' \mid s,a) \ V(s') \right\}
$$
	These equations are not linear $V$ anymore so, the use of a linear solver is not feasible.

---
## See Also
- [[Naive Policy Evaluation]]
