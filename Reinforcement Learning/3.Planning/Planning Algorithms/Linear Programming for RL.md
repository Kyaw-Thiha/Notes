# Linear Programming for finding $V^{*}$
#rl/planning/linear-programming
`Linear Programming` can be used to solve `RL problems` by [[Value Function]] approximation and offline RL.

![Linear Programming|400](https://i.ytimg.com/vi/Bzzqx1F23a8/maxresdefault.jpg)

---
## Derivation
Consider a set of all $V$ that satisfies $V \geq T^{*}V$.
Then,
$$
C = \{ V: V \geq T^{*}V \}
$$
Note an interesting property that for any $V \in C$, by the [[Monotonicity of Bellman Operator|monotonicity]] of $T^{*}$, we have
$$
V \geq T^{*}V
\implies T^{*}V \geq T^{*}(T^{*}V)
= (T^{*})^{2} \ V
$$

Repeating this argument, we get that for any $m \geq 1$,
$$
V \geq (T^{*})^{m} \ V
$$
We take the limit of $m\to \infty$, and use the [[Contraction of Bellman Operator|contraction property]] of [[Bellman Optimality Operator]] to conclude that
$$
V \geq \lim_{ m \to \infty } 
(T^{*})^{m} \ V = V^{*}
$$
This can be interpreted as
- any $V \in C$ is lower bounded by $V^{*}$
- or $V^{*}$ is the function in $C$ that is smaller or equal to any other function in $C$ in pointwise sense
  $V_{1} \leq V_{2} \iff V_{1}(s) \leq V_{2}(s), \ \forall s \in \mathcal{S}$

Hence to find $V^{*}$, we find a $V_{0} \in C$ $s.t.$ $V_{0} \leq V, \ \text{where } V\neq V_{0} \in C$.

---
### Methodology
We can do this by choosing a strictly positive vector $\mu > 0$ with the dimension of $\mathcal{S}$.
$\mu$ can be thought of as probability distribution with a support on $\mathcal{S}$

This results in us solving the following optimization problem
$$
\min_{V \ \in \ C} \mu^{T}V
$$
which can be written as
$$
\begin{align}
\min_{V} \mu^{T}V  
\ , \quad s.t. \ V(s) \geq (T^{*}V)(s) 
\ , \quad \forall s \in \mathcal{S}
\end{align}
$$
This has a linear objective and a set of $|\mathcal{S}|$ nonlinear constraints.
We can convert each of nonlinear constraints to $|\mathcal{A}|$ linear constraints by using the equivalence
$$
\begin{align}
&V(s) \geq \max_{a \in \mathcal{A}}
\left\{  r(s,a) + \gamma \sum_{s'}  
\mathcal{P}(s' \mid s,a) \ V(s')  \right\} \\[6pt]

\iff 
&V(s) \geq  r(s,a) + \gamma \sum_{s'}  
\mathcal{P}(s' \mid s,a) \ V(s') \ , \quad 
\forall a \in \mathcal{A} \\[6pt]
\end{align}
$$
Therefore, we can solve 
$$
\begin{align}
&\min_{V}  \ \mu^{T}V \\[6pt]
&s.t.  \ V(s) \geq r(s,a) + \gamma \sum_{s'}
\mathcal{P}(s' \mid s,a) V(s')
\ , \ \forall(s,a) \in \mathcal{S} \times \mathcal{A}
\end{align}
$$
This is a linear program with $|\mathcal{S} \times \mathcal{A}|$ constraints.

---
### Note on $\mu$
> The choice of $\mu$, as long as $\mu>0$, does not matter

To see this clearly, suppose we find a $\hat{V} \neq V^{*}$ as the minimizer.
As for all $V \in C$, $V \geq V^{*}$, at least at state $s'$, we get $\hat{V}(s') > V^{*}(s')$

If that is the case, we can decrease the objective from $\mu^{T}\hat{V}$ to $\mu^{T}V^{*}$ by the amount of
$$
\underbrace{\mu(s')}_{>0}
\ \underbrace{(\hat{V}(s') - V^{*}(s'))}_{>0}
> 0
$$
So, $\hat{V}$ cannot be strictly larger than $V^{*}$.
if $\mu(s') = 0$ however, we cannot make that argument anymore.

---
## See Also
- [[Value Iteration]]
- [[Policy Iteration]]
- [[Monotonicity of Bellman Operator]]
- [[Contraction of Bellman Operator]]
