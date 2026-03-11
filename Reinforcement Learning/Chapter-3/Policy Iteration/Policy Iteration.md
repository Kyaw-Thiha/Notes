# Policy Iteration
#rl/planning/policy-iteration
`Policy iteration` computes optimal [[Markov Policy|policy]] $\pi^{*}$ based on iteratively applying the following two steps:

![Policy Iteration|300](https://notes-media.kthiha.com/Policy-Iteration/27ab2bb8a7b2c99ebd81c92c6409aafc.png)
- **Policy Evaluation**: Given a policy $\pi_{k}$, compute $V^{\pi_{k}}$ ($(Q^{\pi_{k}})$).
- **Policy Improvement**: 
  Find a new policy $\pi_{k+1}$ that is better than $\pi_{k}$ ($V^{\pi_{k+1}} \geq V^{\pi_{k}}$).
  (with a strict inequality in at least one state, unless at convergence)


---
## How to perform Policy Iteration
- **Policy Evaluation**: We can either solve a [[Linear System Policy Evaluation|linear system of equations]] or perform [[Value Iteration|value iteration(PE)]] to compute the value of policy $\pi_{k}$.
- **Policy Improvement**: Choose a [[Greedy Policy|greedy policy]]

$$
\pi_{k+1}(s) \leftarrow \pi_{g}(s; \ Q^{\pi_{k}})
= \arg\max_{a \in \mathcal{A}} Q^{\pi_{k}}(s,a)
\ , \forall s \in \mathcal{S}
$$
 
---
### Intuition behind using Greedy Policy
Assume at state $s$, we act according to [[Greedy Policy|greedy policy]] $\pi_{g}(s; \ Q^{\pi_{k}})$.
Then afterwards, we follow the policy $\pi_{k}$.

The value of of this new policy is
$$
\begin{align}
&Q^{\pi_{k}}(s; \ \pi_{g}(s; \ Q^{\pi_{k}})) \\[6pt]
&= Q^{\pi_{k}}(s; \ \arg\max_{a \in \mathcal{A}} 
Q^{\pi_{k}}(s,a)) \\[6pt]
&= \max_{a \in \mathcal{A}} Q^{\pi_{k}}(s,a) \\[6pt]
\end{align}
$$

Comparing $\max_{a \in \mathcal{A}} Q^{\pi_{k}}(s,a)$ with the value of following the current policy at state $s$, which is $V^{\pi_{k}}(s) = Q^{\pi_{k}}(s, \ \pi_{k}(s))$, we get that
$$
Q^{\pi_{k}}(s, \ \pi_{g}(s; \ Q^{\pi_{k}}))
= \max_{a \in \mathcal{A}} Q^{\pi_{k}}(s,a)
\geq V^{\pi_{k}}(s)
$$

So the new policy is equal to or better than $\pi_{k}$ at state $s$.

---
## Policy Iteration Algorithm
The `policy improvement` step only required the new policy $\pi_{k+1}$ to have a [[Value Function|value]] larger than the previous policy's $(V^{\pi_{k+1}} \geq V^{\pi_{k}})$.

In this `policy iteration (PI)` algorithm, we will refer to specific case where we pick the new [[Markov Policy|policy]] $\pi_{k+1}$ as the [[Greedy Policy|greedy policy]] $\pi_{g}(Q^{\pi_{k}})$.

Note that
- the [[Value Function|value function]] of $\pi_{k}$ is the [[Fixed Point|fixed point]] of $T^{\pi_{k}}$ 
  by the [[Uniqueness of Fixed Point]] theorem
- the [[Greedy Policy|greedy policy]] satisfies $T^{\pi_{k+1}} Q^{\pi_{k}} = T^{*}Q^{\pi_{k}}$ 
  by [[Greedy Policy of Optimal Value Function is Optimal Policy]]

Hence, we can summarize each iteration as
- **Policy Evaluation**: Given $\pi_{k}$, compute $Q^{\pi_{k}}$.
  Find a $Q$ that satisfies $Q = T^{\pi_{k}}Q$
- **Policy Improvement**: 
  Obtain $\pi_{k+1}$ as a policy that satisfies $T^{\pi_{k+1}}Q^{\pi_{k}} = T^{*}Q^{\pi_{k}}$

---
### Algorithm
1. Initialize $\pi_{0}$ arbitrarily.
2. $k \leftarrow 0$
3. repeat
	- $Q^{\pi_{k}} \leftarrow \text{solution of } Q=T^{\pi_{k}}Q$ (Policy Evaluation: compute $Q^{\pi_{k}}$)
	- $\pi_{k+1} \leftarrow \text{policy s.t. } T^{\pi_{k+1}}Q^{\pi_{k}} = T^{*}Q^{\pi_{k}}$ (Policy Improvement)
	- $k \leftarrow k + 1$
4. until $\pi_{k} = \pi_{k-1}$

---