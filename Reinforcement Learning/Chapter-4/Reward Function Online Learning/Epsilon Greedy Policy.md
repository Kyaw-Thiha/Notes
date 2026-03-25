# Problem with Greedy Policy
#rl/policy/epsilon-greedy 
Recall from [[Online Learning of the Reward Function]] that by selecting
$$
a \leftarrow \pi_{g}(s; \ r) 
= \arg\max_{a \in \mathcal{A}} \ r(s,a)
$$
we would also choose optimal action.

But $r$ is often estimated as $\hat{r}$ using [[Stochastic Approximation(SA)|stochastic approximation]], and this inaccuracy may leads sub-optimal action.

---
## Example
Consider the [[Markov Decision Process (MDP)|MDP]] where we have only one state $s_{1}$ with two actions $a_{1}$ and $a_{2}$.
Let the [[Reward|reward function]] be
$$
\begin{align}
r(s_{1}, a_{1}) = 1 \\[6pt]
r(s_{1}, a_{2}) = 2 \\[6pt]
\end{align}
$$


Suppose that at first episode $t=1$, the agent chose $a_{1}$.
So, its [[Online Learning of the Reward Function|estimates]] will be 
$$
\begin{align}
&\hat{r}_{2}(s_{1}, a_{1})  
= (1 - \alpha_{1}) \times 0 + \alpha_{1} \times 1 > 0
\\[6pt]
&\hat{r}_{2}(s_{1}, a_{2}) = \hat{r}_{1}(s_{1}, a_{2}) = 0
\end{align}
$$

Suppose the agent uses the [[Greedy Policy]].
Then next time agent arrives at state $s_{1}$, 
- it would choose action $a_{1}$ again, and $\hat{r}_{3}(s_{1}, a_{1})$ remain positive.
- since $a_{2}$ is not selected, $\hat{r}_{3}(s_{1}, a_{2})$ remains equal to $0$.

Following [[Greedy Policy]], the optimal action $a_{2}$ is never chosen.

---
## Epsilon Greedy Policy
**Solution**: Force the agent to regularly pick actions other than the one suggested by the [[Greedy Policy|greedy policy]].

For $\epsilon\geq0$ and a [[Reward|reward function]] $\hat{r}$, we define [[Epsilon Greedy Policy|epsilon-greedy strategy]] policy $\pi_{\epsilon}$ as
$$
\pi_{\epsilon}(s; \ \hat{r}) 
= \begin{cases}
\pi_{g}(s; \ \hat{r})  
& \text{probability: } 1-\epsilon \\[6pt]
\text{Uniform}(\mathcal{A})
& \text{probability: } \epsilon \\[6pt]
\end{cases}
$$
There is non-zero probability of selecting any of the actions, so asymptotically all of them are selected infinitely often.

If $\alpha_{t}$ is selected properly, the [[Stochastic Approximation(SA)|SA]] conditions are satisfied and hence, $\hat{r} \to r$ uniformly.

---
## Exploration-Exploitation Tradeoff
The uniform choice of action in $\epsilon$-greedy helps the agent explore all actions.

- **Exploitation** is an optimal action when $\hat{r}$ is exactly equal to $r$.
- But we have uncertainty about the world.
  **Exploration** is exploring other actions which might be better.

---
## See Also
- [[Online Learning of the Reward Function]]
- [[Stochastic Approximation(SA)]]
- [[Greedy Policy]]