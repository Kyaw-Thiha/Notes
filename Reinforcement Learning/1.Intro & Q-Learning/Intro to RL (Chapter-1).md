# Intro to RL 
#rl
This chapter covers the fundamentals of [[Markov Decision Process (MDP)|MDP]], followed by introducing the concept of [[Reward|rewards]], and the [[Value Function|value]]/[[Quality Function|quality function]].

## MDP
`Markov Decision Process (MDP)` is a mathematical framework use to model sequential decision-making.

![MDP|400](https://notes-media.kthiha.com/Markov-Decision-Process-(MDP)/1bac4039479d810db0c24420a5c5014b.png)
[[Markov Decision Process (MDP)|Read More]]

---
## Markov Policy
A `Markov Policy` is a decision rule used in [[Markov Decision Process (MDP)]] that maps the current state of an environment to an action.

![Markov Policy](https://gibberblot.github.io/rl-notes/_images/deterministic_vs_stochastic_policy.png)
[[Policy|Read More]]

---
## Rewards
To find an optimal policy, we need to teach the agent to maximize the [[Reward|expected reward]].

![Expected Reward|400](https://huggingface.co/blog/assets/63_deep_rl_intro/rewards_4.jpg)
[[Reward|Read More]]

## Value Function
This naturally leads to the introduction of [[Value Function]].
![Bellman Equation|500](https://machinelearningmastery.com/wp-content/uploads/2024/07/rl-bellman-equation-mlm.png)
[[Value Function|Read More]]

## Quality Function
`Q-Function` defines the expected cumulative future [[Reward|reward]] of taking a specific action in a given state.

![Quality Function](https://cugtyt.github.io/blog/rl-notes/R/q-func-eq.png)
[[Quality Function|Read More]]

Apart from those, we also formalize the definitions of 
- [[Policy-Induced Transition Kernel]]
- [[Episode]]
- [[Observation]]

---
## See Also
- [Slides](https://amfarahmand.github.io/IntroRL/lectures/lec01.pdf)
- [Book](https://amfarahmand.github.io/IntroRL/lectures/FRL.pdf)
- [[Markov Decision Process (MDP)]]
- [[Policy]]
- [[Policy-Induced Transition Kernel]]
- [[Episode]]
- [[Reward]]
- [[Value Function]]
- [[Quality Function]]
- [[Observation]]

---