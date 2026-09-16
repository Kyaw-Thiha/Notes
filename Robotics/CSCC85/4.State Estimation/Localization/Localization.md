# Localization
#robotics/localization
`Localization` is a process to help robotics system estimate their precise state in a complex environment.

![Localization|500](https://www.exyn.com/hs-fs/hubfs/robot-wakes-up.gif?length=700&name=robot-wakes-up.gif)

In most applications, [[GPS]] can be used to determine the location.
However, it does not work indoors and can be inaccurate.

---

`Probabilistic Localization Model`

The belief $(\text{of how likely particular states are})$ is defined as
$$
Bel(x_{k}) 
= p(x_{k} \ | \ d_{0}, \ d_{1}, \  \dots, \ d_{k})
$$
where
- $Bel(\cdot)$ is the probability of belief
- $x_{k}$ is the current state being estimated
- $d_{i}$, $i\in \{ 0, \dots, k \}$ is the measurement at $\text{step-i}$

`Markovian Localization`
Assuming the [[Markov Property]], we get
$$
Bel(x_{k}) 
= p(x_{k} \ | \ x_{k-1}, \ a_{k-1}, \ s_{k})
$$
where
- $Bel(\cdot)$ is the probability of belief
- $x_{k}$ is the current state 
- $x_{k-1}$ is the previous state
- $a_{k-1}$ is the previous action taken
- $s_{k}$ is the current state

---
## Localization Methods

`Histogram Localization`
Histogram localization is a discrete Bayes filter over a grid of states.
[[Histogram Localization|Read More]]

`Particle-Filter Localization`
Particle-Filter localization is a [[Monte Carlo Methods|Monte Carlo Estimation]] that  represents uncertainty by simulating many particles.
[[Monte-Carlo Localization|Read More]]

---
## See Also
- [Paco's Notes](https://www.cs.utoronto.ca/~strider/docs/C85_Localization.pdf)
- [[GPS]]
- [[Histogram Localization]]
- [[Monte-Carlo Localization]]
