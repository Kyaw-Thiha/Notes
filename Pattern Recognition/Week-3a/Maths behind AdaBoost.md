# Maths behind Adaboost
#ml/ensemble/boosting/adaboost

[[AdaBoost]] has a [[Weak Learner|weak learner]] $WL$ that 
- finds hypothesis with low empirical risk, and 
- receives as input a training set $S = (\ (x_{1}, y_{1}), \dots, (x_{m}, y_{m}) \ )$ labelled by a function $f$.

For each round $t \in \{ 1, \dots, T \}$, `Adaboost` defines a distribution over the samples in $S$:
$$
D^{(t)} \in R_{+}^m
$$
such that $\sum^m_{i=1} D_{i}^{(t)} = 1$.

Then, the `weak learner` $WL$ takes $D^{(t)}$ and $S$ in order to find a weak hypothesis with error
$$
\begin{align}
\epsilon_{t}
&\triangleq L_{D^{(t)}}(h_{t}) \\[6pt]
&\triangleq \sum^m_{i=1} D^{(t)}_{i}
\ \mathbb{1}(h_{t}(x_{i}) \neq y_{i}) \\[6pt]
&\leq \frac{1}{2} - \gamma
\end{align}
$$

Next, hypothesis $h_{t}$ gets assigned a weight $w_{t} = \frac{1}{2} \log\left( \frac{1}{\epsilon_{t}} - 1 \right)$.

After each round is complete, $\mathcal{D}^{(t)}$ is updated to reflect the examples that $h_{t}$ did a good and bad jobs on:
- The examples $h_{t}$ did well get lower probability mass.
- The examples $h_{t}$ did poorly on get higher probability mass.

The output is based on a `weighted sum` of all `weak hypothesis`.

---
## See Also
- [[AdaBoost]]
- [[Weak Learner]]