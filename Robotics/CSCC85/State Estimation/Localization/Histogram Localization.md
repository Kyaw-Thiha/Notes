# Histogram Localization
#robotics/localization/histogram #robotics/localization/bayesian
`Histogram localization` is a discrete [[Bayes Rule|Bayes]] filter over a grid of states.

![Histogram Localization|300](https://www.researchgate.net/profile/Harry-Duong-Nguyen/publication/325809276/figure/fig5/AS:638364235595776@1529209302105/Occupancy-Grid-Map-of-multiple-spaces-stitched-into-a-common-map.png)
It divides the map into a uniform grid of small regions, and estimates the probability per region. 

---
## Main Mechanism

`Process`
The process for estimating the belief values consists of 2 steps.
- `Sensing`: Carrying out sensor measurement
- `Action`: Action is performed, and belief is updated based on the result.
- `Normalize` after each sensing/action step

`Mathematical Model`
The process of computing $B(x_{k})$ is broken into 2 parts:
$$
\begin{align}
Bel(x_{k}) &= p(x_{k} \ | \ x_{k-1}, \ a_{k-1}, \ x_{k}) \\[6pt]
&= \eta \ p(z_{k}| x_{k}) \ 
\sum_{x_{k-1}} 
p(x_{k} \ | \ p_{x_{k-1}} \ , \ a_{k-1})
\ Bel(x_{k-1})
\end{align}
$$
where
- $p(z_{k} \mid x_{k})$ is the `sensor model`
- $\sum_{x_{k-1}} \ p(x_{k} \ | \ p_{x_{k-1}} \ , \ a_{k-1}) \ Bel(x_{k-1})$ is the `motion model`
- $\eta$ is the `normalization constant`

---
`Motion Model`
Note that since the actions of the robot is noisy, all combinations of $(\text{previous state, action})$ need to be taken into account.

`Normalization`
Normalization has to be applied after `each` $\text{sensing / action}$ step.
$$
\eta = \frac{1}{\sum_{x} \ Bel(x)}
$$

---
## Limitations
- `Problem`: Scaling to larger region is computationally expensive.
  `Solution`: Perform localization over multiple scales.
  Localize at larger grid region $(100m \times 100m)$
  Then, localize at smaller grid regions $(1m \times 1m)$

- `Problem`: Real-world spaces may not subdivide cleanly into square cells.
  `Solution`: Ensure update process handle it.
  

We can also consider using [[Monte-Carlo Localization]].

---
## See Also
- [[Localization]]
- [[Monte-Carlo Localization]]