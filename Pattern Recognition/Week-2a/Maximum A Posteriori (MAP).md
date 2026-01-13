### Maximum A Posteriori (MAP) Estimation
Here the `objective` is to find the $\theta^*$ that maximizes the posterior distribution.
By expressing a prior belief on $\theta$, we can constrain the algorithm.
This helps avoid `overfitting.`

$$
\begin{align}
\theta^*
&= \arg \max_{\theta \in \Theta}
\ p(\theta \mid \mathcal{D}) \\[6pt]

&= \arg \max_{\theta \in \Theta}
\frac{p(\mathcal{D} \mid \theta) \ p(\theta)} 
{\mathcal{D}}
\end{align}
$$
We will use the same trick as with `Negative Log Likelihood` to rewrite the optimization as:
$$
\theta^* 
= \arg \min_{\theta \in \Theta}
- \sum^N_{n=1} \log(p(y_{n} \mid \theta))
- \log(p(\theta)) + \log(p(\mathcal{D}))
$$
Note that $\log(p(\mathcal{D}))$ is dropped since $p(\mathcal{D})$ is constant for a fixed dataset.

---
