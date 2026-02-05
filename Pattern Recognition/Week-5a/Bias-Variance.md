# Bias-Variance
Consider a `least squares loss function` for a `regression` problem:

$$
l(h_{S}) = (y - h_{S}(x))^2
$$
where $h_{S}(x)$ is the `ERM minimizer`.

From [[PAC Learning]], we know that $l(h_{S})$ is a random variable dependant on the dataset $S \in \mathcal{D}^m$.

Let's consider the risk of $h_{S}$ by manipulating $l$:
$$
\begin{align}
&l(h_{S}) \\[6pt]
&= (y - h_{S}(x))^2 \\[6pt]
&= (y - \mathbb{E}_{S}[h_{S}(x)]  
+ \mathbb{E}_{S}[h_{S}(x)] - h_{S}(x))^2
& \text{add and subtract the mean} \\[6pt]
&= (y - \mathbb{E}_{S}[h_{S}(x)])^2
+ (\mathbb{E}_{S}[h_{S}(x)] - h_{S}(x))^2  
& \text{expand square}\\[6pt]
& \ + 2 (y - \mathbb{E}_{S}[h_{S}(x)])
\ (\mathbb{E}_{S}[h_{S}(x)] - h_{S}(x))
\end{align}
$$

Note that when computing the risk by taking the expectation with respect to $\mathcal{D}$, the `last term` vanishes.

---
## Understanding Risk in terms of Bias & Variance
$$
\begin{align}
L_{S}(h_{S})
&= \mathbb{E}_{S}[\ l(h_{S}) \ ] \\[6pt]
&= \mathbb{E}_{S}[\ (y - h_{S}(x))^2 \ ] \\[6pt]
&= \underbrace{(\mathbb{E}_{S}[h_{S}(x)] - y)^2}_{\text{Bias}^2}
+ \underbrace{\mathbb{E}_{S}[h_{S}(x) - \mathbb{E}_{S}[ \ (h_{S}(x)])^2 \ ]}_{\text{Variance}}
\end{align}
$$

This means that expected squared loss can be interpreted as
$$
\text{expected loss}
= (\text{bias})^2 + \text{variance} + \text{noise}
$$
where
- $(\text{bias})^2$ is the `Approximation Error`
  The extent to which the average prediction over all datasets differs from the desired regression function.
- $\text{variance}$ is the `Estimation Error`
  The extent to which predictions vary around their average as a function of data.
- $\text{noise}$ is the `variability` inherent in the labels which we can't observe.

[[Bias-Variance Decomposition|Detailed decomposition can be found here.]]

---
## Estimating $(\text{bias})^2$ and $\text{variance}$
We can approximate the `bias` and `variance` terms empirically by sampling multiple datasets.

Let's sample $L$ datasets with $m$ datapoints: $(S_{1}, \ \dots, \ S_{L})$ 
We can then compute the average prediction $h_{S}(x)$ as
$$
\bar{h}_{S}(x) = \frac{1}{L} \sum^L_{i=1} h_{S_{i}}(x)
$$
Then we can estimate as
$$
(\text{bias})^2
= \frac{1}{m} \sum^m_{i=1} (\ \bar{h}_{S}(x_{i}) - y_{i} \ )^2
$$
and
$$
\text{variance} 
= \frac{1}{m} \sum^m_{i=1}
\frac{1}{L} \sum^L_{l=1} 
\ (\ h_{S_{l}}(x_{i}) - \bar{h}_{S}(x_{i}) \ )^2
$$

---
## See Also
- [[Bias-Variance Decomposition]]
- [[Bayesian Approach to Bias-Variance Tradeoff]]
- [[PAC Learning]]
- [[Empirical Risk]]