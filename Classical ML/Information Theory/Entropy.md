# Entropy
#ml/information-theory/entropy

`Entropy` is the measure of uncertainty of a random variable.

![Entropy](https://images.shiksha.com/mediadata/ugcDocuments/images/wordpressImages/2023_02_MicrosoftTeams-image-156.jpg)

It is defined as
$$
H(D) 
= \sum^K_{c=1} P_{c}.\log\left( \frac{1}{P_{c}} \right) = - \sum^K_{c=1} P_{c}.\log (P_{c})
$$
where
- $H$ is `entropy`
- $D$ is the `dataset` over $K$ possible outcomes with probability $P_{c}; \ c\in \{ 1, 2, \dots, K \}$ 

Note that we often use $\log_{2}$ `base-2`, since `information theory` usually deal with binary values.

---
## Range of Uncertainty
$$
0 \leq H(D) \leq \log(K)
$$
where
- $H(D)$ is the `entropy` of dataset $D$
- $K$ is the number of outcomes

### Minimum Uncertainty
For there to be `minimum uncertainty`, consider the `Delta Function` such that $P_{c_{1}} = 1$  and $P_{c_{j \neq 1}} = 0$.
Hence, 
$$
H(D) = \sum^N_{i=1} -P_{c}.\log(P_{c}) = -\log(1) = 0
$$

### Maximum Uncertainty
For there to be `maximum uncertainty`, consider a situation where every outcome is equally likely - $P_{c} = \frac{1}{K}$.
Hence,
$$
\begin{align}
H(D) &= - \sum^N_{i=1} P_{c}\log(P_{c}) \\[6pt]
&= - \sum^K_{i=1} \frac{1}{K} \log\left( \frac{1}{K} \right) \\[6pt]
&= \sum^K_{i=1} \frac{1}{K} \log\left( K \right) \\[6pt]
&= \log\left( K \right) \\[6pt]
\end{align}
$$

---
## See Also
- [[Information Gain]]
- [[Entropy for Binomial Distribution]]

