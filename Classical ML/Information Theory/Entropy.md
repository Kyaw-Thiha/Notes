# Entropy

`Entropy` is the measure of uncertainty of a random variable.

![Entropy](https://images.shiksha.com/mediadata/ugcDocuments/images/wordpressImages/2023_02_MicrosoftTeams-image-156.jpg)

It is defined as
$$
H(D) 
= \sum^K_{c=1} P_{c}.\log\left( \frac{1}{P_{c}} \right) = - P_{c}.\log (P_{c})
$$
where
- $H$ is `entropy`
- $D$ is the `dataset` over $K$ possible outcomes with probability $P_{c}; \ c\in \{ 1, 2, \dots, K \}$ 

Note that we often use $\log_{2}$ `base-2`, since `information theory` usually deal with binary values.
