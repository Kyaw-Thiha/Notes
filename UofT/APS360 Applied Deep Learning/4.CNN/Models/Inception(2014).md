# Inception (GoogLeNet)
[[Inception(2014)|Inception]] attempts to go deeper than [[AlexNet(2012)]] by using up to $22$ [[Convolution Layer|convolution layers]].

![500](https://miro.medium.com/0*q5eMDjUHKqEyo7qY.png)

---
## Inception Block
Uses a mixture of $3 \times 3$ and $5\times 5$ filters in one layer. 

![400](https://media.geeksforgeeks.org/wp-content/uploads/20260507143902524885/convulation_3.webp)

Hence, it shows that we don't need large $7 \times 7$ or $11 \times 11$ filters like [[AlexNet(2012)]].

---
## Pointwise Convolution
[[Pointwise Convolution|Pointwise convolution]] are $1 \times 1$ [[Convolution Layer|convolution operation]] used to mix information across the channel.

![400](https://global.discourse-cdn.com/dlai/original/2X/0/012816e3d8ab7a27f8a728c9e42aee97b943e153.jpeg)

---
## Auxiliary Loss
[[Inception(2014)|Inception Network]] is pretty deep so they are subject to [[Vanishing and Exploding Gradients|vanishing gradient problem.]]

![400](https://gaussian37.github.io/assets/img/dl/concept/inception/7.png)

Their solution is using [[#Auxiliary Loss|intermediate classifiers]]. 

---