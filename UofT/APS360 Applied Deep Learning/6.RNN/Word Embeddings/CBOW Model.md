# Continuous Bag of Words(CBOW) Model
[[CBOW Model|CBOW model]] predicts the center word from a fixed window size of context words.

![|400](https://i.ytimg.com/vi/UqRCEmrv1gQ/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLB_CznHcAtYYdY_fH-Acgflpa7M1g)

Its [[Log Likelihood|log likelihood]] can be defined as
$$
\boxed{
\frac{1}{T} \sum ^{T}_{t=1} \sum_{-c \ \leq \ j \ \leq \ c, \ j\neq 0} \log \ p(w_{t} \mid w_{t+j})
}
$$

---
## Comparism to Skip-Gram
Compared to [[Skip-Gram]],
- [[Skip-Gram]] is better for small datasets and rare words.
- [[CBOW Model]] is faster and better for frequent words.

---
## See Also
- [[Word2Vec]]
- [[Skip-Gram]]
- [[CBOW Model]]
- [[GLoVe]]
