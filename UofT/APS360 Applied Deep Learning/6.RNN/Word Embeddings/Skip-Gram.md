# Skip-Gram
[[Skip-Gram]] predict context words from a given target word.

![300](https://substackcdn.com/image/fetch/$s_!scJv!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F43ab08e4-0e25-46ef-a9fa-3ced2e8e332d_1920x1080.gif)

Its [[Log Likelihood|log likelihood]] can be defined as
$$
\boxed{
\frac{1}{T} \sum ^{T}_{t=1} \sum_{-c \ \leq \ j \ \leq \ c, \ j\neq 0} \log \ p(w_{t+j} \mid w_{t})
}
$$

---
An [[Skip-Gram|n-gram]] is a continguous sequence of $n$ items from a given text.

![image|300](https://notes-media.kthiha.com/Skip-Gram/3bd0171e456401735991f074a1996f35.png)

A [[Skip-Gram|k-skip n-gram]] is an [[Skip-Gram|n-gram]] that can involve a skip operation of size $k$ or smaller.

---
## Skip-Gram Model

![300](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWnNjogUiw8S7ZTQ9A8X-pvyhAnWawh0JdM7VMwPTAd-EGL0gJK3L6hE0&s=10)

Note that the output classifier layer is only used for training.

After training, words that have similar context words will be mapped to similar embeddings.

---
## See Also
- [[Word2Vec]]
- [[Skip-Gram]]
- [[CBOW Model]]
- [[GLoVe]]
