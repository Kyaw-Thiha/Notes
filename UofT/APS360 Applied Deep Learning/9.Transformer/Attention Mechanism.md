# Attention Mechanism
[[Attention Mechanism|Attention]] lets a [[Neural Network|model]] decide which parts of its input are most relevant when producing each part of its output.

![300](https://www.comet.com/site/wp-content/uploads/2023/07/simple-pretty-gif.gif)

---
### Simple Attention Mechanism
- Use [[Neural Network|fully-connected network]] that
	- takes in embeddings, and
	- generate single score for each of them
- Normalize these scores across all inputs using [[Softmax Function|softmax]].
- Multiply each embedding with normalized score and sum

$$
c_{i} = \sum_{j} \alpha_{ij} h_{j}
$$

- Train this [[Neural Network|network]] end-to-end.

---
### Computing Attention Score
Suppose we have two embeddings $a,b \in \mathbb{R}^{d}$. We could use
- **Dot Product**: $\text{score}(a,b) = a^{T} \cdot b$
- **Cosine Similarity**: $\text{score}(a,b) = \frac{a^{T} \cdot b}{||a|| \cdot ||b||}$
- **Bilinear**: $\text{score}(a,b) = a^{T} \ \text{W} \ b$
- **MLP**: $\text{score}(a,b) = \text{Sigmoid}(W[a;b])$

---
## See Also
- [[Transformer]]
- [[Transformer for Language Modelling]]
- [[Self-Attention]]
- [[Attention Mechanism]]
- [[Recurrent Neural Network (RNN)]]

