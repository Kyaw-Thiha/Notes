# Transformer for Language Modelling
Compared to static embeddings like [[Word2Vec]] and [[GLoVe]], [[Transformer]](and [[Recurrent Neural Network (RNN)|RNN]]), 
- can learn contextual embeddings.
- such that embedding of a same word changes according to the context it appear in.

---
## Input Embeddings

![image|500](https://notes-media.kthiha.com/Transformer-for-Language-Modelling/f01449b73b1e8a83d50ed404e6349374.png)

---
## Masked Word Prediction
- Replace $15\%$ of words at random with `[MASK]` token.
- Using the context of non-masked words, predict original value of `[MASK]` token.
- Loss is computed on just the masked word.

---
## Next Sentence Prediction
- Given two sentences, predict if they appear together.
- Create $50\%$ positive and $50\%$ negative pairs of sentences.

---
## See Also
- [[Transformer]]
- [[Transformer for Language Modelling]]
- [[Self-Attention]]
- [[Attention Mechanism]]
- [[Recurrent Neural Network (RNN)]]
- [[Word2Vec]]
- [[GLoVe]]
