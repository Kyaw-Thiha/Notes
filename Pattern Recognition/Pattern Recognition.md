
## Books
[Pattern Recognition & Machine Learning (Chris Bishop)](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf&ved=2ahUKEwjD4_iH8veRAxWkDjQIHUdYOo8QFnoECCEQAQ&usg=AOvVaw0H0pLJQrNHsTN5lRxwibHL)
[Course Link: SYDE 675](https://uwflow.com/course/syde675)
Understanding Machine Learning by Goodfellow

---
## Objective
Apply labels $y \in Y$ to data $x \in X.$
We try to automatically discover some hypothesis $h:X \to Y$.
This is called `Machine Learning`.

---
### What we do in ML
- `Classification`: $f: X\to \{ 0, 1 \}$
- `Regression`: $f: X \to R$
- `Multi-Class Classification`: $f:X \to \{ 0, 1 \}^d$
- `Multi-Class Regression`: $f:X \to \{ 0, 1 \}^d$

We can do more things like
- `Ranking`: $f: X \times X \to \{ -1, 1 \}$
- `Reinforcement Learning`: $f: X\to P(A=a \mid S = s)$
- `Language Modelling`: $f: \text{Token}^n \to P(\text{Token}' = t)$
- `Image Generation`: $f: X \to R^{n\times m}$

---
### Kinds of Machine Learning
- `Generative`
  Learn the joint distribution over inputs and outputs: $P(X,Y)$
- `Discriminative`
  Learn distribution over outputs conditioned on inputs: $P(Y \mid X)$
- `Supervised`
  Given datasets of inputs and outputs $(x,y)_{i} \in X \times Y$
- `Unsupervised`
  Only given input observations $X$

---
