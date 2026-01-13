# Empirical Risk Minimization
#ml/statistical-learning/erm

`Empirical Risk Minimization` is a core principle in machine learning used to select a model by minimizing its average error on a known dataset.

![ERM|300](https://mblogthumb-phinf.pstatic.net/MjAyMDA0MTZfNCAg/MDAxNTg3MDE2ODQ0ODgz.yUmqUzb_On4owI-c8wsqEkOPZpICPDnueCI8tc-_go8g.PODlLlUuTlrLqnhAWqdoM4j9zS6YmSDUqI6NS1PrVzwg.PNG.cheeryun/image.png?type=w800)

---
`Motivation`
We want to pick the best hypothesis, which means minimizing the [[Risk Function]]: 
$$
h^* = \arg \min_{h\in \mathcal{H}} \ L_{D,f}(h)
$$

But the problem is we don't know
- the data distribution $\mathcal{D}$
- the labelling function $f(x)$

Here is what know:
- The `training set` $S  = \{ (x_{1}, y_{1}), \ \dots,  \ (x_{m}, y_{m}) \}$
- For a classifier, we are trying to estimate the probability of correctly labelling a sample

Hence, we can treat this as a `Bernouilli` random variable
  
$$
p_{Bern} \approx \frac{1}{M} \sum^M_{m=1} 
\mathbb{1}(z_{m} = 1)
$$

---
## Empirical Risk
The `empirical risk` can be defined as
$$
L_{S}(h) \triangleq
\frac{1}{M} \sum^M_{m=1} \mathbb{1}(h(x_{m}) \neq y_{m})
$$
where
- $S  = \{ (x_{1}, y_{1}), \ \dots,  \ (x_{m}, y_{m}) \}$ is the `sample`(dataset)
- $M = |S|$ is the `dataset size`
- $h(x_{m})$ is the `prediction` of hypothesis on $x_{m}$ input
- $y_{m}$ is the `true label` of $m^{th}$ example

This can also be considered as `training error`, since this is the error classifier incurs over the training sample.

> `Note`
> We have been able to define this learning paradigm without discussing the learning algorithm $A(S)$ or the expression of hypothesis $h$

`Empirical Risk Minimization (ERM)`
The process of finding $h$ that minimizes $L_{S}(h)$ is called `ERM`.

---
## Problem with ERM: Overfitting
Consider an overfit classification hypothesis that perfectly minimizes the empirical risk
$$
h_{S}(x) = \begin{cases}
y_{i} \ ,   
& \text{if } \exists i \in \{ 1, \dots, M \}  
\text{ s.t. } x_{i} = 1 \\[6pt]

0 & \text{otherwise}
\end{cases}
$$
Then, $L_{S}(h) = 0$ since $y_{i} = h_{S}(x_{i})$, $\forall i \in \{ 1, \dots, M \}$.

> `Overfitting` is when a learned hypothesis performs very well on the training data, but very poorly on the testing data

---

## Inductive Bias

> `Inductive Bias` is built-in assumption that allows the algorithm to extrapolate from data it has seen, to data it hasn't seen before.

## Empirical Risk Minimization with Inductive Bias

Hence to avoid overfitting, we need to provide `ERM` with an `inductive bias`.

We can do this by restricting the hypothesis class $\mathcal{H}$.
Try to pick $\mathcal{H}$ that does not contain any hypothesis $h \in \mathcal{H}$ that can overfit the data.

We can define this `ERM` as
$$
ERM_{\mathcal{H}} \ (S) \in \arg \min_{h \in \mathcal{H}} L_{S}(h)
$$

[[PAC Learning|Read More]]

---
## See Also 
- [[Risk Function]]
- [[PAC Learning]]
