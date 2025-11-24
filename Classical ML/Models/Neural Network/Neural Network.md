# Neural Network
#ml/models/neural-network

A `Neural Network` is a function composed of `linear transformations` and `non-linear activations`.

$$
f_{\theta}: R^n \to R^m
$$

![Neural Network](https://cdn-images-1.medium.com/v2/resize:fit:732/1*KNZZYteeBqkJViS1_LT1CQ.gif)

Another way to see `Neural Network` is as a `Feature Mappings` model which we do not need to manually craft, but is rather learned through data.

---
## Short History
`Perceptron`
The theory of `Perceptron` was invented by [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) in 1958 as a single neuron for binary classification.
It wasn't able to represent the `XOR` gates, which led to the first AI Winter.

`Backpropagation`
In 1980s, `Backprogation` algorithm was invented by [Hinton et. al](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) which allowed any binary function to be reprsented by stacking layers of perceptrons.

---
## Universal Approximation Theorem
A `Feedforward Neural Network` with a `single` hidden layer containing finite number of neurons can approximate any continuous function on compact subsets of $R^N$, under different assumptions of the `Activation Function`:

- Arbitrary width and bounded depth: [Hornik(1989)](https://www.cs.cmu.edu/~epxing/Class/10715/reading/Kornick_et_al.pdf) and [Cybenko(1989)](https://web.njit.edu/~usman/courses/cs675_fall18/10.1.1.441.7873.pdf)
- Arbitrary depth and bounded width: [Lu et. al.(2017)](https://papers.nips.cc/paper_files/paper/2017/file/32cbf687880eb1674a07bf717761dd3a-Paper.pdf)
- Bounded depth and width: [Maiorov and Pinkus(1999)](https://www.sciencedirect.com/science/article/pii/S0925231298001118)

> Most importantly, it shows that two hidden layers are enough to approximate any functions

---
## Activation Functions
`Activation Functions` $\sigma(\cdot)$ introduces `non-linearity` so that the `Neural Network` can represent non-linear functions.
- `Relu`
- [[Sigmoid Function]]
- [[Softmax Function]]
- `Tanh`
- `Gelu`


