# Bias-Variance Decomposition

Based on [[Expected Target Value]], we can decompose [[Expected Test Error]] using [[Expected Test Error Decomposition]] to get

$$
E_{X, Y, D}[(\hat{y}_{D}(x) - y)^2]
= \underbrace{E_{X} [ (\bar{\hat{y}} - \bar{y}(x))^2] }_{\text{Bias}^2}
+ \underbrace{E_{X, D}[(\hat{y}_{D}(x) - \bar{\hat{y}}(x))^2]}_{\text{Variance}}
+ \underbrace{E_{X, Y}[(\bar{y}(x) - y)^2]}_{\text{Noise}}
$$

![[Bias-Variance Dart Board.png]]

`Expected Test Error` comes from 3 different sources
- `Bias` 
  High bias $\implies$ `underfitting`
- `Variance`
  High variance $\implies$ `overfitting`
- `Noise`
  Irreducible unpredictability of data

![[Bias-Variance Plot.png]]
Plotting `Bias-Variance` against `Regularization` factor $\lambda$
