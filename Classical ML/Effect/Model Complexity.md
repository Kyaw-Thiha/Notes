# Model Complexity

`Model Complexity` is a measure of the number of parameters a model has.

![[Model Complexity.png]]
Note that this image was generated using `Polynomial Regression`

Increasing the model complexity reduce the training error consistently, but leads to `overfitting` at very high complexity.

Note that this is only true for `Classical ML Models`, while `Neural Networks` don't really follow this rule due to its `scaling law`.

Also, higher dataset size prevents `overfitting`, and allows the test loss to converge along the training loss.
