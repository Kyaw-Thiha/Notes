# Regularization
`Regularization` is a term added to the [[Loss Function]] to help prevent `overfitting`.
It penalizes the model if it gets too complex.

![[Regularization.png]]

`Regularization` creates a `U-shaped` test error graph which has a minimum. 

At small regularization, the model is `under-regularized`. 
Thus, it is learning the noise and thus `overfitting`.

At higher regularization, the model is `over-regularized`.
The model is forced to be simple and couldn't capture the data properly, leading to `underfitting`.