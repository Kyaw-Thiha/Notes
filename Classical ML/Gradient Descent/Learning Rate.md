# Learning Rate

Recall from [[Gradient Descent]] that we need a parameter to adjust how fast we are moving down the gradient.
This is called `Learning Rate`.

![Learning Rate](https://discuss.datasciencedojo.com/uploads/default/original/1X/808c4d2074b4ab07065cd8b316cd234679d5b31b.png)

$$
w_{t+1} = w_{t} - \lambda \nabla E(w_{t})
$$
where $\lambda$ is the `learning rate`

If learning rate is too small, it learns very slowly.
If learning rate is too big, it jump over minimum and may not converge.
