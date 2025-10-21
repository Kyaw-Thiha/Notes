# Gradient Descent
Many optimization problems do not have closed-form solutions, so we need to use `Gradient Descent` to optimize.

![Gradient Descent](https://mriquestions.com/uploads/3/4/5/7/34572113/steepest-descent-loss-funciton_orig.png)

To update our weights, we use
$$
w_{t+1} = w_{t} - \eta. \nabla E(w_{t})
$$
where
- $w_{t+1}$ is the new weights
- $w_{t}$ is the current weights
- $\eta$ is the `learning rate`
- $\nabla E(w_{t})$ is the `gradient`
Let $E(w_{i})$ be the `objective function`.


## Justification
To justify the `gradient descent formula`, we can use `First Order Taylor Series approximation`.

Think of a curvy surface $E(w_{t})$.
If we zoom in to around point $w_{t}$, the curvy surface starts to look flat.
We can denote it mathematically using `Taylor expansion`.

$$
E(w_{t+1}) = E(w_{t}) + E'(w_{t}).\triangle w
$$
where
- $E(w_{t})$ is the value at current point
- $E'(w_{t})$ is the slope at current point
- $E'(w_{t}).\triangle w$ is how much function changes when moved a bit

Generalizing this to N-D case,
$$
E(w_{t+1}) \approx E(w_{t}) +  (w_{t+1} - w_{t})^T \  \nabla E(w_{t})
$$
where 
- $E(w_{t+1})$: next loss value
- $E(w_{t})$: current loss value
- $\nabla E(w_{t})$: the gradient with respect to weights $w_{t}$
- $w_{t+1} - w_{t}$: small step (direction & distance) to move towards

