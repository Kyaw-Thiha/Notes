# Batch Size
We can use different `Batch Size` to carry out the [[Gradient Descent]] on.

![Batch Size](https://images.ctfassets.net/3vsdo1hyvom2/68onQ2rMp0xMJ21MUiqdxG/6125900ab6e517e1b02a76ef4cda3b9d/batch__stochastic__mini-batch_gradient_descent-Dec-22-2022-04-32-42-4986-AM.png?fm=webp&w=1140&q=99)

## Stochastic Gradient
This carries out the `gradient descent` by sampling a point from a dataset.
$$
\nabla E = \nabla L(w; x_{i}, y_{i})
$$

## Mini Batch Gradient
This computes `gradient descent` on a $K$ batches of data.
$$
\nabla E = \frac{1}{N} \sum^K_{i=1} \nabla L(w; x_{i}, y_{i})
$$

## Batch Gradient
This carries out `gradient descent` on all of the dataset.
$$
\nabla E = \frac{1}{N} \sum^N_{i=1} \nabla L(w; x_{i}, y_{i})
$$


