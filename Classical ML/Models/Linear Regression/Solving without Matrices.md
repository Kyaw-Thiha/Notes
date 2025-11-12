# Solving without Matrices 
#ml/models/classic/linear-regression/solving/no-matrices #math
This is the general [[Loss Function#Cost Function|objective function]].
$$
E(W, b) = \sum^N_{i=1} (\hat{y}_{i} - y_{i})^2
$$

This is the `objective function` for [[Linear Regression]].
$$
\frac{\partial E}{\partial b}  
= \frac{\partial}{\partial b} \sum^N_{i=1} (wx_{i} + b - y_{i})^2 \\ 
$$

To optimize it and find local minima, we need to find its critical point. 

$$
\begin{align}
\frac{\partial E}{\partial b}  &= 0 \\
2. \sum^N_{i=1} (wx_{i} + b - y_{i}) &= 0 \\
\sum^N_{i=1} (wx_{i} - y_{i}) + \sum^N_{i=1}b &= 0 \\
w.\sum^N_{i=1} x_{i} - \sum^N_{i=1} y_{i} + Nb &= 0 \\
b &= \frac{1}{N}.\sum^N_{i=1} y_{i} - w.\frac{1}{N}\sum^N_{i=1} x_{i}   \\
b &= \vec{y} - w.\vec{x}   \\
\end{align}
$$

Now, we substitute that $b$ into the `linear regression` equation.
Note that $\hat{y}_{i}$ is the predicted value.
$$
\begin{align}
\hat{y}_{i} &= w.x_{i} + b^* \\
&= w.x_{i} + \vec{y} - w.\vec{x} \\
&= w.(x_{i} - \vec{x}) + \vec{y}
\end{align}
$$

Since we might end up in `local minima`, we use iterated methods to solve it.
$$
\begin{align}
E(w, b) 
&= \sum^N_{i=1} (\hat{y}_{i} - y_{i})^2 \\
&= \sum^N_{i=1} (w.(x_{i} - \vec{x}) + \vec{y} - y_{i})^2  \\
\end{align}
$$

Now, let's minimize that energy function with respect to $w$.
$$
\begin{align}
\frac{\partial E}{\partial w} &= 0 \\
2.\sum^N_{i=1} [(w(x_{i} - \vec{x}) + (\vec{y} - y_{i}))  . (x_{i} - \vec{x})]
&= 0  \\
\sum^N_{i=1} (w(x_{i} - \vec{x})^2 + (\vec{y} - y_{i})(x_{i} - \vec{x}))
&= 0  \\
w.\sum^N_{i=1} (x_{i} - \vec{x})^2 + \sum^N_{i=1}(\vec{y} - y_{i})(x_{i} - \vec{x}) 
&= 0  \\
w.\sum^N_{i=1} (x_{i} - \vec{x})^2 &= \sum^N_{i=1}(y_{i} - \vec{y})(x_{i} - \vec{x}) \\
 w &=  \frac{\sum^N_{i=1}(y_{i} - \vec{y}) (x_{i} - \vec{x})}{\sum^N_{i=1}(x_{i} - \vec{x}) ^ 2} \\
\end{align} 
$$

