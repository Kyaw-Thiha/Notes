# Backpropagation Example
#ml/models/neural-network/backpropagation 

`Question`
Consider a $3$-layer [[Neural Network]] with $2$ input features.
Hidden layer-$1$ has $3$ nodes and hidden layer-$2$ has $2$ nodes.
The output layer has $1$ node.
We use the [[ReLU|ReLU activation function]]  for the first $2$ hidden layers and the identity function for the output layer.
Using the [[Loss Function|Squared Loss]] as a loss function, derive $\delta^{(k)}$, where $k=3,2,1$

`Diagram`
![[Backpropagation-2.png]]

---
`Solution`

`Layer-1`
$$
\delta^{(3)} 
= \delta^{(3)}_{1}
= \frac{dL}{dz_{1}^{(3)}}
$$

$$
\begin{align}
\frac{dL}{dz_{1}^{(3)}}
&= \frac{dL}{d\hat{y}} .  
\frac{d\hat{y}}{da_{1}^{(3)}} .
\frac{da_{1}^{(3)}}{dz_{1}^{(3)}} \\[6pt]

&= -(y - \hat{y}) \cdot 1 \cdot 1 \\[6pt]
&= \hat{y} - y \\[6pt]
&= \delta_{1}^{(3)}
\end{align}
$$

`Layer-2`
$$
\delta^{(2)}
= \begin{bmatrix}
\delta_{1}^{(2)} \\[6pt]
\delta_{2}^{(2)} \\[6pt]
\end{bmatrix}
= \begin{bmatrix}
\frac{dL}{dz_{1}^{(2)}} \\[6pt]
\frac{dL}{dz_{2}^{(2)}} \\[6pt]
\end{bmatrix}
$$

$$
\begin{align}
\frac{dL}{dz_{1}^{(2)}}

&= \frac{dL}{dz_{1}^{(3)}} \cdot
\frac{dz_{1}^{(3)}}{da_{1}^{(2)}} \cdot
\frac{da_{1}^{(2)}}{dz_{1}^{(2)}} \\[6pt]

&= \delta_{1}^{(3)} \cdot  
w_{11}^{(3)} \cdot \mathrm{ReLU}(z_{1}^{(2)}) \\[6pt]

&= \delta_{1}^{(2)}
\end{align}
$$


$$
\begin{align}
\frac{dL}{dz_{2}^{(2)}}

&= \frac{dL}{dz_{2}^{(3)}} \cdot
\frac{dz_{2}^{(3)}}{da_{2}^{(2)}} \cdot
\frac{da_{2}^{(2)}}{dz_{2}^{(2)}} \\[6pt]

&= \delta_{2}^{(3)} \cdot  
w_{12}^{(3)} \cdot \mathrm{ReLU}(z_{2}^{(2)}) \\[6pt]

&= \delta_{2}^{(2)}
\end{align}
$$


`Layer-3`
$$
\delta^{(2)}
= \begin{bmatrix}
\delta_{1}^{(1)} \\[6pt]
\delta_{2}^{(1)} \\[6pt]
\delta_{3}^{(1)} \\[6pt]
\end{bmatrix}
= \begin{bmatrix}
\frac{dL}{dz_{1}^{(1)}} \\[6pt]
\frac{dL}{dz_{2}^{(1)}} \\[6pt]
\frac{dL}{dz_{3}^{(1)}} \\[6pt]
\end{bmatrix}
$$

$$
\begin{align}
\frac{dL}{dz_{1}^{(1)}}
&= \frac{d}{dz_{1}^{(1)}} L \left( z_{1}^{(2)} \ z_{1}^{(1)}, \ z_{2}^{(2)} \ z_{1}^{(1)}  \right)
\\[6pt]

&= \frac{dL}{dz_{1}^{(2)}}  
\cdot \frac{dz_{1}^{(2)}}{dz_{1}^{(1)}}
+ 
\frac{dL}{dz_{2}^{(2)}}  
\cdot \frac{dz_{2}^{(2)}}{dz_{1}^{(1)}} \\[6pt]

&= \delta_{1}^{(2)}  
\cdot \frac{dz_{1}^{(2)}}{da_{1}^{(1)}}
\cdot \frac{da_{1}^{(a)}}{dz_{1}^{(1)}}
+
\delta_{2}^{(2)}  
\cdot \frac{dz_{2}^{(2)}}{da_{1}^{(1)}}
\cdot \frac{da_{1}^{(a)}}{dz_{2}^{(1)}} \\[6pt]

&= \delta_{1}^{(2)} 
\cdot w_{11}^{(2)}  
\cdot \mathrm{ReLU}'(z_{1}^{(1)})  
+
\delta_{2}^{(2)} 
\cdot w_{21}^{(2)}  
\cdot \mathrm{ReLU}'(z_{1}^{(1)})  
\\[6pt]


&= \mathrm{ReLU}'(z_{1}^{(1)}) 
\left( 
\delta_{1}^{(2)} 
\cdot w_{11}^{(2)}  
+
\delta_{2}^{(2)} 
\cdot w_{21}^{(2)}  
\right)  
\\[6pt]

&= \delta_{1}^{(1)}
\end{align}
$$

Continuing on,
$$
\delta_{2}^{(1)}
= \mathrm{ReLU}'(z_{2}^{(1)}) 
\left( 
\delta_{1}^{(2)} 
\cdot w_{12}^{(2)}  
+
\delta_{2}^{(2)} 
\cdot w_{22}^{(2)}  
\right)  
$$

$$
\delta_{3}^{(1)}
= \mathrm{ReLU}'(z_{3}^{(1)}) 
\left( 
\delta_{1}^{(2)} 
\cdot w_{13}^{(2)}  
+
\delta_{2}^{(2)} 
\cdot w_{23}^{(2)}  
\right)  
$$
