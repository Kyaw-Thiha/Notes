# Convolution Operator
The [[Convolution Operator|1D convolution operator]] is defined as 
$$
(f \ast g)[n] = 
\sum ^{\infty}_{k=-\infty} f[k] \ g[n-k]
$$
where $f$ and $g$ are functions.

![](https://upload.wikimedia.org/wikipedia/commons/b/b9/Convolution_of_spiky_function_with_box2.gif?utm_source=en.wikipedia.org&utm_campaign=parser&utm_content=thumbnail_unscaled)

For continuous version, we have
$$
(f \ast g)[n] = 
\int ^{\infty}_{-\infty} f(k) \ g(n-k) \ dk
$$

---
## 2D Convolution Operator
It can be mathematically defined as 
$$
\begin{align}
y[m,n]
&= \mathbf{I}[m,n] \ast \mathbf{K}[m,n] \\[6pt]
&= \sum^{\infty}_{j=-\infty} \sum^{\infty} 
_{i=-\infty} \mathbf{I}[i,j] \cdot  
\mathbf{K}[m-i, \ n-j]
\end{align}
$$
where $\mathbf{I}$ is the image and $\mathbf{K}$ is the filter kernel.

![300](https://www.aspires.cc/content/images/2024/05/conv-calc.gif)

---
## Applying Convolution
In order to apply [[Convolution Operator|convolution]] of image $\mathbf{I}$ with filter kernal $\mathbf{K}$, 
- Multiply each pixel in $\mathbf{I}$ in range of kernel by corresponding element of kernel $\mathbf{K}$
- Sum all these products and write to a new 2D array
- Slide kernel across all areas of the image 

![300](https://miro.medium.com/v2/resize:fit:1400/1*itcofCIVsGe7rBmciJcmVw.gif)

---
## See Also 
- [Convolutional Neural Network Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- [[Convolutional Neural Network (CNN)]]
- [[Convolution Layer]]
- [[Convolution Operator]]
- [[Pooling Layer]]