# Bilinear Interpolation
#math
`Bilinear interpolation` is a way to estimate a value inside a rectangle when only the values at the four corners are known.

![Bilinear Interpolation|400](https://i.ytimg.com/vi/NnksKpJZEkA/maxresdefault.jpg)

---
## Maths behind Bilinear Interpolation
Let $x,x_{0},x_{1} \in \mathbb{R}$ be arbitrary such that $x_{0} \leq x \leq x_{1}$.
Suppose we know $f(x_{0})$ and $f(x_{1})$.

Then, we can estimate $f(x)$ using `weighted average`
$$
f(x)
= f(x_{0}) \frac{x_{1} - x}{x_{1} - x_{0}}
+ f(x_{1}) \frac{x - x_{0}}{x_{1} - x_{0}}
$$

`Generalizing to 2D`
Let 
- $f_{11} = f(x_{1}, y_{1})$
- $f_{21} = f(x_{2}, y_{1})$
- $f_{12} = f(x_{1}, y_{2})$
- $f_{22} = f(x_{2}, y_{2})$

Then, 
1. Interpolating along $x\text{-axis}$ bottom edge
$$
f(x, y_1) =
f_{11}\frac{x_2 - x}{x_2 - x_1} + f_{21}\frac{x - x_1}{x_2 - x_1}
$$
2. Interpolating along $x\text{-axis}$ top edge
$$
f(x, y_2) =
f_{12}\frac{x_2 - x}{x_2 - x_1} + f_{22}\frac{x - x_1}{x_2 - x_1}
$$
3. Interpolating along $y\text{-axis}$ between the two edges
$$
f(x, y) =
f(x, y_1)\frac{y_2 - y}{y_2 - y_1} + f(x, y_2)\frac{y - y_1}{y_2 - y_1}
$$

This can be combined into
$$
\begin{aligned}
f(x, y) =\;&
f_{11}\frac{x_2 - x}{x_2 - x_1}\frac{y_2 - y}{y_2 - y_1}
+ f_{21}\frac{x - x_1}{x_2 - x_1}\frac{y_2 - y}{y_2 - y_1} \\
&+ f_{12}\frac{x_2 - x}{x_2 - x_1}\frac{y - y_1}{y_2 - y_1}
+ f_{22}\frac{x - x_1}{x_2 - x_1}\frac{y - y_1}{y_2 - y_1}
\end{aligned}
$$

---
## Bilinear Interpolation Kernal

![Bilinear Interpolation Kernal|200](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQzSoz8lMhga_SGtaUw8y-tpaMz2GQF2YZsuQ&s)

`Linear Kernal`
In $1D$, the linear interpolation can be represented as `convolution` of
$$
\hat{f}(x)
= \sum_{n\in \mathbb{Z}}
f[n] \ k(x-n)
$$
where
- $\hat{f}(x)$ is the `interpolated continuous value`
- $f[n]$ is the `the discrete 1D samples`
- $k(t) = \max(0, \ 1-|t|)$ is the `triangle kernal`

---
`2D Bilinear Kernal`
The `bilinear kernal` is a product of two `linear kernals`:
$$
K(x,y) = k(x)\ k(y)
$$
Hence, resampling a `2D discrete image` $I[m,n]$ can be expressed as
$$
\hat{I}(x,y)
= \sum_{m} \sum_{n} I[m,n] \ k(x-m) \ k(y-n)
$$
where
- $(x,y)$ is the `continuous coordinate` in interpolated image $\hat{I}$
- $(m,n)$ is the `integer pixel indices`
- $k(\cdot)$ is the `1D linear kernal`

---