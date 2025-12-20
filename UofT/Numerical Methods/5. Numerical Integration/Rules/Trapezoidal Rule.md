# Trapezoidal Rule
#numerical-methods/interpolatory-quadrature/trapezoid

`Trapezoidal rule` ($n=1$) is [[Interpolatory Quadrature]] that interpolates the function with a `line` between each interval.

![Trapezoidal Rule](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhCsB4sB_f4LfNEdG83SQggvGVPOi9WoQ8B3uhUoUhNdDc_vXQR5vkzg1u21N90t1R2qPXaGC0XSspbZXd7p7fNNIxDFOYuZFD14iqcbq9c4sX_oMifczZgjuy_ZAcLHzX4EIXwotfX_fU/s1600/trapezoid_2_40.gif)

---
`Formula`

![[Trapezoidal Rule Interpolatory.png|500]]

Geometrically, we get
$$
\begin{align}
I(F)  
&\approx T(F) = I(p_{1})\\[6pt]
&= (b-a) \ \frac{F(a) + F(b)}{2} \\[6pt]
&= \underbrace{\frac{b-a}{2}}_{A_{0}} \  
F \underbrace{(a)}_{x_{0}}  
+ \underbrace{\frac{b-a}{2}}_{A_{1}} \ 
F \underbrace{(b_{2})}_{x_{1}}  
\end{align}
$$
---
`Precision`
Computing the precision, we get
$$
\begin{align}
&T(x^0)  = (b-a) \frac{1+1}{2} = b-a = I(x^0) \\[6pt]

&T(x')  = (b-a) \frac{a+b}{2} = \frac{b^2 - a^2}{2} = I(x^1) \\[6pt]

&T(x'')  = (b-a) \frac{a^2 + b^2}{2}  
\neq \frac{b^3 - a^3}{3} = I(x^2) \\[6pt]
\end{align}
$$
Hence, precision $m=2$.

---
## See Also 
- [[Interpolatory Quadrature]]
- [[Midpoint Rule]]
- [[Simpson's Rule]]