# Simpson's Rule
#numerical-methods/interpolatory-quadrature/simpson

`Simpson's rule` ($n=2$) is [[Interpolatory Quadrature]] that interpolates the function with a `quadratic polynomial` between each interval.

![Simpson's Rule](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEggfVnAuG02fIiO3yLw3U9PqQ0tMd2kMqJO7UCaGvv26HRS4HTCzHPo7gLznrrYFB6EHhwaAZs26-AbDrhbOqe0sOYYkY9kbQRtJh3UbfUA_GOJvpEeDUeK1u1KVLL9oNhlZh_2s6wD1Iw/s1600/Simpson_2_40.gif)

---
`Formula`

![[Simpson Rule Interpolatory.png|500]]

Solving the [[Vandermonde Theorem|Transposed Vandermonde]],
$$
\begin{align}
I(F)  
&\approx S(F) = I(p_{2}) \\[6pt]
&= \frac{b-a}{6}  
\left( F(a) + 4F\left( \frac{a+b}{2} \right)  
+ F(b) \right)
\end{align}
$$

In standard form,
$$
\begin{align}
&A_{0} = \frac{b-a}{6}  
& &A_{1} = \frac{2}{3} (b-a)
& &A_{2} = \frac{b-a}{4} \\[6pt]

&x_{0} = a 
& &x_{1} = \frac{a+b}{2}
& &x_{2} = b
\end{align}
$$
---

`Precision`

Computing the precision, we get
$$
\begin{align}
&M(1)
= \frac{b-a}{6}\bigl(1 + 4 + 1\bigr)
= b-a
= I(1) \\[10pt]

&M(x)
= \frac{b-a}{6}\left(
a + 4\cdot \frac{a+b}{2} + b
\right)
= \frac{b-a}{6}\bigl(3a+3b\bigr)
= \frac{b^2-a^2}{2}
= I(x) \\[10pt]

&M(x^2)
= \frac{b-a}{6}\left(
a^2 + 4\left(\frac{a+b}{2}\right)^2 + b^2
\right)
= \frac{b^3-a^3}{3}
= I(x^2) \\[10pt]

&M(x^3)
= \frac{b-a}{6}\left(
a^3 + 4\left(\frac{a+b}{2}\right)^3 + b^3
\right)
= \frac{b^4-a^4}{4}
= I(x^3) \\[10pt]

&M(x^4)
\neq \frac{b^5-a^5}{5}
= I(x^4)
\end{align}
$$

Hence, `precision` $m = 3$.

---
## See Also
- [[Interpolatory Quadrature]]
- [[Midpoint Rule]]
- [[Trapezoidal Rule]]