# Midpoint Rule
#numerical-methods/interpolatory-quadrature/midpoint

`Midpoint rule` ($n=0$) is [[Interpolatory Quadrature]] that interpolates a function with the `midpoint` between each interval.

![Midpoint Rule](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiGsD5leUMxMRV8otzHPJ9wkoSrhowHIoDYdpX-3QMwEU-Yn1u4_k_DVYectFMgoeaKomU6fMlYKnNpZLI3pc-Jf1KLdrvzsBGvAAYHsvFzgfvjEGmECiZXfvoNPLlpO2Y50zCvCB5TsN4/s1600/midpoint_2_40.gif)

---
`Formula`

![[Midpoint Rule Interpolatory.png|500]]
Geometrically, we get
$$
\begin{align}
I(F) &\approx M(F) = I(p_{0}) \\[6pt]
&= \underbrace{(b-a)}_{Weight \ A_{i}}  
\ F \underbrace{\left( \frac{a+b}{2} \right)}_{Node \ x_{0}}
\end{align}
$$

---
`Precision`
Computing the precision, we get
$$
\begin{align} \\
&M(x^0) = M(1) = b-a = I(1) \\[10pt]

&M(x') = (b-a) \ \frac{a+b}{2}  
= \frac{b^2 - a^2}{2} = I(x) \\[10pt]

&M(x'') = (b-a) \left( \frac{a+b}{2} \right)^2
\neq \frac{b^3 - a^3}{3} = I(x^2)
\end{align}
$$

Hence, `precision` $m = 1$

---
`Why precision=1 for Midpoint Rule?`

![[Midpoint Rule Interpolatory Precision.png|500]]

---
## See Also
- [[Interpolatory Quadrature]]
- [[Trapezoidal Rule]]
- [[Simpson's Rule]]

