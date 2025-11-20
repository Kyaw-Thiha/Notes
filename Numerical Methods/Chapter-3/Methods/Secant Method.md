# Secant Method
#numerical-methods/non-linear/methods/secant 

`Formula`
`Secant Method` is essentially [[Newton's Method]], but the derivative is estimated with a secant instead.
$$
x_{k+1} = x_{k} - \frac{F(x_{k}) (x_{k} - x_{k-1})}{F(x_{k}) - F(x_{k-1})}
$$

![Secant Method](https://orionquest.github.io/Numacom/_images/secant.png)

---
`Derivation`
From [[Newton's Method]], we get that 
$$
x_{k+1} = x_{k} - \frac{F(x_{k})}{F'(x_{k})}
$$
We then proceed to approximate $F'(x_{k})$ as
$$
F'(x_{k}) \approx \frac{F(x_{k}) - F(x_{k-1})}{x_{k} - x_{k-1}}
$$

Hence, we get the formula of
$$
x_{k+1} = x_{k} - \frac{F(x_{k}) (x_{k} - x_{k-1})}{F(x_{k}) - F(x_{k-1})}
$$

---

`Not FPI`
The `secant method` is not a [[Fixed Point Iteration (FPI)]].
It cannot be expressed as $x_{k+1} = g(x_{k})$.
This is because there are $2 \ x_{k-1}$ terms.

---
`Analysis`
We cannot directly use [[Fixed Point Theorem (FPT)]] or [[Rate of Convergence Theorem]] to analyze this method.
However with some adjustments, we can prove that 
$$
p = \frac{1 + \sqrt{ 5 }}{2}
$$
This is called `Superlinear Convergence`.
Note that [[Newton's Method]] has a $p=2$ convergence rate.

---
`Reasoning behind faster convergence`
[[Newton's Method]] require $2$ function evaluation per iteration as $F$ and $F'$ aren't the same.
The `secant method` only requires $1$ function evaluation as we only need to compute $F(x_{k})$ since $F(x_{k-1})$ has already been computed in the previous step.

The effective rate of convergence takes into account cost per iteration as well as speed.
Hence, the `secant method` is effectively faster than [[Newton's Method]].

---
`Convergence Guarantee`
The secant method does NOT always converge.

It does not follow the [[Fixed Point Theorem (FPT)]] requirement of $|g'(x)| < 1$

---
## See Also
- [[Newton's Method]]
- [[Bisection Method]]
- [[Hybrid Method]]