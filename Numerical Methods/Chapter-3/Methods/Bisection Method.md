# Bisection Method
#numerical-methods/non-linear/methods/bisection 

`Bisection Method` uses the midpoint to update the values of bounds $[a,b]$ and converge towards root $y=0$.

![Bisection Method](https://media.geeksforgeeks.org/wp-content/uploads/bisection.jpg)

---
`Methodology`
Find an $a < b$ $s.t.$ $F(a) \leq 0 \leq F(b)$.
This means that there is at least $1$ root in $[a, b]$.


Assume $F(a) \leq 0 \leq F(b)$.
```python
Loop until (b-a) is small enough
	Let m = (a+b)/2
	If F(m) <= 0, 
		Let a = m
	Else, 
		Let b = m
Repeat for the interval [m,b] or [a, m]
```

---
`Convergence Guarantee`

For [[Newton's Method]] and [[Secant Method]], if we start off with wrong initial guess, we won't get convergence.

The `Bisection Method` guarantees convergence, but is slower.

With $p=2$ and $c=\frac{1}{2}$, we have a `linear rate of convergence`.

---
## See Also
- [[Newton's Method]]
- [[Secant Method]]
- [[Hybrid Method]]
