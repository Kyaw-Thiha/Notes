# Bisection Method
#numerical-methods/non-linear/methods/bisection 

`Method`
Find an $a < b$ $s.t.$ $F(a) \leq 0 \leq F(b)$.
This means that there is at least $1$ root in $[a, b]$.

Assume $F(a) \leq 0 \leq F(b)$.
Loop until $b-a$ small enough.
	Let $m = \frac{a+b}{2}$+b.
	If $F(m) \leq 0$, let $a=m$.
	Else, let $b=m$.
Repeat for the interval $[m,\ b]$ or $[a,\ m]$
