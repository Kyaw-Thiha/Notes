# Newton Basis Proof
#numerical-methods/interpolation/newton/proof 

`Theorem`

Suppose
$$
\begin{align}
p(x)
= \ &y[x_{0}] + (x - x_{0}) \ y[x_{1}, x_{0}] 
+ (x-x_{0})(x - x_{1}) \ y[x_{2}, x_{1}, x_{0}]
+ \dots \\[6pt]
&+ (x-x_{0}) \dots (x-x_{n-2}) \ y[x_{n-1}, \dots, x_{0}]
+ (x-x_{0}) \dots (x- x_{n-1}) \ y[x_{n}, \dots, x_{0}]
\end{align}
$$

Then, $p(x) \in P_{n}$ and $p(x_{i}) = y_{i}, \ i=0,1, \dots n$

---
`Proof`
Proof by induction

`Base Case`
Case-1: $n=0$
$$
\begin{align}
&p_{0}(x) = y[x_{0}] \\[6pt]
\implies &p_{0} (x_{0}) = y[x_{0}] \\[6pt]
\implies &p_{0} (x_{0}) = y_{0} \\[6pt]
\implies &p_{0}(x) \text{ interpolates } (x_{0}, y_{0})
\end{align}
$$

Case-2: $n=1$
$$
\begin{align}
&p_{1}(x)  
= y[x_{0}] + (x-x_{0}) \ y[x_{1}, x_{0}] \\[6pt]

\implies &p_{1}(x_{0}) = y[x_{0}] = y_{0} \\[3pt]

&p_{1} (x_{1}) = y[x_{0}] + (x_{1} - x_{0})  
\frac{y_{1} - y_{0}}{x_{1} - x_{0}} = y_{1} \\[6pt]

\implies &p_{1}(x) \text{ interpolates }  
(x_{0}, y_{0}), \ (x_{1}, y_{1})
\end{align}
$$

---
`Induction Hypothesis`
1. $p_{n-1}(x) \in P_{n-1}$ is a `unique polynomial` of degree $\leq n-1$ $s.t.$ $p_{n-1}(x_{i}) = y_{i}$ for $i=0, 1, \dots, n-1$ (`first` $n$ polynomials)
$$
p_{n-1}(x) = y[x_{0}] + \dots + (x-x_{0}) \dots (x - x_{n-2}) \ y[x_{n-1}, \dots, x_{0}]
$$

2. $q(x) \in P_{n-1}$ is a `unique polynomial` of degree $\leq n-1$ $s.t.$  $q(x_{i}) = y_{i}$ for $i=1, 2, \dots, n$ (`last` $n$ polynomials)

`Note`: $q(x)$ is $p_{n-1}(x)$ with data shifted $x_{i} \to x_{i+1}$

`WTS`: $p_{n}(x)$ interpolates all $n+1$ data points.

---
`Induction Step`

Consider $p_{n}(x) = p_{n-1}(x) + (x-x_{0}) \dots (x - x_{n-1}) \cdot a_{n}$ 

Then, $p_{n}(x_{i}) = p_{n-1}(x_{i})$ for $i=0, \ 1, \dots, \ n-1$.

We need to choose $a_{n}$ $s.t.$ $p_{n}(x_{n}) = y_{n}$.

Consider 

$$
r(x) = \frac{(x-x_{0}) \ q(x) - (x-x_{n}) \ p_{n-1}(x)}{x-x_{0}}
$$
Then
$$
\begin{align}
&r(x_{0}) = p(x_{0}) = y_{0} 
& \text{by IH 1} \\[6pt]

&r(x_{n}) = q(x_{n}) = y_{n}
& \text{by IH 2} \\[6pt]

&r(x_{i}) = \frac{(x_{i} - x_{0}) \ y_{i}  \\
- (x_{i} - x_{n}) \ y_{i}}{x_{n} - x_{0}}
= y_{i}
\end{align}
$$

Since $r(x) \in P_{n}$, $r(x)$ is a `unique polynomial` of $degree$ $\leq n$ that satisfies $r(x_{i}) = y_{i}, \ i=0, 1, \dots, n$

`If` $p_{n}(x) = p_{n-1}(x) + (r-r_{0})\dots(r-r_{n-1}) \ a_{n}$ also satisfies $p_{n}(x_{i}) = y_{i}, \ i=0, 1, \dots, n$ `then`, $p_{n}(x) = r(x)$

Consider
- `Coefficient` of $x^n$ in $p_{n}(x)$ is $a_{n}$
- `Coefficient` of $x^n$ in $r(x)$ is 
$$
\frac{y[x_{n}, \ \dots, \ x_{1}] - y[x_{n-1}, \ \dots, \ x_{0}]}
{x_{n} - x_{0}} 
= y \ [x_{n}, \ \dots, \ x_{0}]
$$

Thus, 
- $a_{n} = y[x_{n}, \ \dots, \ x_{0}]$
- and $p(x) = p_{n}(x) = p_{n-1}(x) + (x-x_{0})\dots(x-x_{n+1}) \ y[x_{n}, \ \dots, \ x_{0}]$

---
## See Also
- [[Newton Basis]]
- [[Newton Example]]