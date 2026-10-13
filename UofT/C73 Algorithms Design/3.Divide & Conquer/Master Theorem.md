# Master Theorem
The [[#Master Theorem|master theorem]] states that for $T(n) = a \cdot T\left( \frac{n}{b} \right) + c \cdot n^{d}$, we have
$$
T(n) = \begin{cases}
\Theta(n^{d}) & \text{if } a < b^{d} \\[6pt]
\Theta(n^{d} \ \log n) & \text{if } a=b^{d} \\[6pt]
\Theta(n^{\log_{b} (a)}) & \text{if } a > b^{d}
\end{cases}
$$
with
- $(a)$ If $a < b^{d}$, then $T(n) = \Theta(n^{d})$
- $(b)$ If $a=b^{d}$, then $T(n) = \Theta(n^{d} \log n)$
- $(c)$ If $a > b^{d}$, then $T(n) = \Theta(n^{\log_{b}(a)})$

> In the **first case**, the running time is dominated by time splitting the input and recombining the results of the recursive calls.
> 
> In the **second case**, the running time is dominated by the number of recursive calls made.

---
## Proof
We shall reason about the amount of time required by the [[Divide & Conquer|divide-and-conquer algorithm]].

> **Level-0 (Root)**
> At level-$0$(root), we have one call for input of size $n$.
> The time needed by this call, excluding the time needed by recursive calls it makes, is $cn^{d}$. 

> **Level-1**
> At level-$1$, we have $(a)$ calls made by the call at level-0.
> The time needed by these calls, excluding time needed by the calls it makes, is $c\left( \frac{n}{b} \right)^{d}$.
> So, the total time needed by these $a$ calls is $a \ c\left( \frac{n}{b} \right)^{d}$.

> **Level-2**
> At level-$2$, we have $(a^{2})$ calls made by the calls at level-1.
> Each of them work on input of size ($\left( \frac{n}{b^{2}} \right)$.
> So, the total time needed by these $a^{2}$ calls is $a^{2} \ c\left( \frac{n}{b^{2}} \right)^{d}$.

> **General Level**
> At level $i$, we have $a^{i}$ calls. 
> Each of these work on an input of size $\left( \frac{n}{b^{i}} \right)$.
> The total time needed for these calls is $a^{i} \ c\left( \frac{n}{b^{i}} \right)^{d}$.

Thus, the total time required for all the calls at all levels is 
$$
T(n) 
\quad = \quad  \sum^{\log_{b}(n)}_{i=0} a^{i} \ c\left( \frac{n}{b^{i}} \right)^{d}
\quad = \quad  cn^{d} \sum^{\log_{b}(n)}_{i=0} \left( \frac{a}{b^{d}} \right)^{i}
$$

Since $\sum^{\log_{b}(n)}_{i=0} \left( \frac{a}{b^{d}} \right)^{i}$ is just a geometric series of for $\sum^{k}_{i=0} x^{i}$, 
- if $0 \leq x < 1$, then $\sum^{\infty}_{i=0} x^{i} = \frac{1}{1-x}$, and
- for all $x\neq 1$, $\sum ^{k}_{i=0} x^{i} = \frac{x^{k+1} - 1}{x-1}$

---
### Case-1
For [[#Case-1|case-1]], we have $a < b^{d}$.
Using the formula for infinite geometric series, we have
$$
\sum_{i=0}^{\log_b n} \left(\frac{a}{b^d}\right)^i < \sum_{i=0}^{\infty} \left(\frac{a}{b^d}\right)^i = \frac{1}{1 - a/b^d} = \Theta(1).
$$
> So since $T(n) = cn^{d} \ \Theta(1)$, we get $T(n)=\Theta(n^{d})$ for this case.

---
### Case-2
For [[#Case-2|case-2]], we have $a=b^{d}$. Then, we have
$$
\sum_{i=0}^{\log_b n} \left(\frac{a}{b^d}\right)^i 
= \sum_{i=0}^{\log_b n} 1^i 
= 1 + \log_b n 
= \Theta(\log n)
$$
> So since $T(n) = cn^{d} \ \Theta(\log n)$, we get $T(n)=\Theta(n^{d} \log n)$ for this case.

---
### Case-3
For [[#Case-3|case-3]], we have $a > b^{d}$. 
Using the formula for finite geometric series, we have
$$
\sum_{i=0}^{\log_b n} \left(\frac{a}{b^d}\right)^i 
= \frac{\left(\frac{a}{b^d}\right)^{1+\log_b n} - 1} {\underbrace{\frac{a}{b^d} - 1}_{\Theta(1)}} 
= \Theta\!\left(\frac{a^{\log_b n}}{\left(b^{\log_b n}\right)^d}\right) 
= \Theta\!\left(\frac{n^{\log_b a}}{n^d}\right).
$$
using $a^{\log_{b} n} = n^{\log_{b} a}$ and $b^{\log_{b} n} = n$ by definition of logarithms.

So since $T(n) = cn^{d} \ \Theta\left( \frac{n^{\log_{b} a}}{n^{d}} \right)$, we get $T(n)=\Theta(n^{\log_{b} a})$ for this case.

---