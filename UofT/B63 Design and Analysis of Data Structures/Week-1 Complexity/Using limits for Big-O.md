## Using limits to Prove

**Assume** $\exists n_{0}: \ \forall n \geq n_{0} : \ f(n) \geq 0$ and $g(n) \geq 0$.
**Theorem**: If $\lim_{ n \to \infty } \frac{f(n)}{g(n)}$ exists and is finite, then $f(n) \in O(g(n))$.

---
### Examples
**Example-1**:
Prove $\frac{n(n+1)}{2} \in O(n^{2})$.
$$
\begin{align}
\lim_{ n \to \infty } \frac{n(n+1)}{2n^{2}}
&= \lim_{ n \to \infty } \frac{n^{2} + n}{2n^{2}}  
\\[6pt]
&= \frac{1}{2} \lim_{ n \to \infty }  
\frac{n^{2} + n}{n^{2}} \\[6pt]
&= \frac{1}{2} \left( \lim_{ n \to \infty }  
\frac{n^{2}}{n^{2}} + \lim_{ n \to \infty }  
\frac{n}{n^{2}} \right) \\[6pt]
&= \frac{1}{2}(1 + 0) \\[6pt]
&= \frac{1}{2}
\end{align}
$$

**Example-2**: Prove $\ln(n) \in O(n)$.
$$
\begin{align}
\lim_{ n \to \infty } \frac{\ln(n)}{n}
&= \lim_{ n \to \infty } \frac{1}{n}   
& \text{by l'hopital's rule} \\[6pt]
&= 0
\end{align}
$$

---
## Using limits to Disprove

**Theorem**: If $\lim_{ n \to \infty } \frac{f(n)}{g(n)} = \infty$, then $f(n) \notin O(g(n))$.

**Example-1**: Disprove $n^{2} \in O(n)$.
$$
\begin{align}
\lim_{ n \to \infty } \frac{n^{2}}{n}
&= \lim_{ n \to \infty } n \\[6pt]
&= \infty
\end{align}
$$

**Example-2**: Disprove $n \in O(\ln(n))$.
$$
\begin{align}
\lim_{ n \to \infty } \frac{n}{\ln(n)} 
&= \lim_{ n \to \infty } \frac{1}{1/n} \\[6pt]
&= \lim_{ n \to \infty } n \\[6pt]
&= \infty
\end{align}
$$

---
## DNE
If $\lim_{ n \to \infty } \frac{f(n)}{g(n)}$ DNE and is not $\infty$, then there is no conclusion.
This happens with piecewise function.

---