## Using limits to Prove

**Assume** $\exists n_{0}: \ \forall n \geq n_{0} : \ f(n) \geq 0$ and $g(n) \geq 0$.
**Theorem**: If $\lim_{ n \to \infty } \frac{f(n)}{g(n)}$ exists and is finite, then $f(n) \in O(g(n))$.

Let $\lim_{ n \to \infty }\frac{f(n)}{g(n)} = L$.
Then for large $n$, $f(n) \approx L.g(n)$.
Choosing $c > L$(like $c=L+1$), we get $f(n) \leq c.g(n)$.

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
This happens with oscillating functions like piecewise function.

---
### Examples
**Example-1**:
Analyze $g(n) = n(1 + \sin n)$ to see if it is $\in O(n)$.
$$
\lim_{ n \to \infty } \frac{n(1 + \sin n)}{n}
= 1 + \sin n
$$
Hence, it is indefinite by [[Using limits for Big-O|limits]].

Instead note that
$$
\begin{align}
0 &\leq 1 + \sin n \leq 2 \\[6pt]
0 &\leq n(1 + \sin n) \leq 2n
\end{align}
$$
Hence, we get that $f(n) \in O(n)$

**Example-2**
Define 
$$
g(n) = \begin{cases}
1 &\text{if n is even}  \\
n &\text{if n is odd}
\end{cases}
$$
Then, $g(n) \in O(n)$ and $g(n) \notin O(1)$.
But using [[Using limits for Big-O|limits]], 
- $\lim_{ n \to \infty } \frac{g(n)}{n}$ does not exist and is not $\infty$.
- $\lim_{ n \to \infty } \frac{g(n)}{1}$ does not exist and is not $\infty$.

Instead, we can note that
- $1 \leq n$
- $n \leq c.n$

Hence, this allows as to conclude that $g(n) \in O(n)$ and $g(n) \notin O(1)$.

---
### Example with $\Theta(n)$
Prove $n^{2} + n^{\frac{3}{2}} \in \Theta(n^{2})$.

**Step-1**: Prove $n^{2} + n^{\frac{3}{2}} \in O(n^{2})$
$$
\begin{align}
\lim_{ n \to \infty } \frac{n^{2} + n^{3/2}}{n^{2}}
&= \lim_{ n \to \infty } \frac{n^{2}(1 + n^{-1/2})}{n^{2}} \\[6pt]
&= \lim_{ n \to \infty } (1 + n^{-1/2}) \\[6pt]
&= \lim_{ n \to \infty } 1 + \frac{1}{\sqrt{ n }} \\[6pt]
&= 1
\end{align}
$$
**Step-2**: Prove $n^{2} + n^{3/2} \in \Omega(n^{2})$.
First note that
$$
n^{2} + n^{3/2} \in \Omega(n^{2})
\implies n^{2} \in O(n^{2} + n^{3/2})
$$
Using that,
$$
\lim_{ n \to \infty } \frac{n^{2}}{n^{2} (1 + n^{-1/2})}
= 1
$$
**Step-3**:
Since $n^{2} + n^{\frac{3}{2}} \in O(n^{2})$ and $n^{2} + n^{3/2} \in \Omega(n^{2})$, we get $n^{2} + n^{3/2} \in \Theta(n^{2})$.

---