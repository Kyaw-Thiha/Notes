# Big-Theta 
**Informal Definition**: If $f(n) \in O(g(n))$ and $f(n) \in \Omega(g(n))$, 
then $f(n) \in \Theta(g(n))$.

**Formal Definition**:
Let $g \in \mathcal{F}$.
Then, $\Theta(g)$ is the set of functions $f \in \mathcal{F}$ $s.t.$ 
$$
\exists b \in \mathbb{R}^{+}, \ \exists c \in \mathbb{R}^{+}, \exists n_{0} \in \mathbb{N}, \ \forall n \geq n_{0}
\implies 
b.g(n) \leq f(n) \leq c.g(n)
$$


> **Theorem**: $f(n) \in \Theta(g(n))$ $\iff$ $f(n) \in O(g(n))$ and $g(n) \in O(f(n))$.

![Big Theta|400](https://i.ytimg.com/vi/-FORaw3VgQE/maxresdefault.jpg)

The above picture shows $f(n) = \Theta(g(n))$.
[[Big-Theta]] provides a tight bound.

---
### Proof Example-1
Show that $11n^{2} + 14n - 18 \in \Theta(n^{2})$.

**Big-O**
$$
\begin{align}
11n^{2} + 14n - 18 &\leq cn^{2} \\[6pt]
&\leq 11n^{2} + 14n \\[6pt]
&\leq 11n^{2} + 14n^{2} & \text{since }  
n\geq 1 \implies n^{2}\geq 1 \implies 14n^{2} \geq 1 \\[6pt]
&\leq 25n^{2} \\[6pt]
\end{align}
$$

Hence, we choose $c=25$ and $n_{0} = 1$.

**Big-$\Omega$**
$$
\begin{align}
11n^{2} + 14n - 18 &\geq b.n^{2} \\[6pt]
14n \geq 0 \implies 11n^{2} + 14n - 18 &\geq 11n^{2} - 18 \\[6pt]
\end{align}
$$

We then get
$$
\begin{align}
-18 &\geq -n^{2} \\[6pt]
n &\geq 5 \\[6pt]
\implies n_{0} &= 5
\end{align}
$$

Using that, we can bound
$$
\begin{align}
11n^{2} + 14n - 18 &\geq 11n^{2} - n^{2} \\[6pt]
&\geq 10n^{2} \\[6pt]
\implies b&=10
\end{align}
$$
Hence, 
$$
T(n) \in O(n^{2}) \ \cap \ T(n) \in \Omega(n^{2})
\implies T(n) \in \Theta(n^{2})
$$

---
### Proof Example-2
> Let $T(n) = n^{3} - n^{2} + 5$. 
> Show that $T(n) \in \Theta(n^{3})$.

**Big-O**
$$
\begin{align}
n^{3}-n^{2} + 5 &\leq cn^{3} \\[6pt]
n^{3}-n^{2}+5 &\leq n^{3} + 5 \\[6pt]
&\leq n^{3} + n^{3} & \text{since }  
n \geq 5 \implies n^{3} \geq 5 \\[6pt]
&\leq 2n^{3}
\end{align}
$$
Hence, $n_{0}=5$ and $c=2$.

**Big-$\Omega$**
$$
\begin{align}
n^{3}-n^{2}+5 &\geq bn^{3} \\[6pt]
n^{3} - n^{2} + 5 &\geq n^{3} - n^{2} \\[6pt]
&\geq n^{3}\left( 1 - \frac{1}{n} \right) \\[6pt]
&\geq \frac{1}{2} n^{3} &\text{choosing } n=2
\end{align}
$$
Note that we essentially factorized with largest exponent.
Then, we can choose a $n_{0}$ and $b$ pair that works.

Hence, $n_{0}  = 2$ and $b=\frac{1}{2}$.

---
> Show that $n \notin \Theta(n^{2})$.

$$
\begin{align}
\nexists b \in \mathbb{R}^{+}, \ \forall n \in 
\mathbb{N}, \ \nexists n_{0}, n \geq n_{0}
\implies n \geq bn^{2} \\[6pt]
\implies \notin \Theta(n^{2})
\end{align}
$$


---
## See Also
- [[Big-O]]
- [[Big-Omega]]
- [[Time Complexity]]