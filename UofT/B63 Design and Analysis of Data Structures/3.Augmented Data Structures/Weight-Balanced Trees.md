# Weight-Balanced Trees
A [[Weight-Balanced Trees|weight-balanced BST]] is another way to achieve $O(\log n)$ tree height.

At every node $V$ of a [[Weight-Balanced Trees|weight-balanced BST]], 
- $\text{weight}(V.\text{left}) \leq \text{weight}(V.\text{right}) \times 3$
- $\text{weight}(V.\text{right}) \leq \text{weight}(V.\text{left}) \times 3$
where $\text{weight}(V) = \text{size}(V) + 1$.

---
## Examples

![image|200](https://notes-media.kthiha.com/Weight-Balanced-Trees/aac08b65c07163b365a8ae052c605034.png)
![image|200](https://notes-media.kthiha.com/Weight-Balanced-Trees/ac0e6ca261965b756ba8137931fca8ec.png)

---
## Balancing and Rotation
**Single Rotation Counter Clockwise**
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Trees/c0ffef94eca46f0f826301fafd7cbbfd.png)
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Trees/b4d195ecb1b1f39f06bf8f502dacb899.png)

---
**Single Rotation Clockwise**
![image|350](https://notes-media.kthiha.com/Weight-Balanced-Trees/b854028f1ce49b3a8eaf6f6e02295d59.png)

---
**Double Rotation Clockwise -> Counter-Clockwise**
![image|350](https://notes-media.kthiha.com/Weight-Balanced-Trees/bb6cd778b00838b11d0d708ea0d1f07c.png)

---
**Double Rotation Counter-Clockwise -> Clockwise**
![image|350](https://notes-media.kthiha.com/Weight-Balanced-Trees/cc56235a73a0bd28040d165eea9db597.png)

---
## Summary of Rebalancing
![image|350](https://notes-media.kthiha.com/Weight-Balanced-Trees/8d8cb7d220ef28f270fe7e9cb6ff567c.png)

---
### Insertion
![image|300](https://notes-media.kthiha.com/Weight-Balanced-Trees/ab5309c402744a2226f94451c358ebc7.png)

### Deletion
![image|350](https://notes-media.kthiha.com/Weight-Balanced-Trees/2adce452d530f3cbb08fe9073a825e68.png)

---
## WBT Height
**Claim**:
$$h(T) \leq c \times \log(size(T) + 1)$$
where
- $c = \frac{1}{\log\left( \frac{4}{3} \right)}$ and
- $h(T)$ is the height of $T$

**Proof**:
We can do proof by induction.

**Base Case**: $T$ is empty.
$$
h(T) = h(\text{empty}) = 0
$$
Hence,
$$
\begin{align}
&c \times \log(size(T) + 1) \\[6pt]
&= c \times \log(size(0) + 1) \\[6pt]
&= c \times \log(1) \\[6pt]
&= 0
\end{align}
$$
Thus,
$$
h(T) \leq c \times \log(size(T) + 1)
\ , \quad \text{as wanted}
$$

**Induction Hypothesis**:
Suppose that for all $k \in \mathbb{N}$ where $0 \leq k < n$, the height of every [[Weight-Balanced Trees|weight-balanced BST]] of size $k$ is at most $c \times \log(k + 1)$.

**Induction Step**
Let $n_{l} = size(\text{T.left})$.
Let $n_{r} = size(\text{T.right})$.

Then,
$$
\begin{align}
n &= n_{l} + n_{r} + 1 \\[6pt]
n+1 &= n_{l} + n_{r} + 1 + 1 \\[6pt]
&\geq \frac{n_{r} + 1}{3} + n_{r} + 1 + 1 
&\text{since T is balanced} \\[6pt]
&= \frac{4n_{r} + 4}{3} \\[6pt]
&= \frac{4}{3}(n_{r} + 1) \\[6pt]
\end{align}
$$
Hence, we get
$$
n_{r+1} \leq \frac{3}{4} (n+1)
$$

Without loss of generality, assume $h(\text{T.left}) \leq h(\text{T.right})$.
Then,
$$
\begin{align}
h(T) &= 1 + h(\text{T.right}) \\[6pt]
&\leq 1 + c.\log(n_{r} + 1) & \text{by I.H} \\[6pt]
&\leq 1 + c.\log\left( \frac{3}{4}(n+1) \right)  
\\[6pt]
&= 1 + c.\log\left( \frac{3}{4} \right)  
+ c.\log(n+1) \\[6pt]
&= 1 + (-1) + c.\log(n+1) \\[6pt]
&= c.\log(n+1) \\[6pt]
&= c.\log(size(T) + 1) \ , \quad \text{as wanted}
\end{align}
$$

---
## See Also
- [[Augmented AVL Tree]]
- [[Interval Tree]]
- [[Weight-Balanced Trees]]
- [[Weight-Balanced Tree Rotations]]
- [[Augmented Data Structures]]
