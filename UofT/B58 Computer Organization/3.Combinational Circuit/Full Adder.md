# Full Adder
[[Full Adder]] is a [[Half Adder|half adder]], but with a carry-in bit $Z$.
![image|150](https://notes-media.kthiha.com/Full-Adder/c5c67ba24c60e8f392487d3c89bc73f2.png)
Note that $C$ and $Z$ are sometimes represented as $C_{\text{out}}$ and $C_{\text{in}}$.

When $Z=0$, it is exactly like a [[Half Adder|half adder]].
![image|200](https://notes-media.kthiha.com/Half-Adder/dc4dc682a6b4338a1e8c379acd8d8378.png)

When $Z=1$, 
![image|200](https://notes-media.kthiha.com/Full-Adder/2507675ac70df2f5d2e765c770bd872a.png)

---
## Design
Deriving from a truth table and [[Karnaugh Map(K-Map)|K-Map]],
![image|350](https://notes-media.kthiha.com/Full-Adder/df2ec100bf9615c2bd891c51bae4169c.png)

From the [[Karnaugh Map(K-Map)|K-Map]], we then get
$$
S = X \text{ xor } Y \text{ xor } Z
$$
and
$$
\begin{align}
C &= X\cdot Y + X \cdot Z + Y \cdot Z \\[6pt]
&= X \cdot Y + (X \text{ xor } Y) \cdot Z
\end{align}
$$
Note that $(X \text{ xor } Y)$ is in both $C$ and $S$, so it can be reused.

---
## Circuit Diagram
Given $S = X \text{ xor } Y \text{ xor } Z$ and $C= X \cdot Y + (X \text{ xor } Y) \cdot Z$, 
![image|150](https://notes-media.kthiha.com/Full-Adder/846ca4cd2f7ee3ef635db18115dcda56.png)

where 
- $X \cdot Y$: carry generate $(G)$
  whether $X$ and $Y$ generate a carry bit.
- $X \text{ xor } Y$: carry propagate $(P)$
  whether carry will be propagated to $C_{out}$

---
