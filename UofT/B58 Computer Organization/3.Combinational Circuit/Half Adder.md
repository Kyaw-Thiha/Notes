# Half Adder
A [[Half Adder|half adder]] adds two bits to produce a two-bit sum.
It performs the following computations:
![image|300](https://notes-media.kthiha.com/Half-Adder/dc4dc682a6b4338a1e8c379acd8d8378.png)

The sum is expressed as a sum bit $S$ and a carry bit $C$.
![image|150](https://notes-media.kthiha.com/Half-Adder/689b2be22e3f71f0c19aa26a65e7c63d.png)

---
## Implementation
We could analyze the truth table in order the get the following:
$$
C=X\cdot Y
\
\quad
\
\begin{align}
S &= X \cdot \bar{Y} + \bar{X}\cdot Y \\[6pt]
&= X \text{ XOR } Y
\end{align}
$$
![image|300](https://notes-media.kthiha.com/Half-Adder/1ef002fd89fbf0f556d21c1d8f394a71.png)

---
