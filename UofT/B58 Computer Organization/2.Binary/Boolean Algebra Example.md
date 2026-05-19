# Boolean Algebra Example
Consider the following logic function
$$
f(x,y,z)
= \bar{x}yz + \bar{x}y\bar{z} + xz
$$

This gets us the following logic diagram.
![image|350](https://notes-media.kthiha.com/Boolean-Algebra-Example/6371226b3757d5d6e79d98cf3dd906a1.png)

However, we could use [[Boolean Algebra|boolean algebra]] to optimize it to
$$
\begin{align}
f(x,y,z)
&= \bar{x}yz + \bar{x}y\bar{z} + xz \\[6pt]
&= \bar{x}y(z + \bar{z}) + xz \\[6pt]
&= \bar{x}y + xz
\end{align}
$$

Hence, we get a much simpler logic diagram:
![image|350](https://notes-media.kthiha.com/Boolean-Algebra-Example/21d7db4e17d25ace1b9912fbe0fb1a1e.png)

---
## See Also
- [[Boolean Algebra]]
- [[Complementary Metal-Oxide-Semiconductor(CMOS)]]
- [[MOSFET]]
- [[Transistor]]