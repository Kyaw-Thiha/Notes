# Demultiplexer
A [[Demultiplexer]] is a combinational circuit that can switch one common input line to several different output lines.

![|300](https://i.ytimg.com/vi/vMvvggyriCc/maxresdefault.jpg)

---
### Logic Gate Representation
![|300](https://i.ytimg.com/vi/eeWHM3zzK3M/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLA-iBZzBqZh3UV9z8Nz1aBTn-_fgA)

---
## 7-Seg Decoder
A [[Demultiplexer|7-Seg decoder]] can represent different binary numbers by segments of lines.
![image|100](https://notes-media.kthiha.com/Demultiplexer/46d22ec1f33bd70fafad7f22b332090f.png)

This [[Demultiplexer|7-Seg decoder]] can be represented as
![image|200](https://notes-media.kthiha.com/Demultiplexer/8f1bc7bcb6ce01f84a109234a8c0fb37.png)

> Note that the implementation of [[Demultiplexer|decoder]] differs from one [[Demultiplexer|decoder]] to another.

---
### Segment-0
Let's analyze the truth table for $\text{segment-}0$.
![image|350](https://notes-media.kthiha.com/Demultiplexer/2b6ebed434c1f0f90ef6412bc0684f96.png)
This gets us the disjunctive normal form of
$$
\text{Hex}_{0}
= \bar{x}_{3} \cdot \bar{x}_{2} \cdot \bar{x}_{1} 
\cdot \dot{x}_{0}
+ \bar{x}_{3} \cdot x_{2} \cdot \bar{x}_{1} \cdot \bar{x}_{0}
$$

Note that $6$ of the rows are missing.
- These are input values that will never happen.
- They are represented as `x` in the [[Karnaugh Map(K-Map)|K-map]].
- These values can be assigned whatever values you want when constructing the final circuit.

---
### Segment-1
![image|350](https://notes-media.kthiha.com/Demultiplexer/9c2bfed97ed831cfb5e616fb67e5c22a.png)

### Segment-2
![image|350](https://notes-media.kthiha.com/Demultiplexer/069e62f3004eb88dd39e902048654694.png)


---
## See Also
- [A Tutorial](https://www.electronics-tutorials.ws/combination/comb_3.html)
- [GeeksForGeeks Tutorial](https://www.geeksforgeeks.org/electronics-engineering/what-is-demultiplexerdemux/)
- [[Multiplexer]]