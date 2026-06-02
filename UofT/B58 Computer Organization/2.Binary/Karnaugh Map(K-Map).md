# Karnaugh Map
[[Karnaugh Map(K-Map)]] is used to simplify [[Boolean algebra|Boolean algebra expressions]] and minimize logic circuits without complex algebraic calculations.

![|300](https://i.ytimg.com/vi/RO5alU6PpSU/maxresdefault.jpg)

---
## K-Map Rules

![|300](https://www.allaboutcircuits.com/uploads/articles/KMap_Figure3.jpg)

Fill the grid with truth table values
1. Group adjacent $1$s in powers of $2$: $1, \ 2, \ 4, \ 8$
2. Groups must be rectangular.
3. Make groups as large as possible.
4. Every $1$s must be in at least $1$ group.
5. Groups can overlap.
6. The edges wrap around(left touches right, top touches bottom).

Each group gives one simplified term.
The bigger the group, the simpler the term

---
## Deriving back
Based on the groups, 
- ignore the term(s) which is changing
- for fixed term, copy their values

For example, consider the following [[Karnaugh Map(K-Map)|K-map]].
![image|250](https://notes-media.kthiha.com/Karnaugh-Map(K-Map)/43419c42022b40628a5acf702f79ba90.png)

- For orange group, we get $\lnot A \land \lnot B \land \lnot C$.
- For green group, we get $A \land \lnot B \land \lnot D$.
- For pink group, we get $B \land \lnot C \land D$.
- For brown group, we get $\lnot A \land B \land C$.

Hence, we now get 
$$
\begin{aligned}
f(A,B,C,D) = &(\lnot A \land \lnot B \land \lnot C) 
    \lor (A \land \lnot B \land \lnot D) \\[6pt]
    &\lor (B \land \lnot C \land D) \lor (\lnot A \land B \land C)
\end{aligned}
$$

---
## Tips
- Make largest groups first.
- Don't forget edges warping around.
- Ensure your group sizes are of $2^{n}$.

---
## Read More
- [Youtube Video](https://youtu.be/RO5alU6PpSU?si=pFCVsl11YlvHwnGh)

