# Universal Turing Machine
[[Universal Turing Machine(UTM)|Universal Turing Machine]] is a theoretical construction that can simulate the behaviour of other [[Turing Machine|machine]]. 

![UTM|500](https://media.geeksforgeeks.org/wp-content/uploads/20230412160052/img2drawio-(5).png)

---
## Construction of UTM
Without loss of generality, assume the following for  [[Universal Turing Machine(UTM)|machine]] $M$:
- $Q=\{ q_{1}, q_{2}, \dots, q_{n} \}$ is the set of states where $q_{1}$ is the initial state and $q_{n}$ is the final state
- $\tau = \{ \sigma_{1}, \sigma_{2}, \dots, \sigma_{n} \}$ is the set of blank symbols
- Let $q_{1}$ be representable by $1$, $q_{2}$ by $11$, etc.
- Similarly $\alpha_{1}$ is encoded as $1$, $\alpha_{2}$ is encoded as $11$, etc.
- Represent R/W head directions by $1$ for left and $0$ for right
- The symbol $0$ will be used as separator between $1$s

With this scheme, any transition of $M$ can be given as
![Construction of UTM|300](https://media.geeksforgeeks.org/wp-content/uploads/20230413114520/img2drawio-(3)drawio.png)

---
## Implementation
A [[Universal Turing Machine(UTM)|Universal Turing Machine]] $M_{u}$ has an $\text{alphabet}=\{ 0,1 \}$ and a structure of a multi-tape [[Turing Machine]].
![UTM Implementation|300](https://media.geeksforgeeks.org/wp-content/uploads/20230413111235/Untitled-Diagramdrawio-(1).png)

- $M_{u}$ first looks at the contents of Tape-2 and Tape-3 to determine the ID of $M$.
- It then consults Tape-1 to see what $M$ would do with this ID.
- Finally, Tape-2 and Tape-3 will be modified to reflect the result of the move.

---
## See Also
- [Good Explanation from GeeksForGeeks](https://www.geeksforgeeks.org/compiler-design/universal-turing-machine/)