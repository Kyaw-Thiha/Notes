# Strongly Connected Components
`SCC` is the maximum subset of vertices from each other in a directed [[graph]].

---
## Example
Consider the graph below.
![image|300](https://notes-media.kthiha.com/Strongly-Connected-Components-(SCC)/4df46df58b0daa6703a24d1e9876fc85.png)

The [[Strongly Connected Components (SCC)|SCCs]] are
- $\{ e,o \}$
- $\{ m \}$
- $\{ h, f, k, g \}$

---
## Transpose of G
The transpose of $G$, denoted by $G^{T}$ is a [[graph]] with the same vertices as $G$ but the edges are reversed.
E.g:
![image|300](https://notes-media.kthiha.com/Strongly-Connected-Components-(SCC)/462bf027020725b65e16021bc2fd2992.png)

The [[Time Complexity|complexity]] of computing adjacency list of $G^{T}$ is $O(|V| + |E|)$.

> Do not confuse the transpose of $G$ with the complement of $G$, denoted by $G^{c}$.

---
## Complement of G
The complement of $G$ is all possible edges minus all the existing edges.
**Note**: $G^{T}$ has the same [[Strongly Connected Components (SCC)|SCC]] as $G$.

---
## See Also
- [[Graph]]
- [[Kosaraju's SCC Algorithm]]
- [[Time Complexity]]
