# Maximum Lateness Scheduling
**Input**: $n$ jobs of $1,2,\dots, n$ where each job $i$ has
- length: $l(i)$
- deadline: $d(i)$
- release time: $r$ (time we can start scheduling)

**Schedule Definition**:
Schedule $S$ where $S(i) \geq r$ is the start time for job $i$ $s.t.$
$$
[\ S(i), \ S(i) + l(i) \ ) \ \cap \  
[ \ S(j), \ S(j) + l(i) \ ) = \emptyset
$$
with $\forall i\neq j \in \{ 1,2,\dots,n \}$.

**Lateness Definition:**
Then, we define the lateness of $i$ in $S$ as
$$
L(S,i) = \max(S(i) + l(i) - d(i), \ 0)
$$
Using this, we define the lateness of the schedule $S$ as
$$
L(S) = \max_{1 \leq i \leq n} L(S,i)
$$

**Output**: A schedule $S$ with $\min L(S)$.

---
# Example
Consider the following example of $n=4$ and $r=0$.
![image|500](https://notes-media.kthiha.com/Maximum-Lateness-Scheduling/b0aca3b51ab8233757b56a969553bdba.png)

Note that there can be tow optimal solutions.
![image|500](https://notes-media.kthiha.com/Maximum-Lateness-Scheduling/b17b41c95be4f2604d991cd565470b28.png)

---
## Algorithm
```python
Sort jobs by non-decreasing deadline job.
Let 
	d[1...n] be deadlines(sorted).
	l[1...n] be lengths(not necessarily sorted).
F := r
for i=1 to n do
    S(i) := F; \ F:= S(i) + l(i)
return S
```

Running Time: $O(n \log n)$

---
**Inversion Defn**:
Inversion of $S_{1}$ $w.r.t$ $S_{2}$ is a pair of jobs $i,j$ $s.t.$ 
$$
S_{1}(i) < S_{1}(j)
\iff S_{2}(i) > S_{2}(j)
$$

**Theorem**: Schedule $S$ produced by greedy algorithm is optimal.

---
**Preliminaries**:
Let $S^{*}$ be an optimal schedule with fewest inversions $w.r.t$ $S$.
Let $k$ be that $\#$ of inversions.
**Claim**: $k=0$.

For the sake of contradiction, let $k > 0$.

![image|500](https://notes-media.kthiha.com/Maximum-Lateness-Scheduling/16a4a6dcf5f9c7055bc37a8233050078.png)

![image|500](https://notes-media.kthiha.com/Maximum-Lateness-Scheduling/88613e7ca99db1fe41ab4feb34a5ee7c.png)

- $L(S^{'}, \ i^{'}) = L(S^{*}, i^{'})$, $\forall i' \neq i,j$
- $L(S', \ j) \leq L(S^{*}, j)$
- But $L(S', i) \geq L(S^{*}, \ i)$
- But  $L(S^{'}, i) \leq L(S^{*}, \ j)$

---
## Exchange Argument
We start with an initial state.
Then, we massage the state to be more similar to the output of our algorithm.
The importance is that in each step, the massage retains optimality of the algorithm.