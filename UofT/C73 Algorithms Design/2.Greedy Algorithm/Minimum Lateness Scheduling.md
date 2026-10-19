# Minimum Lateness Scheduling
[[Minimum Lateness Scheduling|Minimum lateness scheduling]] aims to minimize the lateness of a set of jobs, given the deadline of each jobs.

![300](https://iq.opengenus.org/content/images/2020/04/Open-Genus.png)

---
## Formal Definition
> **Input**: A finite set $\mathcal{J}$ of $n$ jobs of $1,2,\dots, n$ where each job $j \in \mathcal{J}$ has length $l(j)$, deadline $d(j)$, and release time $r$ (time we can start scheduling).

> **Output**: A schedule of minimum lateness. That is a function $S: \mathcal{J} \to \mathbb{R}$ that
> - (`feasibility`) satisfies $(a)$ and $(b)$ below.
> - (`optimality`) for any function $S': \mathcal{J} \to \mathbb{R}$ that satisfies $(a)$ and $(b)$, $L(S) \leq L(S')$.

---
### Schedule Definition
A [[Minimum Lateness Scheduling|schedule]] is a function $S: \mathcal{J} \to \mathbb{R}$ where $S(j) \geq r$ is the start time for job $j$ $s.t.$
$$
\big[ S(j), \ S(j) + l(j) \big) \ \cap \  
\big[  S(j'), \ S(j') + l(i') \big) = \emptyset
$$
with $\forall j \neq j' \in \{ 1,2,\dots,n \}$. 

Formally, we require scheduler $S$ to satisfy the following `feasibility requirements`:
- $(a)$ $\forall j \in \mathcal{J}, \ S(j) \geq r$
- $(b)$ $\forall j, j' \in \mathcal{J}$, if $j \neq j'$ then the intervals $\big[ S(j), \ S(j) + l(j) \big)$ and $\big[  S(j'), \ S(j') + l(i') \big)$ do not intersect.

---
### Lateness Definition
The [[Minimum Lateness Scheduling|lateness]] of job $j$ in schedule $S$ i the amount by which $j$ misses its deadline.
$$
L(S,j) = \max\big(S(j) + l(j) - d(j), \ 0 \big)
$$
Using this, we define the lateness of the schedule $S$ as
$$
L(S) = \max_{1 \leq i \leq n} L(S,i)
$$


---
## Example
Consider the following example of $n=4$ and $r=0$.
![image|500](https://notes-media.kthiha.com/Maximum-Lateness-Scheduling/b0aca3b51ab8233757b56a969553bdba.png)

Note that there can be multiple optimal solutions.
![image|500](https://notes-media.kthiha.com/Maximum-Lateness-Scheduling/b17b41c95be4f2604d991cd565470b28.png)

---
## Algorithm
```python
MinLatenessSchedule(jobs, r):
	sort jobs by non-decreasing deadline.

	F := r
	for j ∈ sorted jobs do:
	    S(i) := F
	    F:= S(i) + l(i)
	return S
```

---
### Running Time
Running Time: $O(n \log n)$

---
## Proof of Correctness
> **Theorem**: Schedule $S$ produced by greedy algorithm `MinLatenessSchedule()` is optimal.

---
### Inversion Definition
Given $S_{1}$ and $S_{2}$ are [[Minimum Lateness Scheduling|schedules]] of the same set of jobs $\mathcal{J}$, we define [[Minimum Lateness Scheduling|inversion]] of $S_{1}$ $w.r.t$ $S_{2}$ over a pair of jobs $i,j$ as
$$
S_{1}(i) < S_{1}(j)
\iff S_{2}(i) > S_{2}(j)
$$
![image|300](https://notes-media.kthiha.com/Maximum-Lateness-Scheduling/16a4a6dcf5f9c7055bc37a8233050078.png)


---
> **Preliminaries**
> 
> Let $S^{*}$ be an optimal schedule with fewest inversions between $S^{*}$ and $S$.
> Let $k$ be that $\#$ of inversions.
> **Claim**: $k=0$ in which case $S=S^{*}$.

> **Proof by Contradiction**:
> For the sake of contradiction, suppose that $k > 0$.
> Thus, there is a pair of jobs $i'$ and $j'$ such that $i'$ appears before $j'$ in $S^{*}$ but $i'$ appears after $j'$ in $S$. 

> Hence, there should be two jobs $i$ and $j$ such that $i$ is the job immediately before $j$ in $S^{*}$, but $i$ appears after $j$ in $S$ (not necessarily immediately after). Let $S'$ be the schedule with no gaps obtained from $S^{*}$ by swapping order of $i$ and $j$.

![image|500](https://notes-media.kthiha.com/Maximum-Lateness-Scheduling/5be3999511eec425f86086c8959f9fdd.png)

> **No. of inversions**
> Note that no. of inversions between $S'$ and $S$ is $k-1$.
> The inversion between $S^{*}$ and $S$ caused by $i$ and $j$ is now fixed since $j$ appear before $i$ as is in $S$ and no new inversions were created since $i$ and $j$ are next to each other.

> **Sorted Deadline**
> Recall that jobs in $S$ appear in order of increasing deadline so $d(j) \leq d(i)$.

> **Lateness of $i$ and $j$**
> All jobs other than $i$ and $j$ have same lateness in $S'$ as is in $S^{*}$.
> 
> The lateness of $j$ in $S'$ is no worse than in $S^{*}$ since it is moved to an earlier position.
> The lateness of $i$ in $S'$ may have increased relative to its lateness in $S^{*}$, but it is no greater than lateness of $j$ in $S^{*}$.

Thus, we have
$$
\begin{align}
L(i, S')
&= \max(S'(i) + l(i) - d(i), \ 0) \\[6pt]
&= \max(S^{*}(i) + l(i) - d(i), \ 0) \\[6pt]
&\leq \max(S^{*}(i) + l(i) - d(j), \ 0) \\[6pt]
&= L(j, \ S^{*})
\end{align}
$$
Therefore, the maximum lateness in $S'$ is no greater than maximum lateness in $S^{*}$.
$$
L(S') = \max_{m\in \mathcal{J}} L(m, S') 
\leq \max_{m \in \mathcal{J}} L(m, S^{*}) 
= L(S^{*})
$$
> **Contradiction**
> Since $S^{*}$ is optimal and $S'$ is feasible, it must be that $S'$ is also optimal.
> But there are fewer inversions between $S'$ and $S$ than between $S^{*}$ and $S$.  
> 	
> Hence, this is a contradiction. 

---
![image|500](https://notes-media.kthiha.com/Maximum-Lateness-Scheduling/88613e7ca99db1fe41ab4feb34a5ee7c.png)

- $L(S^{'}, \ i^{'}) = L(S^{*}, i^{'})$, $\forall i' \neq i,j$
- $L(S', \ j) \leq L(S^{*}, j)$
- But $L(S', i) \geq L(S^{*}, \ i)$
- But  $L(S^{'}, i) \leq L(S^{*}, \ j)$

---
## Exchange Argument
- We start with an initial state.
- Then, we massage the state to be more similar to the output of our algorithm.
- The importance is that in each step, the massage retains optimality of the algorithm.

In [[Interval Scheduling|promising set argument]], we show that partial solution constructed can always extend to some optimal solution. In [[Interval Scheduling|greedy-stays-ahead argument]], we show that the final answer is at least as good as any optimal solution.

---
## See Also
- [A blog post](https://iq.opengenus.org/scheduling-to-minimize-lateness/#gsc.tab=0)
- [[Greedy Algorithm]]
- [[Interval Scheduling]]
- [[Minimum Lateness Scheduling]]
- [[Fractional Knapsack]]
- [[Huffman Algorithm]]