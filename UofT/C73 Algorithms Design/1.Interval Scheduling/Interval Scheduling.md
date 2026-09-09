# Interval Scheduling
Input: 
- Set of $n$ jobs ($1, \ 2, \dots, \ n$)
- Job $j$ has
	- start time $s(j)$
	- end time $f(j)$
	- and denoted as $[\ s(j), f(j)\ )$
- $(j,i')$ conflict if that intervals intersect

Feasible Set: no two jobs conflict

Output: A max cardinality feasible set.
- Eg: $\{ b,c,d,c \}$
- Eg: $\{ i,j,d,e \}$

$$
\text{max cardinality feasible set} \neq \text{cannot be extended to a feasible superset}
$$

---
## Pseudocode
```python
SORT JOBS BY FINISH TIME
A = ∅; F := -∞
while J != ∅ do
	j = "next" job in J
	if j doesn't conflict with any job s(j) >= F in A, do A = A ∪ {j}
	F = f(j)
return A
```


---
## Wrong Ideas

Naive(wrong) Implementation
```python
INTSCHED(J)
A = ∅
while J != ∅ do
	j = "next" job in J
	if j doesn't conflict with any job in A, do A = A ∪ {j}
	J = J - {j}
return A
```

- **Idea-1**: Sort by start time
  **Counterexample**: Think of a long job at the start.
- **Idea-2**: Sort by length 
  **Counterexample**: Think of a small job that overlaps two longer jobs. Better than Idea-1 though.
- **Idea-3**: Sort by no. of conflict
  **Counterexample**: Consider the following.
![image|300](https://notes-media.kthiha.com/Interval-Schedule/19896c498c595eae210174dd1dc41828.png)

---
## Running Time
- Sort: $O(n \log n)$
- Looping through jobs: $O(n)$
$$
\text{Running Time} = O(n \log n) + O(n) = O(n \log n)
$$

---
### Promising Set(Variant-1)
> The algorithm's current picks are always part of some optimal solution. Links to irrevocability.

**Lemma**: $\forall \text{ iter } t, \ \exists \text{ opt set } A^{*}_{t} \text{ s.t. } A_{t} \subseteq A^{*}_{t}$.

**Induction Step**:
- Assume $A_{t} \subseteq A_{t}^{*}$
- Will show $A_{t+1} \subseteq A^{*}_{t+1}$

Cases:
- **Case-1**: $A_{t+1} = A_{t}$. Take $A_{t+1}^{*} = A^{*}_{t}$.
- **Case-2a**: $A_{t+1} = A_{t} \cup \{ j \}$. $j \in A^{*}_{t} \implies A^{*}_{t+1} = A^{*}_{t}$
- **Case-2b**: $A_{t+1} = A_{t} \cup \{ j \}$. $j \not\in A^{*}_{t}$.
 ![image|400](https://notes-media.kthiha.com/Interval-Schedule/862fba6da4459f75281e8428892746a7.png)
	- $j$ must conflict with some job in $A^{*}_{t}$.
	- $j$ cannot conflict with $2$ or more jobs.
	  (since we would have chosen the job with earlier start time, and thus such $j \not\in A^{*}_{t+1}$)
	- Let $j'$ be the unique job in $A^{*}_{t}$ that $j$ conflicts with.
	- Take $A^{*}_{t+1} = (A^{*}_{t} - \{ j' \}) \cup \{ j \}$.
	- $A_{t} \cup \{ j \} = A_{t+1}$ would $\subseteq A^{*}_{t+1}$ defined above.
	- $A_{n} \subseteq A^{*}_{n}$.
	- There is no $j \in A^{*}_{n} - A_{n}$.

---
## Greedy Stays Ahead Proof(Variant-2)
Let $j_{1},\ j_{2},\ \dots, \ j_{k}$ be jobs in final $A$ in $L-R$ order.
Let $j_{1}^{*}, \ j_{2}^{*}, \dots, \ j^{*}_{n}$ be jobs in any optimal $A^{*}$ in $L-R$ order.

**Lemma**: $\forall i \in [1, \ k], \ f(j_{i}) \leq f(j_{i}^{*})$.
![image|400](https://notes-media.kthiha.com/Interval-Schedule/e65ab69bc483c61993cae084ac1643fa.png)

---
## Generalization
- [[Interval Scheduling|Weighted interval scheduling]]: Instead of optimizing by cardinality, we could weigh each jobs and optimize the weights.
- Optimizing no. of machines required to run all jobs. 
![image|300](https://notes-media.kthiha.com/Interval-Schedule/d9c3d581b69fa521fc534c960713448f.png)

---
