# Interval Scheduling
[[Interval Scheduling|Interval scheduling]] aims to find the largest subset of jobs where no two jobs conflict.

![|300](https://stumash.github.io/Algorithm_Notes/greedy/intervals/scheduling_soln.png)

---
## Formal Definition
**Input**: A set of jobs $\mathcal{J}$ , with start time $s(j)$ and finish time $f(j)$ and denoted as $[\ s(j), f(j)\ )$.
A job pair $(j,i)$ conflict if their intervals intersect: $s(i) < f(j)$.

**Output**: A maximum cardinality subset $A$ of $\mathcal{J}$ with no conflicting jobs.
Formally, a set $A \subseteq \mathcal{J}$ such that
- `feasibility`: for all $j \neq j' \in A$, $j$ does not conflict with $j'$ and
- `optimality`: if $B \subseteq \mathcal{J}$ has the property that for all $j\neq j' \in B$, $j$ does not conflict with $j'$, then $|B| < |A|$.

Note the following: 
$$
\text{max cardinality feasible set} \neq \text{cannot be extended to a feasible superset}
$$

---
## Pseudocode
```python
SORT JOBS BY FINISH TIME

def IntervalScheduling(J):
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
def IntervalScheduling(J):
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
## Promising Set vs Greedy-Stays-Ahead
> In [[#(Lemma-2) Promising Set Lemma|promising set lemma]], we reason about the set in $A$ as it evolves in each iteration, and compare it to an optimal set that may change from each iteration to the next.

> In [[#Greedy Stays Ahead Proof|greedy stays ahead lemma]], we reason about the final set $A$ as it has been computed at the end of the loop and compare it to any fixed optimal set.

---
## Proof Variant-1
### Lemma-1
> $A_{i}$ contains no conflicting jobs.

---
### (Lemma-2) Promising Set Lemma
> The algorithm's current picks are always part of some optimal solution. 
> Links to irrevocability.

**Lemma**: For each $i \in [1\dots |\mathcal{J}|]$, there is some optimal set $A^{*}_{i}$ such that $A_{i} \subseteq A^{*}_{i}$.

#### Proof
We will be proving by induction on $i$. 

> **Base Case**: 
> At $i=0$, it is obvious since $A_{0}$ is empty.

> **Induction Hypothesis**: $A_{i} \subseteq A_{i}^{*} \implies A_{i+1} \subseteq A^{*}_{i+1}$
> Note that $A^{*}_{i+1}$ may be a different optimal set than $A^{*}_{i}$.

> **Induction Step**:
> Assume the induction hypothesis. Then, we have the following cases to consider.

**Cases**:
- **Case-1**: $A_{i+1} = A_{i}$. No job is added to $A$ in iteration $i+1$. Then, we can take $A_{i+1}^{*} = A^{*}_{i}$.
- **Case-2a**: $A_{i+1} = A_{i} \cup \{ j \}$. The job $j$ added happens to be in $A^{*}_{i}$. $j \in A^{*}_{t} \implies A^{*}_{t+1} = A^{*}_{t}$
- **Case-2b**: $A_{i+1} = A_{i} \cup \{ j \}$ and $j \not\in A^{*}_{i}$. This needs non-trivial proof. 

 ![image|400](https://notes-media.kthiha.com/Interval-Schedule/862fba6da4459f75281e8428892746a7.png)
 
For the case of some job $j$ being added to $A$ in iteration $i+1$ and $j \notin A^{*}_{i}$, we can prove as follows:

**Showing j has only one conflict**
- $j$ must conflict with some job in $A^{*}_{i}$.
  Otherwise, we could add $j$ to $A^{*}_{i}$ and obtain feasible set.
- $j$ cannot conflict with $2$ or more jobs.
  (since we would have chosen the job with earlier start time, and thus such $j \not\in A^{*}_{t+1}$)
- If $j \notin A^{*}_{i}$, then there must be exactly one $j^{*} \neq j$ that is $\in A^{*}_{i}$ so that $|A^{*}_{i}| = |A^{*}_{i+1}|$.

**Showing $A_{i+1}^{*}$ contains $A_{i+1}$**
- Let $j'$ be this one unique job in $A^{*}_{t}$ that $j$ conflicts with.
- Take $A^{*}_{i+1} = (A^{*}_{i} - \{ j' \}) \cup \{ j \}$.
- The set $A^{*}_{i+1}$ would then 
	- be feasible
	- has same no. of jobs as optimal set $A^{*}_{i}$
	- contains $A_{i} \cup \{ j \} = A_{i+1}$ (by construction).
- $A_{i} \cup \{ j \} = A_{i+1}$ would $\subseteq A^{*}_{i+1}$ defined above.
- So, $A^{*}_{i+1}$ is an optimal set that contains all the jobs in $A_{i+1}$, as wanted.

---
### Lemma-3
> The set of jobs returned by `IntervalScheduling()` is optimal.

#### Proof
By [[#(Lemma-2) Promising Set Lemma|promising set lemma]], there is an optimal set $A_{n} \subseteq A^{*}$.
**Claim**: $A_{n} = A^{*}$.

Suppose for **contradiction** that there is some job $j \in A^{*} - A_{n}$ as $j \notin A_{n}$.
- Since $A^{*}$ is optimal and feasible, $j$ does not conflict with any other jobs in $A^{*}$.
- Since $A_{n} \subseteq A^{*}$ and $j \in A^{*} - A_{n}$, $j$ does not conflict with any job in $A_{n}$.
- So, when $j$ was considered for inclusion in $A$ by the algorithm, it would have to be added to it. 
- This contradicts that $j \notin A_{n}$

---
## Proof Variant-2
### Lemma-1
> $A_{i}$ contains no conflicting jobs.

---
### Greedy Stays Ahead Lemma
Let $j_{1},\ j_{2},\ \dots, \ j_{k}$ be jobs in final $A$ in left-right order. 
Let $j_{1}^{*}, \ j_{2}^{*}, \dots, \ j^{*}_{m}$ be jobs in any optimal $A^{*}$ in left-right order.

> **Lemma**: $\forall i \in [1, \ k], \ f(j_{i}) \leq f(j_{i}^{*})$.

![image|400](https://notes-media.kthiha.com/Interval-Schedule/e65ab69bc483c61993cae084ac1643fa.png)

#### Proof
Proof by induction on $i$.

> **Base Case**: At $i=1$, first job chosen by algorithm is one with minimum finish time.

> **Induction Hypothesis**: $f(j_{i}) \leq f(j_{i}^{*}) \implies f(j_{i+1}) \leq f(j^{*}_{i+1})$.

> **Induction Step**: Assume the induction hypothesis hold. 
> For sake of contradiction, suppose that $f(j_{i+1}) > f(j^{*}_{i+1})$. 
> This means $j^{*}_{i+1}$ was considered by the [[Greedy Algorithm|greedy algorithm]] before $j_{i+1}$. 
> 
> We then have

$$
f(j_{i}) \leq f(j^{*}_{i}) \leq s(j^{*}_{i+1}) < f(j^{*}_{i+1}) < f(j_{i+1})
$$
where
- first inequality $f(j_{i}) \leq f(j^{*}_{i})$ is by `induction hypothesis`.
- second inequality $f(j^{*}_{i}) \leq s(j^{*}_{i+1})$ is because $A^{*}$ is feasible and jobs indexed left to right.
- third inequality $s(j^{*}_{i+1}) < f(j^{*}_{i+1})$ is because every job starts before it finishes
- fourth inequality $f(j^{*}_{i+1}) < f(j_{i+1})$ is because of our contradiction assumption.

> Thus, the algorithm considered job $j^{*}_{i+1}$ after $j_{i}$ and before $j_{i+1}$, but did not add it to $A$. Otherwise, $j^{*}_{i+1}$ and not $j_{i+1}$ would have been the $(i+1)^{th}$ job added to $A$.

> So, $j^{*}_{i+1}$ starts before $j_{i}$ finishes. This contradicts our earlier inequalities.
> Therefore, $f(j_{i+1}) \leq f(j^{*}_{i+1})$ completing our `induction step`.

---
### Lemma-3(Alternative)
> The set of jobs returned by `IntervalScheduling()` is optimal.

#### Proof
First recall that $k = |A|$ and $m = |A^{*}|$.  
Since $A$ is feasible, $k \leq m$. 

> For the sake of contradiction, suppose that $A$ is not optimal, then $k\neq m \implies k<m$. 
> So $A^{*}$ contains a job $j^{*}_{k+1}$ that is not in $A$.

> By [[#Greedy Stays Ahead Lemma|greedy-stays-ahead lemma]], we have that $f(j_{k}) \leq f(j_{k}^{*})$. 
> Since $A^{*}$ is feasible and its jobs are indexed left-to-right,

$$
f(j^{*}_{k}) \leq s(j^{*}_{k+1}) < f(j^{*}_{k+1})
$$

This means that
- the greedy algorithm considers job $j^{*}_{k+1}$ after job $j_{k}$.
- $j^{*}_{k+1}$ does not conflict with any of the jobs in $A$.

> Thus, the greedy algorithm should have added it to $A$, contradicting that $A$ has only $k$ jobs.

---
## Generalization
- [[Interval Scheduling|Weighted interval scheduling]]: Instead of optimizing by cardinality, we could weigh each jobs and optimize the weights.
- Optimizing no. of machines required to run all jobs. 
![image|300](https://notes-media.kthiha.com/Interval-Schedule/d9c3d581b69fa521fc534c960713448f.png)

---
