## Checklist
> $1.$ Define **objective** being maximized or minimized.

> $2.$ Find the **greedy choice**.
> 
> Common rules:
> $(a)$ Earliest finish time ([[Interval Scheduling|interval scheduling]])
> $(b)$ Earliest deadline ([[Minimum Lateness Scheduling|maximum lateness scheduling]])
> $(c)$ Smallest or largest item first ([[Huffman Algorithm|Huffman]], [[Kruskal's Algorithm|Kruskal's Algo]])
> $(d)$ Best ratio ([[Fractional Knapsack|fractional knapsack]])
> $(e)$ Furthest reach (gas station)
>
> One trick is to look at the extremes since the first choice in optimal solution is usually min or max of something.

> $3.$ **Sort** or use a **heap**.
> Could be sort by key, then one pass.
> Or repeatedly extract the best from the priority queue.

> $4.$ Write the **loop**.
> For each step, pick the best remaining candidate, check feasibility, and decide on either adding or skipping it.

---
### Defining the Objective and Constraints
**Objective**: Quantity after “maximize / minimize / find the largest / fewest / earliest”. Can written as a formula $\text{Eg:} \max |S|$.
 
**Constraints**: defines which solution is legal.
- “no two overlap”, 
- “total weight ≤ W”, 
- “each job finishes by its deadline”, 
- “every point is covered.”

---
### Constraint Types
For different constraint types, and their typical [[Greedy Algorithm|greedy algorithm]]:
- **Conflict / Overlap**: Earliest finish time.
- **Capacity / Budget**: Best ratio (value per unit).
- **Deadlines**: Earliest deadline first.
- **Coverage**: Furthest reach.
- **Merge/Combine Costs**: Smallest-first with heap.

These also exists:
- **Matching / Pairing**: Sort both sides, two-pointer.
  Eg: assign cookies to children, each with different greed factor
- **Ordering** (adjacent swap matters): Sort by exchange-argument comparator.
  Eg: arrange numbers so their concatenation is largest
- **Rolling Selection**: Put into Max-heap + extract-max.
  Eg: IPO where we have starting capital $k$ and projects $(\text{cost, profit})$. Each round you can reinvest the profit you made.
- **Lexicographic** (smallest/largest result): Monotonic stack.
- **Connectivity / Shortest Path**: Kruskal/Prim (MST), Dijkstra (shortest path).

---
### Finding Good Counterexample
> Always start with $n=2$ or $3$. 

Consider the following examples:
- One big, two small
- Small blocker in the middle
- Extreme values
- Equal keys(Ties)
- Almost-fits(capacity exactly one less than what greedy needs)
- Order sensitivity
- Boundary Cases

---

