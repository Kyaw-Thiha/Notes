# Greedy Algorithm
A [[Greedy Algorithm|greedy algorithm]] builds a solution in stages. In each stage, it extends a partial solution to a more complete one [[#Greedy|greedily]] & [[#Irrevocable|irrevocably]].

---
## Terminology
### Greedy
[[#Greedy|Greedily]] means that of the possible extensions to the current partial answer, 
- it chooses one that results in what appears to be best for the next stage 
- without looking ahead to future stages.

### Irrevocable
[[#Irrevocable|Irrevocably]] means that the algorithm never backtracks to undo an earlier decision.
Each stage strictly extends the previous partial answer.

---
## Algorithms
### Interval Scheduling
[[Interval Scheduling|Interval scheduling]] aims to find the largest subset of jobs where no two jobs conflict.

![|300](https://stumash.github.io/Algorithm_Notes/greedy/intervals/scheduling_soln.png)

---
### Maximum Lateness Scheduling
[[Minimum Lateness Scheduling|Minimum lateness scheduling]] aims to minimize the lateness of a set of jobs, given the deadline of each jobs.

![300](https://iq.opengenus.org/content/images/2020/04/Open-Genus.png)

---
### Fractional Knapsack
In [[Fractional Knapsack|fractional knapsack problem]], we aims to maximize the value of objects we can carry in a bag.

![|400](https://mwofqqzjzhhinqamxmxt.supabase.co/storage/v1/object/public/blog-images/88d13d47-10d2-45dc-a734-6b89b6200de0/fractional-knapsack-problem-greedy-approach-guide-1775814454582.png)

---
### Huffman Algorithm
[[Huffman Algorithm|Huffman Algorithm]] can encode symbols in a compressed manner.

![image|350](https://notes-media.kthiha.com/Huffman-Algorithm/1286ada1af9f1ee6b3ae441797bf36a4.png)

---
## Proof Strategies
### Promising Set Argument
In [[Interval Scheduling|promising set argument]], we show that partial solution constructed can always extend to some optimal solution.

### Greedy-Stays-Ahead Argument
 In [[Interval Scheduling|greedy-stays-ahead argument]], we show that the final answer is at least as good as any optimal solution.

### Exchange Argument
In [[Minimum Lateness Scheduling|exchange argument]], we start with an initial state and massage the state to be more similar to the output of our algorithm. The importance is that in each step, the massage retains optimality of the algorithm.

---
## See Also
- [[Greedy Algorithm]]
- [[Interval Scheduling]]
- [[Minimum Lateness Scheduling]]
- [[Fractional Knapsack]]
- [[Huffman Algorithm]]