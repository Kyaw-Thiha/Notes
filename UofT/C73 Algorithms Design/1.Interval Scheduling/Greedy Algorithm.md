# Greedy Algorithm
A [[Greedy Algorithm|greedy algorithm]] builds a solution in stages. In each stage, it extends a partial solution to a more complete one [[#Greedy|greedily]] & [[#Irrevocable|irrevocably]].

---
### Greedy
[[#Greedy|Greedily]] means that of the possible extensions to the current partial answer, 
- it chooses one that results in what appears to be best for the next stage 
- without looking ahead to future stages.

### Irrevocable
[[#Irrevocable|Irrevocably]] means that the algorithm never backtracks to undo an earlier decision.
Each stage strictly extends the previous partial answer.

---
## Examples

### Interval Scheduling
[[Interval Scheduling|Interval scheduling]] aims to find the largest subset of jobs where no two jobs conflict.

![|300](https://stumash.github.io/Algorithm_Notes/greedy/intervals/scheduling_soln.png)

---
