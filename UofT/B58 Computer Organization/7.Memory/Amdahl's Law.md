# Amdahl's Law
[[Amdahl's Law]] is a statement in computer architecture that states that

> The performance gain in task achievable by optimizing a subproblem is limited by the proportion of time spent in that subproblem in the unoptimized task.

Eg: If you only spend $\frac{1}{100}$ of your time doing task-A, then even if you reduce the time required for A to zero, you'll only save $1\%$.

---
## Formula
$$
\frac{1}{1 - f + \frac{1}{P}}
$$
where
- $f$: parallelizable fraction of a program
- $P$: number of processors

---