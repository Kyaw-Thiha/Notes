# Dynamic Programming

Usually we solve `optimization problems`, but we could also solve `decision making problems`.

---
## Checking if Dynamic Programming Helps
1. Analyze problem to discover optimal substructure
2. Sometimes, this requires changing the problem by adding constraints/params.
3. (For `optimization problems`): Focus on computing the **values** of the subproblems.
4. State **clearly** the subproblems to solve. $(*)$
5. Ensure I can solve original problem given solutions to the subproblems.
6. Give recursive formula $({\dagger})$ ([[Bellman Equation for Quality Function|Bellman Equation]]) to compute the sub problems $(*)$.
7. Write pseudcode for computing the [[Bellman Equation for Quality Function|Bellman Equation]] $({\dagger})$.
8. Retrofit the algorithm to compute the optimal object (not just is value).
9. [[Time Complexity|Running time analysis]].

---
## Longest Increasing Sequence
Want:
- find [[Longest Increasing Subsequece(LIS)|LIS]] of $A$
- find [[Longest Increasing Subsequece(LIS)|LIS]] of $A$ ending at pos $i$

The problem we ended up solving up have additional parameters than what we were originally trying to solve. 

---