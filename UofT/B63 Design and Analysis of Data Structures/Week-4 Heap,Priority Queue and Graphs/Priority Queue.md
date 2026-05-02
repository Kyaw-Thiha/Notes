# Priority Queue
A [[Priority Queue|queue]] is an abstract data type that stores its items in First-In, First-Out (FIFO) order.

A [[Priority Queue|priority queue]] is an extension of a queue with the following properties:
- Every item has a priority associated with it.
- An element with higher priority is removed before an element with lower priority.
- If two elements have the same priority, they are served based on their order.

---
## Operations

- `insert(p,j)`: insert job $j$ with priority $p$
- `max()`: read the job with highest priority
- `extract_max()`: 
  Read and remove the job with highest priority.
- `increase-priority(j, p')`: 
  Increase the priority of job $j$ to priority $p'$.

---
