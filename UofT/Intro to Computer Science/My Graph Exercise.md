A robot is navigating a building represented as a grid. Each time it stops, it records its
position as a node in a graph. The robot can move in four directions: **North, East, South, West**.

To represent this graph, we use a special adjacency matrix where each row is a node (position),
and the 4 columns represent the 4 directions. 

Each entry stores the index of the neighbouring node in that direction, or -1 if there is a wall(no connection).

For example, if the robot is at node 5 and can move North to node 1, East to node 6, but has walls
to the South and West, then row 5 looks like:

| Node | North | East | South | West |
|------|-------|------|-------|------|
| 5    | 1     | 6    | -1    | -1   |

The robot navigates the following **2×4 building** (8 rooms total, indexed 0–7 left-to-right,
top-to-bottom):
```
 0  |  1  |  2  |  3
----+-----+-----+----
 4  |  5  |  6  |  7
```

The robot starts at node $0$ and its charging dock is at node $6$. 
The walls in the building create the following adjacency matrix (−1 = wall):

| Node | North | East | South | West |
|------|-------|------|-------|------|
| 0    | -1    | 1    | 4     | -1   |
| 1    | -1    | 2    | -1    | 0    |
| 2    | -1    | 3    | -1    | 1    |
| 3    | -1    | -1   | 7     | 2    |
| 4    | 0     | 5    | -1    | -1   |
| 5    | -1    | -1   | -1    | 4    |
| 6    | -1    | 7    | -1    | -1   |
| 7    | 3     | -1   | -1    | 6    |

The robot uses `DFS` to find a path from its current position back to the charging dock.
DFS always explores directions in order: **North → East → South → West**.

---
1. How many neighbours does node **5** have?
- (a) 0
- (b) 1
- (c) 2
- (d) 3

---

2. Which of the following correctly describes the path `0 → 1 → 2 → 3 → 7 → 6`?

- (a) Not a valid path — node 2 and node 3 are not connected
- (b) Not a valid path — node 3 and node 7 are not connected
- (c) A valid path from node 0 to the charging dock
- (d) A valid path, but it does not end at the charging dock

---

3.  The robot runs DFS starting from node **0**, exploring directions North → East → South → West. Which node is visited **third**?
- (a) 1
- (b) 2
- (c) 3
- (d) 4

---
4. Which of the following gives the correct DFS visit order starting from node **0**, until the charging dock (node 6) is first reached?
- (a) 0 → 1 → 2 → 3 → 7 → 6
- (b) 0 → 4 → 5 → 6
- (c) 0 → 1 → 2 → 3 → 7 → 4 → 6
- (d) 0 → 4 → 5 → 1 → 2 → 3 → 7 → 6

---

5. After DFS visits node **3**, which of the following correctly shows the state of the `visited[]` array? (1 = visited, 0 = not yet visited)

- (a) `visited = [1, 1, 1, 0, 0, 0, 0, 0]`
- (b) `visited = [1, 1, 1, 1, 0, 0, 0, 0]`
- (c) `visited = [1, 1, 1, 1, 1, 0, 0, 0]`
- (d) `visited = [1, 0, 0, 0, 1, 0, 0, 0]`

---

6. At some point during DFS, the `visited[]` array looks like this: `visited = [1, 1, 1, 1, 0, 0, 0, 1]` Which node is DFS currently processing?
- (a) Node 3
- (b) Node 4
- (c) Node 6
- (d) Node 7

---

7. The robot discovers a previously unknown passage connecting node $5$ directly to node $6$ (and back). This edge is added to the adjacency matrix. What graph property does this new passage create?
- (a) A new path that did not previously exist between nodes 5 and 6
- (b) A cycle in the graph
- (c) A disconnected component
- (d) Both (a) and (b)

---

8. After this new passage is added, the robot reruns DFS from node $0$ to find node $6$. 
   Compared to the original DFS, the robot reaches node $6$:
- (a) In fewer steps — DFS takes the new shortcut 0 → 4 → 5 → 6
- (b) In the same number of steps — the new passage does not appear before node 6
is found by the original exploration order
- (c) In more steps — the new cycle forces DFS to explore additional nodes first
- (d) DFS will fail — cycles cause the algorithm to loop forever

---
9. A second robot runs a **faulty** localization routine and incorrectly identifies node **5** as the same location as node **3**. It adds an edge connecting node 5 to node 3 (and back) in the adjacency matrix. When DFS is run from node **0** to find the charging dock (node 6), what happens?

- (a) DFS fails to find node 6 because the false edge permanently redirects it away
from the dock
- (b) DFS still finds node 6, but visits a different set of nodes along the way
- (c) DFS crashes because the adjacency matrix becomes structurally invalid
- (d) DFS finds node 6 faster because the false edge creates a genuine shortcut

---

10. The robot breaks one of its tyres and can now **only move South or West**. The modified DFS explores directions in order: **South → West** only. The robot is placed at node **3**. Which set of nodes can the robot **no longer reach** compared to full movement?

- (a) `{4, 5}`
- (b) `{5}`
- (c) `{1, 2, 5}`
- (d) `{4, 5, 6}`

---

# Answer Key
1. (b)
2. (c)
3. (b)
4. (a)
5. (b)
6. (d)
7. (d)
8. (b)
9. (b)
10. (b)