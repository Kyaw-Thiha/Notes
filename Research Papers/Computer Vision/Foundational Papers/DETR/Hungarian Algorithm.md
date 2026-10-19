# Hungarian Algorithm
#cv/object-detection/hungarian-algorithm 

Hungarian algorithm is an algorithm that can be used to carry out [[Bipartite Matching]] efficiently in polynomial $O(n^3)$ time.

![Hungarian Algorithm](https://i.ytimg.com/vi/ezSx8OyBZVc/maxresdefault.jpg)

## Algorithm

Firstly, set up the cost matrix from [[Matching Cost]] where
- Rows: Predictions
- Columns: Ground Truths

Then, carry out the following algorithm:
1. **Row Reduction**
   For each row, take the minimum value in the row, and subtract it from all elements in a row.
2. **Column Reduction**
   For each column, take the minimum value in the column, and subtract it from all elements.
3. **Lining Up Zeroes**
   Cover all zeroes with minimum number of vertical & horizontal lines
   - If minimum number of lines = $n$, then optimal pair assignments are among the lines
   - Else, continue onto Step-4.
4. **Matrix Adjustment**
   Find the smallest non-zero number.
   Subtract it to all other non-zero numbers.
   Add it to all zeroes that are interested by two lines.
5. **Pair Assignment**
   List out the cells which have values of 0.
   Everytime a cell is chosen, we can't choose another cell from that column.
   Repeat that till we have $n$ cells.

---
## Examples
Let Initial cost matrix be
$$
C =
\begin{bmatrix}
4 & 1 & 3 \\
2 & 0 & 5 \\
3 & 2 & 2
\end{bmatrix}
$$

**Step-1: Row reduction**  
Subtract each row’s minimum from that row:
- Row mins: $[1,\,0,\,2]$

$$
\begin{bmatrix}
3 & 0 & 2 \\
2 & 0 & 5 \\
1 & 0 & 0
\end{bmatrix}
$$

**Step-2: Column reduction**  
Subtract each column’s minimum from that column:
- Column mins: $[1,\,0,\,0]$

$$
\begin{bmatrix}
2 & 0 & 2 \\
1 & 0 & 5 \\
0 & 0 & 0
\end{bmatrix}
$$

**Step-3: Lining Up Zeroes**  
Minimum lines $= 2 < n=3$ 
So, proceed to Step-4.

**Step-4: Matrix Adjustment**  
Smallest uncovered value: $1$.  
- Subtract $1$ from all **uncovered** entries.  
- Add $1$ to all entries covered **twice** (line intersections).

Result:
$$
\begin{bmatrix}
1 & 0 & 1 \\
0 & 0 & 4 \\
0 & 1 & 0
\end{bmatrix}
$$

**Step-3: Lining up zeros**  
Now minimum lines $= 3 = n$ → proceed to assignment.

**Step-5: Pair Assignment**
Valid pairs: $(1,2), (2,1), (3,3)$.
Total cost:
$$
C_{1,2} + C_{2,1} + C_{3,3} \;=\; 1 + 2 + 2 \;=\; 5.
$$

---
## See Also
- [[DETR]]
- [[Matching Cost]]
- [Hungarian Algorithm Blog](https://www.thinkautonomous.ai/blog/hungarian-algorithm/)
- [Hungarian Algorithm Youtube Video](https://youtu.be/ezSx8OyBZVc?si=ulLGc9Upj5fo93r3)

