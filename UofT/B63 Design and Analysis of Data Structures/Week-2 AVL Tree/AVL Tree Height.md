# AVL Tree Height
If there are $n$ nodes, what is the max possible height?
If the height is $h$, then what is the minimum possible number of nodes?

**Solution**:
Recall that in [[AVL Tree]], the heights of the left and right subtrees can differ by at most $1$.

This means that if an AVL tree has a height of $h$, 
- for a tree to have the minimum number of nodes,
- one of its subtrees must have a height of $h-1$,
- and the other must have a height of $h-2$.

Let $\text{minsize(h)}$ be the minimum number of nodes for an [[AVL Tree]] of height $h$. Then,
1. $\text{minsize}(0) = 0$
2. $\text{minsize}(1) = 1$
3. $\text{minsize}(h+2) = 1 + \text{minsize}(h+1) + \text{minsize}(h)$

We can use induction to prove that $\text{minsize}(h) = \text{fibonacci}(h+2) - 1$.
Given the golden ration $\phi$ which equals $\frac{\sqrt{ 5 } + 1}{2} \approx 1.618$, 
$$
\begin{align}
\text{minsize}(h)
&= \frac{\phi^{h+2} - (1 - \phi)^{h + 2}}{\sqrt{ 5 }} - 1 \\[6pt]
&= \frac{\phi^{h+2}}{\sqrt{ 5 }}
- \frac{(1 - \phi)^{h+2}}{\sqrt{ 5 }} - 1 \\[6pt]
&> \frac{\phi^{h+2}}{\sqrt{ 5 }} - 1 - 1 \\[6pt]
&> \frac{\phi^{h+2}}{\sqrt{ 5 }} - 2
\end{align}
$$
We also know that $\text{minsize}(h) \leq n$.
Hence,
$$
\begin{align}
\frac{\phi^{h+2}}{\sqrt{ 5 }} - 2 &< n \\[6pt]
\frac{\phi^{h+2}}{\sqrt{ 5 }} &< n + 2 \\[6pt]
\phi^{h+2} &< \sqrt{ 5 }(n + 2) \\[6pt]
\end{align}
$$
Applying a [[Logarithmic Rules|log]], we get
$$
\begin{align}
(h+2) \log \phi &< \log(\sqrt{ 5 } \ (n+2)) \\[6pt]
h+2 &< \frac{\log(\sqrt{ 5 } \ (n+2))}{\log(\phi)}  
\\[6pt]
&< \frac{\log(\sqrt{ 5 } + \log(n+2))}{\log(\phi)}
\\[6pt]
&< \frac{\log(n+2)}{\log(\phi)}
+ \frac{\log(\sqrt{ 5 })}{\log(\phi)}
\end{align}
$$
Then, we get
$$
\begin{align}
h &< \frac{\log(n+2)}{\log(\phi)}  
+ \frac{\log(\sqrt{ 5 })}{\log(\phi)} - 2 \\[6pt]
&= 1.44 \ \log(n+2) - 2 \\[6pt]
&\in O(\log n)
\end{align}
$$

---
## See Also
- [[AVL Tree]]
- [[AVL Tree Height]]
- [[AVL Tree Algorithms]]
- [[Rotations in AVL Tree]]
- [[Union of AVL Trees]]
- [[Intersection of AVL Trees]]
- [[Difference of AVL Trees]]