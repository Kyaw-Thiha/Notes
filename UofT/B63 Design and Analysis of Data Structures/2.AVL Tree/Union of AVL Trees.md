# Union of AVL Trees
Given two [[AVL Tree|AVL Trees]] $T$ and $V$, return an [[AVL Tree]] with all the keys in $T$ and $V$.

Consider this pseudocode.
![image|300](https://notes-media.kthiha.com/Union-of-AVL-Trees/23572b3761f3f9cfb59300909710ffd4.png)

Union the following trees:
![image|300](https://notes-media.kthiha.com/Union-of-AVL-Trees/b55bdc767851f1aabbeea289af6663f8.png)

**Solution**:
$$
\begin{align}
\text{k} &= \text{root}(T_{2}) \\[6pt]
&= 18
\end{align}
$$
Find $\text{split}(T_{1}, \ 18)$.
Since the root of $T_{1}$ is $11$, we go down $T_{1}$'s right path.
We go down $T_{1}$'s right path.

We hit $25$ which is larger than $18$, so we go down $25$'s left path.
We then hit $20$, which is greater than $18$.
We stop at this point.

$$
\begin{align}
T_{1 < 18} 
&= \text{merge}(( \ (9), \ 10), \ 11, \ \text{NULL}) 
\\[6pt]
&= ( \ (9), \ 10, \ (11))
\end{align}
$$
![image|200](https://notes-media.kthiha.com/Union-of-AVL-Trees/32f247c1754feac7b321ed5e9cabdf79.png)

$$
\begin{align}
T_{1 > 18} 
&= \text{merge}(20 , \ 25, \ 28) \\[6pt]
&= ((20), \ 25, \ 28)
\end{align}
$$
![image|200](https://notes-media.kthiha.com/Union-of-AVL-Trees/617c5405945068bb112721a59d30d53d.png)

Now we recursively union $T_{L} = \text{Union}(T_{1 < 18}, \ T_{2L})$.
![image|300](https://notes-media.kthiha.com/Union-of-AVL-Trees/a1d808db458f0e7f2f745e56181390b9.png)

We have to find $\text{split}(T_{1 < 18} , \ 14)$.

Since all the values in $T_{1 < 18}$ are less than $14$, nothing happens.

Then, we union $T_{1< 18}$ with $13$ to get $T_{1 < 14}$.
![image|200](https://notes-media.kthiha.com/Union-of-AVL-Trees/a1d944c5743d6c3d3602aa6bf33b85a5.png)

- Since no values in $T_{1 < 18}$ are greater than $14$, 
- and only $16$ is greater than $14$ from $T_{2L}$, 
- the union of $T_{1, \ 14< x < 18}$ with $16$ is just $16$.

Then, we get $T_{L}$ by merging with $14$ and $16$.
![image|100](https://notes-media.kthiha.com/Union-of-AVL-Trees/9c40e70675e87eb34fabcc592d1ba1b2.png)

![image|200](https://notes-media.kthiha.com/Union-of-AVL-Trees/8325dc4635eefb765091a7dd0de2fb28.png)

and after a double rotation, we get
![image|200](https://notes-media.kthiha.com/Union-of-AVL-Trees/0aa3db4218891f5c6cbf5532e11b6517.png)

After we find $T_{L}$, we find $T_{R} = \text{Union}(T_{1 > 18}, \ T_{2R})$.
![image|250](https://notes-media.kthiha.com/Union-of-AVL-Trees/fef439be5d8cab73bb120fad465f0459.png)

The root of $T_{2R}$ is $22$, so we split on $22$.
$(T_{1, \ 18 < x < 22}, \ b, \ T_{1 > 22}) = \text{split}(T_{1 > 18}, 22)$

$T_{1, \ 18 < x < 22} = 20$
$$
\begin{align}
T_{1 > 22}
&= (NULL, \ 25, \ 28) \\[6pt]
&= (25, \ (28))
\end{align}
$$

$$
\text{Union}(20, \ \text{left child of } 22) = 20
$$
$$
\text{Union}(25 \ (28), \ 31) = ((25) \ 28 \ (31))
$$

Merging $20$, $22$, and $((25) \ 28 \ (31))$, we get $T_{R}$.
![image|300](https://notes-media.kthiha.com/Union-of-AVL-Trees/fdef9f4a424d45ad90a0272b7ac09e7e.png)

Merging $T_{L}$ with $18$ and $T_{R}$, we can get $T_{1} \cup T_{2}$.
![image|300](https://notes-media.kthiha.com/Union-of-AVL-Trees/4e49784b39a8974e22422ea6a4481720.png)

---
## See Also
- [[AVL Tree]]
- [[AVL Tree Height]]
- [[AVL Tree Algorithms]]
- [[Union of AVL Trees]]
- [[Intersection of AVL Trees]]
- [[Difference of AVL Trees]]
