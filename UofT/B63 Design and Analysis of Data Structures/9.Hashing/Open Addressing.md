# Open Addressing

If we get a [[Hashing|collision]] when we insert a new element, 
- we find a new location to store the new element
- we need to know where we put it. 
	- To do this, we search for a well-defined sequence of other locations in the [[Hash Table|hash table]] until we find one that's not full. 
	- This sequence is called a [[Open Addressing|probe sequence]].

---
### Caveat
Each entry in the [[Hash Table|hash table]] stores a fixed number of $c$ elements. For simplicity, assume that $c=1$. We only use [[Open Addressing|open addressing]] if $c.n < m$.

---
## Probe Sequences
There are $3$ methods for generating a [[Open Addressing|probe sequence]]:
- [[#Linear Probing]]: Try $A[ \ (h(k) + i) \text{ mod } m \ ]$ where $i=0,1,\dots$
- [[#Quadratic Probing]]: Try $A[ \ (h(k) + C_{1}i + C_{2} i^{2}) \text{ mod } m \ ]$
- [[#Double Hashing]]: Try $A[ \ (h(k) + i h'(k)) \text{ mod } m \ ]$ where $h'$ is another [[Hash Function|hash function]].

---
### Linear Probing
For a [[Hash Table|hash table]] of size $m$, key $k$, and [[Hash Function|hash function]] $h(k)$, the probe sequence is calculated as 
$$
S_{i} = (h(k) + i) \ \text{mod} \ m
$$
for $i=0,1,\dots$

> **Note**: The value of $S_{0}$, the home location for the item, is $h(k)$.

The problem with [[#Linear Probing|linear probing]] is `clustering`. When we [[Hashing|hash]] to something within a group of filled locations, we have to probe the whole group until we reach on empty slot.

This increases the size of the cluster. Overall, this results in $2$ keys that didn't necessarily share the same "home" location ending up with almost identical probe sequences.

#### Example
Suppose we start off with an empty array.
Consider this sequence of `inserts`.


![image|250](https://notes-media.kthiha.com/Open-Addressing/1fc7f898b1a8b6eddab63a0e23833f6e.png)
![image|250](https://notes-media.kthiha.com/Open-Addressing/9564e63a8e2a60327dc660ba3c17bd78.png)

---
### Quadratic Probing
In [[#Quadratic Probing|quadratic probing]], the probe sequence is calculated as
$$
S_{i} = (h(k) + C_{1}.i + C_{2}.i^{2}) \ \text{mod} \ m
$$
for $i=0,1,2,\dots$

[[#Quadratic Probing|Quadratic probing]] faces the same problem as [[#Linear Probing|linear probing]]; clusters.

---
### Double Hashing
Use a second, different [[Hash Function|hash function]], $h_{2}(k)$, to calculate the step size. 

The probe sequence is 
$$
S_{i} = (h(k) + i \cdot h_{2}(k)) \ \text{mod} \ m
$$
for $i=0,1,2,\dots$

> **Note**: $h_{2}(k)$ shouldn't have to be $0$ for any value of $k$.

We want to choose a $h_{2}$ $s.t.$ even if $h(k_{1}) = h(k_{2})$, it won't be the case that $h_{2}(k_{1}) = h_{2}(k_{2})$. This way, the two hash functions don't cause collisions on the same pairs of keys.

---
## See Also
- [[Hashing]]
- [[Hash Function]]
- [[Hash Table]]
- [[Direct Addressing]]
- [[Closed Addressing]]
- [[Open Addressing]]
- [[Universal Hashing]]
