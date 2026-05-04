# Hash Table
[[Hash Table|Hashing]] is the process of mapping inputs(keys) to outputs(values) through the use of a [[Hash Table|hash function]].

![image|400](https://notes-media.kthiha.com/Hash-Table/8b27e9bec1481ee9e445811922941916.png)

This allows us to allocate different indexes of the array as storage for different hash function outputs.

---
## Motivation
When working with arrays, finding element without its index takes an [[Time Complexity|average complexity]] of $O(n)$.

This is because in array, elements are not ordered.

With [[hash table]], we could reduce it to $O(1)$.

---
## Hash Function
A [[Hash Table|hash function]] is a mathematical algorithm that maps input data of any size into fixed-size data.

---
### Division Method
The index is defined as the remainder of the division between the key and a chosen number, often prime.

$$
H(\text{key})
= \text{key mod k} 
= \text{key} \  \% \ k
$$

**Why prime modulo?**
- Chosen due to absence of common factors.
- Allows keys to be more evenly distributed and reduces clustering.

![image|300](https://notes-media.kthiha.com/Hash-Table/c8c22c0ab2426dd935d4aa931b4f9b0d.png)
![image|300](https://notes-media.kthiha.com/Hash-Table/c205e6d6739f55e6bdc064dc510e7cdf.png)

---
### Multiplication Method
- The key is multiplied by a constant $A \in (0,1)$.
- Then, the fractional part of the result is multiplied by $m$ which is the desired size of the hash table.
- The final result is then floored to obtain an integer value equating to the index.

$$
H(\text{key})
= \text{floor}(m \ ((\text{key} \times A) \text{ mod } 1))
\quad \text{, where } 0 \ll A < 1
$$

**How to choose A?**
- Simple rational numbers such as $0.5, \ 0.25, \ 0.3$ are avoided.
  This is because they often lead to poorer distributions and clustering.
- Instead choose **irrational numbers** such as **golden ratio** or its reciprocal. This is because they introduce more randomness and variability into [[Hash Table|hash function]], thus helping achieve a more uniform distribution.

**Why multiplicative method over division method?**
- Size of the [[hash table]] does not need to be prime.
- So, uniformly distributing the hash values across the hash table is easier done by simply adjusting the constant $A$.

---
## Dealing with Collisions
Collisions occur when two or more keys have the same [[Hash Table|hash function]] result, and thus are allocated into the same bucket.

There are two methods of handling them: [[#open addressing]] and [[# separate chaining]]

---
### Open Addressing
Elements are always stored in the [[hash table]] itself.
This can be done through **probing** and **rehashing**.

#### Linear Probing
This can be expressed as
$$
H(\text{key})
= (\text{key mod } k) + i
= (\text{key \% } k) + i
$$
Whenever there is a collision at a particular bucket, the key will try the next index in the [[hash table]].

![image|400](https://notes-media.kthiha.com/Hash-Table/8329af78c4de2fd803d0a765d0bb7451.png)

One disadvantage is clustering and slower insertion times.
If collisions are more common at specific key, it will have to pass many consecutive elements.

---
#### Quadratic Probing
This can be expressed as
$$
H(\text{key})
= (\text{key mod } k) + i^{2}
= (\text{key \% } k) + i^{2}
$$
Whenever collision occurs, the key searches at the $1^{2}$ index from collision index, then $2^{2}$ index and so on.

![image|400](https://notes-media.kthiha.com/Hash-Table/066d903f4b328a444bf9fb42748a16ac.png)

- This allows keys to be distributed more sparsely.
- However if multiple keys have same collision, it still could take longer.

---
#### Double Hashing
With rehashing, the increment is decided by the result obtained from passing the hash value from colliding key through another secondary [[Hash Table|hash function]].
![image|400](https://notes-media.kthiha.com/Hash-Table/a8fc9fafd35b536f8900a29652cb8e1e.png)

Note that if second [[Hash Table|hash function]] leads to a filled bucket, then the second hash function is applied once again till an empty bucket is found.

---
### Separate Chaining
Store a linked list of key-value pairs in each bucket of the [[hash table]] that hash to the same index.
![image|400](https://notes-media.kthiha.com/Hash-Table/b065563ce72cd07a3e8a93ae8659fe34.png)

Note that retrieving an element from linked list takes $O(n)$ compared to [[Hash Table|hash table's]] $O(1)$. Hence, it is best to minimize the number of keys attaching to single bucket.

---
### Load Factor
The [[#load factor]] of a [[hash table]] is the ratio between the number of elements in the hash table to the size of the hash table.
$$
\text{load factor}
= \frac{\text{no. of elements}}{\text{size of hash table}}
$$
It signifies how full a [[hash table]] is.
The higher the load factor, the higher the collisions.

Whenever this load factor becomes greater than a threshold, we can
- create a larger hash table
- and rehash all existing data to new table

> Note that rehashing is more efficient with [[#separate chaining]].
> This is because [[#Open Addressing|probing]] in worst-case scenario takes $n$ iterations to find an empty bucket, leading $O(n^{2})$ for rehashing.

---
## See Also
- [Good Article by Alejandro](https://medium.com/@alejandro.itoaramendia/the-hash-table-data-structure-a-complete-guide-27fb7ebed2ff)
