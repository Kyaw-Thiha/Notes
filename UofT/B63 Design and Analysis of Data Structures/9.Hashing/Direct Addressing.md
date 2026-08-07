# Direct Addressing
[[Direct Addressing|Direct addressing]] is a simple technique that works well if the number of keys is reasonably small.

Suppose that an application needs a [[Dynamic Array|dynamic set]] in which each element has a key drawn from the universe 
$$
U = \{ 0, 1, \dots, m-1 \}
$$
where $m$ is not too large.

Furthermore, assume that no two elements have the same key. To represent the [[Dynamic Array|dynamic set]], we can use an array or a `direct-address table`, denoted by 
$$
T \ [0, \dots, m-1]
$$

Each position or spot corresponds to a key in $U$. If the set contains no elements with key $k$, then 
$$
T[k]=\text{NULL}
$$

---
### Dictionary Operations
```python
insert(T, x):
	T[x.key] = x
```

```python
search(T, key):
	return T[key]
```

```python
delete(T, x):
	T[x.key] = NULL
```

Each operation takes $\theta(1)$ time in the worst case.

---
### Implementation
Suppose that we know the key-values are integers from $1$ to $k$. A simple and fast way to rep a dictionary is to 
- allocate an array size $k$, and 
- store an element with key $i$ in the $i^{th}$ cell of the array.

### Problem
The problem with [[Direct Addressing|direct addressing]] is that it only works well if the number of keys are small. If the number of keys is too big, then the array will become huge which is not space efficient.

---
### Examples

#### Example-1
Suppose we are reading a text file and want to store the freq of each letter of the text file. Why is this a good application of [[Direct Addressing|direct addressing]]?

**Soln**:
Since there are only $256$ ASCII characters, we can make an array with length of $256$. Furthermore, we can let the $i^{th}$ cell hold the hold the number of occurances of the $i^{th}$ ASCII character.

#### Example-2
Suppose we are reading a data file of 32-bit ints, and we want to track the frequency of each number. Why is this a bad application of [[Direct Addressing|direct addressing]]?

**Soln**: 
The array would have a size of $2^{32}$.
We can use a [[Hash Table|hash table]] to solve this issue.

---
## See Also
- [[Hashing]]
- [[Hash Function]]
- [[Hash Table]]
- [[Direct Addressing]]
- [[Closed Addressing]]
- [[Open Addressing]]
- [[Universal Hashing]]