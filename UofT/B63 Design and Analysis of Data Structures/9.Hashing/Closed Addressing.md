# Closed Addressing
We can resolve a [[Hashing|collision]] in [[Hashing|hashing]] by `chaining`.
Chaining means storing a linked list at each entry in the [[Hash Table|hash table]]. 

---
## Example
Suppose the following:
- $h(k_{1}) = h(k_{2}) = 1$
- $h(k_{4}) =h(k_{5}) = h(k_{9}) = 3$

Then,
![image|300](https://notes-media.kthiha.com/Closed-Addressing/7bbc5fe09c52e1146132e369940d7237.png)

Assume we can compute the [[Hashing|hash function]] $h$ in constant time. Then, 
- `insert()` takes $\theta(1)$ time.
- `delete()` takes $\theta(1)$ time.
- `search()` is a bit more complicated.

If $m(n-1) < |V|$, then any given [[Hashing|hash function]] will put at least $n$ key-values into some entry of the [[Hash Table|hash table]].

The worst case scenario occurs when $1$ entry has all the elements. Then, search will take $\theta(n)$ time.

---
## See Also
- [[Hashing]]
- [[Hash Function]]
- [[Hash Table]]
- [[Direct Addressing]]
- [[Closed Addressing]]
- [[Open Addressing]]
- [[Universal Hashing]]