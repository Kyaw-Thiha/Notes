# Hashing
[[Hashing]] is a mathematical process that turns any data into a short, fixed-length code.
![300](https://www.thesslstore.com/blog/wp-content/uploads/2018/12/Hashing-Example-1024x492.png)

---
## Introduction
Many applications require a dynamic set that supports only the dictionary operations of 
- `insert()`, 
- `search()` and 
- `delete()`

A [[Hash Table|hash table]] is an effective data structure for implementing dictionaries.

> **Complexity**:
> Although searching for an item in a [[Hash Table|hash table]] can take $\theta(n)$ time in the worst case, in practice [[Hashing|hashing]] performs extremely well. Under reasonable assumptions, the average time to search for an element in a hash table is $O(1)$.

---
## Hash Table
Suppose that the key-values of our elements come from a universe or set $U$. We can allocate a table or array of size $m$, where $m < |U|$.

Then, we can use a [[Hashing|hash function]] 
$$
h:U\to \{ 0, \dots, m-1 \}
$$ 
to decide where to store the element with key-value $x$.

> $x$ gets stored in position $h(x)$ of the [[Hash Table|hash table]].

### Example
Given the set of keys is the set of all integer values from $0$ to $2^{32}-1$, what is a possible [[Hash Function|hash function]] if $m=2^{20}$?

**Soln**:
We can use either `key mod m` or $\text{key} - m$.
The idea is that when we want to access key $k$, we look in $T[f(k)]$ instead of $T[k]$.

If $m < |U|$, then there must be some keys, $k_{1}$ and $k_{2}$ in $U$ $s.t.$ $k_{1} \neq k_{2}$, and yet $h(k_{1}) = h(k_{2})$. This is called a [[Hash Table|collision]].

---
## Collision Resolution
Suppose we have an `address book` and the $N$ section fills up. Where do we put the next entry?

**Soln**:
- Flip to the next page and have an overflow page. ([[Open Addressing]])
- Write a note explaining where to find the rest of the $N$ entries. ([[Closed Addressing]])

A [[Hash Table|collision]] occurs when a hash function maps $2$ keys to the some index. We will look at $2$ general solutions:
- [[Hash Table|Closed Addressing]]: Give explicit directions of next location.
- [[Open Addressing]]: Give a general rule of where to look next.

---
## Hash Function
There are $4$ [[Hash Function|hash functions]] that we will look at. 
They are:
- [[Hash Function#Division Method|Division Method]]
- [[Hash Function#Multiplication Method|Multiplication Method]]
- [[Hash Function#Polynomial Hash|Polynomial Hash]]
- [[Hash Function#FNV-1|Fowler-Noll-Vo(FNV-1)]]

---
## Problem with Hashing
The problem with [[Hashing]] are that when the set of keys are unknown, 
- we can no longer assume a uniform distribution and 
- regular patterns can be found,
- making any deterministic hashing scheme vulnerable to malicious slowdowns.

The solution to the above problem is to create a family of [[Hash Function|hash functions]], and to select one at random. This is called [[Universal Hashing|universal hashing]].

---
## See Also
- [Good Article by Alejandro](https://medium.com/@alejandro.itoaramendia/the-hash-table-data-structure-a-complete-guide-27fb7ebed2ff)
- [[Hashing]]
- [[Hash Function]]
- [[Hash Table]]
- [[Direct Addressing]]
- [[Closed Addressing]]
- [[Open Addressing]]
- [[Universal Hashing]]