# Hash Function
There are $4$ [[Hash Function|hash functions]] that we will look at. 
They are:
- [[#Division Method]]
- [[#Multiplication Method]]
- [[#Polynomial Hash]]
- [[#FNV-1|Fowler-Noll-Vo(FNV-1)]]

---
## Division Method
Assume that each key is an integer.
$$
h(k) = k \ \text{mod} \ m
$$
where $h(k)$ will between $0$ and $m-1$.

> Although this is simple, it is susceptible to regular patterns in keys. This means that there will be a lot of collisions. 
> $\text{I.e}:$ The [[#Division Method]] is sensitive to the value of $m$.

To overcome this issue, we pick $m$ to be a prime number.

---
## Multiplication Method
$$
h(k) = [m \cdot \text{fraction}(k \cdot A)]
$$
In theory, we want to pick a real constant $A$ $s.t.$ $0 < A < 1$.

However in practice, we assume that each key is a $w\text{-bit}$ natural number. We define $A$ by picking a $w\text{-bit}$ constant $S$, $s.t.$ $0 < S < 2^{w}$, and letting 
$$
A = \frac{S}{2^{w}}
$$

Furthermore, we let $m=2^{p}$, for some $p$ $s.t.$ $0 \leq p < w$.
Then, 
$$
\begin{align}
h(k)
&= [m \cdot function(k \cdot A)] \\[6pt]
&= \left[ 2^{p} \cdot function\left( k \cdot \frac{s}{2^{w}} \right) \right] \\[6pt]
&= \left[ 2^{p} \cdot \left(\frac{(k.s) \  
\text{mod} \ 2^{w}}{2^{w}} \right) \right] \\[6pt]
&= \left[ \left(\frac{(k.s) \ \text{mod} \ 2^{w}} 
{2^{w-p}} \right) \right] \\[6pt]
\end{align}
$$

$(k.s) \ \text{mod} \ 2^{w}$ returns the lower $w$ bits of $(k.s)$. Then, dividing by $2^{w-p}$ returns the upper $p$ bits of the lower $w$ bits of $(k.s)$.

> We often want $A$ to be irrational, so we often use the golden ratio for $A$ and work backwards to define $S$.

---
## Polynomial Hash
[[#Polynomial Hash]] is used to hash string typed keys.

A string is a seq of chars encoded using `ASCII code`.
```c
E.g:  H   E   L   L   O
      72  69  76  76  79 
```

Given a string $S = x_{0}, \dots, x_{k-1}$ ,
$$
h(s)
= ( x_{0} \cdot a^{k-1} + x_{1} \cdot a^{k-2} 
+ \dots + x_{k-2} \cdot a + x_{k-1} ) \ \text{mod} \ m
$$
$a$ must be a  non-zero constant and $a \neq 1$.
$I.e:$ $a$ cannot be $0$ or $1$.

Some good values of $a$ are $33$, $37$, $39$, and $41$.

### Pseudocode
```python
c=0
for i in 0 ... k-1:
	c = c.a + x_i
c = c mod m
```

---
## FNV-1
[[#FNV-1]] is a family of [[Hash Function|hash functions]], one for each word size. 
$I.e:$ [[#FNV-1|FNV]] comes in $32$, $64$, $128$, $256$, $512$ and $1024$ bit flavours.

An advantage of [[#FNV-1]] is that it's very easy to implement. 
- You start with an initial [[Hashing|hash value]] called [[#FNV-1|FNV offset basis]].
- For each byte in the input, you multiply the [[Hashing|hash]] by the [[#FNV-1|FNV prime]].
- Finally, you [[Boolean Algebra|XOR]] it with the byte from the input.

They work by chopping the key into 8-bit words.

### Pseudocode for 32-bit FNV-1
![image|300](https://notes-media.kthiha.com/Hash-Function/0165a2c5ea3407223a0418dee996426d.png)

---
## FNV-1a
Like [[#FNV-1]] but you xor the [[Hashing|hash value]] with the byte from the input before you multiply [[Hashing|hash]] with the [[#FNV-1|FNV prime]].

### Pseudocode
![image|300](https://notes-media.kthiha.com/Hash-Function/d5dde3a648cffce53d18d557920fbff5.png)

[[#FNV-1a]] is recommended over [[#FNV-1]] for being more random and uniform.

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
