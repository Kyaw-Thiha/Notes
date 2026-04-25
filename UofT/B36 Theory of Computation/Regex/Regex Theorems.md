# Regex Theorems

## Theorem 7.2
Start with empty string.
Then, you can repeatedly append symbols from $\sum$ to it to build all strings.

Let $S_{\sum}$ be the smallest set such that
- Basis: $\epsilon \in S_{\sum}$
- Induction Step: If $x \in S_{\sum}$ and $a \in \sum$, then $xa \in S_{\sum}$.

## Theorem 7.3
Recusively applying `Definition 7.2` yields $S_{\sum} = \sum^*$ .

> `Defintions 7.2 and 7.3` are useful as it allows as to prove  properties using structural induction.
> 1. Prove a property for base case $\epsilon$
> 2. Assume it holds for string $x$.
> 3. Show it holds for $xa$.

### Example
Defining `reversal` $(x)^R$ recursively.

**Basis**: If $x = \epsilon$, then $(x)^R = \epsilon$.
Reversing empty string gives empty string.

**Induction Step**: 
If $x = ya$, where $y \in \sum^*$ and $a \in \sum$ ($y$ is string & $a$ is symbol)
Then, 
$$
(x)^R = (ya)^R = a(y)^R
$$
Reversing a string ending in $a$ means putting $a$ in front, and reversing the rest of the string.

## Theorem 7.4 (Reversal)
For all strings $x \in \sum$ and $y \in \sum$,
$$
(xy)^R = (y)^R(x)^R
$$
The reverse of string concatenation, is the concatenation of individual reversed strings.

### Proof
> **WTS:**  $P(y): (xy)^R = (y)^R(x)^R$

**Basis**: Let $y = \epsilon$
- $(xy)^R = (x\epsilon)^R = (x)^R$
- $(y)^R(x)^R = \epsilon(x)^R = (x)^R$

In other words,
$$
\begin{align*}
(xy)^R &= (x\epsilon)^R && \text{since $y = \epsilon$, in this case} \\
       &= (x)^R          && \text{since $w\epsilon = w$, for any string $w$} \\
       &= \epsilon (x)^R && \text{since $\epsilon w = w$, for any string $w$} \\
       &= (y)^R (x)^R    && \text{since $(y)^R = \epsilon$, in this case}
\end{align*}
$$


**Induction Hypothesis**
Suppose $P(y')$ holds.
**WTS**: $P(y'a)$ for any $a \in \sum$.

Take an arbitrary $x$.

$$
\begin{align*}
(xy)^R &= (xy'a)^R && \text{since $y = y'a$, in this case} \\
       &= a(xy')^R && \text{by the recursive definition of reversal} \\
       &= a\big((y')^R(x)^R\big) && \text{by the induction hypothesis} \\
       &= \big(a(y')^R\big)(x)^R && \text{by associativity of concatenation} \\
       &= (y'a)^R(x)^R && \text{by the recursive definition of reversal} \\
       &= (y)^R(x)^R && \text{since $y = y'a$, in this case}
\end{align*}

$$
Thus, $P(y'a)$ holds.

## Theorem 7.5 (Language)
A `language` is any subset of $\sum^*$
Eg: Suppose $\sum = \{ a, b \}$.
Then,
- The set $\{ ab, ba, bbb \}$ is a language.
- The set of all strings with even number of $a$ is a language.
- $\sum^*$ itself is a language.

Note that $\emptyset$ and $\{ \epsilon \}$ are different languages since
- $\emptyset$ contain no strings
- $\{ \epsilon \}$ contains exactly one string (empty string)

Note that even though individual string is finite, the language can be infinite.
For eg: set of binary strings $\{ 0, 1 \}^*$ is infinite.

