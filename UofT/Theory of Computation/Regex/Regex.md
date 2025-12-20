# Regex

https://www.cs.toronto.edu/~vassos/b36-notes/notes.pdf
Page: 190
## Notations
- An `alphabet` is a set $\sum$ whose elements are called `symbols`
- A `string` is a finite sequence of symbols from specific `alphabet` $\sum$
- Empty sequence is a `string` denoted as $\epsilon$.
- The set of all strings over $\sum$ is denoted $\sum^*$.

### Operators
- Length (denoted $|x|$): Number of elements in the sequence $x$
- Concatenation (denoted $xy \text{ or } x \circ y$): Joining $y$ after $x$.
  Eg: $abaa \circ babb = abaababb$
- Reversal (denoted $(x)^R$): Listing elements in $x$ in reverse
  Eg: $(abab)^R = (baba)^R$

### Power
For any $k \in N$, we denote $k^{th}$ power of string as $x^k$ by induction on $k$
$$
x^k =
\begin{cases}
\epsilon, & \text{if } k = 0, \\
x^{k-1} \circ x, & \text{if } k > 0.
\end{cases}
$$
### Equality
Two strings $x$ and $y$ are equal (denoted $x = y$) if 
- $|x| = |y|$ and 
- $\forall i \; (1 \leq i \leq |x|), \;\; x[i] = y[i]$ 
  (all symbols of $x$ are equal to their corresponding symbol on $y$)

### Substring
A string $x$ is `substring` if there exists $x'$ and $x''$ (which maybe $\epsilon$) such that $x'xx'' = y$.
A string $x$ is `proper substring` if there exists $x'$ and $x''$ (which CANNOT be $\epsilon$) such that $x'xx'' = y$.

### Prefix
A string $x$ is `prefix` if there exists $x'$ (which can be $\epsilon$) such that $xx' = y$.
A string $x$ is `proper prefix` if there exists $x'$ (which cannot be $\epsilon$) such that $xx' = y$.

### Suffix
A string $x$ is `prefix` if there exists $x'$ (which can be $\epsilon$) such that $x'x = y$.
A string $x$ is `proper prefix` if there exists $x'$ (which cannot be $\epsilon$) such that $x'x = y$.

## Semantic of Regexes
The set of string that matches the regex.
**Notation**: $L(R)$
- $L(\emptyset) = \emptyset$
- $L(\epsilon) = \{ \epsilon \}$
- $L(c) = \{ c \}$ for every $c \in \sum$

**Operators**
- $L((S + T)) = L(S) \ \cup \ L(T)$
- $L((ST)) = L(S)L(T)$
- $L(S*) = L(S)^*$

## Regular Definition
A language $L$ is `regular` $\iff$ there's a regex $R$ s.t $L(R) = L$.
