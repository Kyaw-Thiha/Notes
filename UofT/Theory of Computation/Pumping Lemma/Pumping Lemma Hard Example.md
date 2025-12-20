# Pumping Lemma Palindrome Example

> Let $\Sigma = \{ 0, 1 \}$ and $L = \{ x \in \Sigma^*: x^R = x \}$
> Prove that `Palindrome` L is not regular.

By way of contradiction, suppose $L$ is regular.
Let $n$ be as in `Pumping Lemma`.

`Strategy here is to choose the simplest v`
Let $x = 0^n \ 1 \ 0^n$
Then, $x \in L$ by defn of L.
$|x| = 2n+1 \geq n$

By (i), (ii) and (iii) from [[Pumping Lemma]], $v = 0^j$ for some $j$ where $0 < j \leq n$.
By $(iv)$, pick $k=2$ (pump 2 times)
$u \ v^2 \ w = u \ v \ v \ w = 0^{n+j} \ 1 \ 0^n \notin L$, which is a `contradiction`

