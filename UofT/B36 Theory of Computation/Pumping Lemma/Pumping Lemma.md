If $L$ is a regular language, then there is an integer $n > 0$ with the property that:

For any string $x \in L$ where $|x| \ge n$, there exist strings $u, v, w$ such that:

1. $x = uvw$
2. $v \ne \epsilon$
3. $|uv| \le n$
4. $uv^k w \in L$ for all $k \in \mathbb{N}$

This is known as the `Pumping Lemma` for Regular Languages.
