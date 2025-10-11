# Calculating Error Upper Bound

For any real operation $\circ \in \{+, -, \times, \div\}$,
$$
\mathrm{fl}(x \circ y) = (x \circ y)(1 + \delta),
$$
where $|\delta| \leq \varepsilon$ (machine precision).

Also, use the fact that $fl(x) = x.(1+\delta_{x})$

## Example
If we denote the computer's final result as
$$
\mathrm{fl}\!\left(\frac{x}{z} \cdot y\right),
$$
then what really happens inside is something like
$$
\mathrm{fl}\!\left( \mathrm{fl}\!\left( \frac{\mathrm{fl}(x)}{\mathrm{fl}(z)} \right) \cdot \mathrm{fl}(y) \right),
$$
and each $\mathrm{fl}(\cdot)$ introduces a new small relative error.
