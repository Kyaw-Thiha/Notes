# Big-O Notes

- To find [[Big-O]], overestimate the steps it take.
  To find [[Big-Omega]], find the worst input and underestimate the steps it take.
- $f \in O(g) \text{ and } f \in \Omega(g) \iff f \in \Theta(g)$
  $f \in O(g) \iff g \in \Omega(f)$
- If $f(n) \in O(g(n))$ and $g(n) \in O(h(n))$, then $f(n) \in O(h(n))$.
- If $\lim_{ n \to \infty } \frac{f(n)}{g(n)} \in [0, \infty)$, then $f(n) \in O(g(n))$.
  If $\lim_{ n \to \infty } \frac{f(n)}{g(n)} \in (0, \infty]$, then $f(n) \in \Omega(g(n))$.
  If $\lim_{ n \to \infty } \frac{f(n)}{g(n)} \in (0, \infty)$, then $f(n) \in \Theta(g(n))$

---
### Logarithmic Rules

1. $\log_{b}(xy) = \log_{b}(x) + \log_{b}(y)$
2. $\log_{b}\left( \frac{x}{y} \right) = \log_{b}(x) - \log_{b}(y)$
3. $\log_{b}(x^{y}) = y. \log_{b}(x)$
4. $\log_{b}(x) = \frac{1}{\log_{x}(b)}$
5. $\log_{b}(x) = \frac{\log_{c}(x)}{\log_{c}(b)}$

### Exponential Rules
1. $a^{n}.a^{m} = a^{n+m}$
2. $\frac{a^{n}}{a^{m}} = a^{n-m}$
3. $a^{n}.b^{n} = (a.b)^{n}$
4. $\frac{a^{n}}{b^{n}} = (\frac{a}{b})^{n}$
5. $(b^{n})^{m} = b^{n.m}$

---
## See Also
- [[Time Complexity]]
- [[Big-O]]
- [[Big-Omega]]
- [[Big-Theta]]
- [[Big-O Example]]
- [[Using limits for Big-O]]
- [[Logarithmic Rules]]
- [[Exponential Rules]]