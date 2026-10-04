# Binary Multiplication
## Algorithm
### Binary Representation
 We can represent a binary $X$ as
 $$
X: \underbrace{\boxed{ \ X_{1} \ }}_{\frac{n}{2}}
\underbrace{\boxed{ \ X_{0} \ }}_{\frac{n}{2}}
$$
Hence, we get that
$$
X = X\cdot 2^{n/2} + X_{0}
$$

---
### Binary Representation
Likewise, we can represent a binary $Y$ as
 $$
Y: \underbrace{\boxed{ \ Y_{1} \ }}_{\frac{n}{2}}
\underbrace{\boxed{ \ Y_{0} \ }}_{\frac{n}{2}}
$$
Getting us
$$
Y = Y\cdot 2^{n/2} + Y_{0}
$$

---
### Multiplication Improved Representation
Multiplying the binaries $X$ and $Y$ can be represented in
$$
X \cdot Y = X_{1}Y_{1}2^{n} + (X_{1}Y_{0} + X_{0}Y_{1}) \cdot 2^{n}
+ X_{0}Y_{0}
$$
which has a time complexity of
$$
T(n) = 4 \cdot T\left( \frac{n}{2} \right) + c \cdot n
$$
which is [[Divide & Conquer]] time complexity formula of
- $a=4$
- $b=2$
- $d=1$

Given $4>2^{1}$, we get $T(n) = \Theta(n^{\log_{2} 4})$.

---
### Even Better Representation
We can better represent this by 
$$
X \cdot Y = X_{1}Y_{1}2^{n} + ((X_{1} + X_{0})  \cdot (Y_{1} + Y_{0})
- X_{1}Y_{1} - X_{0}Y_{0}) \cdot 2^{n/2} + X_{0}Y_{0}
$$
which gives us the time complexity of
$$
T(n) = 3 \cdot T\left( \frac{n}{2} \right) + c n
$$
which is
- $a=3$
- $b=2$
- $d=1$

Given $3 > 2^{1}$, we have $T(n) = \Theta(n^{\log_{2}(3)})$.

---
## History
- **1963**: $O(n^{\log_{2}(3)})$
- **Schonhage & Strassen 1971**: $O(n \log n \cdot \log(\log n))$
- **Harvey & Vander Theorem 2019**: $O(n \log n)$

---