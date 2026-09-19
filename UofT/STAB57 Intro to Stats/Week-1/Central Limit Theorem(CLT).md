# Central Limit Theorem(CLT)
> Sample mean will converge to normal distribution when enough sample is taken.

![image|700](https://notes-media.kthiha.com/Central-Limit-Theorem(CLT)/030302b07e1b69bcd3d07cca72980a9a.png)

Concisely, we could define as
$$
\frac{1}{n} \sum X_{i}
\xrightarrow{D} \mathbb{N}\left( E[X_{i}], \ 
\frac{V[X_{i}]}{n} \right)
$$

> [[Central Limit Theorem(CLT)|CLT]] says if we keep increasing the value of $n$ & keep drawing these distributions, at one point it will start to look like a normal density.

---
## Naive Words
- A random variable $X$ can follow some distribution with mean $\mu$ and variance $\sigma^{2}$. 
- If we pick a fixed number of samples $n$ and calculate the sample mean repeatedly, those sample means will have a Normal Distribution with mean $\mu$ and variance $\frac{\sigma^{2}}{n}$.

---
## Formal Definition
Suppose $X_{1}, \ X_{2}, \ \dots$ is an $i.i.d$ sequence of random variables each having finite mean $\mu$ and finite variance $\sigma^{2}$. 
Let sample mean be
$$
X_{n} = \frac{1}{n} \sum^{n}_{i=1} X_{i}
$$
Then according to [[Central Limit Theorem(CLT)|Central Limit Theorem]] as $n\to \infty$,
$$
\boxed{ \
X_{n} \xrightarrow{D} N\left( \mu, \ 
\frac{\sigma^{2}}{n} \right) \ }
$$
or
$$
\boxed{ \ 
\frac{X_{n} - \mu}{\sigma/\sqrt{ n }}
\xrightarrow{D} \mathbb{N}(0,1) \ }
$$

---
