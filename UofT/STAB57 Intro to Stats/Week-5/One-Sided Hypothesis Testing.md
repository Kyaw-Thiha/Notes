# One-Sided Hypothesis Testing
When [[Hypothesis Test|testing null hypothesis]] $H_{0}: \mu = \mu_{0}$ against $H_{1}: \mu \neq \mu_{0}$, we define our rejection region on both sides. 
![image|300](https://notes-media.kthiha.com/One-Sided-Hypothesis-Testing/84441960bca20d687f02f9c2c932777f.png)

When [[Hypothesis Test|testing null hypothesis]] $H_{0}: \mu = \mu_{0}$ against $H_{1}: \mu > \mu_{0}$, we define our rejection region on the right side only. 

![image|300](https://notes-media.kthiha.com/One-Sided-Hypothesis-Testing/5f0f4b852e700f40208b733810406d46.png)

When [[Hypothesis Test|testing null hypothesis]] $H_{0}: \mu = \mu_{0}$ against $H_{1}: \mu < \mu_{0}$, we define our rejection region on the left side only. 
![image|300](https://notes-media.kthiha.com/One-Sided-Hypothesis-Testing/c87035259f70aab18be58dd695ffac94.png)

---
## One-Side Hypothesis Testing
When testing $H_{0}: \mu = \mu_{0}$ against $H_{1}: \mu > \mu_{0}$, 
$$
\text{p-value} = 1 - \Phi\left( \frac{\bar{x} - \mu_{0}}{\sigma/\sqrt{ n }} \right)
$$
When testing $H_{0}:\mu = \mu_{0}$ against $H_{1}:\mu < \mu_{0}$, 
$$
\text{p-value} = \Phi\left( \frac{\bar{x} - \mu_{0}}{\sigma/\sqrt{ n }} \right)
$$
We can use this similar idea for $t, \ \chi^{2}$ or other tests.

---