# Radar Cross Section
#radar/rcs

`RCS` tells you how detectable an object is by [[Radar]].

![RCS](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Sigma_invader_RCS.png/330px-Sigma_invader_RCS.png)

> It can be thought of as the area of an imaginary disk that would return the same echo power.

---
## Formal Definition
RCS $\sigma$ is defined as
$$
\sigma = \lim_{ R \to \infty } 
4\pi \ R^2 \frac{P_{r}}{P_{i}}
$$
where
- $R$ is the `distance` from radar to target
- $P_{r}$ is the `power received` back by radar
- $P_{i}$ is the `power incident` on the target

![Radar Cross Section|300](https://preview.redd.it/a-cool-guide-to-the-radar-cross-section-of-a-flying-object-v0-9zp2njsmk0gc1.jpeg?auto=webp&s=3a1e199d5caec5385f3576045978f1ba8de0dc9d)

---