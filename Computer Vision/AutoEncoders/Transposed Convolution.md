# Transposed Convolution
Unlike [[Convolution Layer|convolution]] which maps $\text{K} \times \text{K}$ to $1$, [[Transposed Convolution|transposed convolution]] maps $1$ pixel to $\text{K} \times \text{K}$.

![300](https://miro.medium.com/v2/resize:fit:1400/1*YwVviBiy2qAp0CwS5CDwmA.gif)

Its output dimension is
$$
o = (i-1) \times s + (k-1) - 2p + op + 1
$$
where
- $o$ is the output dimension
- $i$ is the input dimension
- $s$ is the stride
- $k$ is the kernel size
- $p$ is the padding
- $op$ is the output padding.

---
## How it works
- Take each pixel of input image
- Multiply each value of [[Convolution Layer|kernel]] with input pixel to get weighted kernel
- Insert it in output to create an image
- Where output overlaps, sum them

![250](https://i.redd.it/oooaoghhfaj91.png)

---
## Padding
The effect of padding is opposite to what happens in [[Convolution Layer|convolution layers]]:
- compute output as normal
- remove rows and columns around the perimeter

![image|400](https://notes-media.kthiha.com/Transposed-Convolution/e2b6fac7ad1d055df2a62c52544e15d2.png)

---
## Output Padding
Note that in [[Convolution Layer|convolution layer]], when stride $> 1$, it maps multiple input shapes to the same output shape.

**Eg**: Both $7 \times 7$ and $8 \times 8$ maps to output of $3 \times 3$ for kernel size $3 \times 3$ and stride$=2$.

So when applying [[Transposed Convolution|transposed convolution]], it is ambiguous which output shape it should return.

> [[#Output Padding]] is provided to resolve this ambiguity by effectively increasing the calculated output shape on one side. Note that it does not actually add zero-padding on the output.

---
## Stride
Increasing the stride results in an increase in the upsampling effect.

![image|500](https://notes-media.kthiha.com/Transposed-Convolution/b51087388288f9265b52d6d4123cbd5b.png)

Note that this is opposite to how stride works in [[Convolution Layer|convolution]].

---
## See Also
- [[Convolutional CVAE]]
- [[Convolution Layer]]
- [[Convolution Operator]]