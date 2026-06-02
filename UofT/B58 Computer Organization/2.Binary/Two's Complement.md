# Two's Complement
To represent a negative number in [[binary]], 
- take the positive
- invert all bits,
- and add $1$

![|300](https://i.ytimg.com/vi/sJXTo3EZoxM/maxresdefault.jpg)

---
## Arithmetic
The [[two's complement]] can be used in arithmetic without any problems.
- a positive plus positive is a positive.
- a negative plus negative is a negative.

To subtract, add the negative.

To take the absolute value of a [[two's complement]] negative number, subtract $1$ and invert the bits.

---
- The largest positive $8$-bit signed number is 
$$
0111 \ 1111 = 127
$$
- The largest negative $8$-bit signed number is 
$$
1000 \ 0000 = -128
$$
- The [[Binary|binary form]] of 8-bit signed integer $-1$ is
$$
1111 \ 1111
$$
- For a $n$-bit signed integer, there are $2^{n}$ possible values
	- $2^{n-1}$ are negative numbers $(\text{e.g: 8-bit, -1 to -128})$
	- $2^{n-1}-1$ are positive numbers $(\text{e.g: 8-bit, 1 to 127})$
	- and a zero

---


