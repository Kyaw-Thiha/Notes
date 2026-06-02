# Comparator
A [[comparator]] is a circuit that takes in $2$ input vectors, and determines if first is greater than, less than or equal to second.

---
## Basic Comparator

Consider two [[Binary|binary numbers]] $A$ and $B$.
![image|100](https://notes-media.kthiha.com/Comparator/f90d3dc97895828111cea57f6a9d3dd5.png)

The circuits for this would be
- A == B: $A \cdot B + \bar{A} \cdot \bar{B}$
- A > B: $A \cdot \bar{B}$
- A < B: $\bar{A} \cdot B$

---
## Two-Bit Comparator
Now consider $A$ and $B$ being two bits long.
![image|100](https://notes-media.kthiha.com/Comparator/8c548346efbe49b7df318b02098f33b2.png)

Then, we get
- A == B

$$
\underbrace{(A_{1} \cdot B_{1} + \bar{A}_{1} 
\cdot \bar{B}_{1})}
_{\text{Ensure values of bits 1 are the same}}
\cdot \underbrace{(A_{0} \cdot B_{0} 
+ \bar{A}_{0}\cdot \bar{B}_{0})}
_{\text{Ensure values of bits 0 are the same}}
$$

- A > B

$$
\underbrace{A_{1} \cdot \bar{B}_{1}}_{\text{Check if 1st bit satisfies}}
+ \underbrace{(A_{1} \cdot B_{1} + \bar{A}_{1} 
\cdot \bar{B}_{1})}_{\text{If not, check if first bits are equal}}
\cdot \underbrace{A_{0} \cdot \bar{B}_{0}}_{\text{then do 2nd bit comparism}}
$$

- A < B

$$
\underbrace{\bar{A}_{1} \cdot B_{1}}_{\text{Check if 1st bit satisfies}}
+ \underbrace{(A_{1} \cdot B_{1} + \bar{A}_{1} 
\cdot \bar{B}_{1})}_{\text{If not, check if first bits are equal}}
\cdot \underbrace{\bar{A}_{0} \cdot 
B_{0}}_{\text{then do 2nd bit comparism}}
$$

---
## Comparing Larger Numbers
As number gets larger, the [[Comparator|comparator circuit]] gets more complex. So, its sometimes easier to process the result of subtraction instead.

---