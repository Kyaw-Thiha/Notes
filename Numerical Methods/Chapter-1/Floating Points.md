#math #numerical-methods/floating-point
## Floating-Point System

- **β**: Base (or radix)  
- **t**: Precision  
- **[L, U]**: Exponent range  

By definition, any number $x$ in the floating-point system is represented as:

$$
x = \pm \left( d_0 + \frac{d_1}{\beta} + \frac{d_2}{\beta^2}  + \cdots + \frac{d_{t-1}}{\beta^{(t-1)}}  \right) \, \beta^e
$$

where  

- $0 \leq d_i \leq \beta - 1,\; i = 0, \dots, t-1$  
- $L \leq e \leq U$

A good way to understand is to think of it as 3 different components:
- $\pm$ is represented by the first bit
- $(d_0 + d_1 \beta^{-1} + d_2 \beta^{-2} + \cdots + d_{t-1} \beta^{-(t-1)})$ is the second part called $\text{mantissa}$. This represents the integer + decimal part
- $\beta^e$ represents the 'scaling' part. Note that $e$ is not Euler constant, but rather any specific constant

If it is still hard to understand, think of examples in the base-2 system.

## Normalization
These are rules to essentially ensure every numbers are represented uniquely in the given floating point system.

This helps maximize the no. of numbers represented.
- Ensure $d_{0}$ is always non-zero (except for zero)
- This will lead to $1\leq m <\beta$ (its 1 since $min(d_{0}) = 1$), where $m$ is the mantissa (the 2nd component)

Alternatively,
- Ensure $d_{0}$ is always zero and $d_{1} \neq 0$
- This will lead to $\beta^{-1} \leq m < 1$ (its $\beta^{-1}$ since $min(d_{0} + d_{1}) = 0 + \beta^{-1}$ while its 1 since $d_{0} = 0$)

## Properties
No. of numbers represented: $2(\beta-1).\beta^{t-1}.(U-L+1) + 1$
- $2$ is for $\pm$ 
- $(\beta-1)$ is for leading digit
- $\beta^{t-1}$ is for the remaining digits
- $(U - L +1)$ is for exponent component 
- $+1$ is for zero


Underflow is smallest positive floating no. while overflow is largest positive floating no.
- $\text{Underflow Level} = \text{UFL} = \beta^L$
- $\text{Overflow Level} = \text{OFL} = \beta^{U+1}.(1-\beta^{-1})$

## Rounding
$fl(x)$ = approximation of real number $x$ 

- Chop: $\text{base-}\beta$ expansion of real num $x$ is truncated after $t-1$
- Round to nearest: $fl(x)$ rounds to nearest $x$. Round to even when tied.

Decimal-to-binary and binary-to-decimal conversions can be source of error.

## Machine Precision
Also called machine epsilon, it represents the maximum possible relative error from floating-point representation.
$$
\frac{|fl(x)-x}{x}| \leq \epsilon_{mach}
$$

- When using chopping method, $\epsilon_{\text{mach}} = \beta^{1-t}$
- When using rounding to nearest, $\epsilon_{\text{mach}} = \frac{1}{2}.\beta^{1-t}$

## Subnormals
If we want to represent $0< \text{num} \leq 1$, we can do so by denormalising the floating point.
Note that this has no effect on the machine epsilon tho.

## Exceptional Values
- $Inf$: Infinity, such as when dividing by 0.
- $NaN$: Not a number, such as indeterminate operations like $\frac{Inf}{Inf}$ 

## Floating-Point Arithmetic
- For **Sum** and **Subtraction**: the exponents must match before the mantissa can directly be added/subtacted.
- For **Multiplication**: exponents are summed and mantissa multiplied
- For **Division**: exponents are subtracted, and mantissa divided

**Underflow**: When result gets too small, it can be rounded off to zero.
**Overflow**: When result (mantissa in sum/subtract) gets too big, it will lead to error.
### Examples (Base-10, precision simplified)
#### 1. Sum
$x_1 = 1.23 \times 10^2$  
$x_2 = 4.56 \times 10^1 = 0.456 \times 10^2$  

Now exponents match:  
$1.23 \times 10^2 + 0.456 \times 10^2 = 1.686 \times 10^2$  

---

#### 2. Subtraction
$x_1 = 5.00 \times 10^3$  
$x_2 = 2.50 \times 10^3$  

Same exponents, subtract mantissas:  
$(5.00 - 2.50) \times 10^3 = 2.50 \times 10^3$  

---
#### 3. Multiplication
$x_1 = 2.0 \times 10^2$  
$x_2 = 3.0 \times 10^3$  

Multiply mantissas, add exponents:  
$(2.0 \times 3.0) \times 10^{2+3} = 6.0 \times 10^5$  

---
#### 4. Division
$x_1 = 6.0 \times 10^5$  
$x_2 = 2.0 \times 10^2$  

Divide mantissas, subtract exponents:  
$(6.0 / 2.0) \times 10^{5-2} = 3.0 \times 10^3$  

## Cancellation
When computing a small value by subtracting large quantities, rounding error can dominate the result.
This is called `catastrophic cancellation` or `subtractive cancellation`.

For example, let $\epsilon < \epsilon_{\text{mach}}$ a small positive number.
Then, $(1+\epsilon) - (1-\epsilon) = 1 - 1 = 0$.
However, the true value should have been $2\epsilon$.