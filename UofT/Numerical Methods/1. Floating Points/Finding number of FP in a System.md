# Finding number of FP in a System

Let:
- β = base (usually 2 or 10)  
- p = number of digits in the mantissa/significand (precision)  
- Exponents allowed for normal numbers:  
$e \in [e_{\min}, e_{\max}]$ (inclusive)  
- sign ∈ { + , − }


`Normalized significands`
Have a nonzero leading digit, so the count of distinct significands is:

$$
(\beta - 1)\,\beta^{p - 1}
$$

`Number of exponent choices`
$$
(e_{\max} - e_{\min} + 1)
$$

`Total number of normalized finite numbers`
With a sign bit, the number of normalized finite numbers is:

$$
N_{\text{norm}} = 2\,(\beta - 1)\,\beta^{p - 1}\,(e_{\max} - e_{\min} + 1)
$$

