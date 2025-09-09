## Numerical Method Strategy
#math #numerical-methods
Replacing infinite processes with finite processes, such as replacing integrals or infinite
series with finite sums, or derivatives with finite difference quotients
• Replacing general matrices with matrices having a simpler form
• Replacing complicated functions with simple functions, such as polynomials
• Replacing nonlinear problems with linear problems
• Replacing differential equations with algebraic equations
• Replacing high-order systems with low-order systems
• Replacing infinite-dimensional spaces with finite-dimensional spaces

## Error
Total Error = Computational Error + Propagated Data Error

## Type of Errors
Computational Error = Truncation Error + Rounding Error

Truncation Error:
- Due to say converting infinite to finite series
- Replacing function by poly
- Terminating iterative sequence before convergence
Rounding Error:
- Due to inexactness in representation of numbers

Rounding error tends to dominate in algebraic problems with finite solution algorithms. 
Truncation error tends to dominate in problems of derivatives, integrals, and non-linear problems, which require infinity.

## Absolute vs Relative Error
- $\text{Absolute Error} = \text{Approximate Value} + \text{True Value}$
- $\text{Relative Error} = \frac{\text{Approximate Value}}{True Value}$

## Sensitivity & Conditioning
Sensitivity: How much the output change for a change in input.
$\text{Cond} = \frac{|\Delta\text{Solution}|}{|\Delta \text{Input Data}|} = \frac{|f(\hat{x})-f(x)/f(x)|}{|(\hat{x}-x)/x|}$

- **Conditional** refers to functions
- **Stability** refers to algorithms

## Backward Analysis
- Forward Error: $f(\hat{x}) - f(x)$
- Backward Error: $\hat{x} - x$

## Stability vs Accuracy
Stable: If result produced is relatively insensitive to approximations made during calculations.
Accuracy: How close results are to actual value.

Essentially, stability tells how close the result is to nearby results, while accuracy tells how close the result is to the actual result.

## Floating Points
Conversion to floating points can lead to loss of accuracy due to `truncation` or `rounding`.
Moreover, floating point arithmetic can lead to `overflowing`, `underflowing` and `catastrophic cancellation`.

[[Floating Points|Read More]]