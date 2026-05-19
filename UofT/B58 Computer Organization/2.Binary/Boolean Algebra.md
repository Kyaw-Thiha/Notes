## Axioms of Boolean Algebra
![image|400](https://notes-media.kthiha.com/Binary-Algebra/37a02ab2c2ce0143106771d4e571d27d.png)

These axioms follow the principle of duality.
If we swap $0$s and $1$s, and `AND` and `OR`, the statement are still correct.

Use the prime symbol on the axiom number to indicate duality.

---
## Theorems of One Variable
![image|400](https://notes-media.kthiha.com/Binary-Algebra/7b666312db267e0b9433f2955869d879.png)

---
## Theorems of Two Variables
![image|400](https://notes-media.kthiha.com/Binary-Algebra/0d77b4795f3189b2978f68bb4c6e6bda.png)
![image|400](https://notes-media.kthiha.com/Binary-Algebra/97f8ac0dceacbfd8047f79325221052e.png)

---
## Illegal and Floating Values
We need to handle illegal and floating values.

### Illegal Value X
> Occurs if it is being driven to both $0$ and $1$ at the same time.

This could occur when two nodes are combined without a gate.
This situation is called **contention**.

The [[Electricity|voltage]] here is between $0$ and $V_{DD}$.
$X$ may also represent a value that wasn't initialized.

---
### Floating Value Z
This indicates a node that is neither `HIGH` nor `LOW`.
It may be $0$,$1$, or in between.

It doesn't mean there is error.
If another circuit element drives the node back to a valid logic value, then $Z$ indicates no error.

#### Causes
Causes include
- Forgot to connect a [[Electricity|voltage]] to circuit input
- Assume that an unconnected input is the same as an input with a value of $0$

#### Example
A wire is at $Z$ if it isn't connected to power or ground.
Here's an example:
![image|300](https://notes-media.kthiha.com/Boolean-Algebra/fb76152c3bd986e7c6402f64840a6750.png)

---
### Tristate Buffer
The buffer has three possible output states:
- `HIGH`$(1)$
- `LOW`$(0)$
- `FLOATING`$(Z)$

Suppose $X$ is input, $Y$ is output and $E$ is enable.
When enable is `TRUE`, the [[#Tristate Buffer|tristate buffer]] acts as a simple buffer. When `FALSE`, the output is allowed to float.

---

