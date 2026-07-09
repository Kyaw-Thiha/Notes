# Arithmetic Logic Unit
The [[Arithmetic Logic Unit (ALU)|ALU]] performs all the arithmetic operations and logical operations.

![|250](https://limeup.io/wp-content/uploads/2024/01/ALU-Definition.png)

![image|350](https://notes-media.kthiha.com/Arithmetic-Logic-Unit-(ALU)/51ce763db09d7d6c757495962ca9c08e.png)

---
### ALU Inputs
![image|200](https://notes-media.kthiha.com/Arithmetic-Logic-Unit-(ALU)/d6dac7e2979a8bc25adbf4cf41e7b256.png)

- `A` and `B` are operands.
- The `select bit`(`S`) indicates which operation is being performed. (`S2` is a `mode select bit`, indicating whether the [[Arithmetic Logic Unit (ALU)|ALU]] is in arithmetic or logic mode)
- The `carry bit` `Cin` is used in operations such as incrementing an input value or the overall result.

---
### ALU Outputs
![image|200](https://notes-media.kthiha.com/Arithmetic-Logic-Unit-(ALU)/d6dac7e2979a8bc25adbf4cf41e7b256.png)

- `V`: overflow condition
  The result of the operation could not be stored in the $n$ bits of `G`, meaning the result is incorrect.
- `C`: carry-out bit
- `N`: negative indicator
- `Z`: zero-condition indicator

---
### Arithmetic Component
Fundamentally, it is made of the [[Full Adder|adder/subtractor unit]].
![image|300](https://notes-media.kthiha.com/Arithmetic-Logic-Unit-(ALU)/456590abad73ab5bc1b5f0d874bf6ca4.png)

In addition to `addition` and `subtraction`, other operations can be performed by manipulating what is added to input `B`.
![image|300](https://notes-media.kthiha.com/Arithmetic-Logic-Unit-(ALU)/cff96a828a7c09faf96404110251120a.png)

#### Arithmetic Operation Selection
Based on the values of the `select bit` and the `carry bit`, we can perform any number of basic arithmetic operations.
![image|250](https://notes-media.kthiha.com/Arithmetic-Logic-Unit-(ALU)/98e6ee0a36946c64fbfa90ddb9a4166b.png)

![|150](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZ5rnnsTodKz4y28uNLdXXKwihQoU-8RukRCXybQXqse5p982hemavz_U&s=10)

---
### Logic Component
The [[Multiplexer|multiplexer]] is used to determine which block (logical or arithmetic) goes to the output.
![image|200](https://notes-media.kthiha.com/Arithmetic-Logic-Unit-(ALU)/8811246ad207ff60db15eb3fb2ce2397.png)
If $S_{2}=1$, then logic circuit block is activated.

---
### Full Command List
![|300](https://figures.semanticscholar.org/aca4baa396deaf8e5155654e35e6fb9e5163d876/4-TableII-1.png)

---
## See Also
- [[Control Unit]]
- [[Full Adder]]
- [[Multiplexer]]