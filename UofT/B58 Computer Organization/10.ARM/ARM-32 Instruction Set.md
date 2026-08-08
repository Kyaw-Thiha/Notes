# ARM-32 Instruction Set
Every [[ARM-32 Instruction Set|ARM-32 instruction set]] is a one 32-bit word.

---
## Instruction Skeleton

### Condition
![image|500](https://notes-media.kthiha.com/ARM-32-Instruction-Set/f5d0b0d205f010d900a8456e744df566.png)

The condition code decides if the instruction is even executed.

---
### Instruction Type
The bits $27$-$26$ defines what kind of instruction it is.

![image|500](https://notes-media.kthiha.com/ARM-32-Instruction-Set/c1c8532fb7a34522caad190c73782fb9.png)

- `00`: Data Processing [`ADD/SUB/CMP/MOV`] or [`MUL/SDIV/UDIV`]
- `01`: Load/Store
- `10`: Branch (immediate)
- `00`: Branch (register)

---
## Data Processing

### Immediate Format
![image](https://notes-media.kthiha.com/ARM-32-Instruction-Set/e77cf4fe9f6a97d0e1d741cbb6662f17.png)

Operation codes used in this course are
- `2`: `SUB`. $Rd = Rn - (Rm \mid Imm)$
- `4`: `ADD`. $Rd = Rn + (Rm \mid Imm)$
- `10`: `CMP`. $Rn - (Rm \mid Imm)$. To discard result set flags $S=1$.
- `13`: `MOV`. $Rd = (Rm \mid Imm)$. Rn is ignored.

### Register Format
![image](https://notes-media.kthiha.com/ARM-32-Instruction-Set/a92d5d2bf86b5eb36495af8772f5c420.png)

### Rotated Immediate Encoding
![image|400](https://notes-media.kthiha.com/ARM-32-Instruction-Set/a18a862c0e1a93db2ab1ca2a681c725d.png)

---
## Load/Store

![image](https://notes-media.kthiha.com/ARM-32-Instruction-Set/472354a688b9488519c3a77f784c46dd.png)

`Bit-L`:
- `1`: Load
- `0`: Store

`Bit-B`:
- `1`: Byte-sized Access
- `0`: Word-sized Access

`Bit-U`:
- `1`: address is $Rb + \text{Offset}$
- `0`: address is $Rb - \text{Offset}$

`Offset`: unsigned 12-bit byte offset

---
## Branch (Immediate)

![image](https://notes-media.kthiha.com/ARM-32-Instruction-Set/9b65167d5035792c95762fb08683060b.png)

- If `L=1`, `LR (R14)` is the address of the next instruction.
  I.e: it saves a return address.
- `PC` $:= (\text{address of the branch instruction}) + 8 + (\text{Offset << 2})$

---
## Branch (Register)

![image](https://notes-media.kthiha.com/ARM-32-Instruction-Set/444e16b30a1a18e2d9718e0360ed3c3b.png)

---
## Multiply

![image](https://notes-media.kthiha.com/ARM-32-Instruction-Set/e75d479139a0953e9bf7ce55cd7d765f.png)

---
## Divide

![image](https://notes-media.kthiha.com/ARM-32-Instruction-Set/f27f4e0a4a51cadb18711fe98c5f522c.png)

---
## See Also
- [Cheatsheet this note is based on](https://cmsweb.utsc.utoronto.ca/~leeedw12/cscb58/arm_cheat.pdf)
- [ARM Instruction Set](https://iitd-plos.github.io/col718/ref/arm-instructionset.pdf)
- [[Function Stack]]