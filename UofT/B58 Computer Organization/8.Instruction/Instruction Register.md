# Instruction Register
The [[Instruction Register|instruction register]] takes in the $\text{32-bit}$ instruction fetched from memory, and reads the first $\text{6 bits}$(known as `opcode`) to determine what operation to perform.

![image|350](https://notes-media.kthiha.com/Instruction-Register/b3a993ea3431a683eba8f8af3577dc68.png)

---
## Instruction Decoding
The instructions themselves can be broken down into sections that contain all the information needed to execute the operation.

![image|300](https://notes-media.kthiha.com/Instruction-Register/d84eb7066ad733ce993b48deccaf5d8c.png)

---
### Opcode
The first six digits of the instruction(the `opcode`) will determine the instruction type.

![image|200](https://notes-media.kthiha.com/Instruction-Register/a9c1cc1da9847bdee970673264b61199.png)

---
