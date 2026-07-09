# Control Unit
[[Control Unit]] is essentially a giant [[Finite State Machine]] synchronized to system-wide signals ([[Clocked SR Latch|clock]], `restn`).

![|250](https://cdn1.byjus.com/wp-content/uploads/2022/05/introduction-to-control-unit-1.png)

### Control Signals
Outputs the [[Control Unit|datapath control signals]]:
- `SelxA`,`SelAB`: controls [[Multiplexer|mux outputs]] ([[Arithmetic Logic Unit (ALU)|ALU inputs]])
- `ALUop`: controls [[Arithmetic Logic Unit (ALU)|ALU operations]]
- `LdRA`,`LdRB`: controls loading for [[Register|registers]] `RA`,`RB`

![image|250](https://notes-media.kthiha.com/Control-Unit/a2bea505df462535a65e595fb0897854.png)

Some architecture also output a `done signal` when the computation is complete.

---
## See Also 
- [[Arithmetic Logic Unit (ALU)]]
- [[Finite State Machine]]
- [[Register]]
- [[Multiplexer]]