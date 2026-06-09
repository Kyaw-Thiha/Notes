# Finite State Machine
A [[Finite State Machine|finite state machine]] is an abstract model that captures the operation of a [[Sequential Circuit|sequential circuit]].

It can be defined by
- A finite set of `states`
- A finite set of `transitions` between states, triggered by inputs to the [[Finite State Machine|state machine]]
- `Output values` that are associated with each state or each transition depending on the machine.
- `Start` and `end state` of a machine.

---
## Designing with flip-flops
[[Sequential Circuit|Sequential circuits]] are the basis for operations that require the circuit to remember the past values such as memory and instruction processing.
- These past values are also called states.
- If we need to describe the relation between the current state, and the next side, use combinational circuit.

---
### State Example
With counters, each state is the current number stored in the counter.
![image|350](https://notes-media.kthiha.com/Finite-State-Machine/86f1f89005cb7af7f6c053dcfce734d8.png)
On each clock tick, the circuit transitions from one state to the next, based on the inputs.

---
## State Table
The [[Finite State Machine|state table]] helps illustrate how the states of the circuit change with various input values.
![image|150](https://notes-media.kthiha.com/Finite-State-Machine/4c782a07801a77795ced9300af9773bb.png)
Note that transitions are understood to take place on the clock ticks.

Here, we can see the actual [[Flip-Flop|flip flop]] values instead of state labels.
![image|150](https://notes-media.kthiha.com/Finite-State-Machine/94a774eb6251e66f1da4a70578663419.png)
Note that [[Flip-Flop|flip-flop]] values are both inputs and outputs of circuit here.

---
## FSM Design
The design steps for a [[Finite State Machine|finite state machine]] are:
- draw the [[Finite State Machine|state diagram]]
- derive the [[Finite State Machine|state table]] from the [[Finite State Machine|state diagram]]
- assign [[Flip-Flop|flip-flop configuration]] to each state
- redraw [[Finite State Machine|state table]] with [[Flip-Flop|flip-flop values]]
- derive combinational circuit for output and for each [[Flip-Flop|flip-flop input]]

---
### Example: Sequence Recognizer
> Recognize a sequence of input values, and raise a signal if that input has been seen.

Example: Three high values in a row.
Assumes single input $X$ and a single output $Z$.

---
#### Step-1: State Diagram
In this case, states are labelled with the most recent three input values.
![image|150](https://notes-media.kthiha.com/Finite-State-Machine/6bde0f14327de37092ba8520cf6e8914.png)
Transitions between states are indicated by the values on the transition arrows.

---
#### Step-2: State Table
![image|150](https://notes-media.kthiha.com/Finite-State-Machine/e0192f542fef07774d79b37862cfad6b.png)
Make sure the [[Finite State Machine|state table]] lists
- all states in the [[Finite State Machine|state diagram]]
- all possible inputs at that state

---
#### Step-3: Assign Flip-Flops
Recall that a single [[Flip-Flop|flip-flop]] can store two values $(0 \text{ and } 1)$.
- $\text{One flip-flop} \to 2 \text{ states}$
- $\text{Two flip-flop} \to 4 \text{ states}$
- $\text{Three flip-flop} \to 8 \text{ states}$
- $n \ \text{flip-flop} \to 2^{n} \text{ states}$

Given $n$ states needed, we need $\text{ceiling}(\log_{2} n) \text{ flip-flops}$.

In this case, we need to store $8$ states.
![image|100](https://notes-media.kthiha.com/Finite-State-Machine/6bde0f14327de37092ba8520cf6e8914.png)
Hence, we need $8 \text{ states} = \log_{2} 8 \text{ flip-flops} = 3 \text{ flip-flops}$.

For now, we shall assign a [[Flip-Flop|flip-flop]] to each digit of the state names in the [[Finite State Machine|FSM]] & [[Finite State Machine|state table]].
![image|250](https://notes-media.kthiha.com/Finite-State-Machine/ae4fb3268b9ab0ed997fb199127d96c0.png)

---
#### Step-4: State Table
Mapping states to [[Flip-Flop|flip-flop values]], we get
![image|150](https://notes-media.kthiha.com/Finite-State-Machine/b1c13a9260c82b404220b557febe9d8b.png)
Note that this is NOT the only mapping from state to flip-flop.

---
#### Step-5: Circuit Diagram
- [[Karnaugh Map(K-Map)|K-Map]] for $F_{2}$:
![image|150](https://notes-media.kthiha.com/Finite-State-Machine/58c6a9fb0eb3adb6e6960c472655983e.png)
- [[Karnaugh Map(K-Map)|K-Map]] for $F_{1}$
![image|150](https://notes-media.kthiha.com/Finite-State-Machine/43f20d9493bced7f3e6c78f3007ca78e.png)
- [[Karnaugh Map(K-Map)|K-Map]] for $F_{0}$
![image|150](https://notes-media.kthiha.com/Finite-State-Machine/be4cd6a624003702ea6d957e44a2a1de.png)

Based on these [[Karnaugh Map(K-Map)|K-Maps]], we get this circuit

![image|300](https://notes-media.kthiha.com/Finite-State-Machine/391ebfd218a835c19724da5df45a1d8c.png)
![image|300](https://notes-media.kthiha.com/Finite-State-Machine/20be3a9a419993bc0f4594d57d34ebbc.png)

---
## Moore Machine vs Mealy Machine
There are two ways to derive the circuitry needed for the output values of the [[Finite State Machine|state machine]]:
- `Moore Machine`
  The output of [[Finite State Machine|FSM]] depends solely on the current state.
  (based on entry actions.)
- `Mealy Machine`
  - The output of the [[Finite State Machine|FSM]] depends on the state and the input. (based on input actions)
  - Being in state $X$ can result in different outputs, depending on the input that causes that state.

---
## Timing and State Assignments Issue
If a recognizer is in state `011`, and gets a `0` as input, it moves to `110` as per the [[Finite State Machine|state machine]].
![image|150](https://notes-media.kthiha.com/Finite-State-Machine/4bd76daf4a5cc2f25a5b41e35366365f.png)
- First and last digits should change same time but they can't.
- If first [[Flip-Flop|flip-flop]] changes first, the state will change to `111`.
  The output will go `HIGH` for an instant, which is unexpected behaviour.
- If the second [[Flip-Flop|flip-flop]] changes first, it's fine since the intermediate state `010` does NOT cause unexpected behaviour.

### Solutions
- Whenever possible, make [[Flip-Flop|flip-flop]] assignments such that neighbouring states differ by at most one [[Flip-Flop|flip-flop]] value.
- If the intermediate states are unused in the [[Finite State Machine|state diagram]], you can set the output for these states to provide the output you need.

---