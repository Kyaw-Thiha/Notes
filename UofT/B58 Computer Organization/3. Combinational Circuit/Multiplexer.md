# Multiplexer
A [[multiplexer]] is a combinational circuit that has many data inputs and a single output.

![|300](https://i.ytimg.com/vi/aQlF-9i3fAA/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLAe54_YIK69kZczfSAPUYxtKMS0fA)

---
### Details
A [[multiplexer]] equation looks as follows:
$$
Y = A\bar{R} + BR
$$
where
- $A$ only matters if $\bar{R}$
- $B$ only matters if $R$

![|300](https://circuitdigest.com/sites/default/files/inlineimages/u/Multiplexer-Simulation-using-IC4052.gif)

- An $a:b$ [[multiplexer]] accepts $a$ inputs and returns $b$ outputs.
- Given $M$ selectors, we will have $N=2^{M}$ inputs and $1$ output.

---
### Compacting Multiplexer

An $8:1$ [[multiplexer]] can be seen as having $8$ inputs.
The switch has $3$ bits, which determines which of the $8$ inputs should be reflected in the output.

![|300](https://i.ytimg.com/vi/b0z7YKKCCyY/maxresdefault.jpg)

For example, if switch's bit represent $010$, the second input from the top should be reflected in the output.


---
## See Also
- [GeeksForGeeks Tutorial](https://www.geeksforgeeks.org/digital-logic/multiplexers-in-digital-logic/)
- [[Demultiplexer]]