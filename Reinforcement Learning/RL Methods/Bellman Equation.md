# Bellman Equation
#rl/bellman-equation

Note that when we calculate a [[Value-Based Methods|Value Function]], we need to calculate the value of all the future states, which is computationally expensive 

To solve this, we use a recursive method that consider each state as immediate reward $R_{t+1}$ + discounted value of following state $V(S_{t+1})$.

$$
V_{\pi}(s) = E_{\pi} [R_{t+1} + \gamma. V_{\pi}(S_{t+1}) | S_{t} = s]
$$

![[Bellman Equation.png]]

## See Also
- [[Value-Based Methods]]
