# Reservoir Network

The dynamics are defined as
- $x_{t+1} = f(\mathbf{W} x_{t} + \mathbf{W}_{in} \ u_{t+1} + \mathbf{W}_{fb} \ \hat{y}_{t})$
- $y_{t+1} = g(\mathbf{W}_{out} \ \mathbf{x}_{t+1})$

`Reservoir network` must display the echo state property.
The reservoir will asymptotically wash out initial conditions.q
