# PID Controller
#robotics/controller/pid 

[[PID Controller|PID Controller]] is a controller that uses $3$ terms:
- `Error`
- `Sum of previous T error values`
- `Difference between error and previous error`

![PID Controller|500](https://www.kevsrobots.com/assets/img/how_it_works/pid.jpg)

The 3 components are
- `Proportional`: $k_{p}$ $\times$ $e(t)$
- `Integral`: $k_{p}$ $\times$ $\int e(t) \ dt$
- `Derivative`: $k_{d}$ $\times$ $\frac{d}{dt} e(t)$

---
## P-Controller
This control the value by a proportional change to error only.
```c
while (loop) {
	float x = /* sensor input value */;
	float error = reference - x;
	
	u_p = k_p * error;
	u = u_p + u_d + u_i;
}
```

Its main drawback is its tendency to overshoot and oscillate.
This is due to inertia in the physical system.

![image|300](https://notes-media.kthiha.com/PID-Controller/f7f066c84049d132eefc1957279171ec.png)

The constant $k_{p}$ can be used to compensate it.
But small values of $k_{p}$ will lead to increased convergence time.
And large values of $k_{p}$ will create the oscillation problem.

To fix this, we use the [[#PD Controller]].

---
## PD Controller
Adding in the derivative term, we get
```c
while (loop) {
	float x = /* sensor input value */;
	float error = reference - x;
	float error_derivative = /* Rate of change in error */;
	
	u_p = k_p * error;
	u_d = k_d * error_derivative;
	u = u_p + u_d;
}
```
where $k_{d}$ is significantly smaller than $k_{p}$.

Note that 
$$
\text{input} \propto \frac{de}{dt}
$$
where $\frac{de}{dt}$ is how fast error is changing.

### Discrete Derivative
$$
\frac{de}{dt} = \frac{x_{-1} - x_{0}}{t}
$$
where 
- $x_{-1}$ is the initial state
- $x_{0}$ is the current/final state

### Analyzing Large Error
Suppose the error is large and positive.
Then, it is decreasing due to action of the controller.

Hence,
- `P-Term` $u_{p}$  provide large, positive signal
- `D-Term` $u_{d}$ will provide a control input with opposite sign to $u_{p}$ (since the error is decreasing)
- Since $k_{p} \gg k_{d}$, $| \ \text{diff} \ | < |\text{err|}$ so $u$ is still large, positive signal

### Analyzing Small Error
Suppose the system is approaching the reference value.
Then the error is still positive, but is small.

Hence,
- `P-Term` $u_{p}$  still provide positive signal
- `D-Term` $u_{d}$ will provide a control input with opposite sign to $u_{p}$ (since the error is decreasing)
- Since $| \ \text{diff} \ | > |\text{err|}$, $u_{d}$ can dampen, cancel or counter-act the effects of $u_{p}$

`Limitation`
On some situations, PD controller can achieve equilibrium at a state value that is close to reference, but actually zero.

To fix this, we use the `PID Controller.`

---
## PID Controller
Adding in the integral term, we get

```c
while (loop) {
	float x = /* sensor input value */;
	float error = reference - x;
	float error_difference = /* Rate of change in error */;
	float error_integral = /* Sum of previous T error values */;
	
	u_p = k_p * error;
	u_d = k_d * error_difference;
	u_i = k_i * error_integral;
	u = u_p + u_d + u_i;
}
```
The `integral term` is the integral/sum of past $T$ errors.

Suppose the error is very small (approaching reference).

![image|300](https://notes-media.kthiha.com/PID-Controller/44cf87f577cdd4b7afcd8d0d17f6702a.png)

Then, the `P-Term` $u_{p}$ and `D-Term` $u_{d}$ aren't providing sufficient control input to drive it towards $0$.
Hence, the accumulation of error will allow `I-Term` $u_{i}$ to provide the required push.

Note that the integral term has
$$
\text{input} \propto \int e \ dt \left( = \sum ^{t}_{t=0} e_{t} \Delta t \right)
$$

### Limitation
The optimum constants need to be found empirically.
Badly tuned PID controller will result in
- Failure to converge the system towards reference
- Oscillate too much
- Or be very slow to converge to referece

---
## See Also
- [[Controller]]
- [[Bang-Bang Controller]]
