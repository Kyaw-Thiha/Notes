# Bang-Bang Controller
#robotics/controller/bang-bang

`Bang-Bang Controller` is the simplest [[Controller]] which try to push the state towards the desired reference value.

![Bang-Bang Controller|500](https://www.chi.camp/wp-content/uploads/2016/05/bangbang_control.png)

---
`Simple Controller`
The simplest controller can be thought of as
```c
float acceleration = 0;
float brake = 0;

if (velocity < reference) {
	acceleration = 1;
} else if (velocity > reference) {
	brake = 1;
}
```

Note that when the velocity is actually at the reference, the car will be either accelerating or decelerating most of the time.

---
`Tolerance Bang-Bang Controller`
The tolerance value can be added as
```c
float error = reference - x;

if (error < -threshold) {
	u = a1;
} else if (error > threshold) {
	u = a2;
} else {
	u = 0;
}
```
where
- `threshold` is a small constant that determines how big the `error` value can be
- `u` is a control signal
- `a1` and `a2` are suitable control amounts

Note that this does not have a sort of dampening effect.

In the car example, the controller should apply the accelerator more strongly if current velocity is far away from desired reference, but be more gentle as it approaches the reference.
To achieve this, we need to use [[PID Controller]].

---
## See Also
- [[Controller]]
- [[PID Controller]]
