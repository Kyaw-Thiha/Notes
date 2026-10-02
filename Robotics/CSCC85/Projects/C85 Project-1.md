## Sensor Redundancy
### Belief
- Main thruster adds acceleration along (sin a, cos a), and the side thrusters add it along ±(cos a, sin a).
- Prediction adds gravity, then integrates acceleration into velocity and velocity into position.
- Grows an uncertainty uncertainty every tick,

### Sensor Fusion
- Discontinuity rejection
- Smoothing: EMA filter.
- Stale detection: a sensor that returns the identical value for 100 ticks, or a position history that is flat, is treated as stuck and loses health.
- Gated fusion: 
$$
\frac{\alpha \times \text{health} \times \text{gate}}{1 + \text{uncertainty/tol}^{2}}
$$
where:
- gate = clamp(1 − |reading − belief| / (4 · tol), 0, 1)      [Based on Belief]
- health = 0.97 · health + 0.03 · gate      (kept between 0.05 and 1.0)   [Jumps, etc]
- uncertainty grows every tick, n shrnks when reading gets accepted

$$
belief_new = (1 − w) · belief + w · reading
$$

### Cross-Sensor Redundancy
- Velocity from position. Weight: $0.6$
- Y and VY from the laser.
- Angle from acceleration.

---
## Controller
Sets speed caps VXlim_g and VYlim_g that depend on:
- distance to the pad;
- how much the code trusts its position and velocity estimates
- whether thrusters are missing, since it can't brake sideways without side thrusters
- whether sonar is available to see terrain ahead.

### AllOk
- Rotate to upright.
- Fire the left or right thruster to chase a desired VX proportional to the distance to the pad.
- Fire the main thruster only when falling faster than VYlim_g
- With no sonar, use the laser to brake if the ground is close and isn't confirmed to be the pad.

### MainOnly
- A PID on X sets a desired $V_{x}$, and a second PID on $V_{x}$ sets the desired horizontal acceleration.
- The tilt angle comes from $\arcsin\left( \frac{a_{x}}{35} \right)$. It is capped at about 25° with no sonar, 40° with VX dead, and 8° at the pad.
- The main thruster power is chosen to keep vertical acceleration right at that tilt.
- If a falling lander is near terrain, lift takes priority over sideways thrust.

### Glide
- computes the lift it needs (up_needed) and the horizontal acceleration it wants.
- rotates so the working side thruster points along that combined vector.
- When near walls, an emergency escape pushes away from the closer wall, and the lift term fades as climb rate rises.
- Terminal level-out: within 35 px of the pad (or when the sonar reads under 28 px straight down), it rotates upright and drops

---
