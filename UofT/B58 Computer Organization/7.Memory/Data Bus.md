# Data Bus
Communication between components takes place through groups of wires called a [[Data Bus|bus]] (or [[Data Bus|data bus]]).

- Multiple components can read from a bus, but only one can write to a bus.
- Each component has a [[Tri-State Buffer|tri-state buffer]] that feeds into the [[Data Bus|bus]]. When not reading or writing, the [[Tri-State Buffer|tri-state buffer]] drives `high impedence` to the bus.