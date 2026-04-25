# Design Patterns
`Design Patterns` are standard, reuseable solution to common software design pattern.

### Creational Patterns
These are `design patterns` that define how objects are created by helping increase flexibility.

- [[Abstract Factory]]
  Used for creating family of similar objects
- [[Factory Method]]
  Allow subclass to decide what object to create
- [[Builder]]
  Used to build complex objects step-by-step
- [[Object Pool]]
  Used to reuse expensive objects, instead of repeatedly reallocating memory
- [[Prototype]]
  Used to clone objects (maybe from registry)
- [[Singleton]]
  Used to enforce only 1 instance globally

### Structural Patterns
These are `design patterns` that help compose objects into bigger structure.
- [[Adapter]]
  Make incompatible interfaces work together.
- [[Bridge]]
  Used to enable different changes in a class
- [[Composite]]
  Treat object, and group of objects the same.
  Used for hierarchical, tree structures.
- [[Decorator]]
  Dynamically add behaviour, without editing the classes.
- [[Facade]]
  Provides simplified interface over the complex subsystem.
- [[Flyweight]]
  Share reuseable state objects to save memory.
- [[Proxy]]
  Placeholder to control access.
  Used for lazy loading, security and caching.
- [[Visitor]]
  Used to add new behaviour to a group of class without changing them

## Behavioral Patterns
These are `design patterns` that define clear interaction rules.
- [[Chain of Responsibility]]
  Used to pass a request through handlers, till it is handled
- [[Command]]
  Turn a request into object.
  Used for storing, queuing and undoing the request.
- [[Interpreter]]
  Niche use for defining grammar.
- [[Iterator]]
  Standard way to transverse a collection.
- [[Mediator]]
  Used for centralizing communication between different components
- [[Observer]]
  Used for objects to react to an event
- [[State]]
  Used for implementing state-specific behaviours
- [[Strategy]]
  Swap algorithms at runtime
- [[Template Method]]
  Define skeleton algorithm, which can be overriden by subclasses.

Outside of these, there is also [[Dependency Injection]].

---