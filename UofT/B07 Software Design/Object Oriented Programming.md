# Objected Oriented Programming

`OOP` is a programming paradigm that couples methods and their data together into objects, and treat a program as a collection of objects.

### Inheritance
A `specialized child` class inherits properties & methods from `general parent` class.
In Java, use the keyword `extends`

`Type-Casting`

For `upcasting`, it always possible to do so.
```java
Person p = new Employee();
```

For `downcasting`, you need explicit casting
```java
Person p = new Employee(); 

if (p instanceof Employee){
	Employee e = (Employee)p;
}
```
Note: `Casting` does not create new object.

### Overloading & Overriding
`Overloading`
Defining methods with `same name` but `different signature`

`Overriding`
Using `@override` to redefine methods of `same name & signature` from the parent.

### Common Methods
`super()`
Use to invoke the overloaded constructor or parent's constructor.

`super.methodName()`
Used to invoke method of the parent, especially in `overriden methods`

`equals()`
- `Reflexive`: $x = x$
- `Symmetry`: $x = y \implies y = x$
- `Transitive`: $x = y \ \cap \ y = z \implies x = z$
- `Consistent`: $x = y$ everytime
- For any non-null $x$, $x.equals(null)$ returns False

`hashCode()`
Returns the `hashcode` of the object.
- Comparing by `hashcode` is often faster than using `equals` which is why we have things like `HashTables`
- By default, it returns the `memory address`
- `hashCode()` should be overriden whenever we overrides `equals()`
- A good `hashCode` returns different values for two unequal objects.
- A good way to do it is to multiply the `properties` by a prime number.

`toString()`
Returns string representation of the object
By default, it is `class_name@hex_of_hashcode`

### Polymorphism
An object of a subclass can be used anywhere its parent is used

### Dynamic Binding
The `JVM (Java Virtual Machine)` dynamically binds implementation of methods at runtime

If a method is invoked in the child, `JVM` try to find that method starting from child, and upwards up the parent.

Be careful of typecasting stuffs here

### Encapsulation
Each `module` hides their inner working, and only expose `methods` & `properties` through their designated `API`.
- Allow isolated testing of modules
- Decoupled allows easy reuse of code

`Access Controls`
- `private`: Only accessible from top-level class it is defined in
- `package-private`: Accessible from any classes in the package
- `protected`: Accessible from any classes in the package and subclasses.
- `public`: Accessible from anywhere

### Abstract Classes
Cannot be instantiated with the `new` keyword.
Instead, must be used by `extend`

### Interface
Define common behavior of classes.
Contains only `constants` and `abstract methods`
Can be used with the `implements` keyword

### Generics
There can be `Generic Interfaces`, `Generic Classes` and `Generic Methods`.

Eg:
```java
ArrayList<Integer> A = new ArrayList<Integer>();
```

`Interface Comparable`
```java
public class Point implements Comparable<Point> {
	// class body omitted
	@Override
	public int compareTo(Point p) {
	// implementation omitted
	}
}
```

`Class ArrayList<E>`
- `boolean add(E e)`
- `E.get(int index)`
- `int size()`
- `boolean contains(Object o)`
- `int indexOf(Object o)`

`Class HashSet<E>`
- `boolean add(E e)`
- `int size()`
- `boolean contains(Object o)`

`LinkedHashSet<E>`

### Exceptions
`RuntimeException` and its subclasses are known as `unchecked exceptions`
Eg:
- `NullPointerException`
- `ArrayIndexOutOfBoundsException`
- `ArithmeticException`

All other exceptions are `checked exceptions`, and is forced by the compiler to use `try...catch` block.
Eg:
- `IOException`
- `FileNotFoundException`
- `SQLException`