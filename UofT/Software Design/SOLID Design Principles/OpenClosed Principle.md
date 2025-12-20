# Open/Closed Principle

> `Classes` should be open for extension, but closed for modification.

`Why Open/Closed?`
Without `OCP`, we must use
```python
if (...) else if (...) else if (...) { ... }
```

With `OCP`, we just implements a new object
```java
class Triangle implements Shape {
    public double area() { /* ... */ }
}
```

`Bad Example`
```java
class AreaCalculator {
    public double area(Object shape) {
        if (shape instanceof Circle) {
            Circle c = (Circle) shape;
            return Math.PI * c.radius * c.radius;
        } else if (shape instanceof Rectangle) {
            Rectangle r = (Rectangle) shape;
            return r.width * r.height;
        }
        // If we add Triangle later → must modify here again 😭
        return 0;
    }
}

class Circle { double radius; }
class Rectangle { double width, height; }
```

---
`Good Example`

```java
interface Shape {
    double area();
}
```

```java
class Circle implements Shape {
    double radius;
    Circle(double radius) { this.radius = radius; }

    public double area() {
        return Math.PI * radius * radius;
    }
}

class Rectangle implements Shape {
    double width, height;
    Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    public double area() {
        return width * height;
    }
}
```

```java
class AreaCalculator {
    public double totalArea(List<Shape> shapes) {
        double sum = 0;
        for (Shape s : shapes) {
            sum += s.area();
        }
        return sum;
    }
}
```

```java
public class Main {
    public static void main(String[] args) {
        List<Shape> shapes = List.of(
            new Circle(3),
            new Rectangle(2, 4)
        );

        System.out.println(new AreaCalculator().totalArea(shapes));
    }
}
```