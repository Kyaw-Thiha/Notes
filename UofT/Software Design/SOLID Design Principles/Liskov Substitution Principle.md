# Liskov Substitution Principle

> A `subclass` must be usable anywhere its `superclass` is expected.

---
`Bad Example`

Since `Square` can be thought of as a subset of `Rectangle`, one might be tempted to inherit `Square` from `Rectangle`.

```java
class Rectangle {
    protected int width;
    protected int height;

    public void setWidth(int w) { this.width = w; }
    public void setHeight(int h) { this.height = h; }

    public int area() {
        return width * height;
    }
}
```

```java
class Square extends Rectangle {
    @Override
    public void setWidth(int w) {
        this.width = w;
        this.height = w;  // forces height = width
    }

    @Override
    public void setHeight(int h) {
        this.height = h;
        this.width = h;   // forces width = height
    }
}
```

But this will lead to unexpected behavior when being used.
```java
public static void resizeToWidth(Rectangle r) {
    r.setWidth(10);
    r.setHeight(5);
    System.out.println(r.area()); 
}

resizeToWidth(new Rectangle());
// prints 50

resizeToWidth(new Square());
// prints 25  (because Square forced width=height=5)
```

---
`Good Example`

Use shape abstraction.

```java
interface Shape {
    int area();
}
```

```java
class Rectangle implements Shape {
    protected int width, height;

    public Rectangle(int w, int h) {
        width = w;
        height = h;
    }

    public int area() {
        return width * height;
    }
}
```

```java
class Square implements Shape {
    private int side;

    public Square(int side) {
        this.side = side;
    }

    public int area() {
        return side * side;
    }
}
```

```java
public static void printArea(Shape s) {
    System.out.println(s.area());
}

printArea(new Rectangle(10, 5));  // prints 50
printArea(new Square(5));         // prints 25
```
