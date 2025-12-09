# SOLID

## 1. Single Responsibility Principle (SRP)
A class should have only 1 reason to change.

## 2. Open/Closed principle (OCP)
This is a bad design since we cannot use different date formats.

```java
public class Date {
    int day;
    int month;
    int year;
    
    public Date(int day, int month, int year) {
        this.day = day;
        this.month = month;
        this.year = year;
    }
    
    @Override
    public String toString() {
        return day + "/" + month + "/" + year;
    }
}
```

Instead, we should do

```Java
interface Formatter {
    public String getFormat(int day, int month, int year);
}

class DMYFormatter implements Formatter {
    public String getFormat(int day, int month, int year) {
        return day + "/" + month + "/" + year;
    }
}

public class Date {
    int day;
    int month;
    int year;
    
    Formatter f;
    
    public Date(int day, int month, int year) {
        this.day = day;
        this.month = month;
        this.year = year;
    }
    
    public void setFormatter(Formatter f) {
        this.f = f;
    }
    
    @Override
    public String toString() {
        return f.getFormat(day, month, year);
    }
```

and then, we can run
```java
Date d = new Date(23, 8, 2006);
d.setFormatter(DMYFormatter)
System.out.println(d);
```

## 3. Liskov Substitution Property (LSP)

Subtypes must be substitutable for their base types.


Consider the following example.
Here, the `setWidth` is having to `override` in a awkward way.

```java
class Rectange {
	double width;
	double height;
	
	public Rectangle(double width, double height) {
		//
	}
	
	public void setWidth(double newWidth) {
		width = newWidth;
	}
	
	//
}

class Square extends Rectangle {
	public Square(double side) {
		super(side, side);
	}
	
	@Override
	public void setWidth(double newWidth){
		width = newWidth;
		height = newWidth;
	}
}
```

## 4. Interface Segregation Principle (ISP)

Clients should not be forced to depend on methods that they do not use.

For instance, if say `classA`, and `classB` both implements an `interface`, but only use a subset of those methods from the `interface` (since it is made for both classes), then it is not good.

## 5. Dependency Inversion Principle

High-level modules should not depend on low-level modules.
Abstractions (`abstract class`) should not depend on details (`concrete classes`)
