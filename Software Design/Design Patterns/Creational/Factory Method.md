# Factory Method

> [!summary] Main Idea
> Let subclasses decide which object to create.

`Technique`
- Define the `abstract` class for the products to be created.
- Then, define the `abstract` creator class which have method to create the abstract product
- Then, implement `concrete` classes for both factory & product

---
`Example`

Product Classes 

```java
interface Shape {
    void draw();
}
```

```java
class Circle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a Circle");
    }
}

class Square implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a Square");
    }
}
```

Factory Classes

```java
abstract class ShapeFactory {
    // Factory Method
    abstract Shape createShape();
    
    public Shape getShape() {
        return createShape();
    }
}
```

```java
class CircleFactory extends ShapeFactory {
    @Override
    Shape createShape() {
        return new Circle();
    }
}

class SquareFactory extends ShapeFactory {
    @Override
    Shape createShape() {
        return new Square();
    }
}
```

Driver
```java
public class Main {
    public static void main(String[] args) {
        ShapeFactory factory = new CircleFactory();
        Shape shape = factory.getShape();
        shape.draw();
    }
}
```

---
`ML Example`

Product
```c
#include <iostream>
#include <string>

class NeuralNet {
public:
    virtual void forward() = 0;
    virtual ~NeuralNet() {}
};
```

```c
class CNNModel : public NeuralNet {
public:
    void forward() override {
        std::cout << "Running forward pass on CNN model\n";
    }
};

class MLPModel : public NeuralNet {
public:
    void forward() override {
        std::cout << "Running forward pass on MLP model\n";
    }
};
```

Factory
```c
class ModelFactory {
public:
    // Factory Method
    virtual NeuralNet* createModel() = 0;
    virtual ~ModelFactory() {}
};
```

```c
class CNNFactory : public ModelFactory {
public:
    NeuralNet* createModel() override {
        return new CNNModel();
    }
};

class MLPFactory : public ModelFactory {
public:
    NeuralNet* createModel() override {
        return new MLPModel();
    }
};
```

Then to follow [[OpenClosed Principle]], we use
```c
class FactorySelector {
public:
    static ModelFactory* getFactory(const std::string& type) {
        if (type == "cnn")
            return new CNNFactory();
        if (type == "mlp")
            return new MLPFactory();
        
        throw std::runtime_error("Unknown model type");
    }
};
```

```c
int main() {
    std::string configType = "cnn"; // could come from JSON, CLI, YAML

    ModelFactory* factory = FactorySelector::getFactory(configType);
    NeuralNet* model = factory->createModel();

    model->forward();

    delete model;
    delete factory;
}
```

---
## See Also
- [[Important Design Patterns]]