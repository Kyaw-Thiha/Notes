# Decorator

> [!summary] Main Idea
> Dynamically add behaviours to an object without modifying its classes.

`Technique`
1. Define a `Component Interface` that both base object and decorators implement.
2. Create `concrete` component interfaces.
3. Define an abstract `Decorator` that stores reference to component.
4. Create `concrete` decorators that add behavior before/after the wrapped component.

---
`Code Example`

Component Interface
```java
interface Coffee {
    String getDescription();
    double getCost();
}
```

Concrete Component
```java
class BasicCoffee implements Coffee {
    @Override
    public String getDescription() {
        return "Basic Coffee";
    }

    @Override
    public double getCost() {
        return 2.00;
    }
}
```

Abstract Decorator
```java
abstract class CoffeeDecorator implements Coffee {
    protected Coffee coffee;  // the object being decorated

    public CoffeeDecorator(Coffee coffee) {
        this.coffee = coffee;
    }
}
```

Concrete Decorators
```java
class MilkDecorator extends CoffeeDecorator {
    public MilkDecorator(Coffee coffee) {
        super(coffee);
    }

    @Override
    public String getDescription() {
        return coffee.getDescription() + ", Milk";
    }

    @Override
    public double getCost() {
        return coffee.getCost() + 0.50;
    }
}
```

```java
class SugarDecorator extends CoffeeDecorator {
    public SugarDecorator(Coffee coffee) {
        super(coffee);
    }

    @Override
    public String getDescription() {
        return coffee.getDescription() + ", Sugar";
    }

    @Override
    public double getCost() {
        return coffee.getCost() + 0.20;
    }
}
```

```java
class WhippedCreamDecorator extends CoffeeDecorator {
    public WhippedCreamDecorator(Coffee coffee) {
        super(coffee);
    }

    @Override
    public String getDescription() {
        return coffee.getDescription() + ", Whipped Cream";
    }

    @Override
    public double getCost() {
        return coffee.getCost() + 0.80;
    }
}
```

Main Driver
```java
public class Main {
    public static void main(String[] args) {
        Coffee coffee = new BasicCoffee();
        coffee = new MilkDecorator(coffee);       // add milk
        coffee = new SugarDecorator(coffee);      // add sugar
        coffee = new WhippedCreamDecorator(coffee); // add whipped cream

        System.out.println(coffee.getDescription());
        System.out.println("Cost: $" + coffee.getCost());
    }
}
```

---
`ML Example`

Base Component
```c
#include <iostream>
#include <vector>
#include <memory>
#include <random>

class Layer {
public:
    virtual void forward(std::vector<float>& input) = 0;
    virtual ~Layer() = default;
};
```

Concrete Component
```c
class LinearLayer : public Layer {
private:
    int in, out;

public:
    LinearLayer(int in, int out) : in(in), out(out) {}

    void forward(std::vector<float>& input) override {
        std::cout << "Linear(" << in << " -> " << out << ")\n";
        // Simulate linear transform: normally you'd do W*x + b
    }
};
```

Abstract Decorator
```c
class LayerDecorator : public Layer {
protected:
    std::unique_ptr<Layer> wrapped;

public:
    LayerDecorator(std::unique_ptr<Layer> layer)
        : wrapped(std::move(layer)) {}
};
```

Concrete Decorator
```c
class LoggingDecorator : public LayerDecorator {
public:
    LoggingDecorator(std::unique_ptr<Layer> layer)
        : LayerDecorator(std::move(layer)) {}

    void forward(std::vector<float>& input) override {
        std::cout << "[LOG] Input size: " << input.size() << "\n";
        wrapped->forward(input);
        std::cout << "[LOG] Forward pass complete.\n";
    }
};
```

```c
class DropoutDecorator : public LayerDecorator {
private:
    float dropRate;
    std::default_random_engine rng;
    std::bernoulli_distribution dist;

public:
    DropoutDecorator(std::unique_ptr<Layer> layer, float dropRate)
        : LayerDecorator(std::move(layer)),
          dropRate(dropRate),
          dist(1.0f - dropRate) {}

    void forward(std::vector<float>& input) override {
        std::cout << "[Dropout] rate=" << dropRate << "\n";

        for (float& x : input) {
            if (!dist(rng))
                x = 0.0f;
        }

        wrapped->forward(input);
    }
};
```

Main Driver
```c
int main() {
    std::vector<float> sampleInput(5, 1.0f);

    // Build with decorators
    std::unique_ptr<Layer> model =
        std::make_unique<LoggingDecorator>(
            std::make_unique<DropoutDecorator>(
                std::make_unique<LinearLayer>(5, 3),
                0.5f // dropout rate
            )
        );

    model->forward(sampleInput);

    return 0;
}
```

---

