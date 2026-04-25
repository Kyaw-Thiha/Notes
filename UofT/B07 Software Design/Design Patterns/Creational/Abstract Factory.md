# Abstract Factory

> [!summary] Main Idea
> Provide an interface for creating families of related objects, without specifying their concrete classes.

`Technique`

1. Define `product interfaces`.
2. Define `abstract factory interface` which creates the product interfaces.
3. Implement `concrete factory`, with their own family of `concrete products`

---
`Difference with Factory Method`

- [[Factory Method]] wants subclasses to create ONE object.
- `Abstract Factory Method` wants to create multiple related objects as a consistent family.

---
`Code Example`

`Abstract Product Interfaces`
```java
interface Button {
    void paint();
}

interface Checkbox {
    void check();
}
```

`Abstract Factory Interface`
```java
interface GUIFactory {
    Button createButton();
    Checkbox createCheckbox();
}
```

`Concrete Products`
```java
class WinButton implements Button {
    public void paint() {
        System.out.println("Rendering Windows-style button");
    }
}

class WinCheckbox implements Checkbox {
    public void check() {
        System.out.println("Windows checkbox checked");
    }
}
```

```java
class MacButton implements Button {
    public void paint() {
        System.out.println("Rendering macOS-style button");
    }
}

class MacCheckbox implements Checkbox {
    public void check() {
        System.out.println("macOS checkbox checked");
    }
}
```

`Concrete Factory`
```java
class WinFactory implements GUIFactory {
    public Button createButton() {
        return new WinButton();
    }

    public Checkbox createCheckbox() {
        return new WinCheckbox();
    }
}

class MacFactory implements GUIFactory {
    public Button createButton() {
        return new MacButton();
    }

    public Checkbox createCheckbox() {
        return new MacCheckbox();
    }
}
```

`Driver`
```java
class Application {
    private final Button button;
    private final Checkbox checkbox;

    public Application(GUIFactory factory) {
        this.button = factory.createButton();
        this.checkbox = factory.createCheckbox();
    }

    public void render() {
        button.paint();
        checkbox.check();
    }
}
```

```java
public class Main {
    public static void main(String[] args) {
        GUIFactory factory;

        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("win")) {
            factory = new WinFactory();
        } else {
            factory = new MacFactory();
        }

        Application app = new Application(factory);
        app.render();
    }
}
```

---
`ML-Based Example`

Abstract Product
```c
#include <iostream>
#include <memory>

// Base layer interface
class Layer {
public:
    virtual void forward() = 0;
    virtual ~Layer() = default;
};

// More specific layer interfaces
class Attention : public Layer {};
class FeedForward : public Layer {};
```

Concrete Product
```c
class ConvLayer : public Layer {
public:
    void forward() override {
        std::cout << "Running convolution layer...\n";
    }
};

class MaxPoolLayer : public Layer {
public:
    void forward() override {
        std::cout << "Running max pooling...\n";
    }
};
```

```c
class MultiHeadAttention : public Attention {
public:
    void forward() override {
        std::cout << "Running multi-head self-attention...\n";
    }
};

class TransformerFFN : public FeedForward {
public:
    void forward() override {
        std::cout << "Running transformer feed-forward network...\n";
    }
};
```

Abstract Factory
```c
class ModelFactory {
public:
    virtual std::unique_ptr<Layer> createLayer1() = 0;
    virtual std::unique_ptr<Layer> createLayer2() = 0;
    virtual ~ModelFactory() = default;
};
```

Concrete Factory
```c
class CNNFactory : public ModelFactory {
public:
    std::unique_ptr<Layer> createLayer1() override {
        return std::make_unique<ConvLayer>();
    }

    std::unique_ptr<Layer> createLayer2() override {
        return std::make_unique<MaxPoolLayer>();
    }
};
```

```c
class TransformerFactory : public ModelFactory {
public:
    std::unique_ptr<Layer> createLayer1() override {
        return std::make_unique<MultiHeadAttention>();
    }

    std::unique_ptr<Layer> createLayer2() override {
        return std::make_unique<TransformerFFN>();
    }
};
```

Driver
```c
class Model {
private:
    std::unique_ptr<Layer> layer1;
    std::unique_ptr<Layer> layer2;

public:
    Model(ModelFactory& factory) {
        layer1 = factory.createLayer1();
        layer2 = factory.createLayer2();
    }

    void forward() {
        layer1->forward();
        layer2->forward();
    }
};
```

```c
int main() {
    bool useTransformer = true;

    std::unique_ptr<ModelFactory> factory;

    if (useTransformer)
        factory = std::make_unique<TransformerFactory>();
    else
        factory = std::make_unique<CNNFactory>();

    Model model(*factory);

    model.forward();  
}
```

---
## See Also 
- [[Factory Method]]