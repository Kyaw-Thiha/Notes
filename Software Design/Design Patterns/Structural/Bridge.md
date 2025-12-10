# Bridge

> [!summary] Main Idea
> Used when we have more than 1 dimension of change, and we do not want it explode into many subclasses.

`Technique`
1. Create an `implementation interface`
2. Create an `abstraction class` which have reference to the `implementation interface`
3. Create `concrete abstractions` and `concrete implementations`.

---
`Code Example`

Bridge (Implementation Interface)
```java
interface DrawingAPI {
    void drawCircle(double x, double y, double radius);
}
```

Abstraction Class
```java
abstract class Shape {
    protected DrawingAPI drawingAPI;

    protected Shape(DrawingAPI drawingAPI) {
        this.drawingAPI = drawingAPI;
    }

    abstract void draw();     // high-level API
}
```

Concrete Implementation
```java
class OpenGLAPI implements DrawingAPI {
    public void drawCircle(double x, double y, double radius) {
        System.out.println("OpenGL draws circle at (" + x + ", " + y + ") radius " + radius);
    }
}

class DirectXAPI implements DrawingAPI {
    public void drawCircle(double x, double y, double radius) {
        System.out.println("DirectX draws circle at (" + x + ", " + y + ") radius " + radius);
    }
}
```

Concrete Abstraction
```java
class Circle extends Shape {
    private double x, y, radius;

    public Circle(double x, double y, double radius, DrawingAPI drawingAPI) {
        super(drawingAPI);
        this.x = x;
        this.y = y;
        this.radius = radius;
    }

    @Override
    void draw() {
        drawingAPI.drawCircle(x, y, radius);
    }
}
```

```java
public class Main {
    public static void main(String[] args) {
        Shape c1 = new Circle(1, 2, 5, new OpenGLAPI());
        Shape c2 = new Circle(1, 2, 5, new DirectXAPI());

        c1.draw();
        c2.draw();
    }
}
```

---
`ML Example`

Implementation Interface
```c
#include <iostream>
#include <vector>

class Backend {
public:
    virtual void matmul(const std::vector<float>& A,
                        const std::vector<float>& B,
                        int m, int n, int p) = 0;
    virtual ~Backend() = default;
};
```

```c
class CPUBackend : public Backend {
public:
    void matmul(const std::vector<float>& A,
                const std::vector<float>& B,
                int m, int n, int p) override 
    {
        std::cout << "[CPU] Matrix multiply (" 
                  << m << "x" << n << " * " << n << "x" << p << ")\n";
        // pretend to do computation...
    }
};
```

```c
class GPUBackend : public Backend {
public:
    void matmul(const std::vector<float>& A,
                const std::vector<float>& B,
                int m, int n, int p) override 
    {
        std::cout << "[GPU] Matrix multiply on CUDA (" 
                  << m << "x" << n << " * " << n << "x" << p << ")\n";
        // pretend GPU kernel launch...
    }
};
```

Abstraction Class
```c
class Layer {
protected:
    Backend* backend;
public:
    Layer(Backend* backend) : backend(backend) {}
    virtual void forward() = 0;
    virtual ~Layer() = default;
};
```

```c
class LinearLayer : public Layer {
private:
    std::vector<float> weights;
    std::vector<float> input;
    int m, n; // matrix shapes

public:
    LinearLayer(int m, int n, Backend* backend)
        : Layer(backend), m(m), n(n)
    {
        weights.resize(m * n);
        input.resize(n);
    }

    void forward() override {
        std::cout << "LinearLayer forward:\n";
        backend->matmul(input, weights, 1, n, m);
    }
};
```

Main Driver
```c
int main() {
    CPUBackend cpu;
    GPUBackend gpu;

    // Build model using GPU
    Layer* layer1 = new LinearLayer(10, 20, &gpu);
    layer1->forward();

    // Build another model using CPU
    Layer* layer2 = new LinearLayer(10, 20, &cpu);
    layer2->forward();

    delete layer1;
    delete layer2;
}
```