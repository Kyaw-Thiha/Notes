# Strategy

> [!summary] Main Idea
> Define a family of algorithms, and make them interchangeable at runtime.

`Technique`
- Define a `Strategy Interface`
- Implement multiple `concrete strategies`
- `Context` class receives a strategy object, and uses it

---
`Example`

`Strategy Interface`
```java
interface PaymentStrategy {
    void pay(double amount);
}
```

`Concrete Strategies`
```java
class CreditCardPayment implements PaymentStrategy {
    @Override
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " using Credit Card.");
    }
}

class PayPalPayment implements PaymentStrategy {
    @Override
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " using PayPal.");
    }
}

class CryptoPayment implements PaymentStrategy {
    @Override
    public void pay(double amount) {
        System.out.println("Paid $" + amount + " using Cryptocurrency.");
    }
}
```

`Context Class`
```java
class Checkout {
    private PaymentStrategy strategy;

    public void setStrategy(PaymentStrategy strategy) {
        this.strategy = strategy;
    }

    public void processOrder(double amount) {
        strategy.pay(amount);
    }
}
```

`Main Driver`
```java
public class Main {
    public static void main(String[] args) {
        Checkout checkout = new Checkout();

        checkout.setStrategy(new CreditCardPayment());
        checkout.processOrder(100.0);

        checkout.setStrategy(new PayPalPayment());
        checkout.processOrder(50.0);

        checkout.setStrategy(new CryptoPayment());
        checkout.processOrder(200.0);
    }
}
```

---
`ML Example`

`Strategy Interface`
```c
class Optimizer {
public:
    virtual void step(float& weight, float gradient) = 0;
    virtual ~Optimizer() {}
};
```

```c
class SGD : public Optimizer {
private:
    float lr;

public:
    SGD(float learningRate) : lr(learningRate) {}

    void step(float& weight, float gradient) override {
        weight -= lr * gradient;
    }
};
```

```c
class Adam : public Optimizer {
private:
    float lr, beta1, beta2;
    float m = 0.0f, v = 0.0f;
    int t = 0;

public:
    Adam(float lr=0.001f, float b1=0.9f, float b2=0.999f)
        : lr(lr), beta1(b1), beta2(b2) {}

    void step(float& weight, float gradient) override {
        t++;

        m = beta1 * m + (1 - beta1) * gradient;
        v = beta2 * v + (1 - beta2) * gradient * gradient;

        float m_hat = m / (1 - pow(beta1, t));
        float v_hat = v / (1 - pow(beta2, t));

        weight -= lr * m_hat / (sqrt(v_hat) + 1e-8);
    }
};
```

`Context`
```c
class Trainer {
private:
    Optimizer* optimizer;

public:
    Trainer(Optimizer* opt) : optimizer(opt) {}

    void train(float& weight) {
        for (int epoch = 1; epoch <= 5; ++epoch) {
            float gradient = weight * 2.0f;  
            optimizer->step(weight, gradient);

            std::cout << "Epoch " << epoch 
                      << " | Weight = " << weight << "\n";
        }
    }
};
class Trainer {
private:
    Optimizer* optimizer;

public:
    Trainer(Optimizer* opt) : optimizer(opt) {}

    void train(float& weight) {
        for (int epoch = 1; epoch <= 5; ++epoch) {
            float gradient = weight * 2.0f;  
            optimizer->step(weight, gradient);

            std::cout << "Epoch " << epoch 
                      << " | Weight = " << weight << "\n";
        }
    }
};
```

`Main Driver`
```c
int main() {
    float weight = 5.0f;

    Optimizer* opt = new Adam(0.01f);  
    // You can swap SGD here without changing trainer code:
    // Optimizer* opt = new SGD(0.1f);

    Trainer trainer(opt);
    trainer.train(weight);

    delete opt;
    return 0;
}
```