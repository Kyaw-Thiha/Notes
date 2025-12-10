# Builder

> [!summary] Main Idea
> For creating complex object step-by-step without a huge no. of parameters in constructor.

`Technique`
1. The main class has a `private constructor`.
2. Create a `static nested Builder` class inside the main class.
3. The `builder methods` return `this` for chaining.
4. The `build()` method inside Builder actually create the object.

---
`Code Example`
```java
class House {
    private final String address;
    private final int floors;
    private final boolean hasGarage;
    private final boolean hasPool;
    private final boolean hasGarden;

    // Private constructor receives builder
    private House(Builder builder) {
        this.address = builder.address;
        this.floors = builder.floors;
        this.hasGarage = builder.hasGarage;
        this.hasPool = builder.hasPool;
        this.hasGarden = builder.hasGarden;
    }

    public void display() {
        System.out.println("House at " + address +
            " | floors=" + floors +
            " | garage=" + hasGarage +
            " | pool=" + hasPool +
            " | garden=" + hasGarden);
    }

    // ---------------------------
    // Builder Class
    // ---------------------------
    public static class Builder {
        // Required
        private final String address;

        // Optional with default values
        private int floors = 1;
        private boolean hasGarage = false;
        private boolean hasPool = false;
        private boolean hasGarden = false;

        // Builder constructor enforces required fields
        public Builder(String address) {
            this.address = address;
        }

        public Builder floors(int floors) {
            this.floors = floors;
            return this;
        }

        public Builder garage(boolean value) {
            this.hasGarage = value;
            return this;
        }

        public Builder pool(boolean value) {
            this.hasPool = value;
            return this;
        }

        public Builder garden(boolean value) {
            this.hasGarden = value;
            return this;
        }

        // Build final object
        public House build() {
            return new House(this);
        }
    }
}
```

Driver
```java
public class Main {
    public static void main(String[] args) {
        House house = new House.Builder("123 King Street")
            .floors(2)
            .garage(true)
            .pool(false)
            .garden(true)
            .build();

        house.display();
    }
}
```

---
`ML Example`

Note that modern NN architecture have many parameters.
```c++
NeuralNet net(5 layers, useBatchNorm=true, dropout=0.25, hiddenSizes=[128, 64, 32], activation="ReLU", ...);
```

`Builder Pattern` would simplify this to
```c
auto model = NeuralNet::Builder()
                .layers({128, 64, 32})
                .activation("ReLU")
                .dropout(0.25)
                .batchNorm(true)
                .build();
```

Here is how to implement it.
```c
#include <iostream>
#include <vector>
#include <string>

class NeuralNet {
private:
    std::vector<int> layers;
    std::string activation;
    bool useBatchNorm;
    double dropoutRate;

    // Private constructor — only Builder can create it
    NeuralNet(const std::vector<int>& layers,
              const std::string& activation,
              bool useBatchNorm,
              double dropoutRate)
        : layers(layers),
          activation(activation),
          useBatchNorm(useBatchNorm),
          dropoutRate(dropoutRate) {}

public:
    // Display for demonstration
    void summary() const {
        std::cout << "Neural Network:\n";
        std::cout << " Layers: ";
        for (int l : layers) std::cout << l << " ";
        std::cout << "\n Activation: " << activation
                  << "\n BatchNorm: " << (useBatchNorm ? "Yes" : "No")
                  << "\n Dropout: " << dropoutRate
                  << "\n";
    }

    // ======================================
    // Builder Class
    // ======================================
    class Builder {
    private:
        std::vector<int> layers;
        std::string activation = "ReLU";  // default
        bool useBatchNorm = false;
        double dropoutRate = 0.0;

    public:
        Builder& layersConfig(const std::vector<int>& l) {
            layers = l;
            return *this;
        }

        Builder& activationFunc(const std::string& act) {
            activation = act;
            return *this;
        }

        Builder& batchNorm(bool value) {
            useBatchNorm = value;
            return *this;
        }

        Builder& dropout(double rate) {
            dropoutRate = rate;
            return *this;
        }

        NeuralNet build() {
            return NeuralNet(layers, activation, useBatchNorm, dropoutRate);
        }
    };
};
```
