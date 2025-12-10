# Template

> [!summary] Main Idea
> Used when we want to create skeleton of an algorithm, whose parts can be overriden by subclasses.

`Technique`
1. Define a abstract class with the template method.
2. `Primitive Operations` which can be overriden by subclass.
3. `Hooks` which subclasses may or may not implement.

---
`Code Example`

Abstract Class with Template method
```java
abstract class DocumentExporter {

    // Template Method (algorithm skeleton)
    public final void export(String data) {
        loadData(data);
        String formatted = formatData(data);
        writeFile(formatted);
    }

    // Steps with default implementations
    private void loadData(String data) {
        System.out.println("Loading data: " + data);
    }

    // Step subclasses MUST implement
    protected abstract String formatData(String data);

    // Step subclasses MAY override
    protected void writeFile(String formattedData) {
        System.out.println("Writing file: " + formattedData);
    }
}
```

Concrete Subclasses
```java
class PdfExporter extends DocumentExporter {
    @Override
    protected String formatData(String data) {
        return "[PDF] " + data.toUpperCase();
    }
}
```

```java
class HtmlExporter extends DocumentExporter {
    @Override
    protected String formatData(String data) {
        return "<html><body>" + data + "</body></html>";
    }
}
```

Main Driver
```java
public class Main {
    public static void main(String[] args) {
        DocumentExporter pdf = new PdfExporter();
        DocumentExporter html = new HtmlExporter();

        pdf.export("Hello World");
        html.export("Hello World");
    }
}
```

---
`ML Example`

Base Class with Template Method
```c
#include <iostream>
#include <vector>
#include <cmath>

class Trainer {
public:
    // Template Method: defines the algorithm skeleton
    void trainEpoch(const std::vector<float>& batch) {
        loadBatch(batch);
        auto preds = forward(batch);
        float loss = computeLoss(batch, preds);
        backward(loss);
        optimizerStep();
        log(loss);
    }

protected:
    // Fixed step
    void loadBatch(const std::vector<float>& batch) {
        std::cout << "[Trainer] Loading batch of size " << batch.size() << "\n";
    }

    // Steps subclasses MUST implement
    virtual std::vector<float> forward(const std::vector<float>& batch) = 0;
    virtual float computeLoss(const std::vector<float>& batch,
                              const std::vector<float>& preds) = 0;

    // Optional hooks
    virtual void backward(float loss) {
        std::cout << "[Trainer] Backward pass with loss " << loss << "\n";
    }

    virtual void optimizerStep() {
        std::cout << "[Trainer] Optimizer step\n";
    }

    virtual void log(float loss) {
        std::cout << "[Trainer] Loss = " << loss << "\n\n";
    }
};
```

Concrete 
```c
class RegressionTrainer : public Trainer {
protected:
    std::vector<float> forward(const std::vector<float>& batch) override {
        std::cout << "[Regression] Forward pass (predict y = x * 0.5)\n";
        std::vector<float> preds;
        for (float x : batch) preds.push_back(x * 0.5f);
        return preds;
    }

    float computeLoss(const std::vector<float>& batch,
                      const std::vector<float>& preds) override {
        std::cout << "[Regression] Computing MSE\n";
        float mse = 0.0f;
        for (int i = 0; i < batch.size(); ++i) {
            float error = preds[i] - batch[i];
            mse += error * error;
        }
        return mse / batch.size();
    }
};
```

```c
class ClassificationTrainer : public Trainer {
protected:
    std::vector<float> forward(const std::vector<float>& batch) override {
        std::cout << "[Classification] Forward pass (sigmoid)\n";
        std::vector<float> preds;
        for (float x : batch) preds.push_back(1.0f / (1.0f + std::exp(-x)));
        return preds;
    }

    float computeLoss(const std::vector<float>& batch,
                      const std::vector<float>& preds) override {
        std::cout << "[Classification] Binary cross entropy\n";
        float loss = 0.0f;
        for (int i = 0; i < batch.size(); ++i) {
            float y = batch[i] > 0.5f ? 1.0f : 0.0f;
            float p = preds[i];
            loss += -(y * std::log(p + 1e-6f) + (1 - y) * std::log(1 - p + 1e-6f));
        }
        return loss / batch.size();
    }
};
```

Main Client Driver
```c
int main() {
    std::vector<float> batch = {0.1f, 0.5f, 1.0f, 2.0f};

    RegressionTrainer reg;
    ClassificationTrainer clf;

    std::cout << "--- Regression Training ---\n";
    reg.trainEpoch(batch);

    std::cout << "--- Classification Training ---\n";
    clf.trainEpoch(batch);

    return 0;
}
```

---
