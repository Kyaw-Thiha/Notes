# Memento

> [!summary] Main Idea
> Used to save object's state (snapshot).
> Hence, we can use it to restore later.

`Technique`
1. Define an `Originator`
2. Define a `Memento` to stores internal state of Originator
3. Define a `Caretaker` that manages Mementos, but does not modify them

---
`Code Example`

Memento
```java
class EditorMemento {
    private final String content;

    public EditorMemento(String content) {
        this.content = content;
    }

    public String getSavedContent() {
        return content;
    }
}
```

Originator
```java
class TextEditor {
    private String content = "";

    public void type(String words) {
        content += words;
    }

    public String getContent() {
        return content;
    }

    public EditorMemento save() {
        return new EditorMemento(content);
    }

    public void restore(EditorMemento memento) {
        content = memento.getSavedContent();
    }
}
```

Caretaker
```java
import java.util.Stack;

class History {
    private Stack<EditorMemento> history = new Stack<>();

    public void save(EditorMemento m) {
        history.push(m);
    }

    public EditorMemento undo() {
        if (!history.empty()) {
            return history.pop();
        }
        return null;
    }
}
```

Client Main Driver
```java
public class Main {
    public static void main(String[] args) {
        TextEditor editor = new TextEditor();
        History history = new History();

        editor.type("Hello ");
        history.save(editor.save());

        editor.type("World!");
        history.save(editor.save());

        editor.type(" Extra text...");
        System.out.println("Current: " + editor.getContent());

        // Undo twice
        editor.restore(history.undo());
        System.out.println("After undo 1: " + editor.getContent());

        editor.restore(history.undo());
        System.out.println("After undo 2: " + editor.getContent());
    }
}
```

---
`ML Example`

Memento
```c
struct Checkpoint {
    std::vector<float> weights;
    float learningRate;
    int step;
};
```

Originator
```c
#include <iostream>
#include <vector>

class Model {
private:
    std::vector<float> weights;

public:
    Model(int size) : weights(size, 0.1f) {}

    void trainStep() {
        for (auto &w : weights) {
            w += 0.01f;   // pretend "learning" happens
        }
        std::cout << "[Model] Training step updated weights.\n";
    }

    const std::vector<float>& getWeights() const {
        return weights;
    }

    void setWeights(const std::vector<float>& w) {
        weights = w;
    }
};
```

```c
class Optimizer {
private:
    float learningRate;
    int step;

public:
    Optimizer(float lr) : learningRate(lr), step(0) {}

    void update() {
        step++;
        std::cout << "[Optimizer] step=" << step 
                  << ", lr=" << learningRate << "\n";
    }

    float getLR() const { return learningRate; }
    int getStep() const { return step; }

    void setState(float lr, int step_) {
        learningRate = lr;
        step = step_;
    }
};
```

Originator Method
```c
class TrainingState {
private:
    Model& model;
    Optimizer& opt;

public:
    TrainingState(Model& m, Optimizer& o) : model(m), opt(o) {}

    Checkpoint save() {
        std::cout << "[TrainingState] Saving checkpoint...\n";
        return Checkpoint{
            model.getWeights(),
            opt.getLR(),
            opt.getStep()
        };
    }

    void restore(const Checkpoint& ckpt) {
        std::cout << "[TrainingState] Restoring checkpoint...\n";
        model.setWeights(ckpt.weights);
        opt.setState(ckpt.learningRate, ckpt.step);
    }
};
```

Caretaker
```c
#include <stack>

class CheckpointManager {
private:
    std::stack<Checkpoint> history;

public:
    void save(const Checkpoint& ckpt) {
        history.push(ckpt);
    }

    Checkpoint undo() {
        if (history.empty()) {
            throw std::runtime_error("No checkpoints available!");
        }
        Checkpoint ckpt = history.top();
        history.pop();
        return ckpt;
    }
};
```

Client Main Code
```c
int main() {
    Model model(3);
    Optimizer opt(0.01f);

    TrainingState state(model, opt);
    CheckpointManager manager;

    // Training step 1
    model.trainStep();
    opt.update();
    manager.save(state.save());  // checkpoint 1

    // Training step 2
    model.trainStep();
    opt.update();
    manager.save(state.save());  // checkpoint 2

    // Training step 3
    model.trainStep();
    opt.update();
    std::cout << "\nTraining went bad... restoring!\n";

    // Undo to previous checkpoint
    state.restore(manager.undo());

    return 0;
}
```

---