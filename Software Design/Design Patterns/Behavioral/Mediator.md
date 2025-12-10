# Mediator

> [!summary] Main Idea
> Used when we want to centralize communication into one object.
> If objects talk to each other directly, they might become too tightly coupled.

`Technique`
1. Define a `Mediator` interface.
2. Create a `concrete Mediator`.
3. Create `Colleagues` that hold a reference to the mediator.

---
`Code Example`

Mediator
```java
interface ChatMediator {
    void sendMessage(String message, User user);
    void addUser(User user);
}
```

Concrete Mediator
```java
import java.util.ArrayList;
import java.util.List;

class ChatRoom implements ChatMediator {
    private List<User> users = new ArrayList<>();

    @Override
    public void addUser(User user) {
        users.add(user);
    }

    @Override
    public void sendMessage(String message, User sender) {
        for (User user : users) {
            if (user != sender) {
                user.receive(message);
            }
        }
    }
}
```

Colleagues
```c
abstract class User {
    protected ChatMediator mediator;
    protected String name;

    public User(ChatMediator mediator, String name) {
        this.mediator = mediator;
        this.name = name;
    }

    public abstract void send(String message);
    public abstract void receive(String message);
}
```

Concrete Colleague
```c
class ChatUser extends User {

    public ChatUser(ChatMediator mediator, String name) {
        super(mediator, name);
    }

    @Override
    public void send(String message) {
        System.out.println(name + " sends: " + message);
        mediator.sendMessage(message, this);
    }

    @Override
    public void receive(String message) {
        System.out.println(name + " receives: " + message);
    }
}
```

---
`ML Example`

Interfaces
```c
#include <iostream>
#include <vector>

class TrainingMediator; // forward declaration

class Colleague {
protected:
    TrainingMediator* mediator;
public:
    Colleague(TrainingMediator* m) : mediator(m) {}
    virtual ~Colleague() = default;
};
```

Concrete Colleagues
```c
class Model : public Colleague {
public:
    Model(TrainingMediator* m) : Colleague(m) {}

    float forward(const std::vector<float>& batch) {
        std::cout << "[Model] Forward pass on batch of size " 
                  << batch.size() << "\n";
        return 0.5f; // pretend loss
    }
};
```

```c
class Optimizer : public Colleague {
public:
    Optimizer(TrainingMediator* m) : Colleague(m) {}

    void step(float loss) {
        std::cout << "[Optimizer] Updating weights with loss = " << loss << "\n";
    }
};
```

```c
class DataLoader : public Colleague {
    int batchIdx = 0;
public:
    DataLoader(TrainingMediator* m) : Colleague(m) {}

    std::vector<float> getBatch() {
        std::cout << "[DataLoader] Supplying batch #" << batchIdx << "\n";
        batchIdx++;
        return {1, 2, 3, 4}; // dummy data
    }
};
```

Mediator
```c
class TrainingMediator {
private:
    Model* model;
    Optimizer* optimizer;
    DataLoader* dataloader;

public:
    TrainingMediator()
        : model(nullptr), optimizer(nullptr), dataloader(nullptr) {}

    void registerModel(Model* m) { model = m; }
    void registerOptimizer(Optimizer* opt) { optimizer = opt; }
    void registerDataLoader(DataLoader* dl) { dataloader = dl; }

    void runTrainingStep() {
        std::cout << "\n=== Training Step ===\n";

        auto batch = dataloader->getBatch();
        float loss = model->forward(batch);
        optimizer->step(loss);
    }
};
```

Client Main Driver
```c
int main() {
    TrainingMediator mediator;

    Model model(&mediator);
    Optimizer optimizer(&mediator);
    DataLoader dataloader(&mediator);

    mediator.registerModel(&model);
    mediator.registerOptimizer(&optimizer);
    mediator.registerDataLoader(&dataloader);

    for (int i = 0; i < 3; i++) {
        mediator.runTrainingStep();
    }
}
```

---