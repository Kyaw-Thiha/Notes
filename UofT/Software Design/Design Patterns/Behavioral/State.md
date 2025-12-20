# State

> [!summary] Main Idea
> Used when an object's behaviour changes depending on its internal state.

`Technique`
1. Define a `State` interface
2. Create a `concrete state` implementation
3. Create a `Context` that holds reference to current `State`

---
`Code Example`

State Interface
```java
interface VendingState {
    void insertCoin();
    void selectItem();
    void dispense();
}
```

Context
```java
class VendingMachine {
    private VendingState state;

    public VendingMachine() {
        state = new NoCoinState(this);
    }

    public void setState(VendingState newState) {
        state = newState;
    }

    public void insertCoin()  { state.insertCoin(); }
    public void selectItem()  { state.selectItem(); }
    public void dispense()    { state.dispense(); }
}
```

Concrete States
```java
class NoCoinState implements VendingState {
    private VendingMachine machine;

    public NoCoinState(VendingMachine m) {
        machine = m;
    }

    public void insertCoin() {
        System.out.println("Coin inserted.");
        machine.setState(new HasCoinState(machine));
    }

    public void selectItem() {
        System.out.println("Insert coin first.");
    }

    public void dispense() {
        System.out.println("Insert coin first.");
    }
}
```

```java
class HasCoinState implements VendingState {
    private VendingMachine machine;

    public HasCoinState(VendingMachine m) {
        machine = m;
    }

    public void insertCoin() {
        System.out.println("Coin already inserted.");
    }

    public void selectItem() {
        System.out.println("Item selected.");
        machine.setState(new DispensingState(machine));
    }

    public void dispense() {
        System.out.println("Select item first.");
    }
}
```

```java
class DispensingState implements VendingState {
    private VendingMachine machine;

    public DispensingState(VendingMachine m) {
        machine = m;
    }

    public void insertCoin() {
        System.out.println("Please wait, dispensing...");
    }

    public void selectItem() {
        System.out.println("Already dispensing.");
    }

    public void dispense() {
        System.out.println("Dispensing item!");
        machine.setState(new NoCoinState(machine));
    }
}
```

Client Main Code
```java
public class Main {
    public static void main(String[] args) {
        VendingMachine v = new VendingMachine();

        v.selectItem();    // wrong state
        v.insertCoin();    // transitions to HasCoin
        v.selectItem();    // transitions to Dispensing
        v.dispense();      // transitions back to NoCoin
    }
}
```

---
`ML Example`

State
```c
class TrainingState {
public:
    virtual ~TrainingState() = default;
    virtual void handleEpoch(TrainingSession& session) = 0;
};
```

Context
```c
class TrainingSession {
private:
    std::unique_ptr<TrainingState> state;
    int epoch = 0;
    float bestValLoss = 1e9f;
    int patience = 0;
    int maxPatience;

public:
    TrainingSession(std::unique_ptr<TrainingState> initialState, int maxPatience)
        : state(std::move(initialState)), maxPatience(maxPatience) {}

    void setState(std::unique_ptr<TrainingState> newState) {
        state = std::move(newState);
    }

    void runNextEpoch() {
        if (state) {
            state->handleEpoch(*this);
        }
    }

    // Getters / setters used by states:
    int& getEpoch() { return epoch; }
    float& getBestValLoss() { return bestValLoss; }
    int& getPatience() { return patience; }
    int getMaxPatience() const { return maxPatience; }
};
```

Concrete States
```c
class ValidationState;      // forward declare
class EarlyStoppedState;    // forward declare

class TrainingState : public TrainingState {
public:
    void handleEpoch(TrainingSession& session) override {
        int& epoch = session.getEpoch();
        epoch++;

        std::cout << "[Training] Epoch " << epoch << ": running training step...\n";
        // here you'd call model.forward(), backward(), optimizer.step(), etc.

        // After each training epoch, go to validation
        session.setState(std::make_unique<ValidationState>());
    }
};
```

```c
class ValidationState : public TrainingState {
public:
    void handleEpoch(TrainingSession& session) override {
        std::cout << "[Validation] Evaluating on validation set...\n";

        // pretend we computed a loss
        float valLoss = 0.5f + static_cast<float>(rand() % 100) / 1000.0f;
        std::cout << "  -> val loss = " << valLoss << "\n";

        float& best = session.getBestValLoss();
        int& patience = session.getPatience();
        int maxPatience = session.getMaxPatience();

        if (valLoss < best) {
            best = valLoss;
            patience = 0;
            std::cout << "  -> new best loss! (" << best << ")\n";
        } else {
            patience++;
            std::cout << "  -> no improvement, patience = " << patience << "\n";
        }

        if (patience >= maxPatience) {
            std::cout << "[Validation] Early stopping triggered.\n";
            session.setState(std::make_unique<EarlyStoppedState>());
        } else {
            // go back to training
            session.setState(std::make_unique<TrainingState>());
        }
    }
};
```

```c
class EarlyStoppedState : public TrainingState {
public:
    void handleEpoch(TrainingSession& session) override {
        std::cout << "[EarlyStopped] Training has been stopped. No more epochs.\n";
        // Do nothing, or you could save best model, export, etc.
        // Could also set state to nullptr to prevent further calls.
    }
};
```

Main Client Driver
```c
int main() {
    // Start in Training state, allow 3 epochs of no improvement
    auto initialState = std::make_unique<TrainingState>();
    TrainingSession session(std::move(initialState), /*maxPatience=*/3);

    // Simulate some epochs
    for (int i = 0; i < 10; ++i) {
        session.runNextEpoch();
    }

    return 0;
}
```