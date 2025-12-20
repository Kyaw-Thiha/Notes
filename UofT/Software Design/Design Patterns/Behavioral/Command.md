# Command

> [!summary] Main Idea
> Turning a request into a object so that it can be stored, queued, undone, or passed around.

`Technique`

1. Create a `Command Interface` which have
	- `execute()`
	- `undo()` optional
2. Create `concrete commands`.
3. Create a `Receiver` that actually performs the work.
4. Create an `Invoker` that triggers the command.
5. Create `Client` that create commands and assigns to Invokers

---
`Code Example`

Command
```java
interface Command {
    void execute();
}
```

Receiver
```java
class Light {
    public void turnOn() {
        System.out.println("Light is ON");
    }

    public void turnOff() {
        System.out.println("Light is OFF");
    }
}
```

Concrete Commands
```java
class TurnOnCommand implements Command {
    private Light light;

    public TurnOnCommand(Light light) {
        this.light = light;
    }

    @Override
    public void execute() {
        light.turnOn();
    }
}
```

```java
class TurnOffCommand implements Command {
    private Light light;

    public TurnOffCommand(Light light) {
        this.light = light;
    }

    @Override
    public void execute() {
        light.turnOff();
    }
}
```

Invoker
```java
class RemoteControl {
    private Command command;

    public void setCommand(Command command) {
        this.command = command;
    }

    public void pressButton() {
        if (command != null)
            command.execute();
    }
}
```

Main Driver
```java
public class Main {
    public static void main(String[] args) {
        Light lamp = new Light();

        Command onCommand = new TurnOnCommand(lamp);
        Command offCommand = new TurnOffCommand(lamp);

        RemoteControl remote = new RemoteControl();

        System.out.println("--- Pressing ON ---");
        remote.setCommand(onCommand);
        remote.pressButton();

        System.out.println("--- Pressing OFF ---");
        remote.setCommand(offCommand);
        remote.pressButton();
    }
}
```

---
`ML Example`

Receiver
```c
#include <iostream>
#include <queue>
#include <memory>
#include <vector>

class Model {
public:
    void forward() {
        std::cout << "[Model] Forward pass\n";
    }

    void backward() {
        std::cout << "[Model] Backward pass\n";
    }

    void updateWeights() {
        std::cout << "[Model] Updating weights\n";
    }

    void evaluate() {
        std::cout << "[Model] Running evaluation on validation set\n";
    }
};

class CheckpointManager {
public:
    void save(int step) {
        std::cout << "[Checkpoint] Saving model at step " << step << "\n";
    }
};
```

Command
```c
class Command {
public:
    virtual void execute() = 0;
    virtual ~Command() = default;
};
```

Concrete Commands
```c
class TrainStepCommand : public Command {
private:
    Model& model;
    int stepId;

public:
    TrainStepCommand(Model& model, int stepId)
        : model(model), stepId(stepId) {}

    void execute() override {
        std::cout << "\n[TrainStep] Step " << stepId << "\n";
        model.forward();
        model.backward();
        model.updateWeights();
    }
};
```

```c
class EvalCommand : public Command {
private:
    Model& model;

public:
    EvalCommand(Model& model) : model(model) {}

    void execute() override {
        std::cout << "\n[Eval] Validation\n";
        model.evaluate();
    }
};
```

```c
class SaveCheckpointCommand : public Command {
private:
    CheckpointManager& ckptMgr;
    int stepId;

public:
    SaveCheckpointCommand(CheckpointManager& mgr, int stepId)
        : ckptMgr(mgr), stepId(stepId) {}

    void execute() override {
        ckptMgr.save(stepId);
    }
};
```

Invoker
```c
class TrainingScheduler {
private:
    std::queue<std::unique_ptr<Command>> queue;

public:
    void addCommand(std::unique_ptr<Command> cmd) {
        queue.push(std::move(cmd));
    }

    void run() {
        while (!queue.empty()) {
            auto& cmd = queue.front();
            cmd->execute();
            queue.pop();
        }
    }
};
```

Main Driver
```c
int main() {
    Model model;
    CheckpointManager ckpt;
    TrainingScheduler scheduler;

    int totalSteps = 3;

    for (int step = 1; step <= totalSteps; ++step) {
        // enqueue: train step
        scheduler.addCommand(std::make_unique<TrainStepCommand>(model, step));

        // every step: eval and save (for demo)
        scheduler.addCommand(std::make_unique<EvalCommand>(model));
        scheduler.addCommand(std::make_unique<SaveCheckpointCommand>(ckpt, step));
    }

    // run all scheduled commands
    scheduler.run();

    return 0;
}
```

---
