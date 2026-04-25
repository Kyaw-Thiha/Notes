# Observer

> [!summary] Main Idea
> Define a one-to-many relationship between objects so that when state of one object change, all its dependants are notified automatically.

`Technique`

- `Subject`
	- Maintain a list of `observers`
	- Provides `attach()`, `detach()`, and `notifyObservers()`
- `Observer`
	- Has an `update()` method
	- `Subject` calls this update() method on all observers

---
`Example`

```java
interface Observer {
    void update(float temperature);
}
```

```java
interface Subject {
    void attach(Observer o);
    void detach(Observer o);
    void notifyObservers();
}
```

Concrete Subject
```java
import java.util.ArrayList;
import java.util.List;

class TemperatureSensor implements Subject {

    private List<Observer> observers = new ArrayList<>();
    private float temperature;

    @Override
    public void attach(Observer o) {
        observers.add(o);
    }

    @Override
    public void detach(Observer o) {
        observers.remove(o);
    }

    @Override
    public void notifyObservers() {
        for (Observer o : observers) {
            o.update(temperature);
        }
    }

    public void setTemperature(float newTemperature) {
        this.temperature = newTemperature;
        notifyObservers();  
    }
}
```

Concrete Observers
```java
class PhoneDisplay implements Observer {
    @Override
    public void update(float temperature) {
        System.out.println("Phone Display: temperature now " + temperature);
    }
}

class DashboardDisplay implements Observer {
    @Override
    public void update(float temperature) {
        System.out.println("Dashboard Display: temperature now " + temperature);
    }
}
```

Main Driver

```java
public class Main {
    public static void main(String[] args) {
        TemperatureSensor sensor = new TemperatureSensor();

        Observer phone = new PhoneDisplay();
        Observer dashboard = new DashboardDisplay();

        sensor.attach(phone);
        sensor.attach(dashboard);

        sensor.setTemperature(25.0f);
        sensor.setTemperature(30.5f);
    }
}
```

---
`ML Example`

`Subject`

```c
#include <vector>
#include <cstdlib>   // rand()

class Trainer {
private:
    std::vector<TrainingObserver*> observers;

public:
    void addObserver(TrainingObserver* obs) {
        observers.push_back(obs);
    }

    void train(int maxEpochs) {
        float loss = 1.0f;

        for (int epoch = 1; epoch <= maxEpochs; ++epoch) {
            // Fake training: loss decays with some noise
            loss = loss * 0.9f + float(rand() % 100) / 1000.0f;

            // Notify all observers
            for (auto obs : observers)
                obs->onEpochEnd(epoch, loss);
        }
    }
};
```

`Observer Interface`

```c
class TrainingObserver {
public:
    virtual void onEpochEnd(int epoch, float loss) = 0;
    virtual ~TrainingObserver() {}
};
```

`Observers`

```c
#include <iostream>

class LossLogger : public TrainingObserver {
public:
    void onEpochEnd(int epoch, float loss) override {
        std::cout << "[Logger] Epoch " << epoch << " - Loss: " << loss << "\n";
    }
};
```

```c
#include <fstream>

class CheckpointSaver : public TrainingObserver {
public:
    void onEpochEnd(int epoch, float loss) override {
        if (epoch % 5 == 0) {
            std::cout << "[Checkpoint] Saving model at epoch " << epoch << "\n";
        }
    }
};
```

```c
class EarlyStopping : public TrainingObserver {
private:
    float bestLoss = 1e9;
    int patience = 3;
    int wait = 0;

public:
    bool shouldStop = false;

    void onEpochEnd(int epoch, float loss) override {
        if (loss < bestLoss) {
            bestLoss = loss;
            wait = 0;
        } else {
            wait++;
        }

        if (wait >= patience) {
            shouldStop = true;
            std::cout << "[EarlyStopping] Triggered at epoch " << epoch << "\n";
        }
    }
};
```