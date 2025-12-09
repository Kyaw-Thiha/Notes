# Dependency Inversion Principle

> `High-Level Modules` should not depend on `Low-Level Modules`

---
`Bad Example`

```java
class Heater {
    public void turnOn() { System.out.println("Heater ON"); }
    public void turnOff() { System.out.println("Heater OFF"); }
}

class TemperatureSensor {
    public double read() {
        return 15.0; // just a dummy example
    }
}
```

```java
// ❌ High-level ControlSystem depends on a concrete Heater
class ControlSystem {
    private Heater heater;
    private TemperatureSensor sensor;

    public ControlSystem() {
        this.heater = new Heater();             
        this.sensor = new TemperatureSensor();  
    }

    public void update() {
        if (sensor.read() < 18) {
            heater.turnOn();
        } else {
            heater.turnOff();
        }
    }
}
```

`Good Example`

Introduce abstractions that both high-level and low-level codes rely on.

```java
interface Switchable {
    void turnOn();
    void turnOff();
}

interface TemperatureSensor {
    double read();
}
```

```java
class Heater implements Switchable {
    @Override
    public void turnOn() { System.out.println("Heater ON"); }

    @Override
    public void turnOff() { System.out.println("Heater OFF"); }
}

class RealTemperatureSensor implements TemperatureSensor {
    @Override
    public double read() { return 15.0; }
}
```

```java
class ControlSystem {
    private final Switchable heater;
    private final TemperatureSensor sensor;

    // Dependencies are injected (constructor injection)
    public ControlSystem(Switchable heater, TemperatureSensor sensor) {
        this.heater = heater;
        this.sensor = sensor;
    }

    public void update() {
        if (sensor.read() < 18) {
            heater.turnOn();
        } else {
            heater.turnOff();
        }
    }
}
```

```java
public class Main {
    public static void main(String[] args) {
        Switchable heater = new Heater();
        TemperatureSensor sensor = new RealTemperatureSensor();

        ControlSystem cs = new ControlSystem(heater, sensor);
        cs.update();
    }
}
```