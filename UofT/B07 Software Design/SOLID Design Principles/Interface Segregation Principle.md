# Interface Segregation Principle

> Clients should not be forced to depend on methods they do not use.
> Solution: Use small, focused interface

---
`Bad Example`


```java
interface Device {
    void turnOn();
    void turnOff();
    void setBrightness(int level);
    void playSound(String audio);
    void openDoor();
    void closeDoor();
}
```

Different devices do not need all these methods:
- Light only needs brightness + on/off
- Speaker only needs playSound
- Door only needs open/close
- Heater only needs on/off

```java
class Light implements Device {
    public void turnOn() {}
    public void turnOff() {}
    
    public void setBrightness(int level) {}

    // Useless for a light
    public void playSound(String audio) { 
        throw new UnsupportedOperationException();
    }

    public void openDoor() {
        throw new UnsupportedOperationException();
    }

    public void closeDoor() {
        throw new UnsupportedOperationException();
    }
}
```

---
`Good Example`

```java
interface Switchable {
    void turnOn();
    void turnOff();
}

interface Dimmable {
    void setBrightness(int level);
}

interface Playable {
    void playSound(String audio);
}

interface Openable {
    void open();
    void close();
}
```

```java
class Light implements Switchable, Dimmable {
    public void turnOn() {}
    public void turnOff() {}
    public void setBrightness(int level) {}
}

class Speaker implements Switchable, Playable {
    public void turnOn() {}
    public void turnOff() {}
    public void playSound(String audio) {}
}

class Door implements Openable {
    public void open() {}
    public void close() {}
}

class Heater implements Switchable {
    public void turnOn() {}
    public void turnOff() {}
}
```