# Adapter

> [!summary] Main Idea
> Allow two classes with different interfaces to work together.

`Technique`
- Implement the `target interface`
- Internally hold reference to `adaptee`
- Translate method calls from `target interface` to `adaptee`

---
`Code Example`

Target Interface (What client expects)
```java
interface USBDevice {
    void connectWithUSB();
}
```

Adaptee
```java
class HDMIConnector {
    public void connectWithHDMI() {
        System.out.println("Connected using HDMI");
    }
}
```

Adapter
```java
class HDMItoUSBAdapter implements USBDevice {

    private HDMIConnector hdmi;

    public HDMItoUSBAdapter(HDMIConnector hdmi) {
        this.hdmi = hdmi;
    }

    @Override
    public void connectWithUSB() {
        System.out.println("Adapter converting USB → HDMI...");
        hdmi.connectWithHDMI();
    }
}
```

---
## See Also
- [[Important Design Patterns]]