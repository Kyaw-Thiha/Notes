# Singleton

> [!summary] Main Idea
> Ensure that a class has exactly one instance in the program.

`Techniques`

Make `constructor` private.
Store a single static instance in the class.
Provide public static method like `get_instance`

---
`Code Example`

```java
public class Database {
    private static Database instance;   
    
    private Database() {
    }

    public static synchronized Database getInstance() {
        if (instance == null) {
            instance = new Database();
        }
        return instance;
    }

    public void connect() {
        System.out.println("Connected!");
    }
}
```

`ML Example`

```c
#include <torch/torch.h>

class DeviceManager {
private:
    torch::Device device;

    DeviceManager() {
        if (torch::cuda::is_available())
            device = torch::Device(torch::kCUDA);
        else
            device = torch::Device(torch::kCPU);
    }

public:
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    static DeviceManager& getInstance() {
        static DeviceManager instance;  // Thread-safe in C++11+
        return instance;
    }

    torch::Device getDevice() const {
        return device;
    }
};
```

---
## See Also
- [[Important Design Patterns]]