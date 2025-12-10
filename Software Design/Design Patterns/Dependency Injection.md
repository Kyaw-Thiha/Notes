# Dependency Injection

> [!summary] Main Idea
> Used when the class does not build what it needs, but rather receive them.

---
`Code Example`

Abstract Dependency
```java
interface Notifier {
    void send(String user, String message);
}
```

Concrete Dependencies
```java
class EmailNotifier implements Notifier {
    public void send(String user, String msg) {
        System.out.println("Email to " + user + ": " + msg);
    }
}

class SMSNotifier implements Notifier {
    public void send(String user, String msg) {
        System.out.println("SMS to " + user + ": " + msg);
    }
}
```

High-Level Class
```java
class AlertService {
    private Notifier notifier;

    public AlertService(Notifier notifier) { // constructor injection
        this.notifier = notifier;
    }

    public void sendAlert(String user) {
        notifier.send(user, "Alert triggered!");
    }
}
```

Main Client Driver
```java
Notifier email = new EmailNotifier();
Notifier sms = new SMSNotifier();

AlertService alert1 = new AlertService(email);
AlertService alert2 = new AlertService(sms);

alert1.sendAlert("Kevin");
alert2.sendAlert("Kevin");
```

---
`ML Example`

Abstraction Interface
```c
class Backend {
public:
    virtual ~Backend() = default;
    virtual void matmul(const float* A, const float* B, float* C, int n) = 0;
    virtual void relu(float* X, int n) = 0;
};
```

Concrete Implementation
```c
class CpuBackend : public Backend {
public:
    void matmul(const float* A, const float* B, float* C, int n) override {
        // Simple CPU implementation
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int k = 0; k < n; k++)
                    sum += A[i*n+k] * B[k*n+j];
                C[i*n+j] = sum;
            }
    }
    
    void relu(float* X, int n) override {
        for (int i = 0; i < n; i++)
            X[i] = std::max(0.0f, X[i]);
    }
};
```

```c
class CudaBackend : public Backend {
public:
    void matmul(const float* A, const float* B, float* C, int n) override {
        std::cout << "[CUDA] matmul executed\n";
        // pretend to run kernel
    }
    
    void relu(float* X, int n) override {
        std::cout << "[CUDA] relu executed\n";
    }
};
```

```c
class MockBackend : public Backend {
public:
    void matmul(const float* A, const float* B, float* C, int n) override {
        std::cout << "[MOCK] matmul\n";
    }

    void relu(float* X, int n) override {
        std::cout << "[MOCK] relu\n";
    }
};
```

High Level Class
```c
class InferenceEngine {
public:
    InferenceEngine(Backend* backend) : backend_(backend) {}

    void run(float* input, float* weights, int n) {
        backend_->matmul(input, weights, input, n);
        backend_->relu(input, n);
    }

private:
    Backend* backend_;
};
```

Main Client Driver
```c
int main() {
    float input[4] = {1, -2, 3, -4};
    float weights[4] = {1, 1, 1, 1};

    Backend* cpu = new CpuBackend();
    Backend* cuda = new CudaBackend();
    Backend* mock = new MockBackend();

    InferenceEngine engine1(cpu);
    InferenceEngine engine2(cuda);
    InferenceEngine engine3(mock);

    std::cout << "Running on CPU:\n";
    engine1.run(input, weights, 2);

    std::cout << "\nRunning on CUDA:\n";
    engine2.run(input, weights, 2);

    std::cout << "\nRunning with MOCK:\n";
    engine3.run(input, weights, 2);
}
```

---
