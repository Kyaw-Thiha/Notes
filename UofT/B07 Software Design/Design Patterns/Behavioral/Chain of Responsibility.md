# Chain of Responsibility

> [!summary] Main Idea
> Suppose we have multiple objects that can handle a request.
> Then, we can use CoR to not need the sender to know which one will handle the request.

`Technique`
1. Define a `Handler` interface which have
	- `setNext()` - sets next handler
	- `handle()` - process or pass on the request
2. Create `concrete handlers`
3. Link the `handlers` into a chain so that user can send request to the first one.

---
`Code Example`

Handler Interface
```java
abstract class Handler {
    protected Handler next;

    public Handler setNext(Handler next) {
        this.next = next;
        return next;
    }

    public abstract void handle(String username, String password);
}
```

Concrete Handlers
```java
class UserExistsHandler extends Handler {
    @Override
    public void handle(String username, String password) {
        if (!username.equals("Alice")) {
            System.out.println("User does not exist.");
            return; // stop chain
        }

        System.out.println("User exists.");
        if (next != null) next.handle(username, password);
    }
}
```

```java
class PasswordFormatHandler extends Handler {
    @Override
    public void handle(String username, String password) {
        if (password.length() < 6) {
            System.out.println("Password too short.");
            return;
        }

        System.out.println("Password format OK.");
        if (next != null) next.handle(username, password);
    }
}
```

```java
class AuthorizationHandler extends Handler {
    @Override
    public void handle(String username, String password) {
        if (!username.equals("Alice")) {
            System.out.println("User is unauthorized.");
            return;
        }

        System.out.println("Authorization successful. Access granted!");
    }
}
```

Main Driver (where the chain is built)
```java
public class Main {
    public static void main(String[] args) {

        Handler chain = new UserExistsHandler();
        chain.setNext(new PasswordFormatHandler())
             .setNext(new AuthorizationHandler());

        System.out.println("---- Test 1 ----");
        chain.handle("Bob", "123456");

        System.out.println("\n---- Test 2 ----");
        chain.handle("Alice", "123");
        
        System.out.println("\n---- Test 3 ----");
        chain.handle("Alice", "correctPassword");
    }
}
```

---
`ML Example`

Data Structure
```c
#include <iostream>
#include <vector>
#include <memory>

struct Data {
    std::vector<float> pixels;
    bool corrupted = false;
};
```

Base Handler
```c
class DataHandler {
protected:
    std::unique_ptr<DataHandler> next = nullptr;

public:
    virtual ~DataHandler() = default;

    DataHandler* setNext(std::unique_ptr<DataHandler> nextHandler) {
        next = std::move(nextHandler);
        return next.get();
    }

    virtual void handle(Data& data) {
        if (next) next->handle(data);
    }
};
```

Concrete Handlers
```c
class CorruptionCheckHandler : public DataHandler {
public:
    void handle(Data& data) override {
        if (data.corrupted) {
            std::cout << "[CorruptionCheck] Data is corrupted. Stopping pipeline.\n";
            return; // stop chain
        }
        std::cout << "[CorruptionCheck] Data OK.\n";
        DataHandler::handle(data);
    }
};
```

```c
class ResizeHandler : public DataHandler {
public:
    void handle(Data& data) override {
        std::cout << "[Resize] Resizing image to 256 pixels.\n";
        data.pixels.resize(256);
        DataHandler::handle(data);
    }
};
```

```c
class NormalizeHandler : public DataHandler {
public:
    void handle(Data& data) override {
        std::cout << "[Normalize] Normalizing pixel values.\n";
        for (auto &p : data.pixels) p /= 255.0f;
        DataHandler::handle(data);
    }
};
```

```c
class AugmentHandler : public DataHandler {
public:
    void handle(Data& data) override {
        std::cout << "[Augment] Applying random flip.\n";
        // pretend to flip
        DataHandler::handle(data);
    }
};
```

Main Driver
```c
int main() {
    // Build the processing chain
    std::unique_ptr<DataHandler> chain = std::make_unique<CorruptionCheckHandler>();
    chain->setNext(std::make_unique<ResizeHandler>())
         ->setNext(std::make_unique<NormalizeHandler>())
         ->setNext(std::make_unique<AugmentHandler>());

    // Example 1: clean data
    Data cleanData{ std::vector<float>(100, 128), false };
    std::cout << "\n=== Processing clean data ===\n";
    chain->handle(cleanData);

    // Example 2: corrupted data
    Data badData{ std::vector<float>(100, 255), true };
    std::cout << "\n=== Processing CORRUPTED data ===\n";
    chain->handle(badData);

    return 0;
}
```

---
