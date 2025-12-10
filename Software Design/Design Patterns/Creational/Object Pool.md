# Object Pool

> [!summary] Main Idea
> Creating objects can be expensive.
> So, instead of creating new ones everytime, we reuse objects from a managed pool.

`Technique`
1. Maintain a pool of objects: available objects & used objects.
2. When client request object, return available object, or create a new one.
3. When client returns object, put them in available set instead of destroying it.

---
`Code Example`

```java
class Connection {
    private final int id;

    public Connection(int id) {
        this.id = id;
        System.out.println("Creating connection #" + id);
    }

    public void connect() {
        System.out.println("Using connection #" + id);
    }
}
```

```java
import java.util.*;

class ConnectionPool {
    private final Queue<Connection> available = new LinkedList<>();
    private final Set<Connection> inUse = new HashSet<>();

    private int nextId = 1;
    private final int MAX_POOL_SIZE = 5;

    public Connection acquire() {
        Connection conn;

        if (!available.isEmpty()) {
            conn = available.poll();
        } else if (inUse.size() < MAX_POOL_SIZE) {
            conn = new Connection(nextId++);
        } else {
            throw new RuntimeException("No available connections!");
        }

        inUse.add(conn);
        return conn;
    }

    public void release(Connection conn) {
        if (inUse.remove(conn)) {
            available.offer(conn);
        }
    }
}
```

---
`ML Example`

```c
#include <iostream>
#include <vector>

class Tensor {
private:
    int batchSize;
    int features;
    std::vector<float> data; // pretend this is big, or lives on GPU

public:
    Tensor(int batchSize, int features)
        : batchSize(batchSize),
          features(features),
          data(batchSize * features)
    {
        std::cout << "[Tensor] Allocating buffer: "
                  << "batchSize=" << batchSize
                  << ", features=" << features << "\n";
    }

    void loadBatch(int batchIndex) {
        // Simulate loading data
        std::fill(data.begin(), data.end(), static_cast<float>(batchIndex));
        std::cout << "  -> Loaded batch " << batchIndex << " into tensor\n";
    }

    void forward() {
        // Fake forward pass
        std::cout << "  -> Running forward on batch (size "
                  << batchSize << " x " << features << ")\n";
    }

    void backward() {
        // Fake backward pass
        std::cout << "  -> Running backward on batch\n";
    }
};
```

```c
#include <queue>
#include <memory>
#include <stdexcept>

class TensorPool {
private:
    int batchSize;
    int features;
    std::size_t maxPoolSize;

    std::vector<std::unique_ptr<Tensor>> storage; // owns all tensors
    std::queue<Tensor*> available;                // pointers to free tensors

public:
    TensorPool(int batchSize, int features, std::size_t maxPoolSize)
        : batchSize(batchSize),
          features(features),
          maxPoolSize(maxPoolSize)
    {}

    Tensor* acquire() {
        if (!available.empty()) {
            Tensor* t = available.front();
            available.pop();
            std::cout << "[Pool] Reusing existing tensor\n";
            return t;
        }

        if (storage.size() < maxPoolSize) {
            std::cout << "[Pool] Creating new tensor\n";
            storage.emplace_back(std::make_unique<Tensor>(batchSize, features));
            return storage.back().get();
        }

        throw std::runtime_error("No available tensors in pool and maxPoolSize reached");
    }

    void release(Tensor* t) {
        std::cout << "[Pool] Returning tensor to pool\n";
        available.push(t);
    }
};
```

```c
int main() {
    const int BATCH_SIZE = 64;
    const int FEATURES   = 1024;
    const std::size_t MAX_TENSORS = 2; // we allow at most 2 buffers

    TensorPool pool(BATCH_SIZE, FEATURES, MAX_TENSORS);

    // Simulate 5 training iterations
    for (int step = 0; step < 5; ++step) {
        std::cout << "\n=== Training step " << step << " ===\n";

        // Acquire a tensor from the pool
        Tensor* batchTensor = pool.acquire();

        // Use it
        batchTensor->loadBatch(step);
        batchTensor->forward();
        batchTensor->backward();

        // Return it to the pool
        pool.release(batchTensor);
    }

    return 0;
}
```

---
