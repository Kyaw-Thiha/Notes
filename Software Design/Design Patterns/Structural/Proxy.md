# Proxy

> [!summary] Main Idea
> Used when we want another object to control access to a real object.

`Technique`
1. Create a `Subject Interface`
2. Create the `Real Subject`
3. Create the `Proxy Subject`

---
`Code Example`

Subject Interface
```java
interface FileLoader {
    void displayFile();
}
```

Real Object
```java
class RealFileLoader implements FileLoader {
    private String filename;

    public RealFileLoader(String filename) {
        this.filename = filename;
        loadFromDisk();
    }

    private void loadFromDisk() {
        System.out.println("Loading file from disk: " + filename);
    }

    @Override
    public void displayFile() {
        System.out.println("Displaying file: " + filename);
    }
}
```

Proxy
```java
class FileLoaderProxy implements FileLoader {
    private String filename;
    private RealFileLoader realLoader;
    private boolean hasAccess;

    public FileLoaderProxy(String filename, boolean hasAccess) {
        this.filename = filename;
        this.hasAccess = hasAccess;
    }

    @Override
    public void displayFile() {
        if (!hasAccess) {
            System.out.println("Access denied: " + filename);
            return;
        }

        // Lazy initialization
        if (realLoader == null) {
            realLoader = new RealFileLoader(filename);
        }

        realLoader.displayFile();
    }
}
```

---
`ML Example`

Subject Interface
```c
#include <iostream>
#include <vector>

class Model {
public:
    virtual std::vector<float> predict(const std::vector<float>& input) = 0;
    virtual ~Model() = default;
};
```

Real Subject
```c
class RealModel : public Model {
public:
    RealModel() {
        std::cout << "[RealModel] Loading large model weights from disk...\n";
        // Pretend it takes a long time...
    }

    std::vector<float> predict(const std::vector<float>& input) override {
        std::cout << "[RealModel] Running inference...\n";
        return { 0.42f }; // dummy output
    }
};
```

Proxy Subject
```c
class ModelProxy : public Model {
private:
    RealModel* realModel = nullptr;

    void initIfNeeded() {
        if (realModel == nullptr) {
            std::cout << "[Proxy] Lazy-loading real model now.\n";
            realModel = new RealModel();
        }
    }

public:
    std::vector<float> predict(const std::vector<float>& input) override {
        initIfNeeded();
        return realModel->predict(input);
    }

    ~ModelProxy() {
        delete realModel;
    }
};
```

---