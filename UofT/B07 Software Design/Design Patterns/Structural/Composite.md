# Composite

> [!summary] Main Idea
> Treat single objects, and group of objects in the SAME way.
> Useful when objects have hierarchical tree structure.

`Technique`
1. Define a `Component interface`
2. Define `Composite class` which stores list of Components, and implement operations by forwarding to each child.
3. Define `Leaf class` which cannot have children.

---
`Code Example`

`Component Interface`
```java
interface FileSystemComponent {
    void show(String indent); // indent for pretty printing
}
```

`Leaf`
```java
class FileItem implements FileSystemComponent {
    private String name;

    public FileItem(String name) {
        this.name = name;
    }

    @Override
    public void show(String indent) {
        System.out.println(indent + "- " + name);
    }
}
```

`Composite`
```java
import java.util.ArrayList;
import java.util.List;

class Folder implements FileSystemComponent {
    private String name;
    private List<FileSystemComponent> children = new ArrayList<>();

    public Folder(String name) {
        this.name = name;
    }

    public void add(FileSystemComponent component) {
        children.add(component);
    }

    public void remove(FileSystemComponent component) {
        children.remove(component);
    }

    @Override
    public void show(String indent) {
        System.out.println(indent + "+ " + name);
        for (FileSystemComponent child : children) {
            child.show(indent + "   ");
        }
    }
}
```

`Main Driver`
```java
public class Main {
    public static void main(String[] args) {

        Folder root = new Folder("root");
        Folder documents = new Folder("Documents");
        Folder pictures = new Folder("Pictures");

        FileItem resume = new FileItem("resume.pdf");
        FileItem photo = new FileItem("photo.png");

        documents.add(resume);
        pictures.add(photo);

        root.add(documents);
        root.add(pictures);

        root.show("");
    }
}
```

---
`ML Example`

Component Interface
```c
#include <iostream>
#include <vector>
#include <memory>

class Node {
public:
    virtual void forward() = 0;
    virtual ~Node() = default;
};
```

Leaf
```c
class Linear : public Node {
private:
    int in, out;
public:
    Linear(int in, int out) : in(in), out(out) {}

    void forward() override {
        std::cout << "Linear(" << in << " -> " << out << ")\n";
    }
};
```

```c
class ReLU : public Node {
public:
    void forward() override {
        std::cout << "ReLU\n";
    }
};
```

```c
class Dropout : public Node {
private:
    float rate;
public:
    Dropout(float rate) : rate(rate) {}

    void forward() override {
        std::cout << "Dropout(rate=" << rate << ")\n";
    }
};
```

Composite
```c
class Sequential : public Node {
private:
    std::vector<std::unique_ptr<Node>> children;

public:
    void add(std::unique_ptr<Node> node) {
        children.push_back(std::move(node));
    }

    void forward() override {
        std::cout << "[Sequential Begin]\n";
        for (auto &child : children) {
            child->forward();
        }
        std::cout << "[Sequential End]\n";
    }
};
```

```c
class ResidualBlock : public Node {
private:
    std::unique_ptr<Node> mainPath;
    std::unique_ptr<Node> skipPath;

public:
    ResidualBlock(std::unique_ptr<Node> main, std::unique_ptr<Node> skip)
        : mainPath(std::move(main)), skipPath(std::move(skip)) {}

    void forward() override {
        std::cout << "[Residual Block]\n";
        mainPath->forward();
        skipPath->forward();
        std::cout << "Add(main, skip)\n";
    }
};
```

Driver
```c
int main() {
    Sequential model;

    // Add layers to main model
    model.add(std::make_unique<Linear>(128, 64));
    model.add(std::make_unique<ReLU>());

    // Build a residual block (composite inside a composite)
    auto main = std::make_unique<Sequential>();
    main->add(std::make_unique<Linear>(64, 64));
    main->add(std::make_unique<ReLU>());

    auto skip = std::make_unique<Linear>(64, 64);

    model.add(std::make_unique<ResidualBlock>(std::move(main), std::move(skip)));

    // Final classifier layer
    model.add(std::make_unique<Dropout>(0.5f));
    model.add(std::make_unique<Linear>(64, 10));

    // Run forward pass
    model.forward();

    return 0;
}
```

---
