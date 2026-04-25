# Visitor

> [!summary] Main Idea
> Add new operations to a group of classes without modifying them.

`Technique`
1. Creates `Visitor` interface.
2. Creates the element class that accepts visitors.
3. Implement concrete visitors.

---
`Code Example`

Main Element
```java
interface Item {
    void accept(Visitor visitor);
```

```java
class Book implements Item {
    int price = 20;

    @Override
    public void accept(Visitor visitor) {
        visitor.visit(this);
    }
}

class Fruit implements Item {
    int weight = 2; // kg
    int pricePerKg = 4;

    @Override
    public void accept(Visitor visitor) {
        visitor.visit(this);
    }
}
```

Visitor Interface
```java
interface Visitor {
    void visit(Book book);
    void visit(Fruit fruit);
}
```

Concrete Visitors
```java
class PriceCalculator implements Visitor {
    private int total = 0;

    @Override
    public void visit(Book book) {
        total += book.price;
    }

    @Override
    public void visit(Fruit fruit) {
        total += fruit.pricePerKg * fruit.weight;
    }

    public int getTotal() {
        return total;
    }
}
```

```java
class PrettyPrintVisitor implements Visitor {
    @Override
    public void visit(Book book) {
        System.out.println("Book: $" + book.price);
    }

    @Override
    public void visit(Fruit fruit) {
        System.out.println("Fruit: " 
            + fruit.weight + "kg @ $" + fruit.pricePerKg + "/kg");
    }
}
```

Main Client Driver
```java
List<Item> cart = List.of(new Book(), new Fruit());

PriceCalculator calc = new PriceCalculator();
PrettyPrintVisitor printer = new PrettyPrintVisitor();

for (Item i : cart) {
    i.accept(calc);
    i.accept(printer);
}

System.out.println("Total = $" + calc.getTotal());
```

---
`ML Example`

Base + Concrete Nodes
```c
#include <iostream>
#include <memory>
#include <vector>

class Visitor; // forward declaration

// Base Node
class Node {
public:
    virtual ~Node() = default;
    virtual void accept(Visitor& v) = 0;
};

// Concrete Nodes
class Add : public Node {
public:
    Node* left;
    Node* right;

    Add(Node* l, Node* r) : left(l), right(r) {}

    void accept(Visitor& v) override;
};

class MatMul : public Node {
public:
    Node* A;
    Node* B;

    MatMul(Node* a, Node* b) : A(a), B(b) {}

    void accept(Visitor& v) override;
};

class ReLU : public Node {
public:
    Node* input;

    ReLU(Node* x) : input(x) {}

    void accept(Visitor& v) override;
};
```

Visitor Interface
```c
class Visitor {
public:
    virtual void visit(Add& node) = 0;
    virtual void visit(MatMul& node) = 0;
    virtual void visit(ReLU& node) = 0;
};
```

Concrete Visitors
```c
class ShapeInferenceVisitor : public Visitor {
public:
    void visit(Add& node) override {
        std::cout << "Infer shape: Add(left, right)\n";
    }
    
    void visit(MatMul& node) override {
        std::cout << "Infer shape: MatMul(A, B)\n";
    }
    
    void visit(ReLU& node) override {
        std::cout << "Infer shape: ReLU(input)\n";
    }
};
```

```c
class OptimizeVisitor : public Visitor {
public:
    void visit(Add& node) override {
        std::cout << "Optimize Add: maybe fold constants\n";
    }

    void visit(MatMul& node) override {
        std::cout << "Optimize MatMul: check for transpose fusion\n";
    }

    void visit(ReLU& node) override {
        std::cout << "Optimize ReLU: remove redundant ReLUs\n";
    }
};
```

Main Client Driver
```c
int main() {
    Node* x = new Add(nullptr, nullptr);  // pretend inputs
    Node* y = new Add(nullptr, nullptr);
    Node* sum = new Add(x, y);

    Node* mat = new MatMul(sum, nullptr);
    Node* relu = new ReLU(mat);

    ShapeInferenceVisitor shapePass;
    OptimizeVisitor optPass;

    std::vector<Node*> graph = { sum, mat, relu };

    std::cout << "=== Shape Inference ===\n";
    for (Node* n : graph) {
        n->accept(shapePass);
    }

    std::cout << "\n=== Optimization Pass ===\n";
    for (Node* n : graph) {
        n->accept(optPass);
    }
}
```

---