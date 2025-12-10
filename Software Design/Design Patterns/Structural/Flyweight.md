# Flyweight

> [!summary] Main Idea
> Suppose you have many objects that repeat a lot of internal data.
> Then, we can share the repeated data to save memory.

`Technique`
1. Split object state into two kinds
	- `Intrinsic` state: shared and never change
	- `Extrinsic` state: provided from outside
2. Store all intrinsic state in a shared object (`Flyweight`)
3. Define a `Flyweight Factory` that create flyweights, cache them and return existing ones.

---
`Code Example`

Flyweight
```java
class TreeType {
    private final String name;
    private final String color;
    private final String texture;

    public TreeType(String name, String color, String texture) {
        this.name = name;
        this.color = color;
        this.texture = texture;
    }

    public void draw(int x, int y) {
        System.out.println("Drawing " + name + " tree at (" + x + ", " + y + ")");
    }
}
```

Flyweight Factory
```java
import java.util.HashMap;
import java.util.Map;

class TreeFactory {
    private static Map<String, TreeType> treeTypes = new HashMap<>();

    public static TreeType getTreeType(String name, String color, String texture) {
        String key = name + color + texture;

        if (!treeTypes.containsKey(key)) {
            treeTypes.put(key, new TreeType(name, color, texture));
        }

        return treeTypes.get(key);
    }
}
```

Main Object
```java
class Tree {
    private int x;
    private int y;
    private TreeType type;  // shared flyweight

    public Tree(int x, int y, TreeType type) {
        this.x = x;
        this.y = y;
        this.type = type;
    }

    public void draw() {
        type.draw(x, y);
    }
}
```

```java
import java.util.ArrayList;
import java.util.List;

class Forest {
    private List<Tree> trees = new ArrayList<>();

    public void plantTree(int x, int y, String name, String color, String texture) {
        TreeType type = TreeFactory.getTreeType(name, color, texture);
        trees.add(new Tree(x, y, type));
    }

    public void draw() {
        for (Tree tree : trees) {
            tree.draw();
        }
    }
}
```

Main Driver
```java
public class Main {
    public static void main(String[] args) {
        Forest forest = new Forest();

        // plant 3 million trees, only 2 tree types!
        forest.plantTree(10, 20, "Oak", "Green", "rough");
        forest.plantTree(30, 40, "Oak", "Green", "rough");
        forest.plantTree(50, 60, "Pine", "Dark Green", "smooth");

        forest.draw();
    }
}
```

---
`ML Example`

Flyweight
```java
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>

class EmbeddingVector {
private:
    std::vector<float> values; // big memory block

public:
    EmbeddingVector(int dim) : values(dim, 0.1f) {}

    void print(int tokenId) const {
        std::cout << "Embedding for token " << tokenId 
                  << " (shared vector @" << this << ")\n";
    }
};
```

Flyweight Factory
```java
class EmbeddingFactory {
private:
    static inline std::unordered_map<int, std::shared_ptr<EmbeddingVector>> pool;

public:
    static std::shared_ptr<EmbeddingVector> getEmbedding(int tokenId, int dim) {
        int key = tokenId % 100;  // hash bucket (simulate shared embeddings)

        if (pool.find(key) == pool.end()) {
            pool[key] = std::make_shared<EmbeddingVector>(dim);
        }

        return pool[key];
    }
};
```

Main Object
```c
class Token {
private:
    int id;
    std::shared_ptr<EmbeddingVector> embedding;

public:
    Token(int id, int dim)
        : id(id), embedding(EmbeddingFactory::getEmbedding(id, dim)) {}

    void use() {
        embedding->print(id);
    }
};
```

Main Driver
```c
int main() {
    const int EMBED_DIM = 768;

    Token t1(42, EMBED_DIM);
    Token t2(142, EMBED_DIM);
    Token t3(242, EMBED_DIM);

    t1.use();
    t2.use();
    t3.use();

    return 0;
}
```
