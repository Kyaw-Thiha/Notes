# Iterator

> [!summary] Main Idea
> Traversing a collection without exposing its internal structure.
> This means that the underlying representation of the collection could be array, list, tree or graph.

`Technique`
1. Create an `Iterator` interface with `hasNext()` and `next()`
2. Make the collection produce iterators with `createIterator()`
3. Create `concrete iterator` class

---
`Code Example`

Iterator Interface
```java
interface Iterator<T> {
    boolean hasNext();
    T next();
}
```

Collection
```java
interface IterableCollection<T> {
    Iterator<T> createIterator();
}
```

Concrete Iterator
```java
class NameIterator implements Iterator<String> {
    private NameCollection collection;
    private int index = 0;

    public NameIterator(NameCollection collection) {
        this.collection = collection;
    }

    @Override
    public boolean hasNext() {
        return index < collection.size();
    }

    @Override
    public String next() {
        return collection.get(index++);
    }
}
```

Concrete Collection
```java
class NameCollection implements IterableCollection<String> {
    private String[] names;

    public NameCollection(String[] names) {
        this.names = names;
    }

    @Override
    public Iterator<String> createIterator() {
        return new NameIterator(this);
    }

    public String get(int index) {
        return names[index];
    }

    public int size() {
        return names.length;
    }
}
```

Client Main Driver
```java
public class Main {
    public static void main(String[] args) {
        String[] data = {"Alice", "Bob", "Charlie"};
        NameCollection collection = new NameCollection(data);

        Iterator<String> it = collection.createIterator();

        while (it.hasNext()) {
            System.out.println(it.next());
        }
    }
}
```

---
