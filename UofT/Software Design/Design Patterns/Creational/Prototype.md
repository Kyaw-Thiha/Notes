# Prototype

> [!summary] Main Idea
> Creating new object by copying existing objects, and not calling the constructor.

`Technique`
1. Define a `clone()` method in the base interface or abstract class.
2. Each concrete classes have their own `clone()` logic
3. Optionally, maintain a `Prototype Registry` so that we can clone from reusable prototypes.

---
`Code Example`

`Prototype Interface`
```java
interface Prototype<T> {
    T clone();
}
```

`Concrete Prototype`
```java
class Document implements Prototype<Document> {
    private String title;
    private StringBuilder content; // mutable → deep copy needed

    public Document(String title, StringBuilder content) {
        this.title = title;
        this.content = content;
    }

    @Override
    public Document clone() {
        return new Document(
            this.title,
            new StringBuilder(this.content.toString()) // deep copy
        );
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public void append(String text) {
        this.content.append(text);
    }

    public void show() {
        System.out.println("[" + title + "]");
        System.out.println(content);
    }
}
```

`Prototype Registry`
```java
import java.util.HashMap;
import java.util.Map;

class PrototypeRegistry {
    private static Map<String, Prototype<?>> registry = new HashMap<>();

    public static void addPrototype(String key, Prototype<?> prototype) {
        registry.put(key, prototype);
    }

    public static <T> T getClone(String key) {
        @SuppressWarnings("unchecked")
        T cloned = (T) registry.get(key).clone();
        return cloned;
    }
}
```

`Driver`
```java
public class Main {
    public static void main(String[] args) {

        // ------------------------
        // Register prototypes
        // ------------------------
        PrototypeRegistry.addPrototype("letter",
            new Document("Letter Template",
                new StringBuilder("Dear Sir/Madam,\n\n")));

        PrototypeRegistry.addPrototype("resume",
            new Document("Resume Template",
                new StringBuilder("Name:\nExperience:\nSkills:\n")));

        PrototypeRegistry.addPrototype("invoice",
            new Document("Invoice Template",
                new StringBuilder("Invoice #____\nItems:\n")));
        

        // ------------------------
        // Clone documents
        // ------------------------
        Document myLetter = PrototypeRegistry.getClone("letter");
        Document myResume = PrototypeRegistry.getClone("resume");
        Document myInvoice = PrototypeRegistry.getClone("invoice");

        // Modify each clone independently
        myLetter.append("I am writing to apply...");
        myResume.append("Name: Kevin\nExperience: ...");
        myInvoice.append("Item 1: $20\nItem 2: $30");


        // ------------------------
        // Display
        // ------------------------
        myLetter.show();
        System.out.println();

        myResume.show();
        System.out.println();

        myInvoice.show();
    }
}
```