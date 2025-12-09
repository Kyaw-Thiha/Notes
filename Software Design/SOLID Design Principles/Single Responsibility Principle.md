# Single Responsibility Principle

> Each `class` should handle one main responsibility only.

`Why SRP?`
If we do not follow `SRP`, then 
- One change can break multiple functionalities
- Can't reuse logic easily
- Merge conflicts

---

`Bad Example`

Note that the class `Invoice` is handling many things.

```java
class Invoice {
    double amount;

    public Invoice(double amount) {
        this.amount = amount;
    }

    // BAD: formatting + I/O + business logic all mixed
    public void print() {
        System.out.println("Invoice amount: " + amount);
    }

    public void saveToDatabase() {
        System.out.println("Saving invoice to database...");
    }
}
```


---

`Good Example`


```java
class Invoice {
    double amount;

    public Invoice(double amount) {
        this.amount = amount;
    }

    public double getAmount() {
        return amount;
    }
}
```

```java
class InvoicePrinter {
    public void print(Invoice invoice) {
        System.out.println("Invoice amount: " + invoice.getAmount());
    }
}
```

```java
class InvoiceRepository {
    public void save(Invoice invoice) {
        System.out.println("Saving invoice to database...");
    }
}
```

```java
public class Main {
    public static void main(String[] args) {
        Invoice invoice = new Invoice(100);

        new InvoicePrinter().print(invoice);
        new InvoiceRepository().save(invoice);
    }
}
```