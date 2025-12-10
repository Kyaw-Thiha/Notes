# Interpreter

> [!summary] Main Idea
> Used to define a simple language grammar.
> We can then interpret/evaluate sentences in that language.

`Technique`
1. Define an `Expression Node` that implement `interpret()`
2. Create `Terminal Expressions` 
   Eg: numbers, variables, constants
3. Create `Non-Terminal Expressions`
   Eg: AndExpression, OrExpression, EqualsExpression
4. Build an `Expression Tree`
5. Interpret the tree recursively.

---
`Code Example`

Expression
```java
interface Expression {
    boolean interpret();
}
```

Terminal Expression
```java
class Literal implements Expression {
    private boolean value;

    public Literal(boolean value) {
        this.value = value;
    }

    @Override
    public boolean interpret() {
        return value;
    }
}
```

Non-Terminal Expressions
```java
class AndExpression implements Expression {
    private Expression left;
    private Expression right;

    public AndExpression(Expression left, Expression right) {
        this.left = left;
        this.right = right;
    }

    @Override
    public boolean interpret() {
        return left.interpret() && right.interpret();
    }
}
```

```java
class OrExpression implements Expression {
    private Expression left;
    private Expression right;

    public OrExpression(Expression left, Expression right) {
        this.left = left;
        this.right = right;
    }

    @Override
    public boolean interpret() {
        return left.interpret() || right.interpret();
    }
}
```

Main driver to build Expression Tree
```java
public class Main {
    public static void main(String[] args) {
        Expression expr =
            new OrExpression(
                new AndExpression(
                    new Literal(true),
                    new Literal(false)
                ),
                new Literal(true)
            );

        System.out.println(expr.interpret());
    }
}
```

---
`ML Example`

Context
```c
#include <iostream>
#include <memory>

struct LossContext {
    float mse;
    float l1;
};
```

Expression
```c
class Expression {
public:
    virtual float interpret(const LossContext& ctx) = 0;
    virtual ~Expression() = default;
};
```

Terminal Expressions
```c
class Number : public Expression {
    float value;
public:
    Number(float v) : value(v) {}
    float interpret(const LossContext&) override {
        return value;
    }
};
```

```c
class MSE : public Expression {
public:
    float interpret(const LossContext& ctx) override {
        return ctx.mse;
    }
};
```

```c
class L1 : public Expression {
public:
    float interpret(const LossContext& ctx) override {
        return ctx.l1;
    }
};
```

Non-Terminal Expressions
```c
class Add : public Expression {
    std::unique_ptr<Expression> left, right;
public:
    Add(std::unique_ptr<Expression> l, std::unique_ptr<Expression> r)
        : left(std::move(l)), right(std::move(r)) {}

    float interpret(const LossContext& ctx) override {
        return left->interpret(ctx) + right->interpret(ctx);
    }
};
```

```c
class Multiply : public Expression {
    std::unique_ptr<Expression> left, right;
public:
    Multiply(std::unique_ptr<Expression> l, std::unique_ptr<Expression> r)
        : left(std::move(l)), right(std::move(r)) {}

    float interpret(const LossContext& ctx) override {
        return left->interpret(ctx) * right->interpret(ctx);
    }
};
```

Main Driver to build Expression Tree
```c
int main() {
    // Expression: MSE + (0.1 * L1)
    std::unique_ptr<Expression> expr =
        std::make_unique<Add>(
            std::make_unique<MSE>(),
            std::make_unique<Multiply>(
                std::make_unique<Number>(0.1f),
                std::make_unique<L1>()
            )
        );

    LossContext context{ .mse = 2.0f, .l1 = 5.0f };

    float result = expr->interpret(context);

    std::cout << "Final Loss = " << result << "\n";
}
```

---

