# Makefile
### Basic Usage
```makefile
bb.o : bb.c bb.h rect.h
	gcc -c bb.c
# ^ that's a real TAB character, NOT 8 spaces!
```

- Target(`bb.o`): The file this rule produces
- Prerequisites(`bb.c`,`bb.h`,`rect.h`): Files the target depends on.
- Recipe(`gcc -c bb.c`): Shell command to build the target.

> If `bb.o` is _missing_, OR it's _older_ than at least one of its prerequisites (`bb.c`, `bb.h`, `rect.h`), then run the recipe (`gcc -c bb.c`).

---
### Running `make`
- `make`(no arguments): runs the **first rule** in the Makefile
- `make target`: runs whichever rule's target matches `target`.
- Rules can **recursively trigger** other rules.
- **Order of rules in the file doesn't matter** for correctness.

---
### Build Everything Target
```makefile
all : myexe1 myexe2 myexe3
.PHONY: all
```

`.PHONY: all` tells `make` that `all` is just a label.

### Clean Target
```makefile
clean :
	rm -f *.o myexe1 myexe2 myexe3
.PHONY: clean
```
A clean target to reset your build.

---
## Makefile Variables
```makefile
CFLAGS = -g              # setting a variable inside the Makefile
```

Using it,
```makefile
gcc $(CFLAGS) -c bb.c
```

Overriding from the command line:
```makefile
make CFLAGS='-g -DMY_DEBUG_FLAG'
```

- Environment variables automatically become `make` variables.
- Conversely, `make` variables become environment variables when recipes run.

---
### Complete but Repetitive Makefile
```makefile
mainprog : mainprog.o bb.o rect.o
	gcc -g mainprog.o bb.o rect.o \
		-o mainprog

mainprog.o : mainprog.c bb.h rect.h
	gcc -g -c mainprog.c

bb.o : bb.c bb.h rect.h
	gcc -g -c bb.c

rect.o : rect.c rect.h
	gcc -g -c rect.c
```

---
### Automatic Variables
```makefile
mainprog : mainprog.o bb.o rect.o
	gcc -g $^ -o $@
```
- `$^`: All prerequisites
- `$@`: The target itself
- `$<`: Only first prerequisites

### Pattern Rules
```makefile
%.o : %.c
	gcc -g -c $<
```
One rule to build ANY `.o` from its matching `.c`

Layering extra prerequisites on top of the pattern rule,
```makefile
mainprog.o : bb.h rect.h
bb.o : bb.h rect.h
rect.o : rect.h
```
These are **accumulative, not overriding**.
These add **extra** header-file prerequisites _on top of_ what the pattern rule already implies (the matching `.c` file).

---
### Automatic Prerequisite Listing
```makefile
.depend: mainprog.c bb.c rect.c
	gcc -MM $^ > .depend
include .depend
```

- `gcc -MM` inspects the actual `#include` chains in source files and **automatically generates** the correct prerequisite rules.
- The result is written to a file (conventionally `.depend`, though any name works).
- `include .depend` pulls those generated rules into Makefile.

---
## See Also
- [[C Modularization]]
- [[C Preprocessor]]
- [[Makefile]]