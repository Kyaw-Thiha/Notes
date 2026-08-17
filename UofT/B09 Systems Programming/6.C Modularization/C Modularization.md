# C Modularization

The processing of a `C file` to an executable consists of a preprocessor, compiler, and linker.

![|250](https://intro2oop.sdds.ca/assets/images/14-compiler-72dbcd2e9f0474dac6ec1a0901e05993.png)

- [[C Preprocessor|Preprocessor]]: Handle the macros, and produce actual C code the compiler sees.
- Compiler: Translates preprocessed C code into machine code.
- Linker: Merges object files and libraries into one executable.

---
## Header File
- Macros and types
- Function Prototypes
- Imports

```c
// vector.h

#ifndef VECTOR_H
#define VECTOR_H

#include "complex.h"
typedef struct vector
{
  complex x1, x2;
} vector;

void vector_add(vector *dst, const vector *v1, const vector *v2);
void vector_scale(vector *dst, const complex *c, const vector *v);
void vector_print(const vector *v);

#endif
```

- Note the if wrapping the header file with define.
- It helps prevent the file being imported more than twice. Without it, multiple `.h` files could import it.
- `#pragma once` is the modern alternative of this pattern.

---
### Struct Self-Reference Trick
```c
typedef struct node {
    int i;
    struct node *next;   // "nodetype" not available yet here!
} nodetype;
```

```c
typedef struct node nodetype;   // declare the alias first
// Now "nodetype *" is usable.

struct node {
    int i;
    nodetype *next;             // clean!
};
```

---
### Namespace
Every global function/type/variable name lives in one flat global space across your whole program.

> **Workaround convention**
> Prefix everything with a project-specific string.
> Eg: `GTK+` library uses `gtk_button_new`, `gtk_window_close`.

---
### Global Variables
Used for sharing global variable across multiple `.c` files, but only allocate storage for it once.

- `extern int var;`: declaration only.
- `int var;`: definition. Actual address is allocated.

---
## See Also
- [[C Modularization]]
- [[C Preprocessor]]
- [[Makefile]]