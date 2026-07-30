# C Preprocessor
The [[C Preprocessor]] 
- handles the preprocessor directive macros
- is part of `gcc`
- can be inspected through `gcc -E <file.c>`

---
## Preprocessor Directives
### Macros
```c
#define TABLE_SIZE 20
#define GREETING "Hello"
#define MY_DEBUG_FLAG

#undef GREETING // removes a macro definition
```

### Including files
```c
#include <foo.h>   // look in system-wide directories (e.g. /usr/include)
#include "foo.h"   // ALSO look in the same directory as the current file
```

### Conditional Compilation
```c
#ifdef MY_DEBUG_FLAG
    fprintf(stderr, "x = %d\n", x);
#endif
```

- Toggle debug code: `gcc -DMY_DEBUG_FLAG <file.c>`
- Useful for platform-specific code

---

