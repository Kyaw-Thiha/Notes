# Low-Level File IO
System calls for low level file IO
```c
int open(const char *path, int flags);
int open(const char *path, int flags, int mode);   
// when flags includes O_CREAT

ssize_t read(int fd, void *buf, size_t count);
ssize_t write(int fd, void *buf, size_t count);

off_t lseek(int fd, off_t offset, int origin);      
// origin: SEEK_SET, SEEK_CUR, SEEK_END

int close(int fd);
```

- **`flags`** for `open()`: `O_WRONLY`, `O_RDONLY`, `O_RDWR`, `O_EXCL`, `O_TRUNC`, `O_APPEND`
- On success, `open()` returns a file descriptor.
- When `flags` includes `O_CREAT`, the extra `mode` argument specifies the initial permission bits for the newly created file.
- `lseek()` returns the new offset.

---
## UMask
The `umask` limits/restricts the initial permissions granted at file creation. 
```c
Actual initial permissions = (what open() requests) & ~umask
```

- `077`: Bans all (rwx) permissions for group and others.
- `002`: Bans only write for others.

> **Best Practice**
> At `open()`, request maximally-permissive `0666`.
> Then, let the umask do the actual restricting.
> Unless the file is security-sensitive, in which case request `0600` directly.

---
## Higher Level `stdio.h`
If you have a raw file descriptor but want the convenience of `stdio.h` functions (`fprintf`, `fgets`, etc.):
```c
FILE *fdopen(int fd, const char *mode);
```

- `mode` is again one of `"r"`, `"w"`, `"r+"`, `"w+"`, etc.
- `fclose()` on the resulting `FILE*` will call `close(fd)` for you automatically.

If you have a `FILE*` and want the underlying fd:
```c
int fileno(FILE *stream);
```

---
