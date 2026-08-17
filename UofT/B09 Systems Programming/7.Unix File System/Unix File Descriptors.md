# File Descriptors
Every process has its own file descriptor table(FDT).
File descriptors are simply array indexes into this table.

Three file descriptors already exist when a process is created
- `0`: standard input
- `1`: standard output
- `2`: standard error

Details:
- `open()`(and `dup()`) consumes a free entry in the FDT.
- `close()` frees an entry.
- `open()`/`dup()` always consume the **lowest-numbered free entry** available. Can rely on this behavior to implement I/O redirection and pipelining.
- The FDT is **finite**
- Each FDT entry doesn't hold file data directly.
  It points into a system-wide "open file table"(OFT).

---
### File Descriptor Tables
![|350](https://biriukov.dev/docs/fd-pipe-session-terminal/images/file-descriptor-open-file-description.png)

- **Per-process FDT entry:** just `fd flags` + a pointer into the OFT.
- **System-wide OFT entry:** 
	- stores the **file offset** ("cursor"/current r/w position), 
	- the **status flags** (e.g. `O_APPEND`, `O_NONBLOCK`; these are the flags set at `open()` time), and 
	- a pointer to the i-node.
- **System-wide i-node table entry:** the actual per-file metadata

---
### How Many-to-One Relationships Happen
- Two OFT entries $\to$ same i-node
	- Two separate `open()` calls on the same file 
	- could be from the same process or different processes
- Two FDs $\to$ same OFT entry
	- `dup()` or `dup2()` system calls
- Two processes have FDs $\to$ same OFT entry
	- Launching a child process: the FDT is cloned by default 
	- E.g., via `fork()`

Important details:
- **Crucial detail:** 
	- the file position(cursor) lives in the `OFT entry`, not the `FDT entry`.
- **Consequence:** 
	- if two `FD`s refer to the same `OFT entry`, reading data through one `FD` advances the shared cursor
	- so reading via the other `FD` won't re-read the same data.
- Contrast this with two independent `open()` calls on the same file: 
	- those get separate `OFT entries` with independent cursors
	- so each can read the file from the start independently

---
### Duplicating File Descriptors
```c
int dup(int oldfd);
```
Duplicate: 
- grabs another free FDT entry 
- and makes it refer to the same `OFT entry` as `oldfd`. 
- Returns the new `fd`.

```c
int dup2(int oldfd, int newfd);
```
- Like `dup()`, but you specify exactly which FDT slot (`newfd`) to use.
- If `newfd` was already in use/open, it's closed first.
- Use cases: shell I/O redirection and pipelining.
- Example: 
  The shell syntax `2>&1` is implemented as `dup2(1, 2)`.

---
## See Also
- [[Unix File System]]
- [[Unix File Descriptors]]
- [[Unix Low-Level File IO]]
- [[Unix User]]
- [[Bitwise Operations]]
