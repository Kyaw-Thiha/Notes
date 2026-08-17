## File Redirection

![image|300](https://notes-media.kthiha.com/File-Redirection-&-Pipes/52739af167afe0bddb3080f3e0112621.png)

How a shell implements [[Redirection and Pipelining|redirection]] `command > file`:
1. Before `fork()`: 
	- [[Unix Low-Level File IO|open]] the target file. 
	- Let `fd` be the resulting file descriptor. 
	- If the open fails, there's no point continuing.
2. `fork()`.
	- The parent should close `fd` .
	- The steps below are all for the child.
3. In the child: `dup2(fd, i)` 
	- where `i` is `0`, `1`, or `2` depending on which standard stream you're redirecting (stdin/stdout/stderr).
	- Older style: `close(i)` followed by `dup(fd)`
	  This works because `dup()` always grabs lowest-numbered free FD, which after closing `i`, is exactly `i`.
4. Close the original `fd` in the child 
	- since slot `i` is now the real reference to it 
	- or alternatively, mark `fd` as close-on-exec at the time you originally opened it, so this happens automatically.
5. `exec()` the new program. 
- New program simply talks to standard descriptor `i` as usual.
- but because of the `dup2()` above, `i` now actually refers to the file you opened, not the terminal.

> `open` → `fork` → `dup2` → `close` → `exec`
> 
> Used for `command > output.txt`, `command < input.txt`, `command 2> errors.txt`, etc.

---
## Pipes

![image|300](https://notes-media.kthiha.com/File-Redirection-&-Pipes/ef2146aa63e825327f0182211c5f7718.png)

```c
int pipe(int pipefd[2]);
```

- Creates a **unidirectional** pipe.
	- `pipefd[0]` = the file descriptor for the **read end**.
	- `pipefd[1]` = the file descriptor for the **write end**.

- What if you want **two-way** communication?
  You'd need **two** pipes: one for each direction.

- A pipe is **"unnamed"**: 
	- it has no filename and 
	- doesn't exist anywhere in the file system, 
	- even though you get ordinary-looking file descriptors.

---
### Pipe Hygiene
- **Typical setup:** 
	- exactly one process holds the **write end**, and 
	- exactly one(different) process holds the **read end** 
	- usually a **parent-child** or **sibling** relationship 
	  (since pipe fds only make sense being shared through `fork()`).
- Usually combined with `dup`/`dup2` so that, 
	- a child's **stdin** _is_ the read end, or 
	- its **stdout** _is_ the write end 
- Because of forking and dup-ing, there's often messy intermediate state: 
	- multiple processes and FDs within those processes can all refer to same write end or read end simultaneously.
	- this is natural side effect of `fork()` cloning FD table, and `dup`/`dup2` adding even more references.
- You should close FDs you don't actually need as soon as possible.
	- **How does `read()` know it's hit "end of data"**?
		- Only once every FD referring to the write end has been closed. 
		- If even one process somewhere still holds the write end open, `read()` will just **block** waiting for more data instead of returning EOF.
		- even if that process never actually writes anything!
	- **How does `write()` know there's "no audience" left** ?
		- Only once _every_ FD referring to the read end has been closed

> **Classic bug:** 
> Forgetting to close the unused end of a pipe in the parent (or child) after forking leads to programs that hang forever. 
> Because the "extra" open FD prevents EOF from ever being detected.

---
### Writer's Block
- What if the writer keeps writing, but the reader reads too slowly (or never)?
	- The kernel maintains an internal buffer for unread pipe data but it's finite.
	- Once that buffer fills up, the `write()` call blocks.
	- It hangs until the reader actually reads some data (freeing buffer space) or closes the read end.
- What if the read end gets closed while the writer is still trying to write?
	- This is called a **"broken pipe."**
	- The writer process receives a signal(`SIGPIPE`) upon attempting to `write()`.
	- By default, this signal terminates the process.
	- Classic example: `yes | head -5`. `head` only wants 5 lines and then quits (closing its read end) 

---
### Non-Blocking Write/Read
You can request non-blocking I/O on a per-file-descriptor basis:
```c
int flags = fcntl(fd, F_GETFL);
fcntl(fd, F_SETFL, flags | O_NONBLOCK);
```

Once set, `read()` and `write()` on that fd won't block.
Instead, if the operation can't complete immediately, they:
- return **`-1`**, and
- set `errno` to **`EAGAIN`**

---
## Code Snippets
### Redirect to a file
```c
/* ls > out.txt */

int fd = open("out.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);

pid_t pid = fork();
if (pid == 0) {
    dup2(fd, STDOUT_FILENO);
    close(fd);
    execlp("ls", "ls", NULL);
    exit(1);
}
close(fd); // parent doesn't need it
waitpid(pid, NULL, 0);
```

### Pipe Between Two Children
```c
/* ls | wc -l */

int pfd[2];
pipe(pfd);

pid_t p1 = fork();
if (p1 == 0) {                 // writer: ls
    dup2(pfd[1], STDOUT_FILENO);
    close(pfd[0]); close(pfd[1]);
    execlp("ls", "ls", NULL);
    exit(1);
}

pid_t p2 = fork();
if (p2 == 0) {                 // reader: wc -l
    dup2(pfd[0], STDIN_FILENO);
    close(pfd[0]); close(pfd[1]);
    execlp("wc", "wc", "-l", NULL);
    exit(1);
}

close(pfd[0]); close(pfd[1]);  // parent needs neither end
waitpid(p1, NULL, 0);
waitpid(p2, NULL, 0);
```

### Non-Blocking Read from Pipe
```c
int flags = fcntl(pfd[0], F_GETFL);   // get current flags on the read end
fcntl(pfd[0], F_SETFL, flags | O_NONBLOCK);  

char buf[64];
int n = read(pfd[0], buf, sizeof(buf));      
if (n == -1 && errno == EAGAIN)              
    printf("no data yet\n");
// n == 0  would mean EOF (writer closed its end)
// n  > 0  is the number of bytes actually read
```

---
## See Also
- [[Unix Process Creation]]
- [[File Redirection & Pipes]]
- [[Redirection and Pipelining]]
- [[Unix File System]]
- [[Unix File Descriptors]]