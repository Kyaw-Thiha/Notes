# Unix Process Creation
![image|300](https://notes-media.kthiha.com/Unix-Process-Creation/f088dfe35316f2e285c1010943b0e979.png)

---

**Step-1**: Clone the process (`fork()`)
```c
pid_t fork(void);
```

- Most things get cloned: 
	- memory (global vars, stack, heap, env vars), 
	- current directory, 
	- file descriptor table, 
	- execution state
- After `fork()`, both processes are still running the exact same code.
- Every process has a **process ID(PID)** assigned at birth that never changes.
	- The **old PID** is the original and is called the parent.
	- The **new PID** is the clone and is called the child.
- Both processes continue as though `fork()` itself returned but with different return values:
	- **Child** gets return value **`0`**.
	- **Parent** gets return value $=$ the `child's PID`.
	  (positive, non-zero, non $-1$ value)

```c
pid_t pid = fork();
if (pid == -1) {
    // fork failed
} else if (pid == 0) {
    // I am the child
} else {
    // I am the parent; pid == child's PID
}
```

**Step-2**: Switch to another program (`exec`)
If the child wants to run a different program 
- rather than continuing to run the parent's code,
- it calls one of the `exec` family of system calls
```c
execlp(path, arg0, arg1, ..., (char*)NULL);
```

- **Erased and replaced:** the process's code and data.
- **Preserved across exec:** PID, environment variables, current directory, the file descriptor table

---
## Useful Comands
Listing processes:
- `ps`: List processes
- `pgrep`: Find processes by name, user, etc.
- `top`: Periodically-refreshed process list
- `htop`: `top` but with more features

Terminating processes:
- `kill`: Terminate by **specific PID(s)** you provide.
- `pkill`: Finds processes like `pgrep` does, then kills them.

---
## Children, Zombies and Orphans

### Waiting for a Child
```c
pid_t wait(int *wstatus);
```

Wait for a particular child, or avoid blocking:
```c
pid_t waitpid(pid_t pid, int *wstatus, int options);
```

**Arguments**:
- `pid > 0`: Wait for that specific child
- `pid == -1`: Wait for any child(like plain `wait()`)
- `options == WNOHANG`: Don't hang/block waiting. Return immediately if no child has exited yet.
- `options == 0`: Normal blocking behavior.

**wstatus**:
- **Normal termination**: bits 15–8 = exit status(0–255)
- **Killed by signal**: bits 7–0 = termination signal number (nonzero). One bit is the core dumped flag
- **Stopped by signal**: encodes the stop signal, with a marker byte `0x7F`
- **Continued by signal**: the special value `0xFFFF`

Useful **macros** applied to the `wstatus` value `s`:
- Normal exit: `WIFEXITED(s)`, `WEXITSTATUS(s)`
- Killed by signal: `WIFSIGNALED(s)`, `WTERMSIG(s)`, `WCOREDUMP(s)`
- Stopped/continued: `WIFSTOPPED(s)`, `WIFCONTINUED(s)`

---
### Zombies & Orphans
![image|300](https://notes-media.kthiha.com/Unix-Process-Creation/35bb296479e4dceeba570a38c095872e.png)
#### Zombies
- If the child dies first, but the parent hasn't called `wait()`:
	- The dead child becomes a **"zombie."**
	- The kernel retains an entry in the process table for it (holding onto its exit status).
	- Process listings(`ps`) show status **`Z`** for zombies.
- **Why keep it around at all?** 
  Because the parent might call `wait()` later and needs to be able to retrieve the exit status.

#### Orphans
If the parent dies first, but the child is still running:
- The child becomes an orphan.
- The kernel resets the child's parent `PID` to 1.
- If the (now-adopted) child later dies: 
	- `init` calls `wait()` on it automatically  
	- so it never becomes a permanent zombie.

---
## Code Snippets
### Basic
```c
pid_t pid = fork();
if (pid == 0) {
	/* Child */
    printf("child, pid=%d\n", getpid());
    exit(42);
} else {
	/* Parent */
    int status;
    waitpid(pid, &status, 0); // Wait till child process change state
    if (WIFEXITED(status))
        printf("child exited with %d\n", WEXITSTATUS(status));
}
```

### Running a Program on Fork
```c
pid_t pid = fork();
if (pid == 0) {
    execlp("ls", "ls", "-l", NULL);
    perror("execlp failed"); // only reached if exec fails
    exit(1);
}
waitpid(pid, NULL, 0);
```

### Non-Blocking Wait
```c
pid_t pid = fork();
if (pid == 0) { sleep(3); exit(0); }

int status;
pid_t r;
while ((r = waitpid(pid, &status, WNOHANG)) == 0) {
    printf("still running...\n");
    sleep(1);
}
printf("done: %d\n", r);
```

### Orphan
```c
pid_t pid = fork();
if (pid == 0) {
    sleep(2);
    printf("child's new parent pid = %d\n", getppid()); // will print 1
    exit(0);
}
exit(0); // parent dies immediately, child gets orphaned
```

---
## See Also
- [[Unix Process Creation]]
- [[File Redirection & Pipes]]
- [[Redirection and Pipelining]]
- [[Unix File System]]
- [[Unix File Descriptors]]