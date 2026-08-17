## Broken Pipe
- Broken Pipe is 
	- when you `write()` to a pipe or socket, 
	- but the other end has already closed
	- and your process receives `SIGPIPE`.
- Default action: the process is killed.

Consider the following
```bash
sort bigfile | head -1
```

![image|400](https://notes-media.kthiha.com/Signal-Handler/7c0cf0fc9ac369f4c5f5694cfae72d98.png)

The `SIGPIPE`-kills-by-default behavior lets `sort` terminate promptly instead of wasting CPU.

---
## Overriding this behaviour
Set the action for `SIGPIPE` to **`SIG_IGN`**.
- Once ignored, the process is no longer killed by a broken pipe.
- Instead, the offending `write()` call simply returns -1, with errno set to `EPIPE` 

---
## Handler Limitations

![image|500](https://notes-media.kthiha.com/Signal-Handler/e89b257010725f29499bcd04709715cb.png)

It's unsafe to call some things inside a [[Signal Handler|signal handler]], like
-  `printf()`
	- Suppose your program's normal code is in the middle of running another `printf()` call when a signal arrives.
	- That normal execution gets interrupted, and the signal handler runs.
	- `printf()` maintains internal buffers and bookkeeping variables that it updates as it goes. 
	  If it was interrupted mid-update, those internal structures are currently in a half-finished state.
	- If your handler now also calls `printf()`, it'll try to use/update those same half-finished internal structures, corrupting them.
- `fclose(any_stream)`: exact same underlying problem (shared internal buffer/bookkeeping state).
- `exit()`: because `exit()` internally calls `fclose()` on all open streams to flush buffers as part of its normal cleanup
- Similar library functions like `free()` and `malloc()`
- Any of your own data structures that your "normal" (non-handler) code is also actively using/mutating.

> Inside a handler, you often can't even safely clean up using the normal-looking cleanup functions

> **Rule of Thumb**
> A function is only safe to call from a signal handler if it is **async-signal-safe**.

---
## Handler Strategies
If you need to do anything non-trivial in response to a signal, the standard pattern is: don't do it inside the handler at all. 

Instead:
1. Set up a **global variable** or a **pipe** to serve as a communication channel. 
	- A [[File Redirection & Pipes|pipe]] is generally preferred 
	- because it naturally supports things like `select()` 
	- and blocking-read notification patterns.
2. **Inside the handler**, just do the minimal safe thing: write to that variable or pipe to record the signal.
3. Your normal code **periodically checks** that variable/pipe at safe moments in its own control flow.

---
## Code Snippet
### Self-Pipe Trick
```c
int sigpipe[2];

void handler(int sig) {
    char byte = 1;
    write(sigpipe[1], &byte, 1);     // only async-signal-safe call used
}

pipe(sigpipe);
struct sigaction sa = { .sa_handler = handler, .sa_flags = 0 };
sigemptyset(&sa.sa_mask);
sigaction(SIGCHLD, &sa, NULL);

char byte;
read(sigpipe[0], &byte, 1);          // main loop reacts here, safely
printf("handling SIGCHLD now\n");
```

---
## See Also 
- [[Unix Signals]]
- [[Signal Handler]]
- [[Signal Action]]