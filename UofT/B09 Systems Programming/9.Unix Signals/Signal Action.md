# Setting Signal Actions

![image|400](https://notes-media.kthiha.com/Signal-Action/2d81201d08570b88ea76475bf2504cdb.png)

```c
int sigaction(int sig,
              const struct sigaction *act,
              struct sigaction *oldact);
```

- `sig`: Which signal type this call concerns
- `act`: The **new** action/disposition you want to install
- `oldact`: The **previous** action gets saved here (handy if you want to restore it later)

While using [[File Redirection & Pipes|fork()]] or [[File Redirection & Pipes|exec()]], 
- `fork()`: signal actions/dispositions are **cloned** into the child.
  The child starts out with the exact same [[Signal Handler|handler]] setup as the parent had.
- `exec()`: Any [[Signal Handler|handlers]] installed are **replaced** with the **default action**. But if a signal's disposition was **`SIG_IGN`**(ignored), it **remains ignored** across the exec.

---
## SigAction
```c
struct sigaction {
    void (*sa_handler)(int sig);
        // pointer to your handler function,
        // or the special values SIG_IGN or SIG_DFL
    sigset_t sa_mask;
        // which OTHER signals to mask (block) while your handler is running
        // — set/query this using the sigset_t operations below
    int sa_flags;
        // various option flags 
    void (*sa_restorer)(void);
        // internal use only: not for application code to touch
};
```

- **`sa_handler`** can be:
    - a pointer to your own [[Signal Handler|handler function]],
    - **`SIG_IGN`**: explicitly ignore this signal, 
    - **`SIG_DFL`**: reset to kernel's default action for this signal.
- **`sa_mask`** lets you specify additional signals that should be **automatically blocked** for the duration of your handler's execution.

---
### Sigset Operations
These are the standard building blocks for constructing the `sa_mask` field of a `struct sigaction`.
```c
int sigemptyset(sigset_t *set);            // clear the set (no signals)
int sigfillset(sigset_t *set);             // add ALL signals to the set
int sigaddset(sigset_t *set, int sig);     // add one signal to the set
int sigdelset(sigset_t *set, int sig);     // remove one signal from the set
int sigismember(const sigset_t *set, int sig); // test membership
```

---
### Useful `sa_flags` Values
Flags when installing a handler are
- `SA_NODEFER`: 
	- Don't automatically mask/block this same signal while the handler for it is running.
	- Default Behaviour: the kernel masks the signal being handled even if not explicitly stated in `sa_mask`.
- `SA_RESETHAND`: 
	- Reset the action back to default just before running the handler.
	- Aka "one-shot" handler: after firing once, subsequent occurrences of the signal revert to default behavior.
- `SA_RESTART`: 
	- Auto-restart most interrupted system calls after the handler returns.
	- Default Behaviour: interrupted syscalls fail and return `errno = EINTR`. 
	- Some system calls never auto-restart even with this flag set.

Flags specific to `SIGCHLD` are:
- `SA_NOCLDSTOP`: Don't deliver `SIGCHLD` at all when a child merely **stops** or **continues**
- `SA_NOCLDWAIT`: Don't turn a terminated child into a **zombie**

---
## See Also 
- [[Unix Signals]]
- [[Signal Handler]]
- [[Signal Action]]
