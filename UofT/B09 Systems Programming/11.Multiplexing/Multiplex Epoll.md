# Epoll
![image|450](https://notes-media.kthiha.com/Multiplex-Epoll/53e569218e0a67cc6ff1f0c95e1d55a1.png)

---
## Why epoll scales better
- The kernel remembers the monitored FD list across calls.
- So `epoll_wait()` doesn't need to re-describe or re-scan the whole set every time.
- Cost scales with the number of **active events**, not the total number of monitored FDs.

---
### API
```c
int epoll_create1(int flags);
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *ev);
int epoll_wait(int epfd, struct epoll_event *evs, int n, int timeout);
```

```c
int epoll_create1(int flags);
```
- Returns a new [[Unix File Descriptors|file descriptor]] representing an epoll instance.
- Like a normal FD: has a slot in the FD table, so `close()`, `dup()`, `fork()`, `exec()` semantics all apply to it normally.
- Unlike a normal FD: not meaningful for `read()`/`write()`. Only usable with the other `epoll_*` calls.

---
```c
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *ev);
// Returns 0 on success
```

Options `op`:
- `EPOLL_CTL_ADD`: Start monitoring `fd`.
- `EPOLL_CTL_DEL`: Stop monitoring `fd`(`ev` is unused/ignored).
- `EPOLL_CTL_MOD`: Change what's being monitored for `fd`.

---
```c
int epoll_wait(int epfd, struct epoll_event *evs, int n, int timeout);
```
- `evs`: array to receive ready events.
- `n`: length of that array.
- `timeout`: milliseconds; `-1` = block indefinitely.
- Returns the **count of ready FDs** = number of entries filled into `evs`.
- For each returned entry:
    - `events` → which conditions occurred.
    - `data` → exactly what you stored via `epoll_ctl()`'s `ADD`/`MOD` call for that FD.

---
### Data Structures
```c
struct epoll_event {
    uint32_t     events;
    epoll_data_t data;
};
```

```c
typedef union epoll_data {
    void    *ptr;
    int      fd;
    uint32_t u32;
    uint64_t u64;
} epoll_data_t;
```

- When  calling `epoll_ctl()`, you can store sth in `data`.
- When `epoll_wait()` later reports an event, you get that same value back.
- Commonly stored: 
	- the [[Unix File Descriptors|FD]] being monitored
	- pointer to your own per-connection bookkeeping struct.

---
## Code Snippets
### Epoll Setup
```c
// Array of fds we want to monitor
int epfd = epoll_create1(0);

// Describe what we want to monitor and how
struct epoll_event ev;
ev.events = EPOLLIN; // notify when sfd has data ready to read (level-triggered by default)
ev.data.fd = sfd;    // tag this event with sfd so we know which fd triggered it later

// Register sfd with the epoll instance
epoll_ctl(epfd, EPOLL_CTL_ADD, sfd, &ev);
```

### Epoll Wait Loop
```c
// Buffer to receive up to 10 ready events per epoll_wait() call
struct epoll_event evs[10];

/* Block until at least one registered fd has an event, or forever since timeout is -1.
Returns the number of ready events */
int n = epoll_wait(epfd, evs, 10, -1);

for (int i = 0; i < n; i++) {
    // If the ready fd is the listening socket, it means a new client 
    // is trying to connect
    if (evs[i].data.fd == sfd) {
        // Accept the new connection; 
        // NULL, NULL means we don't care about the client's address here
        int cfd = accept(sfd, NULL, NULL);
        
        // Register the new client fd with epoll 
        // so future data from it triggers events too
        struct epoll_event ev = { .events = EPOLLIN, .data.fd = cfd };
        epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &ev);
    } else {
        // Otherwise, an existing client fd has data ready
        handle_client(evs[i].data.fd);
    }
}
```

### Epoll storing a pointer
```c
// Per-connection state: keep the fd plus a dedicated buffer for this client
struct conn { int fd; char buf[256]; };

// Allocate connection state on the heap so it persists across event loop 
// iterations
struct conn *c = malloc(sizeof(*c));
c->fd = cfd;

struct epoll_event ev;
ev.events = EPOLLIN;
ev.data.ptr = c;               // store your own struct, not just the fd
epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &ev);

// later, in the wait loop:
struct conn *c = evs[i].data.ptr; // recover the connection struct directly 
read(c->fd, c->buf, sizeof(c->buf)); // read into the connection's own buffer
```

### Epoll removing/modifying a monitored fd
```c
epoll_ctl(epfd, EPOLL_CTL_DEL, cfd, NULL);  // stop monitoring, ev unused

struct epoll_event ev = { .events = EPOLLIN | EPOLLOUT, .data.fd = cfd };
epoll_ctl(epfd, EPOLL_CTL_MOD, cfd, &ev);   // now also watch for writability
```

---
## See Also
- [[Multiplexing]]
- [[Multiplex Select-Poll]]
- [[Multiplex Epoll]]