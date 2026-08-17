## Select
[[Multiplex Select-Poll|Select]] can be used for waiting any of several [[Unix File Descriptors|file descriptors]] to become ready.

![image|450](https://notes-media.kthiha.com/Multiplex-Select/ed5472580887d16df9abb65d4d1e3ef2.png)

```c
int select(int n, fd_set *r, fd_set *w, fd_set *e, struct timeval *timeout);
```

Blocks until one of the specified FDs is ready for read/write, or `timeout` expires, or a signal is caught.

---
**Parameters**:
- `n`: `1 + highest FD number` to check across all sets
- `r`: Set of FDs you want to **read** from. `NULL` if not needed.
- `w`: Set of FDs you want to **write** to. `NULL` if not needed.

---
**Return value:**
- `0` → timed out, nothing became ready
- positive → count of ready file descriptors
- `-1` → error

Macros
```c
void FD_ZERO(fd_set *s);      // make empty
void FD_SET(int fd, fd_set *s);   // add fd to the set
void FD_CLR(int fd, fd_set *s);   // remove fd from the set
int  FD_ISSET(int fd, fd_set *s); // query: is fd in the set (after select() returns)?
```

struct timeval
```c
struct timeval {
    time_t      tv_sec;   // seconds
    suseconds_t tv_usec;  // microseconds
};
```

---
## Poll
```c
#include <poll.h>
int poll(struct pollfd *fds, nfds_t nfds, int timeout);
```

Does the same job as `select()`, but cleaner api.

struct pollfd
```c
struct pollfd {
    int   fd;       // file descriptor
    short events;   // requested events (input)
    short revents;  // returned events (output) -- filled in by the kernel
};
```

---
### Key Event Flags
- `POLLIN`: Data available to read
- `POLLOUT`: Ready for writing
- `POLLPRI`: Urgent/out-of-band data available`
- `POLLRDHUP`: peer has shut down the write half of a stream socket connection
- `POLLHUP`: Hang-up, causing the other end to be closed
- `POLLERR`: Error condition
- `POLLNVAL`: The specified FD was **closed** at the time of the call (only returned in `revents`, never set in `events`)

---
### Timeout Argument
- `-1`: Block indefinitely until an FD is ready or signal is caught.
- `0`: Don't block. Just check current readiness and return immediately.
- `> 0`: Block for up to `timeout` milliseconds.

---
### Return Value
- `-1` → error (e.g. `EINTR` if interrupted by a signal handler)
- `0` → timed out, nothing ready
- positive → number of `pollfd` entries with a nonzero `revents`. Each ready FD is counted exactly once, even if several bits are set in its `revents`.

---
## When is a File Descriptor Ready?
![image|450](https://notes-media.kthiha.com/Multiplex-Select-Poll/fee50f7576e62853cbe5e23666214fa3.png)

---
## Problems at Scale
Both `select()` and `poll()` suffer when monitoring large number of [[Unix File Descriptors|file descriptors]].
- Kernel must re-check every listed FD on every call.
- The whole FD-describing data structure must be copied from user to kernel and back on every call.
- The program must scan the entire returned structure afterward to find which FDs are ready.

> **Root cause** 
> The kernel does not remember the FD list between calls. 
> This means their CPU cost scales with the number of FDs monitored, not with the number of actual I/O events.

To solve this, we can use [[Multiplex Epoll|epoll]] instead.

---
## Code Snippets
### Select Basic Usage
```c
// Declare a file descriptor tracking which fds to monitor 
fd_set readfds;

// Clear the set, and add fd1 and fd2
FD_ZERO(&readfds);
FD_SET(fd1, &readfds);
FD_SET(fd2, &readfds);

// select() needs the highest fd number in any set, plus 1
int maxfd = (fd1 > fd2 ? fd1 : fd2) + 1;
int ready = select(maxfd, &readfds, NULL, NULL, NULL);  // block indefinitely

// select() modifies readfds in place 
// This leaves only the fds that are actually ready
if (FD_ISSET(fd1, &readfds))
    printf("fd1 is ready\n");
if (FD_ISSET(fd2, &readfds))
    printf("fd2 is ready\n");
```

### Select with Timeout
```c
struct timeval tv = { .tv_sec = 5, .tv_usec = 0 };  // 5 second timeout

// Adding the fd to the set
fd_set readfds;
FD_ZERO(&readfds);
FD_SET(fd, &readfds);

// Wait up to 5 seconds for fd to become readable
int ready = select(fd + 1, &readfds, NULL, NULL, &tv);
if (ready == 0)
    printf("timed out\n");
```

### Select must rebuild every call
```c
while (1) {
    fd_set readfds;
    // FD_SET gets clobbered by select(), must redo each loop
    FD_ZERO(&readfds);  
    FD_SET(fd, &readfds);

    // Block until fd is readable
    select(fd + 1, &readfds, NULL, NULL, NULL);
	
    // Confirm fd is actually the one that became ready before acting on it	
    if (FD_ISSET(fd, &readfds))
        handle(fd);
}
```

### Poll Basic Usage
```c
// Array of fds we want to monitor
struct pollfd fds[2];

// Watch fds for "ready to read" events
fds[0].fd = fd1; fds[0].events = POLLIN;
fds[1].fd = fd2; fds[1].events = POLLIN;

// Block until at least one fd has an event
// -1 means wait forever
int ready = poll(fds, 2, -1);  // block indefinitely

// poll() fills in revents per-entry rather than mutating the input like select() does
for (int i = 0; i < 2; i++) {
    if (fds[i].revents & POLLIN)
        printf("fd %d is ready\n", fds[i].fd);
}
```

### Poll with timeout
```c
// Array of fds we want to monitor
struct pollfd fds[1] = { { .fd = fd, .events = POLLIN } };

int ready;
while ((ready = poll(fds, 1, 1000)) == -1 && errno == EINTR)
    continue;  // interrupted by signal, just retry

// ready == 0 means the 1-second timeout expired with no events on fd
if (ready == 0)
    printf("timed out\n");
```

### Poll Detecting Hangup Peer
```c
struct pollfd fds[1] = { { .fd = cfd, .events = POLLIN } };
poll(fds, 1, -1);

if (fds[0].revents & POLLHUP)
    printf("peer closed connection\n");
if (fds[0].revents & POLLERR)
    printf("error on socket\n");
```

---
## See Also
- [[Multiplexing]]
- [[Multiplex Select-Poll]]
- [[Multiplex Epoll]]