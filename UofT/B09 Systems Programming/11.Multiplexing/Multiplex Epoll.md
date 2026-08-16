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