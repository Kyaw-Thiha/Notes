# Multiplexing
![image|500](https://notes-media.kthiha.com/Multiplexing/70054cd1c24d36764950971db4dbfe94.png)

---
## Ways to Multiplex
There are $3$ ways to carry out [[Multiplexing|multiplexing]]:
- [[Multiplex Select-Poll|select()]]
- [[Multiplex Select-Poll|poll()]]
- [[Multiplex Epoll|epoll()]]

---
## Self-Pipe Trick
![image|450](https://notes-media.kthiha.com/Multiplexing/9ea41a15c7dddbfab4f33d71c05f8b78.png)

- Create a pipe (`pfd[0]` = read end, `pfd[1]` = write end) **before** installing the signal handler.
- Make both ends non-blocking.
- The signal handler's only job is to `write()` a single byte to the pipe's write end.
- Add pipe's read end to readfds set you're already monitoring.
- Put the monitoring call in a loop that restarts on `EINTR`.
- After the call returns, 
	- check if the pipe's read end is set/ready. If so, a signal arrived. 
	- Drain all bytes from the pipe in a loop using non-blocking `read()` until it fails with `EAGAIN`. 
	- Then perform whatever action the signal should trigger.

---
## Code Snippets
### Self-Pipe Trick
```c
// Create a pipe: pfd[0] is the read end, pfd[1] is the write end
int pfd[2];
pipe(pfd);

// Make both ends non-blocking
fcntl(pfd[0], F_SETFL, O_NONBLOCK);
fcntl(pfd[1], F_SETFL, O_NONBLOCK);

// Signal handler
// The signal handler's entire job is to write one byte into the pipe.
void handler(int sig) {
    char byte = 1;
    write(pfd[1], &byte, 1);   // only safe call in the handler
}

// Register handler for SIGINT
signal(SIGINT, handler);


while (1) {
    // Watch both the "self-pipe" read end and the listening socket 
    // for readability
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(pfd[0], &readfds);
    FD_SET(listen_fd, &readfds);
    int maxfd = (pfd[0] > listen_fd ? pfd[0] : listen_fd) + 1;

    // Block until either fd is ready; retry cleanly if
    // interrupted by a signal
    if (select(maxfd, &readfds, NULL, NULL, NULL) == -1 && errno == EINTR)
        continue;  // restart on EINTR

    // The self-pipe became readable 
    // => a signal was delivered and the handler wrote to it
    if (FD_ISSET(pfd[0], &readfds)) {
        char buf[16];
        while (read(pfd[0], buf, sizeof(buf)) > 0)
            ;  // drain until EAGAIN
        printf("signal handled safely\n");
    }
	
    // The listening socket became readable => normal client activity
    if (FD_ISSET(listen_fd, &readfds)) {
        // handle normal I/O
    }
}
```

---
## See Also
- [[Multiplexing]]
- [[Multiplex Select-Poll]]
- [[Multiplex Epoll]]