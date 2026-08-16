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
## See Also
- [[Multiplexing]]
- [[Multiplex Select-Poll]]
- [[Multiplex Epoll]]