# Socket
A [[Socket|socket]] allows software to send and receive data across a computer network.

![300](https://media.geeksforgeeks.org/wp-content/uploads/20250503164723620815/Socket-in-Computer-Network-2.webp)

It is an IPC(inter-process communication) mechanism like [[File Redirection & Pipes|pipes]], but designed for [[Unix Process Creation|processes]] not related by `fork()`, and may live on different computer.

---
## Pipes vs Sockets
### Pipe
In [[File Redirection & Pipes|pipe]],
- **Addressing**: ends aren't publishable. Only shared via `fork()`.
- **Who can connect**: Only [[Unix Process Creation|processes]] from same fork tree.
- **Direction**: One-way. Need two pipes for bidirectional.
	
### Socket
In [[Socket|sockets]],
- **Addressing**: the server has a publishable address.
- **Who can connect**: Any two unrelated [[Unix Process Creation|processes]].
- **Direction**: [[Unix File Descriptors|File descriptor]] is bidirectional. Read and write on the same fd.

---
## Socket Varieties
```c
int socket(int family, int type, int protocol);
```
- Address Family:
	- `AF_UNIX`: Unix Domain. Address format uses path in filesystem. Local only.
	- `AF_INET`: IPv4. Works over network; has loopback address for local use.
	- `AF_INET6`: IPv6. 
- Socket Type
	- `SOCK_STREAM`: [[Stream Socket]]. Byte-stream oriented.
	- `SOCK_DGRAM`: [[Datagram Socket]]. Message-oriented.

---
## Broken Pipe
Writing to a [[File Redirection & Pipes|pipe]] or [[Socket|socket]] whose other end is closed results in a **broken pipe**.

![image|400](https://notes-media.kthiha.com/Socket/acb9662897ed3415684e4064512bc0d5.png)

- Default behavior: your process receives **`SIGPIPE`**, whose default action is to **kill the process**.
- Fix: set `SIGPIPE`'s disposition to **`SIG_IGN`** (ignore). Then:
	- The process is **not** killed.
	- The offending `write()` instead returns **-1** with `errno == EPIPE`, which you can check and handle gracefully.

---
## See Also
- [[Socket]]
- [[Stream Socket]]
- [[Datagram Socket]]
- [[IP Address]]