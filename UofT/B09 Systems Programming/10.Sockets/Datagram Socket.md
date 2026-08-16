# Datagram Socket
![image|400](https://notes-media.kthiha.com/Datagram-Socket/e0894e49db8b38d43a699ba15fa01dd6.png)

---
### Receiving and Sending

```c
ssize_t recvfrom(int sockfd, void *buffer, size_t length, int flags,
                  struct sockaddr *src_addr, socklen_t *addrlen);
// Returns number of bytes received, 0 on EOF, or -1 on error

ssize_t sendto(int sockfd, const void *buffer, size_t length, int flags,
                const struct sockaddr *dest_addr, socklen_t addrlen);
// Returns number of bytes sent, or -1 on error
```

- First 3 args + return value behave like `read()`/`write()`.
- `flags`: bitmask of socket I/O options; `0` if you don't need any special features.
- `recvfrom()`: `src_addr`/`addrlen` return sender's address

---
## Connect with Datagrams
Even though datagram sockets are connectionless, calling `connect()` on one is useful:

- It records a specific peer address for that socket called a **connected datagram socket**
- After connecting:
	- You can just use `write()`/`send()`
	- The socket will only receive datagrams sent by that specific peer.
- Only the socket that called `connect()` is restricted. The remote socket is unaffected unless it _also_ calls `connect()`.

---
## See Also
- [[Socket]]
- [[Stream Socket]]
- [[Datagram Socket]]
- [[IP Address]]