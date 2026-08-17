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
## Code Snippets
### Datagram Socket
```c
// Create a UDP socket
int fd = socket(AF_INET, SOCK_DGRAM, 0);

// Set up the destination address struct
struct sockaddr_in dest = { .sin_family = AF_INET, .sin_port = htons(9000) };
inet_pton(AF_INET, "127.0.0.1", &dest.sin_addr);

// Send the 4-byte message "ping" to 127.0.0.1:9000
sendto(fd, "ping", 4, 0, (struct sockaddr *)&dest, sizeof(dest));

// Prepare to receive a reply
struct sockaddr_in src;
socklen_t len = sizeof(src);
char buf[64];

// Block waiting for a UDP packet 
// Fills buf with data, src with sender's address, and returns bytes received (or -1 on error)
int n = recvfrom(fd, buf, sizeof(buf), 0, (struct sockaddr *)&src, &len);
```

### Connected Datagram Socket
```c
// Create a UDP socket
int fd = socket(AF_INET, SOCK_DGRAM, 0);

// Set up the destination address struct
struct sockaddr_in peer = { .sin_family = AF_INET, .sin_port = htons(9000) };
inet_pton(AF_INET, "127.0.0.1", &peer.sin_addr);

// just records the peer
connect(fd, (struct sockaddr *)&peer, sizeof(peer));  

write(fd, "hello", 5);            // no need for sendto anymore
read(fd, buf, sizeof(buf));       // only receives datagrams from `peer`
```

---
## See Also
- [[Socket]]
- [[Stream Socket]]
- [[Datagram Socket]]
- [[IP Address]]