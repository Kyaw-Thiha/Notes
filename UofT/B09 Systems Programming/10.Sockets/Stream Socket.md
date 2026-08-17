# Stream Socket


![image|400](https://notes-media.kthiha.com/Stream-Socket/6b3f2e2a0ecd1e09aae7f9a73510e795.png)

Note that the server juggles multiple [[Unix File Descriptors|file descriptors]]:
- One listening file descriptor (`sfd`)
  Waits for new clients.
- One file descriptor per connected client (`cfd`)
  Used to talk to that client.

---
## Core System Calls

### Socket
```c
int socket(int family, int type, int protocol);
// Returns positive socket FD, or -1 on error
```
- `family`: `AF_UNIX`, `AF_INET`, `AF_INET6`
- `type`: `SOCK_DGRAM`, `SOCK_STREAM` (+ advanced low-level types)
- `protocol`: normally `0`

---
### Bind
```c
int bind(int fd, const struct sockaddr *addr, socklen_t addrlen);
```
Used by servers to attach their socket to a well-known address so clients can find them.

---
### Listen
```c
int listen(int fd, int backlog);
```
- `backlog`: max length of the [[Priority Queue|queue]] of pending (not-yet-`accept()`ed) connection requests in the kernel

---
### Accept
```c
int accept(int fd, struct sockaddr *client_addr, socklen_t *addrlen);
// Returns new connected socket FD on success, or -1 on error
```

- `accept()` creates a **brand-new socket** connected to the client.
- The original listening socket (`sockfd`) stays open and keeps listening for more clients.
- `client_addr` / `addrlen` return the peer's address.
  Pass `NULL`/`0` if you don't care about the peer's address.

---
### Connect
```c
int connect(int fd, const struct sockaddr *server_addr, socklen_t addrlen);
// Returns 0 on success, -1 on error
```

- Connects the active (client) socket to the listening socket at `addr`/`addrlen`.
- If `connect()` fails and you want to retry, close the socket, create a new one, and retry.

---
## No Packet Boundaries

![image|500](https://notes-media.kthiha.com/Stream-Socket/173fdbf0e1f5d646ca351e158e9afcba.png)

---
## Code Snippet
### Stream Socket Server
```c
// Create a TCP socket
int sfd = socket(AF_INET, SOCK_STREAM, 0);

// Set up the address to bind to
struct sockaddr_in addr = { .sin_family = AF_INET,
                             .sin_port = htons(8080),
                             .sin_addr.s_addr = INADDR_ANY }; 
bind(sfd, (struct sockaddr *)&addr, sizeof(addr));
listen(sfd, 10);   // backlog of 10 pending connections

// Prepare to capture info about whoever connects
struct sockaddr_in client_addr;
socklen_t len = sizeof(client_addr);


int cfd = accept(sfd, (struct sockaddr *)&client_addr, &len);

// sfd keeps listening; cfd talks to this one client
```

### Stream Socket Client
```c
```c
// Create a TCP socket
int fd = socket(AF_INET, SOCK_STREAM, 0);

// Set up the address of the server we want to connect to
struct sockaddr_in server_addr = { .sin_family = AF_INET,
                                    .sin_port = htons(8080) };
inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

// Attempt to connect to the server
// Blocked until connected, refused, or times out
if (connect(fd,(struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
    perror("connect failed");
    exit(1);
}

// Send the 5-byte message "hello"
write(fd, "hello", 5);
```

### Server Handling Multiple Clients
```c
while (1) {
    int cfd = accept(sfd, NULL, NULL);
    if (fork() == 0) {
        close(sfd);              // child doesn't need the listening socket
        // talk to client via cfd
        read(cfd, buf, sizeof(buf));
        close(cfd);
        exit(0);
    }
    close(cfd);                  // parent doesn't need the client socket
}
```

---
## See Also
- [[Socket]]
- [[Stream Socket]]
- [[Datagram Socket]]
- [[IP Address]]