# IP Address
![image|400](https://notes-media.kthiha.com/IP-Address/53e56594960dcf8a258619bf24699879.png)

---
## Code Snippet
```c
struct sockaddr_in addr;
memset(&addr, 0, sizeof(addr));

addr.sin_family = AF_INET;
addr.sin_port = htons(8080);          // host-to-network byte order
addr.sin_addr.s_addr = INADDR_ANY;    // listen on all local interfaces

// for a client connecting to a specific IP instead:
inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
```

---
## See Also
- [[Socket]]
- [[Stream Socket]]
- [[Datagram Socket]]
- [[IP Address]]
