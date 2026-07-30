# Bitwise Operations
Bitwise Operators:
```c
a = 10001001
b = 00000011

a & b = 00000001  // AND
a | b = 10001011  // OR
a ^ b = 10001010  // XOR
~a    = 01110110  // NOT
```

Shifts:
```c
00011000 << 2 = 01100000     // left shift
00011000 >> 2 = 00000110     // right shift
```
- `a << k` is equivalent to $a \times 2^{k}$
- `a >> k` is equivalent to $\frac{a}{2^{k}}$ (floor division)

To check/set/clear/flip **bit 5** of `b`:
- Let `m = 0010 0000` or `1 << 5`.
- **Check**: `if (b & m)`
- **Set**(to 1): `b = b | m;`
- **Clear**(to 0): `b = b & ~m;`
- **Flip**: `b = b ^ m;`

---
## Permission Bits
```
File type  U G T | R W X | R W X | R W X →
            User    Group   Other
```

`U` = set-user-ID bit, `G` = set-group-ID bit, `T` = sticky bit

```c
S_IRUSR = 0400   // owner read
S_IWUSR = 0200   // owner write
S_IXUSR = 0100   // owner execute
// (similarly S_IRGRP, S_IWGRP, S_IXGRP, S_IROTH, S_IWOTH, S_IXOTH)
```

Sample Usage:
```c
if (s.st_mode & S_IRUSR) { /* owner can read */ }
```

---
## Set-UID/Set-GID
On a directory
- `set-gid`: any new file created inside gets its initial group set to the directory's group.
- `sticky`: 
	- other users cannot delete or rename your files 
	- even if they have write access to the directory.
	- Use case: `/tmp`

On an executable file:
- `set-uid`: 
	- the program runs with the file owner's privilege, not the invoking user's.
	- Use Case: `su` and `sudo`
- `set-gid`: same idea, but for group privilege.

---

