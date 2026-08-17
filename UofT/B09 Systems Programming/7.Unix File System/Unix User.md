## Users & Groups
- `/etc/passwd` stores passwords of users.
	- named `passwd` cox historically stored hashed password
	- now real hashes live elsewhere like `/etc/shadow`
- `/etc/group` stores groups and their members
	- many-to-many between users and groups
- Commands
	- `id`: shows your username, UID and groups.
	- `groups`: shows your groups.

---
## Permissions
- Each file has one owning user and one owning group.
  At creation, set as creator's primary group.
- Permissions are flagged as `[owner][group][other]`.
- `rwxr-x---`
	- Owner: `rwx` (read, write, execute — all)
	- Group: `r-x` (read, execute — no write)
	- Other: `---` (nothing)

---
### Read Access Circumventing No Execution
- Suppose system admin set `-rwxr--r--` on an executable.
- Hence, only owner can run it, but groups and others cannot.

But the group/others still have read permissions.
- You can copy the file you can read.
- You can set your own copy to be executable (`chmod +x [file])` since you own the copy.

> If you want to prevent others from running a program, deny read, not just execute.

---
### Permission Flags for Directories
- `read`: See filenames inside (list directory content).
- `write`: May add/delete files in directories, regardless of who own those individual files.
- `executable`: You may `cd` into directory, and may use pathnames that pass through it.

> `rwx--x--x`
> You can browse directory, but only if told specific filename.

---
## See Also
- [[Unix File System]]
- [[Unix File Descriptors]]
- [[Unix Low-Level File IO]]
- [[Unix User]]
- [[Bitwise Operations]]
