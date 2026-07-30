# Unix File System

## I-Node
- The file system maintains a table of `i-nodes`.
- `i-node number` = array index
- Every file or directory is identified by its `i-node` and not by a filename.
- The i-node stores the file's **metadata**:
    - type: regular file, directory, symbolic link, device, socket, etc
    - permissions
    - owning user, owning group (stored as numerical IDs)
    - size
    - timestamps
    - where the data actually lives on dis
    - other metadata
    - but not the filename since filenames live in directories, not in i-nodes!
- Most of this info is obtainable via the `stat` command or the `stat` system call.

---
## Directory
- A directory is a table mapping filenames to i-node numbers.
- The exact on-disk data structure varies by filesystem.
  You should always access directories portably via the C library functions: `opendir()`, `readdir()`, `closedir()`.
- If two different filenames map to the _same_ i-node number, its called a hard link.

---
### Hard Link
- Makes `path_2` a second filename pointing to same i-node.
```bash
ln path path_2
```

- Corresponding system call: `link()`.
- The special entries `.` and `..` are implemented as hard links.
  (`.` links to the directory itself and `..` links to the parent).
- Hard-linking directories is otherwise disallowed.
- Every i-node stores a reference count.
- Deleting a file (by filename)
	- Decrease the i-node's reference count by 1.
	- If the count is still **positive**, we're done.
	- If the count reaches **zero**, the kernel frees the disk space and the i-node itself.

---
### Symbolic(Soft) Link
- A symlink is a special little file that just contains the pathname of another file.
- Create with the `symlink()` system call, or `ln` command.
```bash
ln -s path linkname
```
- If `path` is relative, it's interpreted relative to the directory `linkname` lives in.
- `ls -l` and `stat` show info about symlink itself. 
  Add `-L` to instead follow link and show info about its target.

---
### Examples
Suppose 
- `myhardlink` is a hard link to `/dir/file` 
- `mysymlink` is a symlink to `/dir/file`.

> **Question-1**: What if you have no access to `/dir` itself?
- `myhardlink` is still accessible (it points directly at i-node).
- `mysymlink` is denied (following it requires resolving the path, which requires traversal(execute) permission on `/dir`).

> **Question-2**: `/dir/file` gets renamed to `/dir/stuff`. 
- `myhardlink` is still working since it points to i-node.
- `mysymlink` is broken(dangling).

> **Question-3**: `/dir/stuff` gets deleted, and a brand-new file is created and named `/dir/file`.
- `myhardlink` still points to the original file's i-node.
- `mysymlink` now resolves to the **new file**, since it just re-reads whatever is currently at the path `/dir/file`.

---
## File Attributes
```c
int stat(const char *path, struct stat *statbuf);
int lstat(const char *path, struct stat *statbuf);
```
- Both return `0` on success, `-1` on error (and set `errno`).
- **Key difference:** if `path` is a symlink, `stat()` follows it and reports on the target. `lstat()` reports on the symlink itself.

---
