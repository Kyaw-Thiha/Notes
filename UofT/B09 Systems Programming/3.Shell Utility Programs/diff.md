# diff
The `diff` command compares two text files, and shows what changed between them.

---
### Basic Usage
```bash
diff old.c new.c        # compare two files
diff -r olddir newdir   # compare two directories recursively
diff -q file1 file2     # brief output (just says if files differ)
```

---
### Exit Codes
Exit Codes:
- `0` if files are identical
- `1` if files are different

Hence, we can use it as
```bash
if diff -q old.c new.c; then
  echo "No changes, skipping build"
else
  echo "Files changed, rebuilding..."
fi
```

---
### Standard Output Format
```bash
2,3d1       → lines 2–3 in old file were deleted (not in new file after line 1)
7c5         → line 7 in old became line 5 in new, but was changed
13a12       → line 12 in new file was added (not in old file at line 13)
```

---
### Unified Format
This is the format akin to `git diff`.
```bash
diff -u old.c new.c
```

which results in this format:
```bash
--- old.c       ← old file
+++ new.c       ← new file

@@ -1,16 +1,15 @@   ← chunk: was lines 1-16, now lines 1-15

 unchanged line
- this line was removed
+ this line was added
 another unchanged line
```

---
