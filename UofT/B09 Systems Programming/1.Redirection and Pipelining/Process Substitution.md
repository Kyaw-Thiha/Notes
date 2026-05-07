# Process Substitution
The main limitation of [[Redirection and Pipelining|pipelining]] is that it only allows $1$ input source and $1$ output destination.
```bash
cmd1 | cmd2 | cmd3
```
But what if we want to take two input files and write to two output files?

---
## Input Process Substitution
```bash
sort <(cmd1) <(cmd2)
```

Under the hood,
- this creates $2$ fake file descriptors: `/dev/fd/70` and `/dev/fd/71`
- runs all $3$ in parallel:
	- `cmd1`: writes to `/dev/fd/70`
	- `cmd2`: writes to `/dev/fd/71`
	- `sort`: reads from `/dev/fd/70` and `/dev/fd/71`.
- `sort` sees two "files", and merges + sort them

---
### Concrete Example
```bash
# Sort the combined output of two directory listings
sort <(ls /bin) <(ls /usr/bin)

# Sort the two files, then compare the diff
diff <(sort file1.txt) <(sort file2.txt)
```

---
## Output Process Substitution
```bash
foo >(cmd1)
```

Analogous to [[Process Substitution|input process substitution]], 
- this creates $1$ fake file descriptor: `/dev/fd/70`
- `foo` writes to `/dev/fd/70`
- `cmd` reads from `/dev/fd/70`

---
### Concrete Example
```bash
# Send output to both a log file and a compression program simultaneously
some_program >(gzip > output.gz) >(tee logfile.txt)
```

```bash
cat bigfile.txt | tee >(grep "ERROR" > errors.txt) >(grep "WARN" > warnings.txt) > /dev/null

# --------------------------------------------------- #

bigfile.txt
     |
     tee ──▶ >(grep "ERROR") ──▶ errors.txt
      |
      └────▶ >(grep "WARN")  ──▶ warnings.txt
```

---
## See Also
- [[Shell]]
- [[Process Substitution]]
- [[Bash Syntax]]
