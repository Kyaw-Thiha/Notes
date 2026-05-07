# Shell General Option Processing
Command `getopts` parses command-line flags like `-n`, `-v`, `-M string` one at a time.

```bash
getopts M:nv myflag
getopts vM:n myflag
getopts nvM: myflag
```

Note that `myflag`, as well as two other variables are updated:
```bash
$myflag      # The flag letter found (`n`, `v`, `M`, or `?` for error)
$OPTARG      # The argument to that flag (like `-M foo`)
$OPTIND      # Index of the next positional argument to process
```

---
### Example
Suppose user runs the following script.
```bash
./tinyscript -n -v -Mfoo -v -M bar abc def -n xyz
```

Then, when we call `getopts M:nv myflag` the $i^{th}$ time:
![image|300](https://notes-media.kthiha.com/Shell-General-Option-Processing/14a5bc636782646f16c6dba8c2b7cfe1.png)

After the loop, `$OPTIND=7`, so `$7` is `abc`.
You do `shift $((OPTIND - 1))` to strip all the flags, leaving `abc def -n xyz` as `$1 $2 $3 $4`.

---
### Explicit Ending
User can add explicit `--` to mark end of options.
```bash
./tinyscript -n -v -Mfoo -v -M bar -- abc def -n xyz
```

Then, when we call `getopts M:nv myflag` the $i^{th}$ time:
![image|300](https://notes-media.kthiha.com/Shell-General-Option-Processing/5458660022114e693dd46b2506cd3cbb.png)

`$OPTIND` becomes 8 instead of 7, because `--` itself occupies a slot. So `$8` is `abc`.

---
### Unknown Option
Suppose user adds an unknown flag `-k`.
```bash
./tinyscript -n -v -Mfoo -k -M bar abc def -n xyz
```

Then, when we call `getopts M:nv myflag` the $i^{th}$ time:
![image|300](https://notes-media.kthiha.com/Shell-General-Option-Processing/8d97e2d0c6f258563e6452de83f193e8.png)

When `-k` is encountered, 
- `getopts` sets `myflag` to `?` 
- and prints `Illegal option -k` to stderr
- but exit code is still $0$, so the loop continues

---
## See Also
- [[Shell]]
- [[Bash Syntax]]
- [[Shell Test Commands]]
- [[Shell General Option Processing]]
- [[Shell Grouping]]
- [[Shell Dot Command]]
- [[Redirection and Pipelining]]
- [[Process Substitution]]
- [[Environment Variables]]
