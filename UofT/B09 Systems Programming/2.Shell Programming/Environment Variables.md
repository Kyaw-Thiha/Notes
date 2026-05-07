# Environment Variables
Every running process on a Unix system carries a collection of [[environment variables]] as part of its state.

---
### Making variable an environment variable

```bash
# Plain shell variable — children won't see this
myvar=hello

# Environment variable — children inherit this
export MYENVVAR=foo

# Or equivalently, two steps:
MYENVVAR=foo
export MYENVVAR
```

---
You can give a child process different environment variables without changing your own shell:
```bash
LC_ALL=C MYNEWENV=foo printenv
```

```bash
x='foo bar'      # assigns "foo bar" to shell variable x
x=foo bar        # tries to run command "bar" with env var x=foo
```

---
### Standard Environment Variables
```bash
$HOME      # Home directory
$TZ        # User timezone  preference
$PATH      # Colon-separated list of directories.
```

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
