# Shell Grouping
Sometimes, we may want to treat multiple commands as one unit. For example to redirect all their output to a single file.

---
### Grouping in Current Shell
```bash
{ grep foo file1 ; ls ; } > file2
```

Note that
```bash
{ grep foo file1 ; ls ; } > file2   # CORRECT
{ grep foo file1 ; ls } > file2     # WRONG — missing ; before }
```

---
### Subshell
```bash
( list ; )
```

Since we are running the command in a different subshell,
```bash
# Curly braces: effects PERSIST after
{ x=hello ; cd / ; }
echo $x        # prints "hello"
pwd            # you're now at /

# Parentheses: effects are LOST after
( x=hello ; cd / ; )
echo $x        # prints nothing — x was never set here
pwd            # you're still wherever you were
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
