# Shell Test Commands
`[ ]` is just a command that takes your tests as individual arguments, and returns an exit code.

---
### Path Tests
```bash
[ -e path ]           # does path exists
[ -f path ]           # does path exists and is regular file
[ -d path ]           # does path exists and is directory
[ -r path ]           # does path exists and is readable
[ -w path ]           # does path exists and is writable
[ -x path ]           # does path exists and is executable
[ path1 -nt path2 ]   # does both path exist and path1 is newer
[ path1 -ot path2 ]   # does both path exist and path1 is older
```

---
### String Comparisms
```bash
[ s1 = s2 ]        # string equality. Also !=, <, > (need escaping/quoting)
[ -n string ]      # string not empty
[ -z string ]      # string empty
```


With `"$v"` instead of `$v`, 
```bash
[ "$v" = xxx ]
[ -n "$v" ]
[ -z "$v" ]
```

---
### Number Comparisms
```bash
[ n1 -eq n2 ]    # integer equality
# Also -ne, -gt, -ge, -lt, -le
```

---
### Logical Connectives
We can also use [[Bash Syntax|logical connectives]] like
```bash
[ ! -e path ]                     # not
[ "$x" -eq 5 -a "$y" -eq 6 ]      # and
[ "$x" -eq 5 -o "$y" -eq 6 ]      # or
```

Note that `-a` has higher precedence than `-o`.

We can also use parantheses with escaping or quoting.
```bash
[ -d dir1 -a ’(’ -d dir2 -o -d dir3 ’)’ ]
```

---
### Common Bug: Variables
The [[Shell Test Commands|test command]] `[ ]` is actually just a program that receives words as arguments. 
```bash
v=""; [ -n $v ]          # shell sees: [ -n ]

v=" "; [ -n $v ]         # shell sees: [ -n ] 

v="x y"; [ -n $v ]       # shell sees: [ -n x y ]. Too many arguments.
```

The fix is to always quote the variables:
```bash
[ -n "$v" ]
```

---
### Old Error
Older shells(`pre-POSIX`) had a buggy `[` that choked even on quoted empty strings. The workaround was to
```bash
[ x != x$v ]
```
prepend a literal `x` to both sides.

However, this still can break when
```bash
# Case-1: $v contain spaces
v="hello world"
[ x != x$v ]    # shell sees: [ x != xhello world ] <- too many arguments

# Case-2: $v starts with special operator
v="!=" 
[ x != x!= ]    # behaviour is undefined / shell-dependent

# Case-3: $v contains ]
v="]" 
[ x != x] ]     # shell sees the first ] as closing the [ command
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
