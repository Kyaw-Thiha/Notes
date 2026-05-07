# Dot Command
There are $3$ different ways to run a [[Bash Syntax|shell script]].
```bash
sh cmds.sh # new shell process 
./cmds.sh # new shell process
. ./cmds.sh # dot command; runs in current shell process
```

This means `. ./cmds.sh` defines [[Bash Syntax|functions]], set [[Bash Syntax|variables]], or use `cd` inside the current shell.

---
## Concrete Use Cases

### Setting Variables
```bash
# cmds.sh
MYVAR="hello"
export PATH="$PATH:/my/tool"
```

```bash
sh cmds.sh        # MYVAR and PATH changes vanish when child exits
. ./cmds.sh       # MYVAR and PATH are now set in YOUR shell ✓
```

---
### Defining Functions
```bash
# cmds.sh
greet() {
    echo "Hello $1"
}
```

```bash
sh cmds.sh        # greet() defined in child, child exits, function gone

. ./cmds.sh       # greet() is now available in YOUR shell
greet Alice       # → Hello Alice
```

---
### Changing Directory
```bash
# cmds.sh 
cd /var/log
```

```bash
sh cmds.sh        # YOU are still where you were
. ./cmds.sh       # YOU are now in /var/log 
```

---
### Real World Examples
```bash
. ~/.bashrc              # reload your bash config into current shell
. ~/.profile             # load environment variables into current shell
. ./venv/bin/activate    # activate a Python virtual environment
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
