# Redirection
Every [[Unix process]] has $3$ standard streams: stdin$(0)$, stdout$(1)$ and stderr$(2)$.

By default, these streams output to the terminal.
Redirection allows connection of these streams to files.

---
### Output Redirection
```bash
command > file.txt      # Write stdout to file (overwrites)
command >> file.txt     # Append stdout to file
command 2> errors.txt   # Write stderr to file
command &> all.txt      # Write both stdout and stderr to file
```

We can also combine streams.
```bash
command 2>&1            # Redirect stderr to wherever stdout goes
command > out.txt 2>&1  # Both to a file (order matters!)
command &> out.txt      # Shorthand for the above (bash/zsh)
```

### Input Redirection
```bash
command < file.txt      # Feed file as stdin to command

command << EOF          # Feed a block of text as stdin
line one
line two
EOF

command <<< "some text" # Feed a single string as stdin
```

---
# Pipelining
A pipe `|` connects the stdout of one command to the stdin or the next, forming a chain.

### Examples
```bash
ls -l | grep ".txt"              # Filter ls output
cat file.txt | sort | uniq       # Sort lines and remove duplicates
ps aux | grep nginx              # Find a running process
cat log.txt | wc -l              # Count lines in a file
du -sh * | sort -h               # Sort files by size
```

We can also pipe stderr too.
```bash
command1 |& command2    # Pipe both stdout and stderr
```

---
## Combining both redirection and pipelining
```bash
# Read from file, process, write to file
sort < input.txt | uniq > output.txt

# Capture stderr separately while piping stdout
command 2>errors.txt | next_command

# Send pipeline output to a file and the terminal simultaneously
command | tee output.txt | wc -l
```

---
## Multi-line Redirection
```bash
cat << EOF
Hello I’m Albert.
You can use variables too
E.g., \$x=$x
EOF
```

Note that `EOF` is not a keyword. It can be anything else:
```bash
cat << BANANA
This is my text
BANANA
```

---
### Variable Expansion
Unquoted marker will by default expand variables.
```bash
x=42
cat << EOF
Hello I'm Albert.
You can use variables too
E.g., \$x=$x
EOF
```

Quoted markers will not expand variables
```bash
x=42
cat << 'EOF'
Hello I'm Albert.
Now $x is $x
EOF
```

```bash
x=42
cat << "EOF"
Hello I'm Albert.
Now $x is $x
EOF
```

---
### Examples

Generating config file.
```bash
name="Alice"
cat << EOF > config.txt
username=$name
home=/home/$name
shell=/bin/bash
EOF
```

Feeding SQL with literal `$` signs.
```bash
cat << 'EOF' | psql mydb
SELECT * FROM users WHERE balance > $1000;
EOF
```

---
## See Also
- [[Shell]]
- [[Process Substitution]]
- [[Bash Syntax]]
