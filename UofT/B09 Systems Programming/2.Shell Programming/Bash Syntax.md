# Bash Syntax

### Printing
```bash
echo hey nice to meet you
```

By default, has newline. To omit,
```bash
echo -n hey nice to meet you
```

To put extra space,
```bash
echo xxx\ \ \ \ yyy zzz
echo ’xxx    yyy zzz’
echo "xxx    yyy zzz"
echo xxx’    ’yyy zzz
```

Escaping characters,
```bash
echo \$v
echo '$v'
echo '$'v
echo "\$v"
```


---
### Variables
```bash
v = xxx
v0 = yyy
echo $v0      # yyy
echo ${v}0    # xxx0
```

Then,
```bash
v = 'Sales Receipt.txt'
ls $v    # 2 Arguments: "Sales", "Receipt.txt"
ls "$v"  # 1 Argument: "Sales Receipt.txt"
ls ${v}  # 2 Arguments: "Sales", "Receipt.txt"
ls "${v}"  # 1 Argument: "Sales Receipt.txt"
```

---
### Arithmetic Expression
```bash
x=$((4 + 1))
y=$(($x * 2))       # $((x * 2)) also OK
echo $(($y + 3))    # $((y + 3)) also OK
```

---
### Command Substitution
We can run a command, capture its stdout, and use it as argument for another command.
```bash
./print-args $(echo ’ aaa bbb ccc ’)     # => 3 arguments, spaces stripped.

./print-args "$(echo ’ aaa bbb ccc ’)"   # => 1 argument, spaces preserved.
```

---
### Shell Script
We can run a shell script using 
```bash
sh myscript
```

But also, what we can do is
```bash
# Inside myscript.bash
#!/bin/sh

# Then, set executable flag on the file
chmod u+x myscript

# Then, we can run it as an executable
./myscript
```

---
### Command Line Arguments
If we were to run the script with arguments,
```bash
./myscrpt foo bar xyz
sh myscript foo bar xyz
```

Then,
```bash
$#    # No. of arguments: 3 
$0    # Name of script: myscript
$1    # First Argument: foo
$2    # Second Argument: bar
$3    # Third Argument: xyz
$*    # Expands arguments to one word: "foo bar xyz"
$@    # Expands arguments to three words: "foo", "bar", "xyz"
```

We can also use the `shift N` keyword to remove `N` parameters.
```bash
$#    # No. of arguments: 2 
$0    # Name of script: myscript
$1    # First Argument: bar
$2    # Second Argument: xyz
$*    # Expands arguments to one word: "bar xyz"
$@    # Expands arguments to two words: "bar", "xyz"
```

---
### Translate
```bash
tr -d 123             # delete '1', '2', and '3'
tr 'a-z' 'A-Z'        # convert lowercase to uppercase
tr -d '\n'            # delete all newlines
tr -s ' '             # squeeze repeated spaces into one
tr ':' '\n'           # replace every colon with a newline (useful for $PATH)
```

---
### File Redirection
```bash
tr -d 123 0< infile 1> outfile
# or
tr -d 123 < infile > outfile
```
- `< infile`: feed `tr` its input from the file
- `> outfile`: send `tr`'s output to a file

[[Redirection and Pipelining|Read more]]

---
### Sequential Composition
```bash
cd B09
ls -l
cd ..
```
is equivalent to
```bash
cd B09 ; ls -l ; cd ..
```
If you want to split one command to multiple lines,
```bash
echo hello B09 \
students
```

---
### Exit Code
Commands give an exit code when done.
```bash
ls /tmp        # succeeds
echo $?        # prints 0

ls /fakedir    # fails (directory doesn't exist)
echo $?        # prints 1 (or 2)
```

---
### Logical Operators
```bash
# Sequential execution, but stop upon first false
mkdir foo && cp myfile foo
```

```bash
# Sequential execution, but stop upon first true.
mkdir foo1 || mkdir foo2 || mkdir foo3
```

```bash
# Logical not: turn 0 to 1, non-0 to 0.
! mkdir foo
```

Note that `&&` and `||` have same precedence, but both are lower than `!`.

---
### Conditional Statement
```bash
if list1 ; then
	list2
elif list3 ; then
	list4
else
	list5
fi
```

---
### While Loop
```bash
while list1 ; do
	list2
done
```
Note the `;` before `do`.

We can also use
```bash
while list1 ; do list2 ; done
```

Note that we can use `break` and `continue`.

---
### For Loop

```bash
for var in word1 word2 ... ; do
	list
done
```

To loop in range,
```bash
for i in $(seq 0 3) ; do ... ; done
```

---
### Filename Patterns
```bash
ls a2/*.py
# shell expands this to:
ls a2/foo.py a2/bar.py a2/hello.py
```

```bash
ls *.py # foo.py, bar.py, hello.py

ls file?.txt # file1.txt, fileA.txt, NOT fileAB.txt

ls [abc]oo.txt # aoo.txt, boo.txt, coo.txt
ls file[135].txt # file1.txt, file3.txt, file5.txt

ls chapter[0-9].txt # chapter1.txt ... chapter9.txt
ls slide[0-9][0-9] # slide01, slide42, slide99

ls file[!0-9].txt # fileA.txt, filex.txt, file_.txt
ls [!aeiou]oo.txt # boo.txt, coo.txt, foo.txt
```

---
### Exit
Command `exit` terminates the whole shell script.

---
## Functions

The function can be defined as
```bash
myfunction() {
	echo "$1"
	echo "$@"
}
```
Note that if we didn't specify `return`, the function's `exit code` is the `exit code` of the last command.

Then, we can call this as
```bash
myfunction foo bar xyz
```

---
#### Using `getopts`
If we use [[Shell General Option Processing|getopts]], we must remember to reset `$OPTIND` since it globally point to next option from the script.
```bash
myfunction() {
    OPTIND=1          # ← reset this first!
    while getopts "M:nv" flag; do
        case $flag in
            n) echo "got -n" ;;
            v) echo "got -v" ;;
            M) echo "got -M: $OPTARG" ;;
        esac
    done
}
```

---
## Local Variables
```bash
myfunc() {
	local x y=hello # x,y local, y inited
	x=hi
	echo "$x" "$y"
}
```

---
### Arrays
```bash
crew=(kermit piggy fozzie)        # set
crew[3]=’sam eagle’               # set by index
echo "${crew[1]}"                 # get by index

crew+=(gonzo ’dr pepper’)         # append
echo ${#crew[@]}                  # number of elements

for c in "${crew[@]}"; do         # all elements, like $@
	...
done

# no prepend feature, but you can always do:
crew=(scooter "${crew[@]}")
```

---
### Associative Arrays
Key-value dictionary. [[#Arrays]] but string indexes.
```bash
declare -A mark

mark=([denise]=4 [bob]=9)                    # preload
mark[charles]=3                              # insert one
mark+=([bob]=7 [alice]=5)                    # insert many

for k in "${!mark[@]}"; do                   # all keys
	echo "$k has ${mark[$k]} marks"         # lookup
done
```

`declare -A` is required. Otherwise, [[Bash Syntax|bash]] assumes integer-indexed [[#Arrays|array]].

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
