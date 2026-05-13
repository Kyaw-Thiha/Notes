# Minterms and Maxterms
> Given any truth table, [[Minterms and Maxterms]] allow us to write a Boolean expression for it.

- [[#Minterms]] look at $Y=1$ and flip $0s$.
- [[#Maxterms]] look at $Y=0$ and flip $1$s.
- Everything else is opposite too.

---
## Minterms
- Look at $Y=1$ rows
- For each row,
	- flip $0$s
	- keep $1$s
	- AND them together
- Combine all rows with OR

![|300](https://i.sstatic.net/QsW6o.png)

> [[Minterms and Maxterms|Minterms]] are based on the inputs, and have no idea on what makes $Y$ true.

It is sum of products(`SOP`).

---
## Maxterms
- Look at $Y=0$ rows
- For each row,
	- flip $1$s
	- keep $0$s
	- OR them together
- Combine all rows with AND

![|300](https://i.sstatic.net/QsW6o.png)

It is product of sums(`POS`).

---
