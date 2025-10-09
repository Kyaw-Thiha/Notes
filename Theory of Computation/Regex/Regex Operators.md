# Regex Operators

Let $L, L'$ be languages over $\sum$.

- `Complemenation`: $\bar{L} = \sum^* - L$
- `Union`: $L \ \cup \ L' = \{ x: x \in L \text{ or } x \in L' \}$ 
- `Intersection`: $L \ \cap \ L' = \{ x : x \in L \text{ and } x \in L' \}$
- `Concatenation`

## Precedence
The higher ones have higher precedence.
1. $*$
2. $\circ$ (concatenate)
3. $+$

**Notes**
- Drop outermost parenthesis 
  $(ST) = ST$
- Equal Precedence -> Right Associate 
  $\epsilon + 0 + 1 = \epsilon + (0 + 1)$

## Example
1. Find regex $R$ such that $L(R) = \left\{  x \in \sum^* : |x| \text{ is even} \right\}$
   - $R = ((0 + 1)(0 + 1))^*$
   - $R = (00 + 01 + 10 + 11)^*$

2. Find regex $R$ for $L(R) = \{ 001, 1001 \}$
   - $R = 001 + 1001$
   - $R = (\epsilon + 1)(001)$

## Equivalence
2 regexes are equivalent iff $L(R) = L(S)$.
**Notation**: $R \equiv S$

## Properties of Regex Equivalence
Two regular languages $R \equiv S$ if $L(R) = L(S)$.
For example, $(0^* 1^*)^* \equiv (0 + 1)^*$

- Commutativity of Union: $(R + S) \equiv (S + R)$
- Associativity of Union: $((R + S) + T) \equiv (R + (S + T))$
- Associativity of Concatenation: $((RS) T) = (R(ST))$
- Left Distributivity: $(R(S + T)) \equiv ((RS) + (RT))$
- Right Distributivity: $((S + T)R) = ((SR) + (ST))$
- Identity for Union: $(R + \epsilon) \equiv R$
- Identity for Concatenation: $(R\epsilon) \equiv R$ and $(\epsilon R) \equiv R$
- Annihilator for Concatenation: $(\emptyset R) = \emptyset$ and $(R \emptyset) = \emptyset$
- Idempotence of Kleene Star: $R^{*^*} \equiv R^*$


$(0 +1)^*(000)(0 +1)^*(111)(0 + 1)^* + (0 +1)^*(111)(0 +1)^*(000)(0 + 1)^*$
