# Nondeterministic Finite State Automata

`NFSA` is a `DFSA` with 2 additional features
- Multiple choices from reading a symbol
- $\epsilon$-transition: can spontaneously move to different state without reading new symbol

> It seems like `NFSA` can mainly be used to represent symbol union.

## Definition
An `NFSA` $M$ accepts a string $x$ means there's a way to
- start at the initial state
- real all of $x$
- end in an accepting state

$$
L(M) = \{ x \in \Sigma^*: M \text{ accepts } x\}
$$



$(0^*01(1+\epsilon))^*$