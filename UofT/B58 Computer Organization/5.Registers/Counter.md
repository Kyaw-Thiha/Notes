## Asynchronous Counter
### Preliminary
Recall that in [[Flip-Flop|T Flip-Flop]], the output is inverted when input $T$ is high. Now, consider a series of [[Flip-Flop|T Flip Flop]] connected in sequence.

![image|200](https://notes-media.kthiha.com/Counter/e9dc2923b0847e93c6407a402b674b6c.png)

More interestingly, we could connect the output of one [[Flip-Flop|flip flop]] to the clock input of the next.

### Ripple Counter
This is a $4\text{-bit}$ [[Counter|ripple counter]].
![image|350](https://notes-media.kthiha.com/Counter/387eeab237aff78a9ea11faa0ad50448.png)
This is an example of `asynchronous circuit`
- the four outputs do not change upon the same clock signal.
- timing isn't synchronised with clock pulse.

### Example
![image|350](https://notes-media.kthiha.com/Counter/2646519e5a91995dcda55def34e8c320.png)

### Timing Diagram
![image|350](https://notes-media.kthiha.com/Counter/c2f002940c9523154cdb65849d9dd7f6.png)

---
## Synchronous Counter
This is a [[Counter|synchronous counter]] since all output $Q$s change upon the same clock signal.

![image|350](https://notes-media.kthiha.com/Counter/07637c14bf14ecd6d2000527394649ac.png)

---
## Counter with Parallel Load
![image|350](https://notes-media.kthiha.com/Counter/8449cd0543bdef9a2991a5cb6f4dfecb.png)
[[Counter|Counters]] are often implemented with `parallel load` and `clear` inputs. This allow the [[Counter|counter]] to be set to whatever value needed.

---
