# Cache
[[Cache]] are structures that makes memory appear closer than it actually is.

---

### Locality
[[Cache|Caches]] rely on spatial and temporal locality.

Examples:
- "Iterating over array": exhibits temporal and spatial locality
- "Executing code": exhibits temporal and spatial locality
- "Accessing from dictionary": does not exhibit locality

---
### Cache Blocks
The [[Cache|cache]] has a few sets of blocks:
- In a [[Cache|direct mapped cache]], each set has one block.
- In a [[Cache|N-way set associative cache]], each set has $N$ blocks.
- In a [[Cache|fully associative cache]], it has one set with all the blocks.

A memory address gets hashed to a set.
Different memory addresses may be hashed to the same set.

### Addresses and Cache
Each load fetches the entire [[Cache|cache block]]; not just a single value.
- The size of the cache block is dependant on the [[Cache|cache]].
- A block is a set of words with closely related addresses.

The easiest way to define a block is to look at its mask.

---
### Bit Masking
A `bit vector` is an integer that should be interpreted as a sequence of bits.

A `mask` is a value that can be used to turn specific bits in a bit vector on and off.

#### Example
Consider a 8-bit memory address which can represent $256$ different addresses.
$$
1010 \ 1010
$$
What if we divide $256$ addresses into 8-byte block?
$$
\underbrace{1010 1}_{\text{block no.}} \quad \underbrace{010}_{\text{block offset}}
$$

---
## Associativity
Most [[Cache|caches]] are smaller than the memory they are caching from so they can't store everything.

If two blocks hash to the same value, they can't be stored.
To avoid that, [[Cache|caches]] are often associative.
- A [[Cache|2-way associative cache]] can store two blocks that hash to same value.
- A [[Cache|fully associative cache]] doesn't have to worry about hash collisions.

---
### Cache Loading and Eviction
Each [[Cache|cache]] has a finite size.
- It can store some maximum number of blocks.
- Based on its associativity, it can store a set of number of blocks with specific hash.

Every time a load is performed from memory, the block must be stored. This means that another block might need to be evicted.

---