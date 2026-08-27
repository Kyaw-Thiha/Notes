# GLoVe
[[GLoVe]] enforces global information into the embeddings.
![300](https://miro.medium.com/v2/resize:fit:1400/1*X88FNisWZNkLZak3uhcpjQ.png)

It helps solve the issue of [[Word2Vec]] not having global info.

---
## Technique
It does this by 
- computing `co-occurance frequency counts` for each word.
	- represented as a matrix 
	- where element $X_{ij}$ denotes number of times word $i$ appears in context of word $j$
- **optimization**: inner product of word vectors should be good predictor of co-occurance frequency.

---
## See Also
- [[Word2Vec]]
- [[Skip-Gram]]
- [[CBOW Model]]
- [[GLoVe]]

