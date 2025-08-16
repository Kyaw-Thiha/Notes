# Tokenization
 #llm/transformers/tokenization #hugging-face/llm/tokenization 

Transformers cannot operate on raw text. They expect **discrete token IDs** as input:
- Words, subwords, or characters → mapped to integers via a **vocabulary**.
- Embedding layers transform these IDs into dense vectors for the model.

Tokenization bridges **human-readable text** and **numerical tensors**.

---

## Types of tokenization

### 1. Word-level tokenization
- Each unique word = one token in the vocab.
- Problems:  
  - Explodes vocab size (`"running"`, `"runner"`, `"ran"` all separate).  
  - **OOV issue**: unseen words → `<unk>` token.  
- Rarely used in modern LLMs.

### 2. Character-level tokenization
- Each character = one token.
- Solves OOV problem (all text reducible to characters).  
- But sequences become very long → inefficient for Transformers.

### 3. Subword tokenization (**modern standard**)
- Break words into frequent subwords or morphemes:
  - `"playing"` → `"play"`, `"##ing"`
  - `"unbelievable"` → `"un"`, `"believe"`, `"able"`
- Balances:
  - Keeps vocab manageable.
  - Covers rare words via composition.
  - Maintains efficiency.

---

## Popular algorithms

- **BPE (Byte-Pair Encoding)**: Iteratively merge most frequent pairs of symbols.
- **WordPiece** (used in BERT): Similar to BPE but optimizes likelihood under a language model.
- **Unigram LM** (used in SentencePiece): Probabilistic, picks subword set that maximizes likelihood.

Most Hugging Face models use one of these (e.g., DistilBERT: WordPiece, GPT-2: BPE).

---

## Hugging Face Tokenizer API

### Loading
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### Encoding text
```python
encoded = tok("Transformers are amazing!")
print(encoded.input_ids)      # list[int]
print(encoded.tokens())       # list[str]
```

### Decoding
```python
decoded = tok.decode(encoded.input_ids)
print(decoded)  # back to readable text
```

---

## Special tokens

Transformers expect special markers in input:
- `[CLS]` — start of sentence (used by classification heads).
- `[SEP]` — separates segments (QA, sentence pairs).
- `<pad>` — padding to equalize length across batch.
- `<mask>` — used in masked language modeling.

`AutoTokenizer` automatically inserts these when appropriate:
```python
tok("Hello world", add_special_tokens=True)
```

---

## Padding, truncation, batching

Transformers need fixed-shape tensors.  
Tokenizers handle this:

```python
batch = tok(
    ["Hello world!", "Transformers are amazing but long sentences might be cut."],
    padding=True, truncation=True, return_tensors="pt"
)
```

- **padding=True** → pad shorter sequences with `<pad>`  
- **truncation=True** → cut longer sequences to model’s max length  
- **attention_mask** → 1 for real tokens, 0 for padding  

---

## Tokenization workflow

1. Raw text → cleaned & lowercased (depending on tokenizer).
2. Split into subwords via BPE/WordPiece/Unigram LM.
3. Add special tokens (`[CLS]`, `[SEP]`, `<pad>`, …).
4. Map tokens → IDs.
5. Batch into tensors (`input_ids`, `attention_mask`, `token_type_ids`).
6. Model consumes these IDs.

---

## Key takeaways

- **Always** use the tokenizer that matches your checkpoint.  
- Subword tokenization is the default for modern Transformers.  
- Hugging Face `AutoTokenizer` abstracts away differences (BPE, WordPiece, Unigram).  
- Padding, truncation, and attention masks are critical for correct batching.  
- Decoding is lossy (some spaces, casing may differ), but roundtripping is usually reliable.

---
## See Also
- [[Pipeline]]
- [[Automodal]]
