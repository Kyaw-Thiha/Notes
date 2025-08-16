# 🤗 Transformers: AutoModel & Task-Specific Heads
 #llm #hugging-face/llm/automodal 

## Backbone vs Head

- **Backbone** = Transformer encoder/decoder producing hidden states  
  Shape: `[batch, seq_len, hidden_size]`.  
- **Head** = Task-specific module on top of backbone:  
  - `AutoModelForSequenceClassification`  
  - `AutoModelForTokenClassification`  
  - `AutoModelForCausalLM`  
  - etc.

---

## Example: Sequence Classification

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt)

batch = tok(["I love this", "I hate this"], padding=True, truncation=True, return_tensors="pt")
out = model(**batch)

probs = torch.softmax(out.logits, dim=-1)
preds = [model.config.id2label[i] for i in probs.argmax(dim=-1).tolist()]
```

---

## Shape Cheat Sheet

- Hidden states (backbone): `[B, T, H]`  
- Classification head: `[B, num_labels]`  
- LM head: `[B, T, vocab_size]`  

---

## Saving & Sharing Models

```python
model.save_pretrained("my_dir")
# reload later:
from transformers import AutoModel
_ = AutoModel.from_pretrained("my_dir")

# Push to Hub
from huggingface_hub import notebook_login
notebook_login()
model.push_to_hub("my-awesome-model")
```

---

## Minimal Theory You Need

- **Tokenization:** Subword tokenization reduces OOVs.  
- **Embeddings:** `input_ids` → dense vectors + positional encodings.  
- **Attention:** Each token contextually attends to others; masks ignore padding/future tokens.  
- **Softmax/logits:** Models output logits for numerical stability; convert later to probabilities.  
- **Max length:** Respect `max_position_embeddings` (e.g. 512 for BERT).  

---

## Pitfalls
- Using backbone (`AutoModel`) alone → you must add your own head/loss.  
- Averaging over padding tokens → always respect `attention_mask`.  
- Over-length sequences silently truncated → control with tokenizer arguments.  

---

## When to use `AutoModel` vs `AutoModelFor...`
- **AutoModel** → feature extraction, custom research, building your own head.  
- **AutoModelFor...** → downstream tasks with ready-to-use heads.  

---