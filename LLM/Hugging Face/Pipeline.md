# 🤗 Transformers: The Pipeline
 #llm #hugging-face/llm/pipeline 

A `pipeline` wraps three stages end-to-end:

1. **Preprocessing** → tokenize raw text into tensors  
2. **Model forward** → run through transformer + head  
3. **Postprocessing** → logits → probabilities/labels

```python
from transformers import pipeline

clf = pipeline("sentiment-analysis")
clf([
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
])
# → [{'label': 'POSITIVE', 'score': ...}, {'label': 'NEGATIVE', 'score': ...}]
```

Why it matters: reproducing these three steps manually gives full control over batching, devices, dtypes, and introspection.

---

## Step 1 — Preprocessing with AutoTokenizer

Responsibilities of the tokenizer:
- Segment text into subwords (BPE or WordPiece).  
- Map tokens → integer `input_ids`.  
- Add model-specific tokens: `[CLS]`, `[SEP]`, etc.  
- Return tensors + `attention_mask` to ignore padding.

```python
from transformers import AutoTokenizer

ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(ckpt)

batch = tok(
    ["I love this!", "I hate this!"], 
    padding=True, truncation=True, return_tensors="pt"
)
```

---

## Step 2 — Model Forward (through the pipeline)

- The `pipeline` automatically picks the right model class (e.g., `AutoModelForSequenceClassification`).  
- Produces logits, then applies postprocessing (softmax + label mapping).  

---

## Step 3 — Postprocessing

- Apply softmax to logits:  
  $$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$  
- Convert indices → human-readable labels using `config.id2label`.  

---

## Manual reproduction of pipeline

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt)

def classify(texts):
    batch = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        ids = probs.argmax(dim=-1).tolist()
    return [{"label": model.config.id2label[i], "score": probs[k, i].item()}
            for k, i in enumerate(ids)]
```

---

## Pitfalls & Fixes
- Tokenizer-model mismatch → always load from the same checkpoint.  
- Variable-length sequences → use `padding=True`.  
- Silent truncation → set `truncation=True` and check `max_length`.  

---

## When to use `pipeline`
- **Quick demos** or baselines.  
- **One-liners** for standard tasks (sentiment, QA, generation).  
- **NOT** for production-scale inference (use manual approach instead).  

---