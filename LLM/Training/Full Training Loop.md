# Full Training Loop Without Trainer API (Pure PyTorch)

> **Goal**: Learn how to fine-tune a Transformer model using a **manual training loop in PyTorch**, without Hugging Face’s `Trainer`.

---

## 1. Why Write a Custom Loop?

While `Trainer` simplifies training, sometimes you want:
- **Full control** over training steps  
- Integration with **custom loss functions**  
- Advanced logging, gradient manipulation, or research experiments  

For this, you can write your own loop with PyTorch.

---

## 2. Setup: Data & Tokenization

```python
from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_fn(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized = raw_datasets.map(tokenize_fn, batched=True)
```

---

## 3. DataLoader Preparation

```python
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized["validation"], batch_size=8, collate_fn=data_collator
)
```

---

## 4. Load Model

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2
)
```

---

## 5. Optimizer & Scheduler

```python
from torch.optim import AdamW
from transformers import get_scheduler

optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = len(train_dataloader) * 3  # 3 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
```

---

## 6. Training Loop

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

progress = 0
epochs = 3

for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress += 1
        if progress % 100 == 0:
            print(f"Step {progress} | Loss: {loss.item():.4f}")
```

---

## 7. Evaluation Loop

```python
import torch
from datasets import load_metric

metric = load_metric("glue", "mrpc")

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

final_score = metric.compute()
print(final_score)
```

---

## 8. Summary Workflow

1. Load & tokenize dataset (`datasets` + `AutoTokenizer`)  
2. Create `DataLoader`s with collator for padding  
3. Load pretrained model (`AutoModelFor...`)  
4. Define optimizer + learning rate scheduler  
5. Write training loop (`model.train()`, forward, backward, optimizer step)  
6. Write evaluation loop (`model.eval()`, no gradients, compute metrics)  

---

## Key Points

- You control **everything**: batching, optimizer, scheduler, loss, metrics.  
- Mirrors how PyTorch training usually works.  
- More verbose than `Trainer`, but essential for research flexibility.  

---
