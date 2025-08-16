# Hugging Face Accelerate API

> **Goal**: A practical + theory guide to using Hugging Face’s **Accelerate API** for efficient training & inference across CPUs, GPUs, and TPUs.

---

## 1. What is Accelerate?

`accelerate` is a lightweight library from Hugging Face that makes it **easy to train and run models on multiple devices** (CPU, multi-GPU, TPU) with minimal code changes.  

It wraps around **PyTorch/XLA/DeepSpeed/FSDP** and abstracts device placement, distributed training, and mixed precision.

---

## 2. Why Use Accelerate?

- **Seamless device placement**: CPU → single GPU → multi-GPU → TPU without rewriting code.  
- **Distributed training** made simple (DDP, FSDP, DeepSpeed).  
- **Mixed precision** (FP16, BF16) enabled with a flag.  
- **Checkpointing & logging** are integrated.  
- Keeps training loop **close to pure PyTorch** (unlike `Trainer`).

---

## 3. Installation

```bash
pip install accelerate
```

To configure:

```bash
accelerate config
```

You’ll be asked about:
- Compute environment (local, multi-GPU, etc.)  
- Mixed precision (yes/no)  
- Logging setup  

Then launch with:

```bash
accelerate launch train.py
```

---

## 4. Core Usage

The heart of Accelerate is the `Accelerator` class.

```python
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

accelerator = Accelerator()

# Load model & tokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Data
from datasets import load_dataset
raw_datasets = load_dataset("glue", "mrpc")

def tokenize_fn(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized = raw_datasets.map(tokenize_fn, batched=True)
train_dataloader = DataLoader(tokenized["train"], shuffle=True, batch_size=16)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Prepare objects for distributed/mixed precision training
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
```

---

## 5. Training Loop with Accelerate

```python
from transformers import get_scheduler

num_training_steps = len(train_dataloader) * 3
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

for epoch in range(3):
    model.train()
    for batch in train_dataloader:
        # Move batch to device automatically
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        accelerator.backward(loss)   # handles scaling in mixed precision
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    accelerator.print(f"Epoch {epoch} complete")
```

---

## 6. Evaluation with Accelerate

```python
from datasets import load_metric
metric = load_metric("glue", "mrpc")

model.eval()
for batch in train_dataloader:
    with torch.no_grad():
        outputs = model(**batch)
    predictions = torch.argmax(outputs.logits, dim=-1)
    metric.add_batch(predictions=accelerator.gather(predictions),
                     references=accelerator.gather(batch["labels"]))

final_score = metric.compute()
accelerator.print(final_score)
```

---

## 7. Advanced Features

- **Mixed Precision Training**
  - Enable FP16 or BF16 globally by setting `mixed_precision="fp16"` in `Accelerator()`.
- **Gradient Accumulation**
  - Control how many steps to accumulate before backprop.
- **Integration with DeepSpeed & FSDP**
  - Add `--config_file` and Accelerate handles partitioning/optimizer sharding.
- **Checkpointing**
  - Use `accelerator.save()` / `accelerator.load_state()` to save/load distributed states correctly.

---

## 8. Benefits

- Minimal code changes compared to raw PyTorch.  
- Run the same script on **CPU, single GPU, multi-GPU, or TPU**.  
- Easy integration with **Transformers** models and datasets.  
- Supports **research flexibility** + **production deployment**.  

---

## 9. When to Use Accelerate?

- When `Trainer` feels too restrictive.  
- For custom training loops that still need **scalability**.  
- For large models that need **multi-GPU or distributed training**.  
- When experimenting with **DeepSpeed/FSDP optimizations**.  

---

## 10. Quick Workflow Recap

1. Install and configure `accelerate`.  
2. Wrap model, optimizer, dataloader(s) with `accelerator.prepare()`.  
3. Replace `loss.backward()` with `accelerator.backward(loss)`.  
4. Use `accelerator.print()` for safe distributed logging.  
5. Use `accelerator.gather()` to collect predictions across devices.  
6. Launch with `accelerate launch script.py`.  

---

## References
- [Accelerate Docs](https://huggingface.co/docs/accelerate)  
- [Hugging Face Course — Accelerate](https://huggingface.co/learn/llm-course)  
- [DeepSpeed + FSDP integration](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)  

---
