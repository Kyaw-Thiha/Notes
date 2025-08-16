# Fine-Tuning with the Trainer API
 #hugging-face/llm/trainer 

> **Goal**: Learn how to fine-tune a pretrained Transformer model on a custom dataset using Hugging Face’s `Trainer` class.

---

## 1. What is the Trainer API?

The **Trainer API** is Hugging Face’s high-level training loop wrapper for PyTorch models.  
It manages:
- Training + evaluation loops  
- Gradient updates  
- Logging & checkpointing  
- Mixed precision training  
- Multi-GPU / distributed training  
- (Optional) Upload to Hugging Face Hub  

This allows you to **fine-tune models with minimal boilerplate**.

---

## 2. Preparing the Dataset

We use the `datasets` library to load and preprocess data.

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
```

### Tokenization
Every dataset must be tokenized with the **same tokenizer** used by the checkpoint.

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_fn(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized = raw_datasets.map(tokenize_fn, batched=True)
```

---

## 3. Data Collator

A **data collator** dynamically pads batches during training, saving compute.

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

---

## 4. Load the Model

Pick a task-specific head. For classification:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2
)
```

> ⚠️ You may see warnings about random head initialization — this is expected when adapting a pretrained backbone to a new task.

---

## 5. Define Training Arguments

`TrainingArguments` configures everything: batch size, evaluation strategy, logging, saving, and device usage.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="test-trainer",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    push_to_hub=False,   # set True to upload to HF Hub
)
```

---

## 6. Initialize the Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

- `tokenizer`: ensures Trainer auto-manages padding if `data_collator` not specified  
- `train_dataset` / `eval_dataset`: must be tokenized datasets  

---

## 7. Fine-Tune the Model

Run training with:

```python
trainer.train()
```

This will:
- Run epochs over the dataset  
- Log training/eval metrics  
- Save checkpoints in `output_dir`  

---

## 8. Evaluation

After training:

```python
metrics = trainer.evaluate()
print(metrics)
```

---

## 9. Upload to Hub (Optional)

If `push_to_hub=True` in `TrainingArguments`:

```python
trainer.push_to_hub()
```

This makes your fine-tuned model available to others.

---

## 10. Summary Workflow

| Step | Description |
|------|-------------|
| 1 | Load dataset (`datasets`) |
| 2 | Tokenize with matching checkpoint tokenizer |
| 3 | Create data collator (dynamic padding) |
| 4 | Load task-specific model (`AutoModelFor...`) |
| 5 | Define training arguments (`TrainingArguments`) |
| 6 | Wrap everything in `Trainer` |
| 7 | Call `trainer.train()` |
| 8 | Evaluate + (optionally) push to Hub |

---

## Why Start with Trainer?

- Provides a **baseline** with almost no custom code.  
- Handles all boilerplate (saving, resuming, distributed training).  
- Easy to extend when moving to custom loops later.  

---
