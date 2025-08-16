# Weights & Biases (W&B) for Experiment Tracking
 #hugging-face/llm/wandb

> **Goal**: Learn how to integrate **Weights & Biases (W&B)** with Hugging Face models to track training metrics, hyperparameters, and artifacts.

---

## 1. What is W&B?

[W&B](https://wandb.ai/) is a **machine learning experiment tracking platform**.  
It helps you:
- Log training/evaluation metrics in real-time  
- Visualize learning curves and comparisons  
- Store and version datasets, models, configs  
- Collaborate with teammates by sharing dashboards  

It integrates directly with **PyTorch, Hugging Face Transformers, Accelerate, and Trainer**.

---

## 2. Installation & Setup

```bash
pip install wandb
wandb login
```

- You’ll be prompted for your API key (available in your W&B account).  
- Once logged in, all runs automatically sync to your dashboard.

---

## 3. Basic Usage in PyTorch

```python
import wandb
import torch

wandb.init(project="bert-finetuning", config={"epochs": 3, "lr": 5e-5})

for epoch in range(3):
    train_loss = 0.3 - 0.05*epoch
    eval_acc = 0.7 + 0.05*epoch
    wandb.log({"train_loss": train_loss, "eval_acc": eval_acc})

wandb.finish()
```

This logs metrics and creates interactive plots in the W&B dashboard.

---

## 4. Integration with Hugging Face Trainer

Hugging Face’s `Trainer` natively supports W&B.  
Simply install `wandb` and set `report_to="wandb"` in `TrainingArguments`.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="test-trainer",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    report_to="wandb",   # Enable W&B logging
    run_name="bert-mrpc", # Custom run name
)
```

When you call `trainer.train()`, metrics will automatically stream to W&B.

---

## 5. Integration with Accelerate

When using Accelerate, initialize W&B manually:

```python
import wandb
from accelerate import Accelerator

accelerator = Accelerator()
wandb.init(project="bert-accelerate")

for epoch in range(3):
    loss = 0.2
    accelerator.print(f"Epoch {epoch}, Loss {loss}")
    wandb.log({"epoch": epoch, "loss": loss})
```

---

## 6. Logging Hyperparameters

You can store configuration details for reproducibility:

```python
wandb.init(
    project="bert-finetuning",
    config={
        "model": "bert-base-uncased",
        "batch_size": 16,
        "lr": 5e-5,
        "epochs": 3,
    }
)
```

This creates a **config table** in the W&B dashboard.

---

## 7. Logging Model Artifacts

You can upload and version models or datasets:

```python
import wandb

artifact = wandb.Artifact("bert-mrpc", type="model")
artifact.add_file("pytorch_model.bin")
wandb.log_artifact(artifact)
```

Artifacts allow you to store models and reuse them in other runs.

---

## 8. Common Features

- **Dashboards**: Real-time plots for metrics (loss, accuracy, F1).  
- **Run comparison**: Overlay multiple runs to compare hyperparameters.  
- **Hyperparameter sweeps**: Automate searching across learning rates, batch sizes, etc.  
- **Artifacts**: Track datasets and models as versioned objects.  
- **Collaboration**: Share dashboards with your team.  

---

## 9. Example End-to-End (Trainer + W&B)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import wandb

# Initialize wandb
wandb.init(project="hf-trainer-wandb")

# Data
dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

dataset = dataset.map(tokenize_fn, batched=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# TrainingArguments with W&B
args = TrainingArguments(
    output_dir="test-trainer",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    report_to="wandb",
    run_name="bert-mrpc"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer
)

trainer.train()
```

Now all metrics will stream into your W&B dashboard automatically.

---

## 10. Summary

- **W&B** is a powerful experiment tracking platform.  
- Integrates seamlessly with **Trainer** (`report_to="wandb"`).  
- For custom loops, log with `wandb.log()`.  
- Use **artifacts** to store models and datasets.  
- Great for **hyperparameter sweeps** and team collaboration.  

---

## References
- [W&B Docs](https://docs.wandb.ai/)  
- [Hugging Face Docs: W&B Integration](https://huggingface.co/docs/transformers/main_classes/trainer#trainer-integrations)  
- [W&B Sweeps](https://docs.wandb.ai/guides/sweeps)  

---
