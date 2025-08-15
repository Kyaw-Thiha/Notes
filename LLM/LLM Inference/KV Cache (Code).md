#llm/inference/kv-cache 
# KV Cache Code
KV (Key-Value) caching is a core optimization that dramatically improves the speed and efficiency of **decoder-only language models** like GPT. 

[[KV Cache|Read More]]

## In Python (without libraries)
```python
import math
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class SimpleTransformerDecoder:
    def __init__(self, d_model):
        self.d_model = d_model
        self.key_cache = []
        self.value_cache = []
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)

    def self_attention(self, q, K, V):
        scores = np.dot(q, K.T) / math.sqrt(self.d_model)
        weights = softmax(scores)
        return np.dot(weights, V)

    def decode_step(self, token_embedding):
        # Step 1: Linear projections
        q = token_embedding @ self.W_q
        k = token_embedding @ self.W_k
        v = token_embedding @ self.W_v

        # Step 2: Append K and V to cache
        self.key_cache.append(k)
        self.value_cache.append(v)

        # Step 3: Stack cached K/V
        K = np.stack(self.key_cache, axis=0)
        V = np.stack(self.value_cache, axis=0)

        # Step 4: Apply attention
        output = self.self_attention(q, K, V)
        return output

# Simulate input
decoder = SimpleTransformerDecoder(d_model=4)
for step in range(5):
    token_embed = np.random.randn(4)  # Random embedding
    output = decoder.decode_step(token_embed)
    print(f"Step {step+1}, Output: {output}")
```


## Real World KV Cache (with library)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

# Encode prompt (prefill phase)
prompt = "The meaning of life is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Step 1: Prefill
with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=True)
    logits = outputs.logits
    past_key_values = outputs.past_key_values  # ← KV cache here

# Step 2: Decode one token using cache
next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)

# Step 3: Generate next token with past_key_values (cached K/V)
with torch.no_grad():
    new_outputs = model(input_ids=next_token, use_cache=True, past_key_values=past_key_values)
    new_logits = new_outputs.logits
    new_past = new_outputs.past_key_values  # updated cache

# Decode result
generated_token = tokenizer.decode(next_token[0])
print(f"Next token: {generated_token}")
```

## See Also
- [[KV Cache]]