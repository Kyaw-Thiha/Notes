#ml #llm/transformers/encoders/positional-encoding 
## 🧭 Positional Embeddings
Positional embeddings are used in Transformers to inject order information into input sequences, since the model itself is permutation-invariant (i.e., it doesn't inherently understand token positions).

## 📐 Why Positional Embeddings?
In Transformer architectures, tokens are processed in parallel with no recurrence or convolution. To provide sequence order, we add or concatenate a positional embedding vector to each input token vector.

Let $x_i$ be the embedding of the $i$-th token. Then the input to the Transformer is:
$$z_{i}=x_{i}+p_{i}$$
where $p_i$ is the positional embedding corresponding to position $i$.

## 📊 Mathematical Explanation

Let $d$ be the embedding dimension. The positional embedding can be represented as:

- Additive: $z_i = x_i + p_i$
- Learned: $p_i$ is a learned vector
- Fixed: $p_i$ is a deterministic function (e.g., sinusoid)

## 🧪 Mathematical Example

Let $d=4$(model dimension) and input sentence be ["I", "love", "math"]. Suppose token embeddings (simplified) are:

x_0 = [1, 0, 0, 0]
x_1 = [0, 1, 0, 0]
x_2 = [0, 0, 1, 0]

Assume positional embeddings:

p_0 = [0.1, 0.2, 0.3, 0.4]
p_1 = [0.2, 0.3, 0.4, 0.5]
p_2 = [0.3, 0.4, 0.5, 0.6]

Then:

z_0 = [1.1, 0.2, 0.3, 0.4]
z_1 = [0.2, 1.3, 0.4, 0.5]
z_2 = [0.3, 0.4, 1.5, 0.6]

## 🌊 Sinusoidal Positional Embeddings (Used in original Transformer)

These are fixed embeddings that use sinusoids of different frequencies.
$$
\text{PE}_{\text{pos, 2i}}
= \sin\left( \frac{pos}{10000^{\frac{2i}{d}}} \right)
$$

$$
\text{PE}_{\text{pos, 2i+1}}
= \cos\left( \frac{pos}{10000^{\frac{2i}{d}}} \right)
$$
Where:
- $pos$: position
- $i$: dimension index
- $d$: total dimension

This allows the model to interpolate/extrapolate to unseen sequence lengths and encode relative distance information.

## 🧠 BERT Positional Embeddings

BERT uses learned positional embeddings. It adds three embeddings:
- Token embedding
- Segment embedding
- Positional embedding $p_i$

Each $p_i$ is a learned vector of size $d$ stored in a positional embedding table (up to max length, e.g., 512).
$$
z_{i} = x_{i} + p_{i} + s_{i}
$$
Where:
- $x_i$: token embedding
- $p_i$: learned position embedding
- $s_i$: segment embedding

Pros:
- Task-specific
- Fully learnable

Cons:
- Cannot extrapolate to longer sequences

## 🧠 RoBERTa Positional Embeddings

RoBERTa uses the same learned embeddings as BERT — no architectural changes. However:
- Trained with longer sequences (e.g., 514 tokens)
- Removes Next Sentence Prediction (NSP), not related to positional encoding but affects training

RoBERTa retains learned position vectors $p_i$ for positions up to a fixed length.

## 🧠 GPT-2 Positional Embeddings

GPT-2 also uses learned positional embeddings, but in a causal language model.
$$
z_{i} = x_{i} + p_{i}
$$

GPT-2 differs from BERT in that:
- It is decoder-only
- Uses causal masking (only looks at previous tokens)
- Still uses fixed-size learned $p_i$ vectors up to context length (usually 1024)
No segment embeddings are used.

## 📌 Comparison Table

| Model       | Positional Embeddings | Type          | Notes                      |
| ----------- | --------------------- | ------------- | -------------------------- |
| Transformer | Sinusoidal (fixed)    | Deterministic | Encodes relative distances |
| BERT        | Learned               | Trainable     | + Segment embeddings       |
| RoBERTa     | Learned               | Trainable     | Trained on longer inputs   |
| GPT-2       | Learned               | Trainable     | Decoder-only               |
## 🔍 Further Notes

- Learned embeddings allow better flexibility but do not generalize beyond trained length.
- Sinusoidal embeddings allow extrapolation and make relative distance computation easier.
- Newer models (e.g., T5, ALiBi, Rotary Embeddings) aim to address limitations of both.

## 📚 Suggested Readings

- Attention Is All You Need (original Transformer paper)
- BERT: Pre-training of Deep Bidirectional Transformers
- RoBERTa: A Robustly Optimized BERT Pretraining Approach
- Language Models are Unsupervised Multitask Learners (GPT-2)

## See Also
- [[Positional Encoding (Short)]]
- [[Encoder]]