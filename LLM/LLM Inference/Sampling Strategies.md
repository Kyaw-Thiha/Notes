#llm/inference/sampling 
# 🎲 Sampling Strategies in Language Models
![[token_selection.png]]
Once an LLM finishes the **prefill phase**, it enters the **decode phase**, where it generates tokens *one-by-one*. At each step, it must choose **the next token** based on predicted probabilities. This decision process is called **sampling**.

Sampling strategies control **creativity**, **coherence**, and **repetition** in generated text.

---

## 🧮 From Logits to Tokens

At each step, the model outputs **logits**: raw scores for each token in the vocabulary. These are converted into probabilities using the softmax function:

$$
P(t_i) = \frac{e^{z_i / T}}{\sum_{j=1}^V e^{z_j / T}}
$$

- $z_i$: the logit for token $i$
- $T$: **temperature** (explained below)
- $V$: vocabulary size

---

## 🔥 Temperature Sampling

### 🎛️ What it does:
- Controls randomness and *confidence* in output.
- Applies to logits before softmax: divides each by **temperature** $T$.

### 🌡️ Behavior:
- $T = 1$: Normal distribution.
- $T < 1$: **Sharper** distribution → more deterministic.
- $T > 1$: **Flatter** distribution → more diverse and creative.

> Example: A temperature of 2.0 makes unlikely tokens more likely.

---

## 🔝 Top-k Sampling

### 🧰 Description:
- Only the **top $k$** tokens (by probability) are considered.
- Remaining logits are masked (set to $-\infty$ before softmax).

$$
\text{TopK}(z, k) = 
\begin{cases}
z_i & \text{if } i \in \text{Top-}k \text{ indices} \\
-\infty & \text{otherwise}
\end{cases}
$$

### ✅ Use When:
- You want bounded randomness.
- $k=50$ is a common default.

---

## 🔝 Top-p (Nucleus) Sampling

### 🧰 Description:
- Dynamically selects the smallest number of tokens whose cumulative probability ≥ $p$.
- Examples: $p = 0.9$ includes top tokens until 90% of total probability mass is covered.

### ✅ Use When:
- You want **context-aware diversity**.
- More flexible than top-k.

---

## 🔁 Managing Repetition

LLMs often **repeat phrases** or stick to high-probability words. We apply **penalties** to discourage that.

### 📉 Presence Penalty
- Penalizes tokens that have already appeared, regardless of frequency.
- Pushes the model to explore new ideas.

### 📉 Frequency Penalty
- Scales penalty by how often a token appears in generated output.
- Formula:
  $$
  \text{logit}_i := \text{logit}_i - \lambda \cdot \text{count}_i
  $$

- $\lambda$ is the penalty factor.
- Helps avoid loops or redundancy.

---

## 📏 Controlling Length

We often want to restrict how much the model generates.

### ⛔ Methods:
- **Token Limit:** Set `min_new_tokens` or `max_new_tokens`.
- **Stop Sequences:** Stop when a specific pattern is generated (e.g., `\n\n`, `</s>`).
- **EOS Token:** The model will stop when a special token (e.g., `<|endoftext|>`) is emitted.

> Example: Use `max_new_tokens=100` and stop sequence `"###"` for concise completions.

---

## 🌟 Beam Search (Deterministic Alternative)
![[beam_search.png]]
Unlike sampling, **Beam Search** tracks multiple possible sequences.

### 🧠 How it works:
1. Start with top $b$ most likely tokens (beam width $b$).
2. Extend each sequence with all possible next tokens.
3. Keep only top $b$ new candidates by cumulative log probability.
4. Repeat until EOS or max length.
5. Return the highest scoring sequence.

### ✅ Use When:
- You want the **most likely** output, not the most creative.
- Works well in translation, summarization.

---
## 📊 Summary Table

| Strategy        | Deterministic | Randomness Control | Output Diversity | Notes                         |
|-----------------|---------------|---------------------|------------------|-------------------------------|
| Greedy Decoding | ✅ Yes        | ❌ None             | ❌ Low           | Always picks max probability |
| Temperature     | ❌ No         | ✅ Via $T$          | ✅ Medium-High   | Adjusts entropy              |
| Top-k Sampling  | ❌ No         | ✅ Via $k$          | ✅ Controlled    | Clips low-probability tokens |
| Top-p Sampling  | ❌ No         | ✅ Via $p$          | ✅ Context-aware | Dynamic truncation           |
| Beam Search     | ✅ Mostly     | ❌ Fixed            | ❌ Low           | Multiple candidates          |

---
## 🧠 When to Use What?

| Goal                     | Recommended Strategy        |
|--------------------------|-----------------------------|
| Creative writing         | Top-p with $p=0.9$, $T=1.0$ |
| Deterministic outputs    | Greedy or Beam Search       |
| Controlled diversity     | Top-k with $k=40$           |
| Avoiding loops           | Add repetition penalties    |
| Short, structured output | Use stop sequences          |

---

## 🔗 See Also

- [[Inference Process]]
- [[Decoder]]
- [[Self-Attention]]
- [[KV Cache]]
