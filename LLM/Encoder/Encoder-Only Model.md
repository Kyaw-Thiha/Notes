#ml #llm/transformers/encoders  
# Encoder-Only Models

## Definition
Encoder-only models are transformer architectures that consist solely of the **encoder stack** from the Transformer architecture.  
They are specialized for **understanding tasks** where the entire input is available at once.

---
## Key Characteristics
- Utilizes **self-attention** to capture relationships between all input tokens bidirectionally.
- **No decoder component**: optimized for understanding rather than generation.
- **No causal masking**: tokens attend to both left and right context during training.
- Pretrained using objectives like **Masked Language Modeling (MLM)**.

---
## Primary Use Cases
- ✅ Text classification (e.g., sentiment analysis)
- ✅ Named Entity Recognition (NER)
- ✅ Semantic similarity & embeddings
- ✅ Extractive Question Answering
- ✅ Textual entailment
- ✅ Information retrieval

---
## Popular Encoder-Only Models
| Model          | Notes                                              |
| -------------- | -------------------------------------------------- |
| **BERT**       | Introduced MLM and Next Sentence Prediction (NSP). |
| **RoBERTa**    | Improved training of BERT (more data, no NSP).     |
| **DistilBERT** | Lightweight BERT with ~40% fewer parameters.       |
| **ALBERT**     | Parameter sharing, factorized embeddings.          |
| **DeBERTa**    | Disentangled attention, better relative positions. |

---
## Code Implementation
- [[Encoder-Only Model(Code)#Minimal Encoder (without library)|Minimal Encoder]]
- [[Encoder-Only Model(Code)#Pytorch Encoder|Pytorch Encoder]]
- [[Encoder-Only Model(Code)#BERT Real-World Model (with Tokenizer & MLM)|BERT Encoder]]

---
## ⚖️ Strengths
- Deep **bidirectional contextual understanding**.
- Pretrained models are highly effective when fine-tuned.
- Strong performance on most **NLP understanding tasks**.

## ⚠️ Limitations
- Cannot perform **text generation** natively.
- Limited by **max input length** (e.g., 512 tokens in BERT).
- Computationally intensive for large models.

---
## 🧩 Related Architectures
- [[Decoder-Only Models (GPT)]]
- [[Encoder-Decoder Models (T5, BART)]]

---
## 🔗 References
- [Vaswani et al., "Attention is All You Need" (2017)](https://arxiv.org/pdf/1706.03762)
- [Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)](https://arxiv.org/pdf/1810.04805)

## See Also
- [[Encoder]]
- [[Encoder-Only Model(Code)]]
- [[Self-Attention]]
