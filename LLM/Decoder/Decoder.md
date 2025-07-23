#ml #llm/transformers/decoders 
# Decoder
A **decoder** generates output sequences (e.g., translations) one token at a time by: 
- Attending to previous target tokens (via **masked self-attention**) 
- Attending to encoder outputs (via **encoder-decoder attention**) This allows it to condition each output token on both: 
	- Previously generated tokens 
	- The entire input sequence 

--- 
## ✅ What’s Inside a Decoder? 
Each decoder consists of: 
1. **Target Embeddings**: For previously generated tokens + [[Positional Encoding (Short)|positional encodings]]. 
2. **Masked Multi-Head Self-Attention**: Looks only at earlier positions. ([[Attention Mask#2. **Causal Mask (Lower Triangular)**|Causal Mask]])
3. **Encoder-Decoder Attention**: Attends to encoder output. 
4. **Feedforward Neural Network (FFN)**: Per-token transformation. 
5. **Add & LayerNorm**: Residual connections after each sublayer. Multiple such decoder **layers are stacked** (e.g., 6 layers in original Transformer). 

### Decoder Layer Structure 

Target Embedding 
	↓ 
Masked Multi-Head Self-Attention 
	↓ 
Add & Norm 
	↓ 
Encoder-Decoder Attention 
	↓ 
Add & Norm 
	↓ 
Feedforward Network 
	↓ 
Add & Norm 
	↓ 
Decoder Output 

--- 
## ✅ Mathematical Walkthrough 
We demonstrate one decoder layer using simple values. 

**Step 1**: Target Input 
Assume 3 tokens with 2D embeddings: $$ Y = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} $$ 
**Step 2**: Masked Self-Attention 
Project to Q, K, V (assume identity for simplicity): $$ Q = K = V = Y $$ Compute scores: $$ S = \frac{QK^T}{\sqrt{2}} = \frac{YY^T}{\sqrt{2}} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix} $$ Apply a causal mask (hide future tokens): $$ M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix} $$ Apply mask: $$ S_{\text{masked}} = S + M $$ Then apply softmax row-wise to get attention weights: $$ A_{\text{masked}} = \text{softmax}(S_{\text{masked}}) $$ Final output: $$ \text{MaskedAttn}(Y) = A_{\text{masked}} \cdot V $$ 
**Step 3**: Encoder-Decoder Attention 
Use encoder output $E$ (same as earlier encoder example): $$ E = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} $$Project Y and E: 
- Q from Y 
- K, V from E 

Assume identity projections again: $$ Q = Y,\quad K = V = E $$Then: $$ S = \frac{QE^T}{\sqrt{2}} = \frac{YE^T}{\sqrt{2}} $$ Softmax over rows of $S$ gives cross-attention weights: $$ \text{CrossAttn}(Y, E) = \text{softmax}(S) \cdot E $$ 
**Step 4**: Feedforward Network Same as encoder: $$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$

**Step 5**: Add & LayerNorm 
Each of the three sublayers is wrapped in: 
- Residual connection: $x + \text{sublayer}(x)$ 
- [[Layer Normalization]] 

--- 
## ✅ Python Code: Minimal Decoder Layer (No Libraries) 
```python 
import numpy as np 

def softmax(x): 
	e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
	return e_x / np.sum(e_x, axis=-1, keepdims=True) 

def causal_mask(size): 
	return np.triu(np.ones((size, size)) * -np.inf, k=1) 

	def decoder_layer(Y, E, W_q, W_k, W_v, W_qe, W_ke, W_ve, W1, b1, W2, b2): 
	# Masked Self-Attention (Q = K = V = Y) 
	Q = Y @ W_q 
	K = Y @ W_k 
	V = Y @ W_v 
	dk = Q.shape[-1] 
	scores = Q @ K.T / np.sqrt(dk) 
	scores += causal_mask(Y.shape[0]) 
	attention_weights = softmax(scores) 
	self_attn_output = attention_weights @ V 
	
	# Encoder-Decoder Attention (Q from Y, K/V from E) 
	Qe = self_attn_output @ W_qe 
	Ke = E @ W_ke 
	Ve = E @ W_ve 
	cross_scores = Qe @ Ke.T / np.sqrt(dk) 
	cross_attn_weights = softmax(cross_scores) 
	encdec_output = cross_attn_weights @ Ve 
	
	# Feedforward Network 
	ff_input = encdec_output 
	hidden = np.maximum(0, ff_input @ W1 + b1) 
	
	# ReLU 
	output = hidden @ W2 + b2 
	
	return output 
```

For more code examples, 
- [[Decoder(Code)#Minimal Decoder (without libraries)|Minimal Example]]
- [[Decoder(Code)#Real-World Example (using Pytorch)|Real-World Example]]

--- 
## ✅ Why Use Masked Self-Attention? 
- Prevents "cheating" by disallowing access to future tokens during training. 
- Ensures outputs are generated autoregressively during inference. 

--- 
## ✅ Summary 

| Component             | Purpose                                    |     |
| --------------------- | ------------------------------------------ | --- |
| Masked Self-Attention | Attend to previous tokens only             |     |
| Encoder-Decoder Attn  | Attend to encoder outputs                  |     |
| Feedforward Network   | Non-linear transformation per token        |     |
| Add & Norm            | Stabilize and enable deep stacking         |     |
| Layer Stacking        | Deeper understanding + longer dependencies |     |

--- 
## See Also 
- [[Encoder-Decoder Attention]] 
- [[Decoder(Code)]]
- [[Decoder-Only Model]]
- [[Self-Attention]] 
- [[Attention Mask]] 
- [[Layer Normalization]] 
- [[Transformer Architecture]]