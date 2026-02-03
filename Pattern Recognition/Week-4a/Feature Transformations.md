# Feature Transformations

`Centering`
$$
x_{i,j} \leftarrow x_{i,j} - \text{mean}_{i} (x_{i,j})
$$
---

`Unit Range`
$$
x_{i,j} \leftarrow \frac{x_{i,j} - \min_{i}\{ x_{i,j} \}}
{\max_{i} \{ x_{i,j} \} - \min_{i} \{ x_{i,j} \}}
$$

---
`Standardization`
$$
x_{i,j} \leftarrow
\frac{x_{i,j} - \text{mean}_{i}(x_{i,j})}
{\text{std}_{i}(x_{i,j})}
$$
---

`Clipping`
$$
x_{i,j} \leftarrow \text{sign}(x_{i,j}) \ 
\max\{ b, \ |x_{i,j}| \}
$$
---

`Sigmoidal Term Frequency (TF)`
$$
x_{i,j} \leftarrow
\frac{1}{1 + \exp(b \ x_{i,j})}
$$
where $b$ is defined by the user

---
`Logarithmic Term Frequency (TF)` 
$$
x_{i,j} \leftarrow \log(b + x_{i,j})
$$
where $b$ is defined by user

---
