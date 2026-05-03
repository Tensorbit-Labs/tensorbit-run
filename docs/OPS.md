# Operation Catalog

All operations in tensorbit-run. Each operation has a C core reference
implementation (portable, no dependencies) and optional accelerated backends.

## Linear Operations

### sparse_linear

```
y = x · W^T + bias    (with N:M sparsity via mask)

x:    [in_features]
W:    [out_features, in_features]  (dense, pruned = 0.0)
mask: [out_features, in_features / M]  (packed uint8)
bias: [out_features]  (optional)
y:    [out_features]
```

**Complexity:** O(out_features × in_features × N/M)

For 2:4 sparsity (50% pruned): skips half the multiplications.

### dense_linear

```
y = x · W^T + bias

x:    [in_features]
W:    [out_features, in_features]
bias: [out_features]  (optional)
y:    [out_features]
```

**Complexity:** O(out_features × in_features)

Standard dense matrix-vector product. Used for unpruned layers (embedding, lm_head, norms).

## Normalization

### rms_norm

```
RMS(x) = sqrt(mean(x²) + ε)
y = x / RMS(x) * weight

x:      [n]
weight: [n]  (learned scale, optional)
y:      [n]
```

Used in Llama/Mistral architectures instead of LayerNorm.

### layer_norm

```
μ = mean(x)
σ² = var(x)
y = (x - μ) / sqrt(σ² + ε) * weight + bias

x:      [n]
weight: [n]  (learned scale, optional)
bias:   [n]  (learned shift, optional)
y:      [n]
```

## Activation Functions

### gelu

```
GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))

x: [n]
y: [n]
```

Gaussian Error Linear Unit. tanh approximation (matches PyTorch `nn.GELU(approximate='tanh')`).

### silu

```
SiLU(x) = x · σ(x) = x / (1 + exp(-x))

x: [n]
y: [n]
```

Sigmoid Linear Unit. Also known as Swish. Used in Llama MLP gating.

## Softmax

### softmax

```
exp(x_i) = exp(x_i - max(x))
y_i = exp(x_i) / Σ exp(x_j)

With optional mask:
exp(x_i) = exp(x_i + mask_i - max(x + mask))
Mask values of -∞ suppress positions (e.g., causal masking).

x:    [n]
mask: [n]  (optional, added before exp)
y:    [n]
```

Numerically stable: subtracts max before exponentiation to prevent overflow.

## Attention

### scaled_dot_product

```
S = Q · K^T / √d
if causal: S[i][j] = -∞ for j > i
if mask: S += mask
A = softmax(S)
Y = A · V

Q: [n_heads, seq_len, head_dim]
K: [n_heads, seq_len, head_dim]
V: [n_heads, seq_len, head_dim]
Y: [n_heads, seq_len, head_dim]
```

**Complexity:** O(n_heads × seq_len² × head_dim)

### rope (Rotary Position Embedding)

```
For position m, dimension pair (2i, 2i+1):
  θ_i = 1 / base^(2i/d)
  cos_mi = cos(m · θ_i)
  sin_mi = sin(m · θ_i)
  q[2i]'   = q[2i]·cos_mi   - q[2i+1]·sin_mi
  q[2i+1]' = q[2i+1]·cos_mi + q[2i]·sin_mi
  (same for k)

q, k: [n_heads, seq_len, head_dim]  (modified in-place)
```

**Complexity:** O(n_heads × seq_len × head_dim)

Uses `base = rope_theta` (typically 10000.0 for Llama, 1000000.0 for Llama 3).

## Residual Connection

### residual_add

```
y = x + residual    (element-wise)

x:        [n]
residual: [n]
y:        [n]
```

## Embedding

### embedding

```
y[t] = table[token_ids[t]]    (row copy)

token_ids: [n_tokens]  (int32 indices)
table:     [vocab_size, embedding_dim]
y:         [n_tokens, embedding_dim]
```

## Convolution

### conv2d

```
y = conv2d(x, kernel, stride, padding) + bias

x:      [batch, in_channels, H, W]
kernel: [out_channels, in_channels, kH, kW]
bias:   [out_channels]  (optional)
y:      [batch, out_channels, H', W']

H' = (H + 2·padding - kH) / stride + 1
W' = (W + 2·padding - kW) / stride + 1
```

Implemented as direct convolution (im2col + GEMM for larger kernels is a planned optimization).

For ViT patch embedding: stride = patch_size, padding = 0, no bias.

## Backend Support Matrix

| Operation | CPU Ref | SIMD (AVX2) | SIMD (NEON) | CUDA | Metal | Vulkan |
|-----------|---------|-------------|-------------|------|-------|--------|
| sparse_linear | ✓ | ✓ | ✓ | ✓ | - | - |
| dense_linear | ✓ | ✓ | ✓ | ✓ | - | - |
| rms_norm | ✓ | ✓ | ✓ | ✓ | - | - |
| layer_norm | ✓ | - | - | ✓ | - | - |
| gelu | ✓ | ✓ | ✓ | ✓ | - | - |
| silu | ✓ | ✓ | ✓ | ✓ | - | - |
| softmax | ✓ | - | - | ✓ | - | - |
| scaled_dot_product | ✓ | - | - | ✓ | - | - |
| rope | ✓ | ✓ | ✓ | ✓ | - | - |
| residual_add | ✓ | ✓ | ✓ | ✓ | - | - |
| embedding | ✓ | - | - | ✓ | - | - |
| conv2d | ✓ | - | - | - | - | - |

✓ = implemented, - = planned
