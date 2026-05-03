# Tensorbit Binary (`.tb` / `.tbm`) Format Specification

## `.tb` File Format

A single tensor in the Tensorbit Binary format. Produced by tensorbit-core,
consumed by tensorbit-run.

### Header (4096 bytes, packed)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `magic` | `0x31304254` ("TB01" LE) |
| 4 | 4 | `version` | Format version (1) |
| 8 | 4 | `nm_n` | N in N:M sparsity |
| 12 | 4 | `nm_m` | M in N:M sparsity |
| 16 | 8 | `num_weights` | Total weight elements |
| 24 | 8 | `num_mask_bytes` | Total mask bytes |
| 32 | 8 | `weights_offset` | Offset to weight data (always 4096) |
| 40 | 8 | `masks_offset` | Offset to mask data |
| 48 | 1 | `precision` | 0=FP32, 1=FP16, 2=BF16, 3=FP64 |
| 49 | 4047 | `reserved` | Zero-padding |

Total header size: exactly 4096 bytes (enforced by `static_assert`).

### Data Layout

```
[0..4095]     Header
[4096..W]     Weight data (dense, pruned = 0.0, little-endian)
[W..end]      Mask data (1 byte per M-sized group)
```

### Mask Packing

For group `g` at `masks_data[g]`:
- Bit 0 = weight[g*M + 0] is kept (1) or pruned (0)
- Bit 1 = weight[g*M + 1] is kept (1) or pruned (0)
- ...
- Bit (M-1) = weight[g*M + M-1]

For M ≤ 8, one byte per group. For M = 16, two bytes per group. For M = 32, four bytes per group.

### Example: 2:4 Sparsity

```
M = 4, N = 2
Group size: 4 weights → 1 byte mask

Mask byte 0x05 (0b0101):
  Bit 0 = 1 → weight[0] kept
  Bit 1 = 0 → weight[1] pruned
  Bit 2 = 1 → weight[2] kept
  Bit 3 = 0 → weight[3] pruned

Mask byte 0x0A (0b1010):
  Bit 0 = 0 → weight[0] pruned
  Bit 1 = 1 → weight[1] kept
  Bit 2 = 0 → weight[2] pruned
  Bit 3 = 1 → weight[3] kept
```

## `.tbm` Container Format

The Tensorbit Model archive bundles multiple `.tb` blobs and model metadata
into a single file for deployment.

### File Layout

```
┌──────────────────────────────────────────────────────────────┐
│  .tb blob layer_0    (offset L₀, 4096-byte header + data)    │
│  .tb blob layer_1    (offset L₁)                               │
│  .tb blob layer_2    (offset L₂)                               │
│  ...                                                           │
│  .tb blob layer_N    (offset Lₙ)                               │
├──────────────────────────────────────────────────────────────┤
│  JSON index          (UTF-8, array of layer descriptors)       │
├──────────────────────────────────────────────────────────────┤
│  uint32_t            (4 bytes, little-endian, index length)    │
└──────────────────────────────────────────────────────────────┘
```

### JSON Index Schema

```json
{
  "architecture": "llama",
  "config": {
    "hidden_size": 4096,
    "num_heads": 32,
    "num_kv_heads": 32,
    "intermediate_size": 11008,
    "vocab_size": 32000,
    "num_layers": 32,
    "max_seq_len": 2048,
    "norm_eps": 1e-06,
    "rope_theta": 10000.0
  },
  "tensors": [
    {
      "name": "model.embed_tokens.weight",
      "offset": 0,
      "shape": [32000, 4096],
      "nm_n": 0,
      "nm_m": 0,
      "dtype": "fp32",
      "num_weights": 131072000,
      "num_mask_bytes": 0
    },
    {
      "name": "model.layers.0.self_attn.q_proj.weight",
      "offset": 524288000,
      "shape": [4096, 4096],
      "nm_n": 2,
      "nm_m": 4,
      "dtype": "fp32",
      "num_weights": 16777216,
      "num_mask_bytes": 4194304
    }
  ]
}
```

### Loading Algorithm

```
1. Open file, get size S
2. Read 4 bytes at offset S-4 → index_len (uint32 LE)
3. Read index_len bytes at offset S-4-index_len → JSON index
4. Parse JSON: extract architecture, config, tensors array
5. For each tensor in the array:
   a. Seek to tensor.offset
   b. Read and validate 4096-byte .tb header (magic, version)
   c. Map weights at tensor.offset + header.weights_offset
   d. Map masks at tensor.offset + header.masks_offset
6. Model is now ready for zero-copy inference
```

### manifest.json (Directory Mode)

For development, tensors can be stored as individual `.tb` files:

```json
{
  "architecture": "llama",
  "config": { ... },
  "tensors": [
    {
      "name": "model.layers.0.self_attn.q_proj.weight",
      "file": "layer_0_q_proj.tb",
      "shape": [4096, 4096],
      "nm_n": 2,
      "nm_m": 4,
      "dtype": "fp32"
    }
  ]
}
```

This is equivalent to the `.tbm` container format but uses separate files.

## Precision Types

| Code | Name | Bytes | C Type | Description |
|------|------|-------|--------|-------------|
| 0 | FP32 | 4 | `float` | IEEE 754 single precision |
| 1 | FP16 | 2 | `uint16_t` | IEEE 754 half precision |
| 2 | BF16 | 2 | `uint16_t` | Brain floating point (truncated FP32) |
| 3 | FP64 | 8 | `double` | IEEE 754 double precision |

## Version History

| Version | Changes |
|---------|---------|
| 1 | Initial format: single tensor, 4096-byte header, N:M masks |
