# Architecture Overview

## Project Identity

**tensorbit-run** is the bare-metal C++20/C11 inference engine for the Tensorbit
Labs P-D-Q pipeline. It is stage four (Execution) — consuming `.tb` / `.tbm` models
produced by tensorbit-core and refined by tensorbit-distill and tensorbit-quant.

## Directory Layout

```
tensorbit-run/
├── CMakeLists.txt
├── .clang-format
├── .gitignore
├── README.md
│
├── include/tensorbit/run/
│   ├── common.h            # C core: error codes, platform, logging, allocator
│   ├── common.hpp          # C++20 wrapper: Logger, Result<T,E>, vformat macros
│   ├── tensor.h            # C core: TbTensor, TBHeader, utility functions
│   ├── tensor.hpp          # C++ wrapper: RAII Tensor with std::span
│   ├── model.h             # C core: TbModel, TbLayerDesc, .tb/.tbm loading API
│   ├── model.hpp           # C++ wrapper: Model class, zero-copy mmap
│   ├── backend.h           # C core: TbBackend, op dispatch, registry
│   ├── backend.hpp         # C++ wrapper: BackendRegistry, op convenience wrappers
│   ├── ops.h               # C core: op function declarations (all F32)
│   └── ops.hpp             # C++ wrapper: TransformerRunner forward pass
│
├── src/
│   ├── main.cpp            # CLI entry point: load, tokenize, generate
│   ├── model.c             # .tbm loader: mmap, JSON parser, create
│   ├── model.cpp           # C++ layer compilation unit
│   ├── ops/                # C core reference implementations
│   │   ├── linear.c        # sparse N:M linear, dense linear
│   │   ├── norm.c          # RMSNorm, LayerNorm
│   │   ├── activation.c    # GELU, SiLU
│   │   ├── softmax.c       # numerically stable softmax with mask
│   │   ├── attention.c     # scaled dot-product attention
│   │   ├── rope.c          # rotary position embedding
│   │   ├── embedding.c     # embedding lookup
│   │   ├── residual.c      # residual add
│   │   └── conv2d.c        # conv2d (im2col + GEMM)
│   └── backend/
│       ├── cpu/            # CPU backend dispatch wrappers + registry
│       ├── simd/           # SIMD backend (future)
│       └── cuda/           # CUDA backend (future)
│
├── tests/
│   ├── test_linear.c       # sparse/dense linear correctness
│   ├── test_norm.c         # RMS/LayerNorm correctness
│   ├── test_softmax.c      # softmax correctness
│   ├── test_attention.c    # attention correctness
│   ├── test_tbm.c          # .tbm create/load round-trip
│   ├── test_integration.cpp# C++ wrapper integration
│   └── test_all.sh         # test runner
│
├── examples/
│   ├── desktop/            # Desktop inference demo
│   └── esp32/              # ESP-IDF component (future)
│
├── scripts/
│   └── setup.sh            # Dev environment setup
│
└── docs/
    ├── ARCHITECTURE.md     # This file
    ├── TBM_FORMAT.md       # .tb/.tbm binary format spec
    └── OPS.md              # operation catalog
```

## Dual-Layer Architecture (C Core + C++ Wrapper)

### C Core (Pure C11)

All inference kernels are written in C11 with zero dependencies:

- **No STL**: Uses raw arrays, `memcpy`, `malloc`/`free`
- **No exceptions**: Returns `TbError` codes
- **No RTTI**: No virtual dispatch (uses function pointer tables)
- **No OS deps**: `printf` for logging, `mmap`/`VirtualAlloc` for file mapping (platform-specific)
- **ESP32 ready**: Compiles under ESP-IDF and PlatformIO without modification

### C++ Wrapper (C++20)

Thin RAII layer providing:

- **`Result<T,E>`**: Custom `std::expected` replacement (C++20 compatible)
- **`Tensor`**: Move-only RAII wrapper with `std::span` shape access
- **`Model`**: mmap-based zero-copy model loading
- **`Logger`**: Thread-safe singleton with `std::vformat`
- **`BackendRegistry`**: Runtime hardware detection and op dispatch
- **`TransformerRunner`**: Full autoregressive forward pass

## Dependency Graph

```
tb-run (executable)
  └── tensorbit-run-cpp (static lib, C++20)
        ├── common.hpp → Logger, Result<T,E>
        ├── tensor.hpp → RAII Tensor
        ├── model.hpp  → Model class
        ├── backend.hpp → BackendRegistry, op wrappers
        ├── ops.hpp    → TransformerRunner
        └── tensorbit-run-c (static lib, C11)
              ├── model.c     → .tbm loader, JSON parser
              ├── ops/*.c     → all reference kernels
              └── backend/cpu/*.c  → CPU backend dispatch
```

## Key Design Decisions

### 1. C/Pointer-Based Tensor

The C core uses a simple struct with raw data pointer and shape array:

```c
typedef struct {
    void*    data;
    size_t   shape[8];
    uint8_t  rank;
    TbDtype  dtype;
    TbDevice device;
    bool     owns_data;
} TbTensor;
```

No templating, no virtual dispatch — just raw pointers and metadata. The C++
`Tensor` class wraps this with RAII and provides type-safe accessors.

### 2. Backend Dispatch

Each op has a function pointer in the `TbBackend` struct. The registry stores
backends sorted by priority. At dispatch time, the best available backend is
selected for each op individually:

```c
const TbBackend* be = tb_backend_get_best(TB_OP_SPARSE_LINEAR);
be->ops[TB_OP_SPARSE_LINEAR](output, inputs, n_inputs, params);
```

### 3. Zero-Copy Model Loading

On desktop platforms, `.tbm` files are memory-mapped (`mmap` / `MapViewOfFile`).
The `Model::get_weights()` and `Model::get_mask()` methods return `Tensor` objects
that wrap pointers directly into the mapped region — no data copy occurs.

### 4. N:M Sparse Inference

The CPU reference performs sparse linear by iterating over packed N:M bitmasks:

```c
for each output feature j:
  acc = bias[j]
  for each group g of M input features:
    mbyte = mask[j * groups_per_row + g]
    for k in 0..M:
      if (mbyte & (1 << k)):
        acc += x[g*M + k] * w[j*in_feat + g*M + k]
  y[j] = acc
```

For 2:4 sparsity (the most common), this skips 50% of multiplications.

### 5. Logging — Following tensorbit-core Conventions

Logger uses `std::vformat` + `std::make_format_args` (C++20). **All format args
must be lvalues** — no literals, no ternaries, no arithmetic expressions.

### 6. No External Dependencies

- No JSON library (hand-rolled parser for `.tbm` index)
- No Eigen3 (not needed for inference — matmul is direct loops)
- No CUDA/cuBLAS required (optional backend)
- No GoogleTest (custom `EXPECT_TRUE`/`EXPECT_FLOAT_EQ` macros)

## Memory Budget

| Component | Size | Notes |
|-----------|------|-------|
| Weights (FP32, 7B, 2:4 sparse) | 28 GB | Dense storage in .tb |
| Masks (2:4 packed) | 1.75 GB | 1 byte per 4 weights |
| KV Cache (32 layers, seq=2048, hidden=4096) | ~2 GB | FP32 |
| Activations (1 token) | ~16 KB | Per-layer buffers reused |
| **Engine binary** | ~50 KB | C core stripped |
| **Total runtime overhead** | ~2 GB + model size | Zero-copy for weights |

## License

Apache License 2.0 — Tensorbit Labs
