# tensorbit-run

**The Execution Runtime** — bare-metal, dependency-free C++20 inference engine.

`tensorbit-run` is the engine that sits inside the hardware and drives. It reads
Tensorbit Binary (`.tb` / `.tbm`) models produced by tensorbit-core, tensorbit-distill,
and tensorbit-quant, and executes them at the absolute theoretical limit of the chip.

Part of the **Tensorbit Labs P-D-Q pipeline**:
```
.safetensors → [tensorbit-core: Prune] → .tb → [tensorbit-distill] → [tensorbit-quant] → [tensorbit-run]
```

## Key Capabilities

| Feature | Description |
|---------|-------------|
| **Zero-Copy Memory** | Maps `.tbm` files directly into hardware RAM via mmap — no duplicate copies |
| **N:M Sparse Inference** | Executes N:M structured sparse linear layers, skipping pruned weights at runtime |
| **CPU Reference Backend** | Portable C11 kernels, autovectorizable, compiles on any C++20 compiler |
| **SIMD Backend** | AVX2/AVX-512 (x86) and NEON (ARM) accelerated kernels |
| **CUDA Backend** | NVIDIA GPU acceleration for large models |
| **Dynamic Compute Routing** | Detects available hardware and routes ops to the best backend |
| **Micro-Footprint** | C core compiles to kilobytes — flashable onto ESP32 and IoT sensors |
| **Full Transformer Support** | Llama/Mistral-style decoder with RMSNorm, RoPE, GELU/SiLU, KV cache |
| **Hybrid C/C++** | Pure C11 core (no STL, no exceptions) + C++20 desktop wrapper |

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel -j4

# Run with mock model (no real model needed)
./bin/tb-run --mock --prompt "hello world"

# Run with real model
./bin/tb-run --model model.tbm --prompt "Once upon a time"

# Run tests
bash tests/test_all.sh
```

## Command-Line Options

```
tensorbit-run [OPTIONS]

Model Loading:
  --model <path>       .tbm model file to load
  --model-dir <path>   Directory with manifest.json

Inference:
  --prompt <text>      Input text prompt
  --max-tokens N       Max tokens to generate (default: 50)
  --temperature T      Sampling temperature (default: 0.8)

Mock Mode (testing without real models):
  --mock               Create and run a mock model
  --mock-hidden N      Hidden size (default: 256)
  --mock-intermediate N Intermediate size (default: 1024)
  --mock-layers N      Number of layers (default: 2)
  --mock-vocab N       Vocab size (default: 100)
  --mock-seq-len N     Max sequence length (default: 64)
  --temp-dir <path>    Temp file directory (default: .)

Other:
  --help               Show help
  --version            Show version
```

## Build Options

| CMake Option | Default | Description |
|-------------|---------|-------------|
| `TENSORBIT_BUILD_TESTS` | ON | Build test suite |
| `TENSORBIT_BACKEND_SIMD` | ON | Enable AVX2/NEON acceleration |
| `TENSORBIT_BACKEND_CUDA` | OFF | Enable NVIDIA CUDA GPU backend |
| `TENSORBIT_BACKEND_METAL` | OFF | Enable Apple Metal GPU backend |
| `TENSORBIT_BACKEND_VULKAN` | OFF | Enable Vulkan GPU backend |
| `TENSORBIT_BACKEND_NPU` | OFF | Enable NPU backend interface |

## Model Format

tensorbit-run uses the `.tbm` (Tensorbit Model) container format:

```
.tbm file structure:
┌──────────────────────────────────────────────────┐
│  .tb blob layer_0    (4096-byte header + data)   │
│  .tb blob layer_1                                 │
│  ...                                              │
├──────────────────────────────────────────────────┤
│  JSON index          (array of layer descriptors) │
├──────────────────────────────────────────────────┤
│  uint32_t            (4 bytes: index byte length) │
└──────────────────────────────────────────────────┘
```

Each `.tb` blob contains a 4096-byte header with magic number, N:M sparsity config,
weight/mask offsets, and data type, followed by dense weight data and packed N:M bitmasks.

See [`docs/TBM_FORMAT.md`](docs/TBM_FORMAT.md) for the complete specification.

## Architecture

tensorbit-run has two compilation profiles sharing the same kernels:

| Layer | Language | Features | Targets |
|-------|----------|----------|---------|
| **C Core** | C11 | No STL, no exceptions, no RTTI, no OS deps | ESP32, bare metal, PlatformIO |
| **C++ Wrapper** | C++20 | RAII Tensor/Model, mmap, CLI, Backend registry | Linux, macOS, Windows |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for internals.

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Project internals, dependency graph, design decisions |
| [`docs/TBM_FORMAT.md`](docs/TBM_FORMAT.md) | .tb and .tbm binary format specification |
| [`docs/OPS.md`](docs/OPS.md) | Operation catalog with math definitions |

## Tech Stack

C11, C++20, CMake 3.22+, GCC 13+ / Clang 16+ / MSVC 2022

## License

This project is **dual-licensed**.
- Open source use: Licensed under the [GNU AGPLv3](LICENSE). You may use, modify, and distribute the code under the terms of the AGPL, which requires all modifications and larger works to be licensed under the same license and requires making source code available to network users.

- Commercial use: If you wish to use this library in a proprietary product without the copyleft obligations of the AGPL, a separate commercial license is available. Please contact us for details.
