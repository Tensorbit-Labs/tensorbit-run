# TODO.md — tensorbit-run Remaining Enhancements

This document tracks features and improvements deferred to future releases
for the tensorbit-run inference engine.

---

## 1. Architecture: Support non-Llama model architectures

**File:** `include/tensorbit/run/ops.hpp:108-304`

### Current state
The `TransformerRunner` forward pass hardcodes Llama/Mistral-style
layer components: `self_attn.q_proj`, `mlp.gate_proj`, `mlp.up_proj`,
`mlp.down_proj`. The RMSNorm→Attention→RMSNorm→MLP loop structure is
assumed. Only the naming prefix is auto-detected (Llama, GPT-J, Phi).

### What's missing
Different architectures use fundamentally different structures:
- **GPT-J**: `attn.q_proj`, `mlp.fc_in`, `mlp.fc_out` (single FC, no gate/up/down)
- **Falcon**: `self_attention.query_key_value` (fused QKV), parallel attention + MLP
- **Phi**: `mixer.Wqkv` (fused QKV), different residual placement
- **Gemma**: RMSNorm before AND after attention (different residual structure)
- **DeepSeek**: Multi-head latent attention, MoE layers

### Suggested fix
- Define an architecture descriptor struct with field patterns for each
  model family (embedding, attention, MLP, norms, LM head).
- The `TransformerRunner` reads the descriptor and dispatches to
  architecture-specific forward pass functions.
- Start with Llama (working), add GPT-J support next (most similar),
  then extend to fused QKV architectures.

---

## 2. Memory: mmap fallback missing on Windows

**File:** `src/model.c:45-46`

### Current state
On Linux/WSL, if `mmap` fails, there's a `malloc + read` fallback that
loads the entire file into heap memory. On Windows, if `MapViewOfFile`
fails, it returns `TB_ERR_FILE_OPEN` with no fallback. Large models
that can't be address-mapped on Windows will fail to load.

### Suggested fix
- On Windows, if `MapViewOfFile` fails, fall back to `VirtualAlloc +
  ReadFile` to load the file into a committed virtual memory region.
- Or implement a chunked-read pattern: keep the file handle open and
  read tensor weights on-demand rather than loading the entire file.

---

## 3. Memory: On-demand tensor loading instead of full file mmap

**File:** `src/model.c:445-550`

### Current state
The model loader uses `mmap` (or `malloc` fallback) to map the entire
.tbm file into memory. For a 30 GB 7B model, this requires 30 GB of
virtual address space, which fails on WSL1 (7.7 GB process limit) and
on Windows with fragmented address space.

### Suggested fix
- On mmap/malloc failure, keep the file descriptor open and implement
  a lazy-load pattern: `tb_model_get_weight_tensor()` reads just the
  requested tensor's weights from disk via `pread()`.
- Cache recently accessed tensors with an LRU eviction policy.
- This reduces peak memory from O(file_size) to O(largest_tensor),
  fitting 7B models into ~4-6 GB physical RAM.
- This is also the pattern needed for ESP32 (flash-backed loading).

---

## 4. Runtime: SIMD flags crash on older x86_64 CPUs

**File:** `CMakeLists.txt:42-47`

### Current state
When `CMAKE_SYSTEM_PROCESSOR` is `x86_64`, the CMake unconditionally
adds `-mavx2 -mfma` compile flags. Running the resulting binary on a
pre-Haswell CPU (2013 or older, some low-power Atoms) causes immediate
`SIGILL` (illegal instruction).

### Suggested fix
- Add a CMake option `TENSORBIT_ENABLE_AVX2` (default ON) that users
  can disable on old hardware.
- Implement runtime CPU feature detection with `__builtin_cpu_supports()`
  or `CPUID` check, then dispatch to AVX2 or scalar paths at runtime.
- Keep the scalar fallback kernels as a safety net.

---

## 5. Runtime: ARM64 NEON SIMD not implemented

**File:** CMakeLists.txt:46, `src/ops/*.c`

### Current state
ARM64 NEON is declared with `-DTB_SIMD_NEON=1` but no NEON-accelerated
kernel implementations exist. All ops fall through to scalar C code.
This is a future placeholder for Apple Silicon and AWS Graviton.

### Suggested fix
- Implement NEON SIMD kernels for the most impactful ops: `dense_linear`,
  `rms_norm`, `silu`, `residual_add`.
- Use `#ifdef TB_SIMD_NEON` guards following the existing AVX2 pattern.
- Validate on Apple M1/M2 and AWS Graviton3.

---

## 6. Logging: Inconsistent use of TB_LOG_* vs bare fprintf

**Files:** `src/main.cpp`, `tests/test_*.c`

### Current state
`model.c` uses `TB_LOG_WARN`/`TB_LOG_ERROR`/`TB_LOG_INFO` macros with
the `[tb-run]` prefix. But `main.cpp` uses bare `fprintf(stderr, ...)`
and `fprintf(stdout, ...)` for informational output (model loading
info goes to stderr, generated text goes to stdout). Test files use
bare `printf(...)`. Output is inconsistent across components.

### Suggested fix
- Convert `main.cpp` info output to `TB_LOG_INFO` for model metadata.
- Keep generated text on stdout for pipeability (`./tb-run | cat`).
- Convert test `printf` to `TB_LOG_INFO` for consistent formatting.

---

## 7. CLI: `--temp-dir` help text disagrees with code default

**File:** `src/main.cpp:332, 364`

### Current state
Help text says default is `/tmp` (Unix-only), but the code default is `"."`
(current directory). On Windows, `/tmp` doesn't exist. On Linux, `"."`
may not be writable if running from a protected directory.

### Suggested fix
- Use `std::filesystem::temp_directory_path()` for the default.
- Update help text to match.
- On Windows, this resolves to `%TEMP%`; on Linux to `/tmp`.

---

## 8. Sampling: Hardcoded random seed produces deterministic output

**File:** `src/main.cpp:107`

### Current state (fixed May 2026)
RNG is now seeded from `std::random_device` instead of hardcoded `42`.
Temperature-based sampling now produces genuinely different outputs on
each run.

### Remaining consideration
`std::random_device` may not be truly random on all platforms (some
MinGW builds return deterministic values). Consider fallback to
`std::chrono::system_clock::now().time_since_epoch().count()` if
`rd.entropy() == 0.0`.

---

## 9. Dtypes: Only FP32 weights natively supported (INT4 dequant added May 2026)

**Files:** `include/.../model.hpp`, `include/.../model.h`, `src/model.c`, `include/.../ops.hpp`

### Current state
All linear algebra kernels assume `float*` inputs. The .tbm format supports
`fp32`, `fp16`, `bf16`, `fp64`, `int4`, `int8`. **INT4 is now supported** via
`Model::get_weight_fp32()` which dequantizes INT4/INT8 weights to FP32 at
load time. No kernel changes were needed — dequantization happens before the
linear dispatch, producing a temporary FP32 tensor consumed by existing kernels.

### What's done
- `TbDtype` enum: added `TB_DTYPE_INT8 = 7`, `TB_DTYPE_INT4 = 8`
- `TbLayerDesc`: added `scale_count` and `group_size` fields
- `tb_model_get_scale_tensor()`: zero-copy access to scale data in the mmap
- `Model::get_weight_fp32()`: returns FP32 weights for any dtype.
  FP32 → zero-copy view. INT4/INT8 → allocates temp buffer, dequantizes.
- `ops.hpp`: all `get_weights()` calls replaced with `get_weight_fp32()`
- JSON parser in `model.c`: recognizes `"int4"`/`"int8"` dtype strings,
  parses `scale_count` and `group_size` fields

### Remaining
- **FP16/BF16**: Same approach — add dequant branch in `get_weight_fp32()`.
  FP16 uses IEEE 754 half-precision unpacking. BF16 uses bit-shift `u16 << 16`.
- **Native kernel support**: Currently dequant+GEMM is two passes. A fused
  INT4 kernel that reads nibbles, multiplies by scales, and accumulates in
  one pass would be ~20% faster. Deferred until performance profiling.
- **CUDA INT4**: The CUDA dispatch (`backend/cuda/registry.c`) still expects
  FP32. Until `get_weight_fp32()` causes the C++ side to pass FP32, this
  works transparently. But for GPU-side dequant (faster), the CUDA kernels
  need INT4 support directly.

---

## 10. Backends: CUDA registry compile errors under TENSORBIT_BACKEND_CUDA

**File:** `src/backend/cuda/registry.c:197, 212`

### Current state (fixed May 2026)
- Removed non-existent `be->type` field assignment.
- Changed `nullptr` to `NULL` (file is compiled as C11, not C++).

### Remaining verification
The CUDA backend has never been build-tested with `TENSORBIT_BACKEND_CUDA=ON`
on a machine with CUDA installed. The registry may have additional link-time
errors from mismatched function signatures between the C declarations in
`registry.c` and the C++ definitions in `kernels.cu`.

---

## 11. ESP32: Scaffold is incomplete and untested

**File:** `src/model_esp32.c`

### Current state (fixed May 2026)
- Replaced nonexistent `parse_tbm_json()` with `tb_parse_tbm_index()`.
- Removed references to nonexistent `TbLayerDesc` fields (`weights_offset`,
  `masks_offset`). Weights offset now computed from `tbm_offset + 4096`.
- Changed `TB_DEVICE_HOST` to `TB_DEVICE_CPU`.
- Fixed `owns_data = true` to `owns_data = 1`.

### What's still missing for functional ESP32 inference
- **ESP-IDF project structure**: CMakeLists.txt, sdkconfig, partition table.
- **Flash sizing**: A quantized .tbm for a 7B model at INT4 is ~3.6 GB.
  ESP32-S3 has 16 MB flash max. Only tiny models (150K params) fit.
  Consider ESP32-P4 (32 MB PSRAM) or ESP32-S3 with external SPI flash.
- **Quantized inference**: ESP32 lacks FPU. All compute must be INT8/INT4.
- **KV cache in PSRAM**: PSRAM access from C requires explicit `IRAM_ATTR`.
- **No C++ stdlib**: `TransformerRunner` uses C++20 — needs porting to pure C.
- **On-demand streaming**: The current scaffold malloc's each tensor's
  worth of data per call to `tb_model_get_weight_tensor`, which leaks
  memory on repeated calls.
- **Wi-Fi/BLE model download**: OTA update from companion app.

### Suggested approach
For ESP32, target sub-100M param models with INT4 quantization.
The `model_esp32.c` file provides the flash-read primitives; the
remaining work is the C inference kernel port and ESP-IDF integration.

---

## 12. Memory: 1D tensor shape handling

**File:** `src/model.c:344-354` (shape parser), `include/.../ops.hpp` (layer dispatch)

### Current state
The tensor index shape parser always reads up to 2 dimensions. 1D tensors
(e.g., layernorm weights `[4096]`) have `shape[1] = 0` (from `memset`).
The `TransformerRunner` assumes all weight tensors are 2D. 1D tensors
cause incorrect dispatch and potential crashes.

### Suggested fix
- Store tensor `rank` (1 or 2) alongside `shape[]` in `TbLayerDesc`.
- In the JSON parser, detect rank from how many dimension values were
  parsed. Store shape[1] = 1 for rank-1 tensors.
- In `TransformerRunner` ops, handle rank-1 tensors correctly (LayerNorm
  weight, RMSNorm weight are 1D, not 2D).

---

## 13. I/O: `--temp-dir` creates files in CWD on default

**File:** `src/main.cpp:364`

### Current state
The `--temp-dir` defaults to `"."` which creates test/model files in the
current working directory. If run from a read-only filesystem, this fails.

### Suggested fix
- Use `std::filesystem::temp_directory_path()` as described in item #7.
- Clean up temp files on exit with a RAII guard.

---

## 14. Windows: ANSI path limitation in CreateFileA

**File:** `src/model.c:36`

### Current state
On Windows, `CreateFileA` is used for file opening. This uses ANSI encoding
and can't handle Unicode paths or long paths (>260 chars). Modern Windows
systems support `\\?\` prefix for extended-length paths.

### Suggested fix
- Use `CreateFileW` with `MultiByteToWideChar(CP_UTF8, ...)` for Unicode
  path support.
- Or use `fopen` on Windows for simplicity (handles UTF-8 paths correctly
  on Windows 10 version 1903+).

---

## 15. Tests: Test files created in CWD

**File:** `tests/test_tbm.c:23, 233`

### Current state
`test_tbm.c` creates files (`test_model.tbm`, `test_full.tbm`) in the
current working directory. If tests are run from a read-only directory
or with unexpected CWD, they fail.

### Suggested fix
- Use `tmpfile()` or `mkstemp()` for temporary test files.
- Or create them in the build directory (which `test_all.sh` `cd`s into
  before running).

---

## 16. Tokenizer: Demo character-level tokenizer cannot encode real text

**File:** `src/main.cpp:21-73` — `SimpleTokenizer`

### Current state
The `SimpleTokenizer` class maps single ASCII characters to IDs with a
hardcoded vocabulary of ~50 tokens (26 lowercase letters + digits + a
few punctuation characters). It can tokenize prompts like `"hello"` for
demos but cannot encode arbitrary text (no uppercase, no unicode, no
subword units). Real models use 32,000–128,000 token vocabularies with
BPE/SentencePiece subword tokenization (Mistral 7B = 32,000 tokens).

The comment on line 22-23 explicitly says:
```
In production, use a proper BPE/SentencePiece tokenizer.
```

### Why this matters
Without a model-compatible tokenizer, the engine cannot perform real
inference — the token IDs produced by `SimpleTokenizer` have no relation
to the actual model vocabulary. Even if the .tbm loads correctly, token
generation produces garbage because token IDs don't map to the embeddings
the model was trained on.

### Suggested fix
- Integrate the HuggingFace `tokenizers` library (Rust bindings, pip-installable,
  provides BPE/WordPiece/SentencePiece/Unigram tokenizers).
- Or integrate `sentencepiece` C++ library directly (header-only, no Python dependency).
- Or integrate `tiktoken` (OpenAI's Rust+Python tokenizer, used by GPT models).
- Load the `tokenizer.json` or `tokenizer.model` file alongside the `.tbm` file.
- Add a `--tokenizer PATH` CLI flag to specify the tokenizer file.
- The tokenizer should support `encode(prompt)` → token IDs and `decode(token_ids)` → text.
- Fall back to `SimpleTokenizer` only when no tokenizer file is provided
  (for backward compatibility with demo/test mode).

---

## Summary

| # | Item | Category | Severity |
|---|------|----------|----------|
| 1 | Support non-Llama architectures | Inference | High |
| 2 | Windows mmap fallback | I/O | Medium |
| 3 | On-demand tensor loading | Memory | High |
| 4 | SIMD flags crash on old CPUs | Runtime | Medium |
| 5 | ARM64 NEON not implemented | SIMD | Medium |
| 6 | Inconsistent log formatting | Logging | Low |
| 7 | --temp-dir help/code mismatch | CLI | Low |
| 8 | Deterministic RNG (fixed) | Sampling | Low |
| 9 | Only FP32 natively (INT4/INT8 via get_weight_fp32) | Inference | Medium |
| 10 | CUDA registry untested | Backend | Medium |
| 11 | ESP32 scaffold incomplete | Platform | High |
| 12 | 1D tensor shape handling | Memory | Medium |
| 13 | Temp files in CWD | I/O | Low |
| 14 | Windows ANSI path limitation | I/O | Low |
| 15 | Test files in CWD | Testing | Low |
| 16 | Tokenizer: character-level demo, no BPE/SentencePiece | Inference | High |

**Last updated:** May 2026 — post-audit state.
