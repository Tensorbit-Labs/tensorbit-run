#ifndef TENSORBIT_RUN_BACKEND_H
#define TENSORBIT_RUN_BACKEND_H

#include "tensorbit/run/common.h"
#include "tensorbit/run/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Operation types
 * ================================================================ */

typedef enum {
    TB_OP_SPARSE_LINEAR = 0,
    TB_OP_DENSE_LINEAR,
    TB_OP_RMS_NORM,
    TB_OP_LAYER_NORM,
    TB_OP_GELU,
    TB_OP_SILU,
    TB_OP_SOFTMAX,
    TB_OP_ATTENTION_QKV,
    TB_OP_SCALED_DOT_PRODUCT,
    TB_OP_ROPE,
    TB_OP_RESIDUAL_ADD,
    TB_OP_EMBEDDING,
    TB_OP_CONV2D,
    TB_OP_COUNT
} TbOpType;

/* ================================================================
 * Operation parameter structs
 * ================================================================ */

typedef struct {
    int num_heads;
    int num_kv_heads;
    int head_dim;
} TbQkvParams;

typedef struct {
    float eps;
} TbNormParams;

typedef struct {
    float scale;
    bool  causal;
    int   seq_len;
} TbSdpParams;

typedef struct {
    int   position;
    float theta;
    int   head_dim;
} TbRopeParams;

typedef struct {
    int stride;
    int padding;
} TbConv2dParams;

/* ================================================================
 * Generic op dispatch function pointer
 * All ops use this uniform signature with type-erased pointers.
 * Each op knows the real types and casts internally.
 * ================================================================ */

typedef int (*TbOpFn)(void* output, const void** inputs, int n_inputs, const void* params);

/* ================================================================
 * Backend
 * ================================================================ */

typedef struct {
    const char* name;
    int         priority;
    TbOpFn      ops[TB_OP_COUNT];
} TbBackend;

/* ================================================================
 * Backend registry
 * ================================================================ */

/* Max registered backends */
#define TB_MAX_BACKENDS 8

/* Register a backend. Returns 0 on success. */
int tb_backend_register(const TbBackend* backend);

/* Get the best backend for a given op. Returns NULL if none found. */
const TbBackend* tb_backend_get_best(TbOpType op);

/* Get number of registered backends */
int tb_backend_count(void);

/* Get backend by index */
const TbBackend* tb_backend_get(int index);

/* ================================================================
 * Convenience dispatch
 * ================================================================ */

int tb_dispatch_sparse_linear(TbTensor* y, const TbTensor* x,
                               const TbTensor* w, const TbTensor* mask,
                               const TbTensor* bias);

int tb_dispatch_dense_linear(TbTensor* y, const TbTensor* x,
                              const TbTensor* w, const TbTensor* bias);

int tb_dispatch_rms_norm(TbTensor* y, const TbTensor* x,
                          const TbTensor* weight, float eps);

int tb_dispatch_layer_norm(TbTensor* y, const TbTensor* x,
                            const TbTensor* weight, const TbTensor* bias, float eps);

int tb_dispatch_gelu(TbTensor* y, const TbTensor* x);
int tb_dispatch_silu(TbTensor* y, const TbTensor* x);

int tb_dispatch_softmax(TbTensor* y, const TbTensor* x, const TbTensor* mask);

int tb_dispatch_attention_qkv(TbTensor* q, TbTensor* k, TbTensor* v,
                               const TbTensor* x,
                               const TbTensor* wq, const TbTensor* wk, const TbTensor* wv,
                               const TbTensor* mq, const TbTensor* mk, const TbTensor* mv,
                               int num_heads, int num_kv_heads, int head_dim);

int tb_dispatch_scaled_dot_product(TbTensor* y,
                                    const TbTensor* q, const TbTensor* k, const TbTensor* v,
                                    const TbTensor* mask, float scale, bool causal);

int tb_dispatch_rope(TbTensor* q, TbTensor* k, int position, float theta, int head_dim);

int tb_dispatch_residual_add(TbTensor* y, const TbTensor* x, const TbTensor* residual);

int tb_dispatch_embedding(TbTensor* y, const int* token_ids, int n_tokens,
                           const TbTensor* embedding);

int tb_dispatch_conv2d(TbTensor* y, const TbTensor* x, const TbTensor* kernel,
                        const TbTensor* bias, int stride, int padding);

/* Initialize the CPU backend (always available) */
int tb_backend_cpu_init(void);

/* Initialize the CUDA backend (requires TB_HAS_CUDA) */
#ifdef TB_HAS_CUDA
int tb_backend_cuda_init(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* TENSORBIT_RUN_BACKEND_H */
