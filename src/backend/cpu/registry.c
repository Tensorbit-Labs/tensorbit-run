#include "tensorbit/run/backend.h"
#include "tensorbit/run/ops.h"

#include <string.h>

/* ================================================================
 * Declare dispatch functions from other CPU backend files
 * ================================================================ */

extern int tb_cpu_sparse_linear_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_dense_linear_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_rms_norm_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_layer_norm_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_gelu_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_silu_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_softmax_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_attention_qkv_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_scaled_dot_product_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_rope_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_residual_add_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_embedding_dispatch(void*, const void**, int, const void*);
extern int tb_cpu_conv2d_dispatch(void*, const void**, int, const void*);

/* ================================================================
 * Backend registry — simple static array
 * ================================================================ */

static TbBackend g_backends[TB_MAX_BACKENDS];
static int       g_num_backends = 0;

int tb_backend_register(const TbBackend* backend) {
    if (g_num_backends >= TB_MAX_BACKENDS) return -1;
    g_backends[g_num_backends++] = *backend;
    return 0;
}

const TbBackend* tb_backend_get_best(TbOpType op) {
    const TbBackend* best = NULL;
    int              best_prio = -1;

    for (int i = 0; i < g_num_backends; i++) {
        if (g_backends[i].ops[op] != NULL) {
            if (g_backends[i].priority > best_prio) {
                best_prio = g_backends[i].priority;
                best = &g_backends[i];
            }
        }
    }
    return best;
}

int tb_backend_count(void) { return g_num_backends; }

const TbBackend* tb_backend_get(int index) {
    if (index < 0 || index >= g_num_backends) return NULL;
    return &g_backends[index];
}

/* ================================================================
 * CPU backend initialization
 * ================================================================ */

int tb_backend_cpu_init(void) {
    TbBackend cpu_backend;
    memset(&cpu_backend, 0, sizeof(cpu_backend));
    cpu_backend.name = "cpu";
    cpu_backend.priority = 0;

    cpu_backend.ops[TB_OP_SPARSE_LINEAR] = tb_cpu_sparse_linear_dispatch;
    cpu_backend.ops[TB_OP_DENSE_LINEAR] = tb_cpu_dense_linear_dispatch;
    cpu_backend.ops[TB_OP_RMS_NORM] = tb_cpu_rms_norm_dispatch;
    cpu_backend.ops[TB_OP_LAYER_NORM] = tb_cpu_layer_norm_dispatch;
    cpu_backend.ops[TB_OP_GELU] = tb_cpu_gelu_dispatch;
    cpu_backend.ops[TB_OP_SILU] = tb_cpu_silu_dispatch;
    cpu_backend.ops[TB_OP_SOFTMAX] = tb_cpu_softmax_dispatch;
    cpu_backend.ops[TB_OP_ATTENTION_QKV] = tb_cpu_attention_qkv_dispatch;
    cpu_backend.ops[TB_OP_SCALED_DOT_PRODUCT] = tb_cpu_scaled_dot_product_dispatch;
    cpu_backend.ops[TB_OP_ROPE] = tb_cpu_rope_dispatch;
    cpu_backend.ops[TB_OP_RESIDUAL_ADD] = tb_cpu_residual_add_dispatch;
    cpu_backend.ops[TB_OP_EMBEDDING] = tb_cpu_embedding_dispatch;
    cpu_backend.ops[TB_OP_CONV2D] = tb_cpu_conv2d_dispatch;

    return tb_backend_register(&cpu_backend);
}

/* ================================================================
 * Convenience dispatch — assembly of params, backend lookup
 * ================================================================ */

int tb_dispatch_sparse_linear(TbTensor* y, const TbTensor* x, const TbTensor* w,
                               const TbTensor* mask, const TbTensor* bias) {
    const TbBackend* be = tb_backend_get_best(TB_OP_SPARSE_LINEAR);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    const void* arr[] = {x, w, mask, bias};
    return be->ops[TB_OP_SPARSE_LINEAR](y, arr, 4, NULL);
}

int tb_dispatch_dense_linear(TbTensor* y, const TbTensor* x, const TbTensor* w,
                              const TbTensor* bias) {
    const TbBackend* be = tb_backend_get_best(TB_OP_DENSE_LINEAR);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    const void* arr[] = {x, w, bias};
    return be->ops[TB_OP_DENSE_LINEAR](y, arr, 3, NULL);
}

int tb_dispatch_rms_norm(TbTensor* y, const TbTensor* x, const TbTensor* weight, float eps) {
    const TbBackend* be = tb_backend_get_best(TB_OP_RMS_NORM);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    TbNormParams p = {eps};
    const void*  arr[] = {x, weight};
    return be->ops[TB_OP_RMS_NORM](y, arr, 2, &p);
}

int tb_dispatch_layer_norm(TbTensor* y, const TbTensor* x, const TbTensor* weight,
                            const TbTensor* bias, float eps) {
    const TbBackend* be = tb_backend_get_best(TB_OP_LAYER_NORM);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    TbNormParams p = {eps};
    const void*  arr[] = {x, weight, bias};
    return be->ops[TB_OP_LAYER_NORM](y, arr, 3, &p);
}

int tb_dispatch_gelu(TbTensor* y, const TbTensor* x) {
    const TbBackend* be = tb_backend_get_best(TB_OP_GELU);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    const void* arr[] = {x};
    return be->ops[TB_OP_GELU](y, arr, 1, NULL);
}

int tb_dispatch_silu(TbTensor* y, const TbTensor* x) {
    const TbBackend* be = tb_backend_get_best(TB_OP_SILU);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    const void* arr[] = {x};
    return be->ops[TB_OP_SILU](y, arr, 1, NULL);
}

int tb_dispatch_softmax(TbTensor* y, const TbTensor* x, const TbTensor* mask) {
    const TbBackend* be = tb_backend_get_best(TB_OP_SOFTMAX);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    const void* arr[] = {x, mask};
    return be->ops[TB_OP_SOFTMAX](y, arr, 2, NULL);
}

int tb_dispatch_attention_qkv(TbTensor* q, TbTensor* k, TbTensor* v, const TbTensor* x,
                               const TbTensor* wq, const TbTensor* wk, const TbTensor* wv,
                               const TbTensor* mq, const TbTensor* mk, const TbTensor* mv,
                               int num_heads, int num_kv_heads, int head_dim) {
    int ret;
    ret = tb_dispatch_sparse_linear(q, x, wq, mq, NULL);
    if (ret) return ret;
    ret = tb_dispatch_sparse_linear(k, x, wk, mk, NULL);
    if (ret) return ret;
    ret = tb_dispatch_sparse_linear(v, x, wv, mv, NULL);
    if (ret) return ret;
    (void)num_heads;
    (void)num_kv_heads;
    (void)head_dim;
    return TB_OK;
}

int tb_dispatch_scaled_dot_product(TbTensor* y, const TbTensor* q, const TbTensor* k,
                                    const TbTensor* v, const TbTensor* mask, float scale,
                                    bool causal) {
    const TbBackend* be = tb_backend_get_best(TB_OP_SCALED_DOT_PRODUCT);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    TbSdpParams  p = {scale, causal, 0};
    const void*  arr[] = {q, k, v, mask};
    return be->ops[TB_OP_SCALED_DOT_PRODUCT](y, arr, 4, &p);
}

int tb_dispatch_rope(TbTensor* q, TbTensor* k, int position, float theta, int head_dim) {
    const TbBackend* be = tb_backend_get_best(TB_OP_ROPE);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    TbRopeParams p = {position, theta, head_dim};
    const void*  arr[] = {k};
    return be->ops[TB_OP_ROPE](q, arr, 1, &p);
}

int tb_dispatch_residual_add(TbTensor* y, const TbTensor* x, const TbTensor* residual) {
    const TbBackend* be = tb_backend_get_best(TB_OP_RESIDUAL_ADD);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    const void* arr[] = {x, residual};
    return be->ops[TB_OP_RESIDUAL_ADD](y, arr, 2, NULL);
}

int tb_dispatch_embedding(TbTensor* y, const int* token_ids, int n_tokens,
                           const TbTensor* embedding) {
    const TbBackend* be = tb_backend_get_best(TB_OP_EMBEDDING);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    const void* arr[] = {token_ids, &n_tokens, embedding};
    return be->ops[TB_OP_EMBEDDING](y, arr, 3, NULL);
}

int tb_dispatch_conv2d(TbTensor* y, const TbTensor* x, const TbTensor* kernel,
                        const TbTensor* bias, int stride, int padding) {
    const TbBackend* be = tb_backend_get_best(TB_OP_CONV2D);
    if (!be) return TB_ERR_BACKEND_NOT_FOUND;
    TbConv2dParams p = {stride, padding};
    const void*    arr[] = {x, kernel, bias};
    return be->ops[TB_OP_CONV2D](y, arr, 3, &p);
}
