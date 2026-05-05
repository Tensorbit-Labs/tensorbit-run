/// @file registry.c
/// @brief CUDA backend registry — GPU dispatch wrappers with real kernels.
/// @ingroup tensorbit-run
///
/// When compiled with TENSORBIT_BACKEND_CUDA, this provides GPU-accelerated
/// inference by calling the CUDA kernels in `kernels.cu`.  Ops that don't
/// have GPU implementations (attention, embedding, conv2d, layer_norm,
/// softmax) fall back to CPU.

#include "tensorbit/run/backend.h"
#include "tensorbit/run/ops.h"

#include <string.h>

// CUDA kernel declarations (defined in kernels.cu, callable from C via C++ link)
// Inlined here as extern function prototypes — actual symbols resolved by linker.
int tb_cuda_sparse_linear_f32(float*, const float*, const float*, const uint8_t*,
                               size_t, size_t, int, int);
int tb_cuda_dense_linear_f32(float*, const float*, const float*, const float*,
                              size_t, size_t);
int tb_cuda_rms_norm_f32(float*, const float*, const float*, size_t, float);
int tb_cuda_gelu_f32(float*, const float*, size_t);
int tb_cuda_silu_f32(float*, const float*, size_t);
int tb_cuda_rope_f32(float*, float*, int, int, int, int, float);
int tb_cuda_residual_add_f32(float*, const float*, const float*, size_t);
int tb_cuda_softmax_f32(float*, const float*, size_t, size_t, const float*);
int tb_cuda_embedding_f32(float*, const int*, const float*, int, int);
int tb_cuda_layer_norm_f32(float*, const float*, const float*, const float*,
                            size_t, float);
int tb_cuda_conv2d_f32(float*, const float*, const float*, const float*,
                        int, int, int, int, int, int, int, int);
int tb_cuda_sdp_attention_f32(float*, const float*, const float*, const float*,
                               int, int, int, int, float, _Bool);

// ================================================================
// CUDA dispatch wrappers
// ================================================================

static int cuda_linear_s(void* y, const void** in, int n, const void* p) {
    if (n < 4) return TB_ERR_INVALID_ARG;
    const TbLinearParams* lp = (const TbLinearParams*)p;
    int nm_n = lp ? (lp->nm_n > 0 ? lp->nm_n : 2) : 2;
    int nm_m = lp ? (lp->nm_m > 0 ? lp->nm_m : 4) : 4;
    TbTensor*       yo = (TbTensor*)y;
    const TbTensor* xi = (const TbTensor*)in[0];
    const TbTensor* wi = (const TbTensor*)in[1];
    const TbTensor* mi = (const TbTensor*)in[2];
    return tb_cuda_sparse_linear_f32(
        tb_tensor_f32(yo), tb_tensor_cf32(xi), tb_tensor_cf32(wi),
        (const uint8_t*)mi->data, wi->shape[0], wi->shape[1], nm_n, nm_m);
}

static int cuda_linear_d(void* y, const void** in, int n, const void* p) {
    (void)p;
    if (n < 2) return TB_ERR_INVALID_ARG;
    TbTensor*       yo = (TbTensor*)y;
    const TbTensor* xi = (const TbTensor*)in[0];
    const TbTensor* wi = (const TbTensor*)in[1];
    const TbTensor* bi = (n >= 3 && in[2]) ? (const TbTensor*)in[2] : nullptr;
    /* Use y's shape for out_features to handle tied embeddings */
    size_t out_f = 1;
    for (int i = 0; i < yo->rank; i++) out_f *= yo->shape[i];
    return tb_cuda_dense_linear_f32(
        tb_tensor_f32(yo), tb_tensor_cf32(xi), tb_tensor_cf32(wi),
        bi ? tb_tensor_cf32(bi) : nullptr, out_f, wi->shape[1]);
}

static int cuda_rms_norm(void* y, const void** in, int n, const void* p) {
    (void)p;
    if (n < 1) return TB_ERR_INVALID_ARG;
    TbTensor*       yo  = (TbTensor*)y;
    const TbTensor* xi  = (const TbTensor*)in[0];
    const TbTensor* wi  = (n >= 2 && in[1]) ? (const TbTensor*)in[1] : nullptr;
    size_t elem = 1;
    for (int i = 0; i < xi->rank; i++) elem *= xi->shape[i];
    return tb_cuda_rms_norm_f32(tb_tensor_f32(yo), tb_tensor_cf32(xi),
                                 wi ? tb_tensor_cf32(wi) : nullptr, elem, 1e-5f);
}

static int cuda_gelu(void* y, const void** in, int n, const void* p) {
    (void)p;
    if (n < 1) return TB_ERR_INVALID_ARG;
    TbTensor*       yo = (TbTensor*)y;
    const TbTensor* xi = (const TbTensor*)in[0];
    size_t elem = 1;
    for (int i = 0; i < xi->rank; i++) elem *= xi->shape[i];
    return tb_cuda_gelu_f32(tb_tensor_f32(yo), tb_tensor_cf32(xi), elem);
}

static int cuda_silu(void* y, const void** in, int n, const void* p) {
    (void)p;
    if (n < 1) return TB_ERR_INVALID_ARG;
    TbTensor*       yo = (TbTensor*)y;
    const TbTensor* xi = (const TbTensor*)in[0];
    size_t elem = 1;
    for (int i = 0; i < xi->rank; i++) elem *= xi->shape[i];
    return tb_cuda_silu_f32(tb_tensor_f32(yo), tb_tensor_cf32(xi), elem);
}

static int cuda_rope(void* y, const void** in, int n, const void* p) {
    const TbRopeParams* rp = (const TbRopeParams*)p;
    if (!rp) return TB_ERR_INVALID_ARG;
    TbTensor* q = (TbTensor*)y;
    TbTensor* k = (TbTensor*)in[0];
    (void)n;
    return tb_cuda_rope_f32(tb_tensor_f32(q), tb_tensor_f32(k),
                             (int)q->shape[0], (int)k->shape[0], (int)q->shape[2],
                             rp->position, rp->theta);
}

static int cuda_residual(void* y, const void** in, int n, const void* p) {
    (void)p;
    if (n < 2) return TB_ERR_INVALID_ARG;
    TbTensor*       yo = (TbTensor*)y;
    const TbTensor* ai = (const TbTensor*)in[0];
    const TbTensor* bi = (const TbTensor*)in[1];
    size_t elem = 1;
    for (int i = 0; i < ai->rank; i++) elem *= ai->shape[i];
    return tb_cuda_residual_add_f32(tb_tensor_f32(yo), tb_tensor_cf32(ai),
                                     tb_tensor_cf32(bi), elem);
}

// Ops with GPU kernels
static int cuda_sdp(void* y, const void** in, int n, const void* p) {
    const TbSdpParams* sp = (const TbSdpParams*)p;
    TbTensor* yo = (TbTensor*)y;
    const TbTensor* qi = (const TbTensor*)in[0];
    const TbTensor* ki = (const TbTensor*)in[1];
    const TbTensor* vi = (const TbTensor*)in[2];
    (void)n;
    return tb_cuda_sdp_attention_f32(
        tb_tensor_f32(yo), tb_tensor_cf32(qi), tb_tensor_cf32(ki), tb_tensor_cf32(vi),
        (int)qi->shape[0], (int)qi->shape[1], (int)ki->shape[1], (int)qi->shape[2],
        sp ? sp->scale : 1.f, sp ? sp->causal : true);
}

static int cuda_softmax(void* y, const void** in, int n, const void* p) {
    (void)p;
    if (n < 1) return TB_ERR_INVALID_ARG;
    TbTensor* yo = (TbTensor*)y;
    const TbTensor* xi = (const TbTensor*)in[0];
    const TbTensor* mi = (n >= 2 && in[1]) ? (const TbTensor*)in[1] : nullptr;
    return tb_cuda_softmax_f32(tb_tensor_f32(yo), tb_tensor_cf32(xi),
                                xi->shape[0], xi->shape[1],
                                mi ? tb_tensor_cf32(mi) : nullptr);
}

static int cuda_embedding(void* y, const void** in, int n, const void* p) {
    (void)p;
    if (n < 2) return TB_ERR_INVALID_ARG;
    TbTensor* yo = (TbTensor*)y;
    const TbTensor* ti = (const TbTensor*)in[0];
    const TbTensor* ei = (const TbTensor*)in[1];
    return tb_cuda_embedding_f32(tb_tensor_f32(yo), (const int*)ti->data,
                                  tb_tensor_cf32(ei), (int)ti->shape[0], (int)ei->shape[1]);
}

static int cuda_layer_norm(void* y, const void** in, int n, const void* p) {
    (void)p;
    if (n < 1) return TB_ERR_INVALID_ARG;
    TbTensor* yo = (TbTensor*)y;
    const TbTensor* xi = (const TbTensor*)in[0];
    const TbTensor* wi = (n >= 2 && in[1]) ? (const TbTensor*)in[1] : nullptr;
    const TbTensor* bi = (n >= 3 && in[2]) ? (const TbTensor*)in[2] : nullptr;
    size_t elem = 1;
    for (int i = 0; i < xi->rank; i++) elem *= xi->shape[i];
    return tb_cuda_layer_norm_f32(tb_tensor_f32(yo), tb_tensor_cf32(xi),
                                   wi ? tb_tensor_cf32(wi) : nullptr,
                                   bi ? tb_tensor_cf32(bi) : nullptr, elem, 1e-5f);
}

static int cuda_conv2d(void* y, const void** in, int n, const void* p) {
    (void)p;
    if (n < 2) return TB_ERR_INVALID_ARG;
    TbTensor* yo = (TbTensor*)y;
    const TbTensor* xi = (const TbTensor*)in[0];
    const TbTensor* wi = (const TbTensor*)in[1];
    const TbTensor* bi = (n >= 3 && in[2]) ? (const TbTensor*)in[2] : nullptr;
    return tb_cuda_conv2d_f32(
        tb_tensor_f32(yo), tb_tensor_cf32(xi), tb_tensor_cf32(wi),
        bi ? tb_tensor_cf32(bi) : nullptr,
        (int)xi->shape[0], (int)xi->shape[1], (int)xi->shape[2], (int)xi->shape[3],
        (int)wi->shape[0], (int)wi->shape[2], 1, 0);
}

// ================================================================
// CUDA backend vtable
// ================================================================

static TbBackend g_cuda_backend;

int tb_backend_cuda_init(void) {
    TbBackend* be = &g_cuda_backend;
    memset(be, 0, sizeof(*be));

    be->name     = "cuda";
    be->priority = 100;

    be->ops[TB_OP_SPARSE_LINEAR]      = cuda_linear_s;
    be->ops[TB_OP_DENSE_LINEAR]       = cuda_linear_d;
    be->ops[TB_OP_RMS_NORM]           = cuda_rms_norm;
    be->ops[TB_OP_LAYER_NORM]         = cuda_layer_norm;
    be->ops[TB_OP_GELU]               = cuda_gelu;
    be->ops[TB_OP_SILU]               = cuda_silu;
    be->ops[TB_OP_SOFTMAX]            = cuda_softmax;
    be->ops[TB_OP_SCALED_DOT_PRODUCT] = cuda_sdp;
    be->ops[TB_OP_ROPE]               = cuda_rope;
    be->ops[TB_OP_EMBEDDING]          = cuda_embedding;
    be->ops[TB_OP_RESIDUAL_ADD]       = cuda_residual;
    be->ops[TB_OP_CONV2D]             = cuda_conv2d;
    be->ops[TB_OP_ATTENTION_QKV]      = NULL;

    return tb_backend_register(be);
}
