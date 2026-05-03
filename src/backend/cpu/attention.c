#include "tensorbit/run/backend.h"
#include "tensorbit/run/ops.h"

#include <math.h>

/* ================================================================
 * Scaled dot-product attention dispatch
 * inputs[0] = q, inputs[1] = k, inputs[2] = v, inputs[3] = mask (optional)
 * params = TbSdpParams
 * ================================================================ */

int tb_cpu_scaled_dot_product_dispatch(void* output, const void** inputs, int n_inputs,
                                        const void* params) {
    const TbSdpParams* sp = (const TbSdpParams*)params;

    TbTensor*       y = (TbTensor*)output;
    const TbTensor* q = (const TbTensor*)inputs[0];
    const TbTensor* k = (const TbTensor*)inputs[1];
    const TbTensor* v = (const TbTensor*)inputs[2];
    const TbTensor* mask = (n_inputs >= 4 && inputs[3]) ? (const TbTensor*)inputs[3] : NULL;

    int n_heads = (int)q->shape[0];
    /* q_seq_len may differ from kv_seq_len in decode (padded Q handles this) */
    int kv_seq_len = (int)k->shape[1];
    int head_dim = (int)q->shape[2];
    int seq_len = kv_seq_len;  /* use K's sequence length for KV dimension */

    bool  causal = sp ? sp->causal : true;
    float scale = sp ? sp->scale : (1.0f / sqrtf((float)head_dim));

    if (q->rank == 2) {
        seq_len = (int)q->shape[0] / n_heads;
    }

    return tb_cpu_scaled_dot_product_f32(n_heads, seq_len, head_dim, tb_tensor_cf32(q),
                                          tb_tensor_cf32(k), tb_tensor_cf32(v),
                                          mask ? tb_tensor_cf32(mask) : NULL, tb_tensor_f32(y),
                                          causal, scale);
}

/* ================================================================
 * QKV projection dispatch
 * Handled at a higher level via three separate sparse linear calls.
 * This stub exists for backend registration completeness.
 * ================================================================ */

int tb_cpu_attention_qkv_dispatch(void* output, const void** inputs, int n_inputs,
                                   const void* params) {
    (void)output;
    (void)inputs;
    (void)n_inputs;
    (void)params;
    return TB_OK;
}
