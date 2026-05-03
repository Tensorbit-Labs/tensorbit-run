#include "tensorbit/run/backend.h"
#include "tensorbit/run/ops.h"

int tb_cpu_rope_dispatch(void* output, const void** inputs, int n_inputs, const void* params) {
    const TbRopeParams* rp = (const TbRopeParams*)params;
    if (!rp) return TB_ERR_INVALID_ARG;

    TbTensor* q = (TbTensor*)output;
    TbTensor* k = (TbTensor*)inputs[0];

    int n_q_heads = (int)q->shape[0];
    int n_kv_heads = (int)k->shape[0];
    int head_dim = (int)q->shape[2];

    (void)n_inputs;

    return tb_cpu_rope_gqa_f32(n_q_heads, n_kv_heads, head_dim,
                               tb_tensor_f32(q), tb_tensor_f32(k),
                               rp->position, rp->theta);
}
