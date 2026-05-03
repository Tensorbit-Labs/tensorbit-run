#include "tensorbit/run/backend.h"
#include "tensorbit/run/ops.h"

int tb_cpu_sparse_linear_dispatch(void* output, const void** inputs, int n_inputs,
                                   const void* params) {
    if (n_inputs < 4) return TB_ERR_INVALID_ARG;

    TbTensor*       y = (TbTensor*)output;
    const TbTensor* x = (const TbTensor*)inputs[0];
    const TbTensor* w = (const TbTensor*)inputs[1];
    const TbTensor* mask = (const TbTensor*)inputs[2];
    const TbTensor* bias = (const TbTensor*)inputs[3];

    size_t out_features = w->shape[0];
    size_t in_features = w->shape[1];

    /* Read N:M from params (TbLinearParams), not hardcoded */
    const TbLinearParams* lp = (const TbLinearParams*)params;
    int nm_n = 2;
    int nm_m = 4;
    if (lp) {
        nm_n = lp->nm_n > 0 ? lp->nm_n : 2;
        nm_m = lp->nm_m > 0 ? lp->nm_m : 4;
    }

    return tb_cpu_sparse_linear_f32(out_features, in_features, tb_tensor_cf32(x),
                                     tb_tensor_cf32(w), (const uint8_t*)mask->data,
                                     bias ? tb_tensor_cf32(bias) : NULL, tb_tensor_f32(y), nm_n,
                                     nm_m);
}

int tb_cpu_dense_linear_dispatch(void* output, const void** inputs, int n_inputs,
                                   const void* params) {
    (void)params;
    if (n_inputs < 2) return TB_ERR_INVALID_ARG;

    TbTensor*       y = (TbTensor*)output;
    const TbTensor* x = (const TbTensor*)inputs[0];
    const TbTensor* w = (const TbTensor*)inputs[1];
    const TbTensor* bias = (n_inputs >= 3 && inputs[2]) ? (const TbTensor*)inputs[2] : NULL;

    /* Use y's first dimension for out_features — handles tied embeddings
       where w has more rows than the vocabulary size. */
    size_t out_features = 1;
    for (int i = 0; i < y->rank; i++) out_features *= y->shape[i];

    return tb_cpu_dense_linear_f32(out_features, w->shape[1], tb_tensor_cf32(x), tb_tensor_cf32(w),
                                    bias ? tb_tensor_cf32(bias) : NULL, tb_tensor_f32(y));
}
