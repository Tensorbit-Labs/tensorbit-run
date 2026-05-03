#include "tensorbit/run/backend.h"
#include "tensorbit/run/ops.h"

int tb_cpu_rms_norm_dispatch(void* output, const void** inputs, int n_inputs, const void* params) {
    const TbNormParams* p = (const TbNormParams*)params;
    float eps = p ? p->eps : 1e-6f;

    TbTensor*       y = (TbTensor*)output;
    const TbTensor* x = (const TbTensor*)inputs[0];
    const TbTensor* weight = (n_inputs >= 2) ? (const TbTensor*)inputs[1] : NULL;

    return tb_cpu_rms_norm_f32(tb_tensor_nelem(x), tb_tensor_cf32(x),
                                weight ? tb_tensor_cf32(weight) : NULL, tb_tensor_f32(y), eps);
}

int tb_cpu_layer_norm_dispatch(void* output, const void** inputs, int n_inputs, const void* params) {
    const TbNormParams* p = (const TbNormParams*)params;
    float eps = p ? p->eps : 1e-5f;

    TbTensor*       y = (TbTensor*)output;
    const TbTensor* x = (const TbTensor*)inputs[0];
    const TbTensor* weight = (n_inputs >= 2) ? (const TbTensor*)inputs[1] : NULL;
    const TbTensor* bias = (n_inputs >= 3) ? (const TbTensor*)inputs[2] : NULL;

    return tb_cpu_layer_norm_f32(tb_tensor_nelem(x), tb_tensor_cf32(x),
                                  weight ? tb_tensor_cf32(weight) : NULL,
                                  bias ? tb_tensor_cf32(bias) : NULL, tb_tensor_f32(y), eps);
}
