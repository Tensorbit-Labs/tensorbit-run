#include "tensorbit/run/backend.h"
#include "tensorbit/run/ops.h"

int tb_cpu_gelu_dispatch(void* output, const void** inputs, int n_inputs, const void* params) {
    (void)n_inputs;
    (void)params;
    TbTensor*       y = (TbTensor*)output;
    const TbTensor* x = (const TbTensor*)inputs[0];
    return tb_cpu_gelu_f32(tb_tensor_nelem(x), tb_tensor_cf32(x), tb_tensor_f32(y));
}

int tb_cpu_silu_dispatch(void* output, const void** inputs, int n_inputs, const void* params) {
    (void)n_inputs;
    (void)params;
    TbTensor*       y = (TbTensor*)output;
    const TbTensor* x = (const TbTensor*)inputs[0];
    return tb_cpu_silu_f32(tb_tensor_nelem(x), tb_tensor_cf32(x), tb_tensor_f32(y));
}
