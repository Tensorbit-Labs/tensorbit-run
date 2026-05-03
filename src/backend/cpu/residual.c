#include "tensorbit/run/backend.h"
#include "tensorbit/run/ops.h"

int tb_cpu_residual_add_dispatch(void* output, const void** inputs, int n_inputs,
                                  const void* params) {
    (void)params;
    (void)n_inputs;
    TbTensor*       y = (TbTensor*)output;
    const TbTensor* x = (const TbTensor*)inputs[0];
    const TbTensor* residual = (const TbTensor*)inputs[1];

    return tb_cpu_residual_add_f32(tb_tensor_nelem(x), tb_tensor_cf32(x), tb_tensor_cf32(residual),
                                    tb_tensor_f32(y));
}
