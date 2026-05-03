#include "tensorbit/run/backend.h"
#include "tensorbit/run/ops.h"

int tb_cpu_softmax_dispatch(void* output, const void** inputs, int n_inputs, const void* params) {
    (void)params;
    TbTensor*       y = (TbTensor*)output;
    const TbTensor* x = (const TbTensor*)inputs[0];
    const TbTensor* mask = (n_inputs >= 2) ? (const TbTensor*)inputs[1] : NULL;
    return tb_cpu_softmax_f32(tb_tensor_nelem(x), tb_tensor_cf32(x),
                               mask ? tb_tensor_cf32(mask) : NULL, tb_tensor_f32(y));
}
