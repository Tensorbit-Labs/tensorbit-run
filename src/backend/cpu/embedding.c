#include "tensorbit/run/backend.h"
#include "tensorbit/run/ops.h"

int tb_cpu_embedding_dispatch(void* output, const void** inputs, int n_inputs,
                               const void* params) {
    (void)params;
    const int* token_ids = (const int*)inputs[0];
    int        n_tokens = n_inputs > 1 ? *(const int*)inputs[1] : 1;
    const TbTensor* table = (const TbTensor*)inputs[2];

    return tb_cpu_embedding_f32(token_ids, n_tokens, (int)table->shape[1], tb_tensor_cf32(table),
                                 tb_tensor_f32((TbTensor*)output));
}
