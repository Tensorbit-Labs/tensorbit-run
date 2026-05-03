#include "tensorbit/run/ops.h"
#include <string.h>

int tb_cpu_residual_add_f32(size_t n, const float* TB_RESTRICT x, const float* TB_RESTRICT residual,
                              float* TB_RESTRICT y) {
    if (!x || !residual || !y) return TB_ERR_INVALID_ARG;

    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] + residual[i];
    }

    return TB_OK;
}
