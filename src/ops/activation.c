#include "tensorbit/run/ops.h"
#include <math.h>

int tb_cpu_gelu_f32(size_t n, const float* TB_RESTRICT x, float* TB_RESTRICT y) {
    if (!x || !y) return TB_ERR_INVALID_ARG;

    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        float v3 = v * v * v;
        float inner = TB_SQRT_2_PI * (v + 0.044715f * v3);
        float tanh_val = tanhf(inner);
        y[i] = 0.5f * v * (1.0f + tanh_val);
    }

    return TB_OK;
}

int tb_cpu_silu_f32(size_t n, const float* TB_RESTRICT x, float* TB_RESTRICT y) {
    if (!x || !y) return TB_ERR_INVALID_ARG;

    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }

    return TB_OK;
}
