#include "tensorbit/run/ops.h"
#include <math.h>

int tb_cpu_softmax_f32(size_t n, const float* TB_RESTRICT x, const float* TB_RESTRICT mask,
                        float* TB_RESTRICT y) {
    if (!x || !y) return TB_ERR_INVALID_ARG;

    float max_val = -INFINITY;
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        if (mask) v += mask[i];
        if (v > max_val) max_val = v;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        if (mask) v += mask[i];
        float e = expf(v - max_val);
        y[i] = e;
        sum += e;
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < n; i++) {
            y[i] *= inv_sum;
        }
    }

    return TB_OK;
}
