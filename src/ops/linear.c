#include "tensorbit/run/ops.h"
#include <math.h>

int tb_cpu_sparse_linear_f32(size_t out_features, size_t in_features,
                              const float* TB_RESTRICT x,
                              const float* TB_RESTRICT w,
                              const uint8_t* TB_RESTRICT mask,
                              const float* TB_RESTRICT bias,
                              float* TB_RESTRICT y, int nm_n, int nm_m) {
    if (!x || !w || !mask || !y) return TB_ERR_INVALID_ARG;
    if (nm_m < 2 || nm_n >= nm_m) return TB_ERR_INVALID_ARG;

    size_t groups_per_row = in_features / nm_m;

    for (size_t j = 0; j < out_features; j++) {
        float acc = bias ? bias[j] : 0.0f;
        const float*     w_row = w + j * in_features;
        const uint8_t*   m_row = mask + j * groups_per_row;

        for (size_t g = 0; g < groups_per_row; g++) {
            uint8_t mbyte = m_row[g];
            for (int k = 0; k < nm_m; k++) {
                if (mbyte & (uint8_t)(1u << k)) {
                    size_t idx = g * nm_m + k;
                    acc += x[idx] * w_row[idx];
                }
            }
        }
        y[j] = acc;
    }

    return TB_OK;
}

int tb_cpu_dense_linear_f32(size_t out_features, size_t in_features,
                             const float* TB_RESTRICT x,
                             const float* TB_RESTRICT w,
                             const float* TB_RESTRICT bias,
                             float* TB_RESTRICT y) {
    if (!x || !w || !y) return TB_ERR_INVALID_ARG;

    for (size_t j = 0; j < out_features; j++) {
        float acc = bias ? bias[j] : 0.0f;
        const float* w_row = w + j * in_features;

        for (size_t i = 0; i < in_features; i++) {
            acc += x[i] * w_row[i];
        }
        y[j] = acc;
    }

    return TB_OK;
}
