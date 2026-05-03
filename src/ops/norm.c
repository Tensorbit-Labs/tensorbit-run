#include "tensorbit/run/ops.h"
#include <math.h>

int tb_cpu_rms_norm_f32(size_t n, const float* TB_RESTRICT x, const float* TB_RESTRICT weight,
                         float* TB_RESTRICT y, float eps) {
    if (!x || !y) return TB_ERR_INVALID_ARG;

    float sum_sq = eps;
    for (size_t i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }

    float rms = 1.0f / sqrtf(sum_sq / (float)n);

    if (weight) {
        for (size_t i = 0; i < n; i++) {
            y[i] = x[i] * rms * weight[i];
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            y[i] = x[i] * rms;
        }
    }

    return TB_OK;
}

int tb_cpu_layer_norm_f32(size_t n, const float* TB_RESTRICT x, const float* TB_RESTRICT weight,
                           const float* TB_RESTRICT bias, float* TB_RESTRICT y, float eps) {
    if (!x || !y) return TB_ERR_INVALID_ARG;

    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) {
        mean += x[i];
    }
    mean /= (float)n;

    float var = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var = var / (float)n + eps;
    float inv_std = 1.0f / sqrtf(var);

    if (weight && bias) {
        for (size_t i = 0; i < n; i++) {
            y[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
        }
    } else if (weight) {
        for (size_t i = 0; i < n; i++) {
            y[i] = (x[i] - mean) * inv_std * weight[i];
        }
    } else if (bias) {
        for (size_t i = 0; i < n; i++) {
            y[i] = (x[i] - mean) * inv_std + bias[i];
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            y[i] = (x[i] - mean) * inv_std;
        }
    }

    return TB_OK;
}
