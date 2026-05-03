#include "tensorbit/run/ops.h"
#include <math.h>
#include <string.h>

int tb_cpu_scaled_dot_product_f32(int n_heads, int seq_len, int head_dim,
                                   const float* TB_RESTRICT q, const float* TB_RESTRICT k,
                                   const float* TB_RESTRICT v, const float* TB_RESTRICT mask,
                                   float* TB_RESTRICT y, bool causal, float scale) {
    if (!q || !k || !v || !y) return TB_ERR_INVALID_ARG;

    size_t head_stride = (size_t)seq_len * head_dim;

    for (int h = 0; h < n_heads; h++) {
        const float* qh = q + h * head_stride;
        const float* kh = k + h * head_stride;
        const float* vh = v + h * head_stride;
        float*       yh = y + h * head_stride;

        float* scores = (float*)tb_malloc(sizeof(float) * seq_len * seq_len);
        if (!scores) return TB_ERR_OOM;

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    s += qh[i * head_dim + d] * kh[j * head_dim + d];
                }
                s *= scale;

                if (mask) s += mask[i * seq_len + j];
                if (causal && j > i) s = -INFINITY;

                scores[i * seq_len + j] = s;
            }
        }

        for (int i = 0; i < seq_len; i++) {
            float max_val = -INFINITY;
            for (int j = 0; j < seq_len; j++) {
                if (scores[i * seq_len + j] > max_val) max_val = scores[i * seq_len + j];
            }

            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                float e = expf(scores[i * seq_len + j] - max_val);
                scores[i * seq_len + j] = e;
                sum += e;
            }

            float inv_sum = 1.0f / sum;
            for (int j = 0; j < seq_len; j++) {
                scores[i * seq_len + j] *= inv_sum;
            }
        }

        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    acc += scores[i * seq_len + j] * vh[j * head_dim + d];
                }
                yh[i * head_dim + d] = acc;
            }
        }

        tb_free(scores);
    }

    return TB_OK;
}
